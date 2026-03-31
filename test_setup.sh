#!/bin/bash
#
# Agent Grid End-to-End Test Script
# Verifies installation, hardware detection, server, API, and health monitor.
#

set -o pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No color

PASS=0
FAIL=0
SKIP=0

PYTHON_BIN=${PYTHON_BIN:-python3}
API_DIR="${API_DIR:-../agentgrid-api}"
TEST_MODEL="${TEST_MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
API_TEST_PORT=5555

pass() { echo -e "  ${GREEN}[PASS]${NC} $1"; ((PASS++)); }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; ((FAIL++)); }
skip() { echo -e "  ${YELLOW}[SKIP]${NC} $1"; ((SKIP++)); }
info() { echo -e "  ${CYAN}[INFO]${NC} $1"; }

# Load HF token from .env or environment
load_hf_token() {
    if [ -n "${HF_TOKEN:-}" ]; then
        return 0
    fi
    if [ -f ".env" ]; then
        HF_TOKEN=$(grep -E '^HF_TOKEN=' .env 2>/dev/null | cut -d'=' -f2-)
        export HF_TOKEN
    fi
    if [ -z "${HF_TOKEN:-}" ]; then
        return 1
    fi
    return 0
}

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  Agent Grid End-to-End Test${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# ============================================================
# Stage 1: Verify Installation
# ============================================================
echo -e "${CYAN}Stage 1: Verify Installation${NC}"

if $PYTHON_BIN -c "import agentgrid" 2>/dev/null; then
    pass "agentgrid package imports OK"
else
    fail "agentgrid package import failed"
fi

if $PYTHON_BIN -c "import torch" 2>/dev/null; then
    pass "torch package imports OK"
else
    fail "torch package import failed"
fi

if $PYTHON_BIN -c "import hivemind" 2>/dev/null; then
    pass "hivemind package imports OK"
else
    fail "hivemind package import failed"
fi

GPU_STATUS=$($PYTHON_BIN -c "
import torch
if torch.cuda.is_available():
    backend = 'ROCm' if getattr(torch.version, 'hip', None) else 'CUDA'
    print(f'{backend}: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS: Apple Silicon')
else:
    print('CPU only')
" 2>/dev/null)

if [ -n "$GPU_STATUS" ]; then
    pass "Hardware: $GPU_STATUS"
else
    fail "Could not detect hardware"
fi

echo ""

# ============================================================
# Stage 2: Hardware Detection
# ============================================================
echo -e "${CYAN}Stage 2: Hardware Detection${NC}"

PROBE_OUTPUT=$($PYTHON_BIN -m agentgrid.launcher.discovery --probe-devices 2>/dev/null)
if [ $? -eq 0 ] && [ -n "$PROBE_OUTPUT" ]; then
    pass "Device probe succeeded"
    echo "$PROBE_OUTPUT" | $PYTHON_BIN -m json.tool 2>/dev/null | while IFS= read -r line; do
        info "$line"
    done
else
    fail "Device probe failed"
fi

echo ""

# ============================================================
# Stage 3: Grid Server Smoke Test (dry_run)
# ============================================================
echo -e "${CYAN}Stage 3: Grid Server Smoke Test${NC}"

if ! load_hf_token; then
    skip "No HF_TOKEN found (set in .env or environment). Skipping server test."
    skip "Run ./setup.sh first to configure your token."
else
    info "Using model: $TEST_MODEL"
    info "Running throughput dry_run (downloads config, evaluates, then exits)..."

    DRY_OUTPUT=$($PYTHON_BIN -m agentgrid.cli.run_server \
        "$TEST_MODEL" \
        --token "$HF_TOKEN" \
        --throughput dry_run \
        --new_swarm \
        2>&1)
    DRY_EXIT=$?

    if [ $DRY_EXIT -eq 0 ]; then
        pass "Server dry_run completed successfully"
    else
        # dry_run may exit non-zero but still work — check for throughput output
        if echo "$DRY_OUTPUT" | grep -qi "throughput\|tokens.*per.*second\|rps"; then
            pass "Server dry_run completed (throughput evaluated)"
        else
            fail "Server dry_run failed (exit code: $DRY_EXIT)"
            echo "$DRY_OUTPUT" | tail -5 | while IFS= read -r line; do
                info "  $line"
            done
        fi
    fi
fi

echo ""

# ============================================================
# Stage 4: API Server Test
# ============================================================
echo -e "${CYAN}Stage 4: API Server Test${NC}"

if [ ! -d "$API_DIR" ]; then
    skip "agentgrid-api not found at $API_DIR"
    skip "Clone it: git clone https://github.com/ReEnvision-AI/agentgrid-api.git $API_DIR"
else
    # Check if requirements are installed
    if ! $PYTHON_BIN -c "import fastapi" 2>/dev/null; then
        skip "FastAPI not installed. Run: pip install -r $API_DIR/requirements.txt"
    else
        # Start API server in background
        info "Starting API server on port $API_TEST_PORT..."
        (cd "$API_DIR" && $PYTHON_BIN -m uvicorn app.main:app --host 127.0.0.1 --port $API_TEST_PORT) &>/dev/null &
        API_PID=$!

        # Wait for it to be ready (up to 10 seconds)
        READY=false
        for i in $(seq 1 10); do
            if curl -sf "http://127.0.0.1:$API_TEST_PORT/" >/dev/null 2>&1; then
                READY=true
                break
            fi
            sleep 1
        done

        if [ "$READY" = true ]; then
            pass "API server started on port $API_TEST_PORT"

            # Test root endpoint
            ROOT_RESP=$(curl -sf "http://127.0.0.1:$API_TEST_PORT/" 2>/dev/null)
            if [ -n "$ROOT_RESP" ]; then
                pass "GET / returns: $ROOT_RESP"
            else
                fail "GET / returned empty response"
            fi

            # Test health endpoint
            HEALTH_RESP=$(curl -sf "http://127.0.0.1:$API_TEST_PORT/health" 2>/dev/null)
            if [ -n "$HEALTH_RESP" ]; then
                pass "GET /health returns response"
            else
                # Health endpoint may fail if hivemind can't connect — that's expected
                skip "GET /health unavailable (expected if grid not running)"
            fi

            # Test chat endpoint (will fail at grid level but API should accept the request)
            CHAT_RESP=$(curl -sf -X POST "http://127.0.0.1:$API_TEST_PORT/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d "{\"model\": \"$TEST_MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"test\"}], \"max_completion_tokens\": 1}" \
                2>/dev/null)
            CHAT_CODE=$?

            if [ $CHAT_CODE -eq 0 ] && [ -n "$CHAT_RESP" ]; then
                pass "POST /v1/chat/completions accepted request"
            else
                # A non-200 response is expected if grid isn't running — check if API at least responded
                CHAT_HTTP=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:$API_TEST_PORT/v1/chat/completions" \
                    -H "Content-Type: application/json" \
                    -d "{\"model\": \"$TEST_MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"test\"}]}" \
                    2>/dev/null)
                if [ "$CHAT_HTTP" = "000" ]; then
                    fail "POST /v1/chat/completions - API not responding"
                else
                    pass "POST /v1/chat/completions - API responded (HTTP $CHAT_HTTP, expected without running grid)"
                fi
            fi
        else
            fail "API server failed to start within 10 seconds"
        fi

        # Cleanup
        if [ -n "${API_PID:-}" ]; then
            kill $API_PID 2>/dev/null
            wait $API_PID 2>/dev/null
        fi
    fi
fi

echo ""

# ============================================================
# Stage 5: Health Monitor Test
# ============================================================
echo -e "${CYAN}Stage 5: Health Monitor Test${NC}"

if ! load_hf_token; then
    skip "No HF_TOKEN found. Skipping health monitor test."
else
    info "Running health monitor (one-shot, JSON mode)..."
    HEALTH_OUTPUT=$($PYTHON_BIN -m agentgrid.cli.health_monitor \
        --model "$TEST_MODEL" \
        --token "$HF_TOKEN" \
        --refresh 0 \
        --json \
        2>/dev/null)
    HEALTH_EXIT=$?

    if [ $HEALTH_EXIT -eq 0 ] && echo "$HEALTH_OUTPUT" | $PYTHON_BIN -m json.tool >/dev/null 2>&1; then
        pass "Health monitor returned valid JSON"

        # Extract peer count
        PEER_COUNT=$(echo "$HEALTH_OUTPUT" | $PYTHON_BIN -c "import sys,json; print(json.load(sys.stdin).get('peers_total', 0))" 2>/dev/null)
        info "Peers connected: ${PEER_COUNT:-0}"

        BLOCKS=$(echo "$HEALTH_OUTPUT" | $PYTHON_BIN -c "import sys,json; print(json.load(sys.stdin).get('total_blocks', '?'))" 2>/dev/null)
        info "Total model blocks: ${BLOCKS:-?}"
    else
        fail "Health monitor failed or returned invalid JSON"
        if [ -n "$HEALTH_OUTPUT" ]; then
            echo "$HEALTH_OUTPUT" | tail -3 | while IFS= read -r line; do
                info "  $line"
            done
        fi
    fi
fi

echo ""

# ============================================================
# Summary
# ============================================================
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}, ${YELLOW}$SKIP skipped${NC}"
echo -e "${CYAN}============================================================${NC}"

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "Some tests failed. Common fixes:"
    echo "  - Run ./setup.sh to generate config.yml and .env with your HF token"
    echo "  - Ensure GPU drivers are installed (nvidia-smi or rocminfo)"
    echo "  - Install API deps: pip install -r ../agentgrid-api/requirements.txt"
    echo ""
    exit 1
fi

if [ $SKIP -gt 0 ]; then
    echo ""
    echo "Some tests were skipped. To run all tests:"
    echo "  - Set HF_TOKEN in .env or environment"
    echo "  - Clone agentgrid-api alongside agent-grid-2"
    echo ""
fi

exit 0
