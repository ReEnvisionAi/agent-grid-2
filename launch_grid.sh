#!/bin/bash
#
# Agent Grid Unified Launcher
# Starts the inference server (and optionally the API) using config.yml settings.
# Run ./setup.sh first to generate configuration files.
#

set -m
set -o pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
SERVER_PID=""
SERVER_PGID=""
API_PID=""

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN=python
    else
        echo "Error: python3 is required but not found." >&2
        exit 1
    fi
fi

cleanup() {
    echo ""
    echo "Shutting down..."

    # Stop API server
    if [ -n "${API_PID:-}" ] && kill -0 "$API_PID" 2>/dev/null; then
        echo "Stopping API server (PID: $API_PID)..."
        kill "$API_PID" 2>/dev/null || true
        wait "$API_PID" 2>/dev/null || true
    fi

    # Stop grid server
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping grid server (PID: $SERVER_PID)..."
        if [ -n "${SERVER_PGID:-}" ]; then
            kill -TERM "-${SERVER_PGID}" 2>/dev/null || true
            sleep 1
            kill -KILL "-${SERVER_PGID}" 2>/dev/null || true
        else
            kill "$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
    fi

    echo "Cleanup complete."
}

trap cleanup SIGINT SIGTERM EXIT

# --- Check for config.yml ---
if [ ! -f "config.yml" ]; then
    echo "No config.yml found. Running setup wizard..."
    echo ""
    bash setup.sh
    echo ""
    if [ ! -f "config.yml" ]; then
        echo "Error: Setup wizard did not generate config.yml." >&2
        exit 1
    fi
fi

# --- Load .env ---
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Warning: HF_TOKEN is not set. Gated models will fail to load."
fi

# --- Detect public IP ---
PUBLIC_IP=""
if command -v curl >/dev/null 2>&1; then
    PUBLIC_IP=$(curl -fsS --connect-timeout 5 ipinfo.io/ip 2>/dev/null)
fi
if [ -z "$PUBLIC_IP" ]; then
    if command -v hostname >/dev/null 2>&1; then
        PUBLIC_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi
fi
if [ -z "$PUBLIC_IP" ] && command -v ipconfig >/dev/null 2>&1; then
    PUBLIC_IP=$(ipconfig getifaddr en0 2>/dev/null)
    if [ -z "$PUBLIC_IP" ]; then
        PUBLIC_IP=$(ipconfig getifaddr en1 2>/dev/null)
    fi
fi
if [ -z "$PUBLIC_IP" ]; then
    echo "Warning: Unable to determine public IP; defaulting to 127.0.0.1"
    PUBLIC_IP="127.0.0.1"
fi

echo "============================================================"
echo "  Agent Grid Launcher"
echo "============================================================"
echo "  Public IP: $PUBLIC_IP"
echo "  Config:    config.yml"
echo "============================================================"
echo ""

# --- Start the grid server ---
CMD=(
    "$PYTHON_BIN" -m agentgrid.cli.run_server
    -c config.yml
    --public_ip "$PUBLIC_IP"
)

if [ -n "${HF_TOKEN:-}" ]; then
    CMD+=(--token "$HF_TOKEN")
fi

echo "Starting grid server..."
"${CMD[@]}" &
SERVER_PID=$!
SERVER_PGID=$(ps -o pgid= "$SERVER_PID" 2>/dev/null | tr -d ' ')
echo "  Grid server PID: $SERVER_PID"

# --- Optionally start the API server ---
LAUNCH_API=${LAUNCH_API:-false}
API_PORT=${API_PORT:-5000}
API_DIR="${API_DIR:-../agentgrid-api}"

if [ "$LAUNCH_API" = "true" ] && [ -d "$API_DIR" ]; then
    echo ""
    echo "Starting API server on port $API_PORT..."
    (cd "$API_DIR" && "$PYTHON_BIN" -m uvicorn app.main:app --host 0.0.0.0 --port "$API_PORT") &
    API_PID=$!
    echo "  API server PID: $API_PID"
elif [ "$LAUNCH_API" = "true" ]; then
    echo "Warning: API directory '$API_DIR' not found. Skipping API server."
fi

echo ""
echo "Grid is running. Press Ctrl+C to stop."
echo ""

# Wait for the server process
wait "$SERVER_PID"
