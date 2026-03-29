#!/bin/bash

# Cross-platform swarm launcher (macOS & Linux)
set -m
set -o pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
PYTHON_PID=""
PYTHON_PGID=""

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN=python
    else
        echo "Error: python3 is required but not found." >&2
        exit 1
    fi
fi

cleanup() {
    if [ -n "${PYTHON_PID:-}" ]; then
        echo "Shutting down the server and its child processes..."
        if kill -0 "$PYTHON_PID" 2>/dev/null; then
            if [ -n "${PYTHON_PGID:-}" ]; then
                kill -TERM "-${PYTHON_PGID}" 2>/dev/null || true
                sleep 1
                kill -KILL "-${PYTHON_PGID}" 2>/dev/null || true
            else
                kill "$PYTHON_PID" 2>/dev/null || true
            fi
            wait "$PYTHON_PID" 2>/dev/null || true
        fi
        echo "Cleanup complete"
    fi
}

trap cleanup SIGINT SIGTERM EXIT

if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Please create a .env file with required variables."
    exit 1
fi

if [ ! -f "models" ]; then
    echo "Error: models file not found. Please create a models file with a list of models."
    exit 1
fi

source models

echo "Please select a model:"
for i in "${!MODELS[@]}"; do
    echo "$((i + 1)). ${MODELS[i]}"
done

while true; do
    read -r -p "Enter the number of your choice (1-${#MODELS[@]}): " choice
    if [[ "$choice" =~ ^[1-9][0-9]*$ && "$choice" -le "${#MODELS[@]}" ]]; then
        MODEL=${MODELS[$((choice - 1))]}
        break
    else
        echo "Invalid choice. Please enter a number between 1 and ${#MODELS[@]}."
    fi
done

PORT=${PORT:-31331}
ALLOC_TIMEOUT=${ALLOC_TIMEOUT:-6000}
ATTN_CACHE_TOKENS=${ATTN_CACHE_TOKENS:-264000}
DISK_SPACE=${DISK_SPACE:-120GB}
INFERENCE_MAX_LENGTH=${INFERENCE_MAX_LENGTH:-136192}
P2P_FILE=${P2P_FILE:-./dev.id}

PUBLIC_IP=""
if command -v curl >/dev/null 2>&1; then
    PUBLIC_IP=$(curl -fsS ipinfo.io/ip 2>/dev/null)
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
    echo "Warning: Unable to determine public IP automatically; defaulting to 127.0.0.1"
    PUBLIC_IP="127.0.0.1"
fi

source .env

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN is not set in .env."
    exit 1
fi

DETECTED_SETTINGS=$("$PYTHON_BIN" -c 'import torch
device = "cpu"
dtype = "float32"
quant = "none"
backend = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    is_rocm = getattr(torch.version, "hip", None) is not None
    if is_rocm:
        dtype = "float16"
        quant = "int4_weight_only"
        backend = "rocm"
    else:
        dtype = "float16"
        quant = "int4_weight_only"
        backend = "cuda"
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    device = "mps"
    dtype = "float16"
    quant = "none"
    backend = "mps"
print(device, dtype, quant, backend)
' 2>/dev/null) || true

read -r DETECTED_DEVICE DETECTED_DTYPE DETECTED_QUANT DETECTED_BACKEND <<<"${DETECTED_SETTINGS:-cpu float32 none cpu}"

DEVICE=${AG_DEVICE:-$DETECTED_DEVICE}
TORCH_DTYPE=${AG_TORCH_DTYPE:-$DETECTED_DTYPE}
QUANT_TYPE=${AG_QUANT_TYPE:-$DETECTED_QUANT}
WARMUP_TOKENS_INTERVAL=${AG_WARMUP_TOKENS_INTERVAL:-}

if [ "$DEVICE" != "cuda" ]; then
    QUANT_TYPE="none"
fi

if [ "$DETECTED_BACKEND" = "rocm" ]; then
    echo "Using AMD ROCm (HIP) backend: device=$DEVICE, torch_dtype=$TORCH_DTYPE, quant_type=$QUANT_TYPE"
else
    echo "Using device=$DEVICE, torch_dtype=$TORCH_DTYPE, quant_type=$QUANT_TYPE"
fi

CMD=(
    "$PYTHON_BIN" -m agentgrid.cli.run_server
    --public_ip "$PUBLIC_IP"
    --device "$DEVICE"
    --torch_dtype "$TORCH_DTYPE"
    --quant_type "$QUANT_TYPE"
    --port "$PORT"
    --token "$HF_TOKEN"
    --attn_cache_tokens "$ATTN_CACHE_TOKENS"
    #--inference_max_length "$INFERENCE_MAX_LENGTH"
    --identity_path "$P2P_FILE"
    --throughput eval
    --new_swarm
    "$MODEL"
)

if [ -n "$WARMUP_TOKENS_INTERVAL" ]; then
    CMD+=(--warmup_tokens_interval "$WARMUP_TOKENS_INTERVAL")
fi

"${CMD[@]}" &

PYTHON_PID=$!
PYTHON_PGID=$(ps -o pgid= "$PYTHON_PID" | tr -d ' ')

echo "Server started with PID: $PYTHON_PID (PGID: $PYTHON_PGID). Press Ctrl+C to stop."

wait "$PYTHON_PID"
