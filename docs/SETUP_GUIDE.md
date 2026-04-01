# AgentGrid Complete Setup Guide

Step-by-step instructions for setting up the **AgentGrid server** (serves distributed model blocks) and **AgentGrid API** (OpenAI-compatible REST API) on every supported platform.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the Repositories](#2-clone-the-repositories)
3. [Create a Virtual Environment](#3-create-a-virtual-environment)
4. [Install AgentGrid (by GPU type)](#4-install-agentgrid-by-gpu-type)
5. [Install the API Server](#5-install-the-api-server)
6. [Configure Your Environment](#6-configure-your-environment)
7. [Start the Grid Server](#7-start-the-grid-server)
8. [Start the API Server](#8-start-the-api-server)
9. [Verify Everything Works](#9-verify-everything-works)
10. [Windows WSL2 Setup](#10-windows-wsl2-setup)
11. [Available Models](#11-available-models)
12. [Ports and Networking](#12-ports-and-networking)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Prerequisites

### All Platforms

| Requirement | Details |
|-------------|---------|
| Python | 3.11 or higher (3.11, 3.12, 3.13 supported) |
| Git | Any recent version |
| RAM | 8 GB+ recommended |
| HuggingFace token | Required for gated models (Nemotron, etc.) - get one at https://huggingface.co/settings/tokens |

### Linux with NVIDIA GPU

- NVIDIA driver 535 or later
- No separate CUDA toolkit install needed (PyTorch wheel bundles CUDA 12.4)
- Verify with: `nvidia-smi`

### Linux with AMD GPU (ROCm)

- Supported GPUs: AMD Instinct MI300, RDNA3 (RX 7900 series), AMD AI Max 395 (gfx1151)
- ROCm 6.2+ installed from [AMD ROCm repos](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
- Verify with: `rocminfo` and `rocm-smi`
- See [ROCM_SETUP.md](./ROCM_SETUP.md) for detailed AMD instructions

### macOS (Apple Silicon)

- M1, M2, M3, or M4 chip
- macOS 13 (Ventura) or later
- MPS backend is used automatically
- Note: torchao quantization is not available on macOS

### Windows

- **WSL2 is required** (hivemind's libp2p daemon only runs on Linux/macOS)
- Windows 10 version 2004+ or Windows 11
- See [Section 10](#10-windows-wsl2-setup) for full WSL2 setup
- See [WSL2_SETUP.md](./WSL2_SETUP.md) for detailed Windows instructions

---

## 2. Clone the Repositories

**Option A: Public repos (HTTPS)**

```bash
git clone https://github.com/ReEnvisionAi/agent-grid-2.git
git clone https://github.com/ReEnvision-AI/agentgrid-api.git
```

**Option B: Private org repos (GitHub CLI)**

```bash
# Install GitHub CLI first: https://cli.github.com/
gh auth login
gh repo clone ReEnvisionAi/agent-grid-2
gh repo clone ReEnvision-AI/agentgrid-api
```

**Checkout the feature branch:**

```bash
cd agent-grid-2 && git checkout claude/refactor-agent-grid-Z6h7b && cd ..
cd agentgrid-api && git checkout claude/refactor-agent-grid-Z6h7b && cd ..
```

---

## 3. Create a Virtual Environment

```bash
python3.11 -m venv ~/agentgrid-env
source ~/agentgrid-env/bin/activate
pip install --upgrade pip
```

> On macOS, you may need `python3` instead of `python3.11` if installed via Homebrew.

---

## 4. Install AgentGrid (by GPU type)

### NVIDIA GPU (Linux or WSL2)

```bash
cd ~/agent-grid-2
pip install -e ".[full]"
```

This installs PyTorch with CUDA 12.4, torchao, triton, and all inference dependencies.

### AMD GPU with ROCm (Linux)

```bash
cd ~/agent-grid-2

# Step 1: Install PyTorch ROCm wheel FIRST
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.2

# Step 2: Install agent-grid with ROCm extras
pip install -e ".[rocm]"
```

**For AMD AI Max 395 (gfx1151)**, add these environment variables to your shell or `.env`:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export AGENTGRID_DISABLE_CUDA_GRAPHS=1
```

### Apple Silicon (macOS)

```bash
cd ~/agent-grid-2
pip install -e ".[macos]"
```

MPS is detected automatically. Quantization is not available (torchao doesn't support macOS).

### CPU Only (any platform)

```bash
cd ~/agent-grid-2
pip install -e ".[inference]"
```

Useful for running the API server without local GPU inference.

---

## 5. Install the API Server

```bash
cd ~/agentgrid-api
pip install fastapi fastapi-cli sse-starlette uvicorn environs
```

> The API connects to the distributed grid as a **client** - it does not need a GPU. It works identically on all platforms.

---

## 6. Configure Your Environment

### Option A: Interactive Setup Wizard (Recommended)

```bash
cd ~/agent-grid-2
bash setup.sh
```

The wizard will:
- Detect your GPU hardware (NVIDIA CUDA, AMD ROCm, Apple MPS, or CPU)
- Suggest optimal defaults for dtype and quantization
- Ask for your HuggingFace token
- Generate `config.yml`, `.env`, and optionally `../agentgrid-api/.env`

### Option B: Manual Configuration

**Create `~/agent-grid-2/.env`:**

```bash
HF_TOKEN=hf_your_token_here
LAUNCH_API=true
API_PORT=5000
```

**Create `~/agentgrid-api/.env`:**

```bash
HUGGING_FACE_HUB_TOKEN=hf_your_token_here
API_VERSION=0.3.9
```

**Create `~/agent-grid-2/config.yml`:**

```yaml
# NVIDIA example
converted_model_name_or_path: Qwen/Qwen2.5-Coder-32B-Instruct
device: cuda
torch_dtype: float16
quantization: int4_weight_only
port: 31331
identity_path: ./server.id
attn_cache_tokens: 264000
throughput: eval
```

| GPU Type | device | torch_dtype | quantization |
|----------|--------|-------------|--------------|
| NVIDIA | `cuda` | `float16` | `int4_weight_only` |
| AMD ROCm | `cuda` | `float16` | `int4_weight_only` |
| Apple Silicon | `mps` | `float16` | `none` |
| CPU | `cpu` | `float32` | `none` |

> Note: AMD ROCm uses `cuda` as the device type because PyTorch ROCm reports through `torch.cuda`.

---

## 7. Start the Grid Server

### Option A: Unified Launcher (Recommended)

```bash
cd ~/agent-grid-2
bash launch_grid.sh
```

This script:
- Checks for `config.yml` (runs setup wizard if missing)
- Loads environment from `.env`
- Detects your public IP
- Starts the grid server
- Optionally launches the API server if `LAUNCH_API=true`

### Option B: Manual Start

```bash
cd ~/agent-grid-2
source ~/agentgrid-env/bin/activate
python -m agentgrid.cli.run_server --config config.yml
```

### Option C: Swarm Mode (Multiple Models)

```bash
cd ~/agent-grid-2
bash start_swarm.sh
```

Presents an interactive menu to choose which model to serve.

---

## 8. Start the API Server

Open a **separate terminal** (or use `launch_grid.sh` with `LAUNCH_API=true`):

```bash
source ~/agentgrid-env/bin/activate
cd ~/agentgrid-api
python -m uvicorn app.main:app --host 0.0.0.0 --port 5000
```

> **Important:** Always use `python -m uvicorn`, not just `uvicorn`. This ensures the virtual environment's Python is used, not the system Python.

---

## 9. Verify Everything Works

### Health Check

```bash
# Overall grid health
curl http://localhost:5000/health

# Specific model health
curl http://localhost:5000/health/Qwen/Qwen2.5-Coder-32B-Instruct
```

### Chat Completion Test

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "messages": [{"role": "user", "content": "Hello! Write a Python hello world."}],
    "max_tokens": 100
  }'
```

### Rich Terminal Health Monitor

```bash
source ~/agentgrid-env/bin/activate
python -m agentgrid.cli.health_monitor
```

Add `--refresh 10` for auto-refresh every 10 seconds, or `--json` for machine-readable output.

### Using with OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:5000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
print(response.choices[0].message.content)
```

---

## 10. Windows WSL2 Setup

### Step 1: Enable WSL2

Open **PowerShell as Administrator**:

```powershell
wsl --install
```

Restart your computer when prompted.

### Step 2: Install Ubuntu

From Microsoft Store, install **Ubuntu 22.04 LTS**. Launch it and create your username/password.

### Step 3: GPU Driver (NVIDIA only)

Install the **Windows** NVIDIA driver (NOT inside WSL2). WSL2 automatically passes the GPU through.

- Download from: https://www.nvidia.com/drivers
- Verify inside WSL2: `nvidia-smi`

### Step 4: Install Python 3.11

Inside WSL2 (run `wsl` from PowerShell to enter):

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install -y python3.11 python3.11-venv python3.11-dev git curl
```

### Step 5: Install GitHub CLI (for private repos)

```bash
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update && sudo apt install gh -y
gh auth login
```

### Step 6: Follow Standard Setup

Now follow [Sections 2-9](#2-clone-the-repositories) above. All commands run inside WSL2.

### Accessing from Windows

The API is accessible from Windows at `http://localhost:5000` (WSL2 automatically forwards ports).

---

## 11. Available Models

| Model | Full ID | Parameters | Use Case |
|-------|---------|------------|----------|
| Qwen 2.5 Coder | `Qwen/Qwen2.5-Coder-32B-Instruct` | 32B | Code generation, technical tasks |
| Nemotron Super | `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5` | 49B | General purpose (requires HF token) |
| GPT-OSS | `unsloth/gpt-oss-20b-BF16` | 20B | General purpose, smaller footprint |

All models are served through the distributed grid. The API server routes requests to the appropriate model based on the `model` field in the request.

---

## 12. Ports and Networking

| Service | Default Port | Configurable Via |
|---------|-------------|------------------|
| Grid Server (DHT/P2P) | 31331 | `config.yml` → `port` |
| API Server | 5000 | `API_PORT` env var |

**Public DHT Bootstrap Peers:**

```
/dns4/sociallyshaped.net/tcp/8788/p2p/QmSt3bPSboHuBNfgB3tPrjGnW1D3xFRPyvrmi2x7TiZ3qR
/ip4/52.14.122.164/tcp/8788/p2p/QmT5mCzypk1HwyEaZ9JbKRoypG235i5KghUoFo32VDaTEZ
```

These are used by default. For a private grid, specify custom peers in `config.yml` under `initial_peers`.

---

## 13. Troubleshooting

### `ModuleNotFoundError: No module named 'pkg_resources'`

The pre-built hivemind wheel in `deps/` avoids this. Make sure you're installing from the repo directory:

```bash
cd ~/agent-grid-2
pip install -e ".[full]"
```

### `ImportError: cannot import name 'PreTrainedModel' from 'transformers'`

This happens when torchao is installed but incompatible with your torch version. For the API server, torchao is not needed:

```bash
pip uninstall torchao -y
```

The API's `requirements.txt` uses `agent-grid[inference]` (not `[full]`) to avoid this.

### `FlashAttention2 has been toggled on, but it cannot be used`

Already fixed in the latest code. Pull the latest:

```bash
cd ~/agentgrid-api
git pull origin claude/refactor-agent-grid-Z6h7b
```

### Wrong Python / wrong uvicorn

If you see errors from `/usr/bin/uvicorn` or Python 3.10, the system Python is being used instead of the venv:

```bash
# Always use this form:
python -m uvicorn app.main:app --host 0.0.0.0 --port 5000

# NOT this:
uvicorn app.main:app --host 0.0.0.0 --port 5000
```

### `fatal: not a git repository`

You downloaded the repo as a ZIP instead of cloning. Use `git clone`:

```bash
git clone https://github.com/ReEnvisionAi/agent-grid-2.git
```

### Private repo authentication fails

Use GitHub CLI for org repos:

```bash
gh auth login
gh repo clone ReEnvisionAi/agent-grid-2
```

### `RequirementParseError: Invalid URL given`

Upgrade pip first (fresh venvs have old pip that can't parse `file:./` URLs):

```bash
pip install --upgrade pip
```

### AMD ROCm: `torch.cuda.is_available()` returns False

- Verify ROCm is installed: `rocminfo`
- Ensure you installed the ROCm PyTorch wheel: `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.2`
- For AI Max 395: `export HSA_OVERRIDE_GFX_VERSION=11.0.0`

---

## Quick Reference: Platform Cheat Sheet

### Linux + NVIDIA

```bash
python3.11 -m venv ~/agentgrid-env && source ~/agentgrid-env/bin/activate
pip install --upgrade pip
cd ~/agent-grid-2 && pip install -e ".[full]"
cd ~/agentgrid-api && pip install fastapi fastapi-cli sse-starlette uvicorn environs
cd ~/agent-grid-2 && bash setup.sh
bash launch_grid.sh
# In another terminal:
source ~/agentgrid-env/bin/activate && cd ~/agentgrid-api
python -m uvicorn app.main:app --host 0.0.0.0 --port 5000
```

### Linux + AMD ROCm

```bash
python3.11 -m venv ~/agentgrid-env && source ~/agentgrid-env/bin/activate
pip install --upgrade pip
cd ~/agent-grid-2
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.2
pip install -e ".[rocm]"
cd ~/agentgrid-api && pip install fastapi fastapi-cli sse-starlette uvicorn environs
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # AI Max 395 only
export AGENTGRID_DISABLE_CUDA_GRAPHS=1   # Recommended for ROCm
cd ~/agent-grid-2 && bash setup.sh
bash launch_grid.sh
# In another terminal:
source ~/agentgrid-env/bin/activate && cd ~/agentgrid-api
python -m uvicorn app.main:app --host 0.0.0.0 --port 5000
```

### macOS + Apple Silicon

```bash
python3 -m venv ~/agentgrid-env && source ~/agentgrid-env/bin/activate
pip install --upgrade pip
cd ~/agent-grid-2 && pip install -e ".[macos]"
cd ~/agentgrid-api && pip install fastapi fastapi-cli sse-starlette uvicorn environs
cd ~/agent-grid-2 && bash setup.sh
bash launch_grid.sh
# In another terminal:
source ~/agentgrid-env/bin/activate && cd ~/agentgrid-api
python -m uvicorn app.main:app --host 0.0.0.0 --port 5000
```

### Windows (WSL2) + NVIDIA

```powershell
# In PowerShell (one-time):
wsl --install
# Restart, then open Ubuntu from Start menu
```

```bash
# Inside WSL2 — one-time setup:
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install -y python3.11 python3.11-venv python3.11-dev git curl

# Then follow "Linux + NVIDIA" steps above
```

### Windows (WSL2) + AMD ROCm

```bash
# Inside WSL2 — install ROCm per AMD docs, then follow "Linux + AMD ROCm" steps above
```
