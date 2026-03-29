# Windows Setup Guide (WSL2)

Agent Grid requires WSL2 (Windows Subsystem for Linux) on Windows. Native Windows is not supported due to the hivemind P2P networking layer requiring Linux.

## Why WSL2?

Agent Grid depends on [hivemind](https://github.com/learning-at-home/hivemind), which uses a libp2p daemon for peer-to-peer networking. This daemon is only available for Linux and macOS. Additionally, the server uses fork-based multiprocessing and POSIX signals for process management, which are not available on Windows.

**WSL2 provides near-native Linux performance** and full GPU passthrough, so there is no meaningful performance penalty.

## Prerequisites

- **Windows 11** (or Windows 10 build 19041+)
- **NVIDIA GPU** with latest Game Ready or Studio drivers (for CUDA)
  - AMD ROCm on WSL2 is only supported for AMD Instinct GPUs, not consumer RDNA GPUs
- **16GB+ RAM** recommended

## Setup

### 1. Enable WSL2

Open PowerShell as Administrator:
```powershell
wsl --install -d Ubuntu-22.04
```

Restart your computer when prompted. After restart, Ubuntu will finish installing and ask you to create a username and password.

### 2. Update Ubuntu

```bash
sudo apt update && sudo apt upgrade -y
```

### 3. Install Python 3.11+

```bash
sudo apt install -y python3.11 python3.11-venv python3-pip git curl
```

### 4. Verify GPU Access

For NVIDIA GPUs, the Windows GPU driver automatically provides CUDA support inside WSL2:
```bash
nvidia-smi
```

You should see your GPU listed. If not, update your NVIDIA drivers on Windows.

### 5. Create a Virtual Environment

```bash
python3.11 -m venv ~/agentgrid-env
source ~/agentgrid-env/bin/activate
```

### 6. Install PyTorch

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Verify:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

### 7. Clone and Install Agent Grid

```bash
cd ~
git clone https://github.com/ReEnvisionAi/agent-grid-2.git
cd agent-grid-2
pip install -e ".[full]"
```

### 8. Clone the API (Optional)

```bash
cd ~
git clone https://github.com/ReEnvision-AI/agentgrid-api.git
```

### 9. Run the Setup Wizard

```bash
cd ~/agent-grid-2
./setup.sh
```

### 10. Launch the Grid

```bash
./launch_grid.sh
```

## Tips

### Accessing WSL2 Files from Windows
Your WSL2 files are at `\\wsl$\Ubuntu-22.04\home\<username>\` in Windows Explorer.

### Port Forwarding
WSL2 shares the network with Windows by default. Services running on `localhost:31331` in WSL2 are accessible from Windows at the same address.

### Persistent Sessions
Use `tmux` or `screen` to keep the grid running after closing the terminal:
```bash
sudo apt install -y tmux
tmux new -s agentgrid
./launch_grid.sh
# Detach with Ctrl+B, then D
# Reattach with: tmux attach -t agentgrid
```

### Memory Configuration
WSL2 defaults to using 50% of system RAM. To increase this, create `C:\Users\<username>\.wslconfig`:
```ini
[wsl2]
memory=24GB
processors=8
```

Then restart WSL2: `wsl --shutdown` from PowerShell.

## AMD GPU on Windows

AMD ROCm support in WSL2 is currently limited to **AMD Instinct (data center) GPUs** only. Consumer RDNA GPUs (RX 7900, AI Max 395, etc.) are **not supported** in WSL2 for compute workloads.

**If you have an AMD consumer GPU**, your options are:
1. **Dual-boot Linux** — Full ROCm support on native Linux (see [ROCM_SETUP.md](ROCM_SETUP.md))
2. **Run CPU-only in WSL2** — Works but significantly slower
3. **Use the API client only** — Connect to a grid running on Linux/macOS machines with GPUs

## Troubleshooting

**`nvidia-smi` not found in WSL2:**
- Update your NVIDIA GPU driver on Windows to the latest version
- Do NOT install CUDA or drivers inside WSL2 — the Windows driver provides them

**Slow file I/O:**
- Keep your project files inside WSL2's filesystem (`~/`), not on mounted Windows drives (`/mnt/c/`)
- Mounted Windows drives have poor I/O performance in WSL2

**Network connectivity issues:**
- If the grid can't connect to peers, check Windows Firewall rules
- WSL2 may need port forwarding for inbound connections: `netsh interface portproxy add v4tov4 listenport=31331 listenaddress=0.0.0.0 connectport=31331 connectaddress=$(wsl hostname -I | awk '{print $1}')`
