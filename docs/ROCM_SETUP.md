# AMD ROCm Setup Guide

Run Agent Grid on AMD GPUs using ROCm (HIP). PyTorch ROCm exposes AMD GPUs through the standard `torch.cuda` API, so the vast majority of Agent Grid works without modification.

## Supported Hardware

| GPU | Architecture | Status |
|-----|-------------|--------|
| AMD Instinct MI300X/MI300A | CDNA3 (gfx942) | Officially supported by ROCm |
| AMD Instinct MI250X/MI250 | CDNA2 (gfx90a) | Officially supported by ROCm |
| AMD Radeon RX 7900 XTX | RDNA3 (gfx1100) | Officially supported by ROCm |
| AMD Ryzen AI Max+ 395 | RDNA3.5 (gfx1151) | Community-supported via nightly builds |
| AMD Radeon RX 7600/7800 | RDNA3 (gfx1102) | Community-supported |

## Prerequisites

- **Linux** (Ubuntu 22.04+ or RHEL 9+)
- **ROCm 6.4.1+** (ROCm 7.x recommended for best compatibility)
- **Python 3.11+**

## Installation

### 1. Install ROCm

Follow the [official ROCm installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/).

For Ubuntu:
```bash
# Add ROCm repository
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.4.60401-1_all.deb
sudo apt install ./amdgpu-install_6.4.60401-1_all.deb
sudo amdgpu-install --usecase=rocm

# Verify
rocminfo | grep gfx
```

### 2. Install PyTorch ROCm

**For officially supported GPUs (MI300, RX 7900 XTX):**
```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.2
```

**For AMD AI Max 395 (gfx1151) — use nightly builds:**
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm7.2
```

Or use AMD's dedicated gfx1151 nightlies:
```bash
pip install --pre torch --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
```

### 3. Verify PyTorch sees the GPU
```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('ROCm:', torch.version.hip)"
```

Expected output:
```
GPU available: True
Device: AMD Radeon Graphics
ROCm: 6.4.43482-d62f6a171
```

### 4. Install Agent Grid
```bash
pip install -e ".[rocm]"
```

### 5. Run the setup wizard
```bash
./setup.sh
```

The wizard will auto-detect your AMD GPU and suggest appropriate defaults (float16 dtype, int4_weight_only quantization).

## Configuration Notes

### Quantization
INT4 and INT8 weight-only quantization via torchao is supported on AMD GPUs. The setup wizard defaults to `int4_weight_only` for ROCm, same as CUDA.

### torch.compile
Agent Grid uses `torch.compile(mode='reduce-overhead')` on ROCm (vs `max-autotune` on NVIDIA). Enable with `--compile-block`.

### CUDA Graphs
ROCm supports HIP graphs through the `torch.cuda.CUDAGraph()` API. If you encounter issues, disable them:
```bash
export AGENTGRID_DISABLE_CUDA_GRAPHS=1
```

### Memory (AI Max 395)
The AMD AI Max 395 has unified memory (up to 128GB). On Linux, you can increase the GPU-accessible VRAM beyond the BIOS default using TTM kernel parameters:
```bash
# Check current allocation
cat /sys/class/drm/card0/device/mem_info_vram_total

# Increase via kernel parameter (requires reboot)
# Add to GRUB: amdgpu.ttm_pages=<pages>
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AGENTGRID_DISABLE_CUDA_GRAPHS` | Set to `1` to disable CUDA/HIP graph capture |
| `HSA_OVERRIDE_GFX_VERSION` | Override GPU architecture (e.g. `11.0.0` for gfx1100 compat) |
| `PYTORCH_HIP_ALLOC_CONF` | HIP memory allocator config (similar to `PYTORCH_CUDA_ALLOC_CONF`) |

## Troubleshooting

**`torch.cuda.is_available()` returns False:**
- Verify ROCm is installed: `rocminfo`
- Ensure your user is in the `video` and `render` groups: `sudo usermod -aG video,render $USER`
- Check you installed the ROCm PyTorch wheel, not the CPU/CUDA one

**HIP kernel errors at runtime:**
- Try setting `HSA_OVERRIDE_GFX_VERSION` to a supported target (e.g. `11.0.0`)
- For gfx1151, ensure you're using the nightly PyTorch builds

**Out of memory:**
- Use `int4_weight_only` quantization to reduce memory usage
- Reduce `--attn_cache_tokens` and `--num_blocks`

**Flash Attention not working:**
- Flash Attention on ROCm is experimental for RDNA 3.x GPUs
- Set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` to try it
- If it fails, Agent Grid falls back to standard attention automatically
