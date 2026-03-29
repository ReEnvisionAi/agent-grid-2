#
# Copyright (c) 2025 ReEnvision AI, LLC. All rights reserved.
#
# This software is the confidential and proprietary information of
# ReEnvision AI, LLC ("Confidential Information"). You shall not
# disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with ReEnvision AI, LLC.
#

"""
Device abstraction utilities for CUDA / ROCm (HIP) / MPS backends.

PyTorch ROCm exposes AMD GPUs through the ``torch.cuda`` API with
``device.type == "cuda"``.  The helpers here detect the *actual* backend
so that code can branch when CUDA and HIP behaviour diverges (e.g. CUDA
graphs, torch.compile tuning).
"""

from __future__ import annotations

import functools
import os

import torch


def is_rocm() -> bool:
    """Return ``True`` when PyTorch was built against ROCm / HIP."""
    return getattr(torch.version, "hip", None) is not None


def is_gpu_available() -> bool:
    """Return ``True`` when any GPU (CUDA or ROCm) is usable."""
    return torch.cuda.is_available()


def gpu_backend_name() -> str:
    """Human-readable name for the active GPU backend."""
    if not torch.cuda.is_available():
        return "None"
    return "ROCm" if is_rocm() else "CUDA"


def get_gpu_name(index: int = 0) -> str:
    """GPU device name (works for both CUDA and ROCm)."""
    if torch.cuda.is_available() and index < torch.cuda.device_count():
        return torch.cuda.get_device_name(index)
    return "Unknown"


@functools.lru_cache(maxsize=1)
def supports_cuda_graphs() -> bool:
    """Check whether CUDA/HIP graph capture is likely to work.

    On NVIDIA CUDA this is always ``True``.  On ROCm the support depends
    on the GPU architecture and driver version, so we do a trivial
    trial capture and cache the result.  The env-var
    ``AGENTGRID_DISABLE_CUDA_GRAPHS=1`` forces graphs off.
    """
    if os.environ.get("AGENTGRID_DISABLE_CUDA_GRAPHS", "").strip() in ("1", "true", "yes"):
        return False

    if not torch.cuda.is_available():
        return False

    if not is_rocm():
        return True  # NVIDIA CUDA — graphs always supported

    # ROCm: attempt a trivial graph capture to verify support.
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        a = torch.zeros(1, device="cuda")
        with torch.cuda.stream(s):
            b = a + 1  # noqa: F841
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            b = a + 1  # noqa: F841
        g.replay()
        return True
    except Exception:
        return False
