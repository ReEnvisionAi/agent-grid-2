#
# Copyright (c) 2025 ReEnvision AI, LLC. All rights reserved.
#
# This software is the confidential and proprietary information of
# ReEnvision AI, LLC ("Confidential Information"). You shall not
# disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with ReEnvision AI, LLC.
#

"""Utility helpers for discovering available models and devices."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence

import torch


def list_models(models_file: Path | str) -> List[str]:
    """Read available model identifiers from a models file.

    The default shell-based ``models`` file defines a ``MODELS`` bash array. We source the file in a
    subprocess and print the expanded elements. For JSON/YAML documents (ending with ``.json`` or
    ``.yml``/``.yaml``) we parse the array directly.
    """

    path = Path(models_file)
    if not path.exists():
        raise FileNotFoundError(f"Models file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".json"}:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            models = data.get("models")
        else:
            models = data
        if not isinstance(models, Sequence):
            raise ValueError("JSON models file must be an array or contain a 'models' array")
        return [str(item) for item in models]

    if suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyYAML is required to parse YAML models files") from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            models = data.get("models")
        else:
            models = data
        if not isinstance(models, Sequence):
            raise ValueError("YAML models file must be an array or contain a 'models' array")
        return [str(item) for item in models]

    # Default: bash file with MODELS array
    cmd = ["bash", "-lc", f"source {quote(path)}; printf '%s\n' \"${{MODELS[@]}}\""]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    models = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return models


def probe_devices() -> Dict[str, Dict[str, bool | int]]:
    """Return a summary of available compute backends (CUDA, ROCm, MPS, CPU)."""

    from agentgrid.utils.device_utils import is_rocm

    info: Dict[str, Dict[str, bool | int]] = {
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        "mps": {
            "available": getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(),
        },
        "cpu": {"available": True},
    }

    if info["cuda"]["available"]:  # type: ignore[index]
        devices = []
        for idx in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(idx)
            devices.append({"id": idx, "name": name})
        info["cuda"]["devices"] = devices  # type: ignore[index]

    # Add ROCm-specific metadata when running on AMD HIP
    if torch.cuda.is_available() and is_rocm():
        info["rocm"] = {
            "available": True,
            "hip_version": getattr(torch.version, "hip", "unknown"),
            "backend": "HIP",
        }
        # Relabel the backend so callers know this is ROCm, not NVIDIA CUDA
        info["cuda"]["backend"] = "ROCm (HIP)"  # type: ignore[index]
    elif torch.cuda.is_available():
        info["cuda"]["backend"] = "NVIDIA CUDA"  # type: ignore[index]

    return info


def quote(path: Path) -> str:
    return "'" + str(path).replace("'", "'\\''") + "'"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Agent Grid discovery utilities")
    parser.add_argument("--models-file", type=str, help="Path to models file", default=None)
    parser.add_argument("--list-models", action="store_true", help="Print available models")
    parser.add_argument("--probe-devices", action="store_true", help="Print available compute devices")
    args = parser.parse_args(argv)

    if not args.list_models and not args.probe_devices:
        parser.error("Specify at least one action (--list-models or --probe-devices)")

    if args.list_models:
        models_path = args.models_file or os.environ.get("AGENTGRID_MODELS_FILE", "models")
        models = list_models(models_path)
        for model in models:
            print(model)

    if args.probe_devices:
        info = probe_devices()
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

