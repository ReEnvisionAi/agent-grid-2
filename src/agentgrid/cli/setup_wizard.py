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
Interactive setup wizard for Agent Grid. Generates config.yml and .env files
so the server can be launched with minimal command-line arguments.
"""

import getpass
import os
import sys
from pathlib import Path


def _detect_hardware():
    """Detect available hardware and suggest defaults."""
    device = "cpu"
    dtype = "float32"
    quant = "none"
    gpu_name = None

    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            dtype = "float16"
            quant = "int4_weight_only"
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = "CUDA GPU"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
            dtype = "float16"
            quant = "none"
            gpu_name = "Apple Silicon"
    except ImportError:
        pass

    return device, dtype, quant, gpu_name


def _load_models_file(path: str = "models") -> list:
    """Parse the bash-style models file."""
    models = []
    if not os.path.exists(path):
        return models
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                models.append(line.strip('"'))
            elif line.startswith("'") and line.endswith("'"):
                models.append(line.strip("'"))
    return models


def _prompt(question: str, default: str = "", secret: bool = False) -> str:
    """Prompt with a default value shown in brackets."""
    suffix = f" [{default}]" if default else ""
    prompt_str = f"{question}{suffix}: "
    if secret:
        value = getpass.getpass(prompt_str)
    else:
        value = input(prompt_str)
    return value.strip() or default


def _prompt_choice(question: str, options: list, default: int = 1) -> int:
    """Prompt user to pick from numbered options. Returns 0-based index."""
    print(f"\n{question}")
    for i, opt in enumerate(options):
        marker = " (default)" if i + 1 == default else ""
        print(f"  {i + 1}. {opt}{marker}")
    while True:
        raw = input(f"Enter choice [1-{len(options)}] (default {default}): ").strip()
        if not raw:
            return default - 1
        try:
            choice = int(raw)
            if 1 <= choice <= len(options):
                return choice - 1
        except ValueError:
            pass
        print(f"  Invalid choice. Please enter 1-{len(options)}.")


def _read_existing_env(path: str) -> dict:
    """Read key=value pairs from an existing .env file."""
    env = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    env[key.strip()] = value.strip()
    return env


def main():
    print("=" * 60)
    print("  Agent Grid Setup Wizard")
    print("=" * 60)

    # Detect hardware
    device, dtype, quant, gpu_name = _detect_hardware()
    print(f"\nDetected hardware: {gpu_name or 'CPU only'}")
    print(f"  Suggested: device={device}, dtype={dtype}, quantization={quant}")

    # Read existing .env if present
    existing_env = _read_existing_env(".env")

    # --- Model selection ---
    models = _load_models_file("models")
    if models:
        models.append("(enter custom model path)")
        idx = _prompt_choice("Select a model:", models, default=1)
        if idx == len(models) - 1:
            model = _prompt("Enter model name or path")
        else:
            model = models[idx]
    else:
        model = _prompt("Enter model name or path (e.g. Qwen/Qwen2.5-Coder-32B-Instruct)")

    if not model:
        print("Error: model is required.")
        sys.exit(1)

    # --- Device ---
    device = _prompt("Device (cuda/mps/cpu/auto)", default=device)

    # --- Torch dtype ---
    dtype = _prompt("Torch dtype (float16/bfloat16/float32/auto)", default=dtype)

    # --- Quantization ---
    if device in ("cuda", "auto"):
        quant = _prompt("Quantization (none/int4_weight_only/int8_weight_only)", default=quant)
    else:
        print(f"\nQuantization disabled for device={device}")
        quant = "none"

    # --- Compile block ---
    compile_block = False
    if device in ("cuda", "auto"):
        resp = _prompt("Enable torch.compile for blocks? (y/n)", default="n")
        compile_block = resp.lower() in ("y", "yes")

    # --- Port ---
    port = _prompt("Server port", default="31331")

    # --- HuggingFace token ---
    default_token = existing_env.get("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
    token_display = f"{'*' * 8}...{default_token[-4:]}" if len(default_token) > 4 else ""
    if token_display:
        print(f"\nExisting HF token found: {token_display}")
        use_existing = _prompt("Use existing token? (y/n)", default="y")
        if use_existing.lower() in ("y", "yes"):
            hf_token = default_token
        else:
            hf_token = _prompt("HuggingFace token", secret=True)
    else:
        hf_token = _prompt("HuggingFace token (required for gated models)", secret=True)

    # --- Public name ---
    public_name = _prompt("Public name for leaderboard (optional)", default="")

    # --- Number of blocks ---
    num_blocks = _prompt("Number of blocks to serve (leave empty for auto)", default="")

    # --- Swarm mode ---
    swarm_idx = _prompt_choice(
        "Swarm mode:",
        ["Join public swarm (default)", "Join private swarm (provide peer address)", "Start new swarm"],
        default=1,
    )
    initial_peers = None
    new_swarm = False
    if swarm_idx == 1:
        initial_peers = _prompt("Enter initial peer multiaddr(s) (space-separated)")
    elif swarm_idx == 2:
        new_swarm = True

    # --- Identity file ---
    identity_path = _prompt("Identity file path", default="./server.id")

    # --- Attention cache ---
    attn_cache = _prompt("Attention cache tokens", default="264000")

    # --- API ---
    launch_api = _prompt("Also launch the API server? (y/n)", default="y")
    launch_api = launch_api.lower() in ("y", "yes")
    api_port = "5000"
    if launch_api:
        api_port = _prompt("API server port", default="5000")

    # --- Generate config.yml ---
    print("\n" + "-" * 60)
    print("Generating configuration files...")

    config_lines = []
    config_lines.append(f"# Agent Grid configuration - generated by setup wizard")
    config_lines.append(f"converted_model_name_or_path: {model}")
    config_lines.append(f"device: {device}")
    config_lines.append(f"torch_dtype: {dtype}")
    config_lines.append(f"quantization: {quant}")
    config_lines.append(f"port: {port}")
    config_lines.append(f"identity_path: {identity_path}")
    config_lines.append(f"attn_cache_tokens: {attn_cache}")
    config_lines.append(f"throughput: eval")

    if compile_block:
        config_lines.append(f"compile-block: true")
    if public_name:
        config_lines.append(f"public_name: {public_name}")
    if num_blocks:
        config_lines.append(f"num_blocks: {num_blocks}")
    if new_swarm:
        config_lines.append(f"new_swarm: true")
    elif initial_peers:
        config_lines.append(f"initial_peers: [{initial_peers}]")

    config_content = "\n".join(config_lines) + "\n"

    config_path = Path("config.yml")
    if config_path.exists():
        overwrite = _prompt(f"{config_path} already exists. Overwrite? (y/n)", default="n")
        if overwrite.lower() not in ("y", "yes"):
            print("Skipping config.yml")
        else:
            config_path.write_text(config_content)
            print(f"  Wrote {config_path}")
    else:
        config_path.write_text(config_content)
        print(f"  Wrote {config_path}")

    # --- Generate .env ---
    env_lines = []
    env_lines.append("# Agent Grid environment - generated by setup wizard")
    if hf_token:
        env_lines.append(f"HF_TOKEN={hf_token}")
    env_lines.append(f"LAUNCH_API={'true' if launch_api else 'false'}")
    env_lines.append(f"API_PORT={api_port}")
    env_content = "\n".join(env_lines) + "\n"

    env_path = Path(".env")
    if env_path.exists():
        overwrite = _prompt(f"{env_path} already exists. Overwrite? (y/n)", default="n")
        if overwrite.lower() not in ("y", "yes"):
            print("Skipping .env")
        else:
            env_path.write_text(env_content)
            print(f"  Wrote {env_path}")
    else:
        env_path.write_text(env_content)
        print(f"  Wrote {env_path}")

    # --- Generate agentgrid-api .env if launching API ---
    if launch_api:
        api_env_dir = Path("../agentgrid-api")
        if api_env_dir.exists():
            api_env_lines = []
            api_env_lines.append("# AgentGrid API environment - generated by setup wizard")
            if hf_token:
                api_env_lines.append(f"HUGGING_FACE_HUB_TOKEN={hf_token}")
            if initial_peers:
                api_env_lines.append(f'INITIAL_PEERS=["{initial_peers}"]')
            api_env_content = "\n".join(api_env_lines) + "\n"

            api_env_path = api_env_dir / ".env"
            if api_env_path.exists():
                overwrite = _prompt(f"{api_env_path} already exists. Overwrite? (y/n)", default="n")
                if overwrite.lower() not in ("y", "yes"):
                    print(f"Skipping {api_env_path}")
                else:
                    api_env_path.write_text(api_env_content)
                    print(f"  Wrote {api_env_path}")
            else:
                api_env_path.write_text(api_env_content)
                print(f"  Wrote {api_env_path}")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print(f"  Run ./launch_grid.sh to start the grid.")
    print("=" * 60)


if __name__ == "__main__":
    main()
