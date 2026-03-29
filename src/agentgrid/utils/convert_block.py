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
Tools for converting transformer blocks, applying quantization and/or tensor parallelism
"""
import re
from enum import Enum
from typing import Optional, Sequence

import tensor_parallel as tp
import torch
import torch.nn as nn
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from tensor_parallel.slicing_configs import get_bloom_config
from transformers import PretrainedConfig

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


class QuantType(Enum):
    NONE = 0
    INT8 = 1  # Deprecated alias for INT8_WEIGHT_ONLY
    NF4 = 2  # Deprecated alias for INT4_WEIGHT_ONLY
    INT8_WEIGHT_ONLY = 3  # 8-bit weight-only quantization via torchao
    INT4_WEIGHT_ONLY = 4  # 4-bit weight-only quantization via torchao


def convert_block(
    block: nn.Module,
    block_index: int,
    config: PretrainedConfig,
    tensor_parallel_devices: Sequence[torch.device],
    output_device: torch.device,
    quant_type: QuantType,
    freeze: bool = True,
    adapters: Optional[Sequence[str]] = None,
    **kwargs,
) -> tp.TensorParallel:
    """
    Optimize a transformer block for use in a Agent Grid server, apply tensor parallelism and/or LLM.8bit quantization

    :note: some optimizations will modify the input block in-place!
    :param block: a single transformer block, either pre-trained or newly initialized
    :param config: HF transformers config for the full model
    :param tensor_parallel_devices: if specified, use tensor parallelism to split the model between these devices
    :note: if there is only a single device, model wil still be wrapped with TensorParallel (for uniformity)
    :param output_device: if tensor_parallel_devices is True, output
    :param quant_type: quantization type
    :param freeze: if True (default), make all module parameters non-trainable
    :return: a module that acts like the original block, but runs with all specified optimizations

    """
    if freeze:
        block.requires_grad_(False)

    tensor_parallel_devices = tuple(
        device if isinstance(device, torch.device) else torch.device(device)
        for device in tensor_parallel_devices
    )

    if output_device.type == "mps":
        attn_impl = getattr(config, "_attn_implementation", None)
        if attn_impl != "eager":
            logger.info(
                "Forcing attention implementation to 'eager' for MPS backend instead of %s", attn_impl
            )
            config._attn_implementation = "eager"

    if len(tensor_parallel_devices) > 1:
        unique_types = {device.type for device in tensor_parallel_devices}
        if unique_types != {"cuda"}:
            raise ValueError(
                "Tensor parallelism currently supports GPU (CUDA/ROCm) devices only. "
                f"Detected device types: {sorted(unique_types)}"
            )

    if output_device.type == "mps" and not hasattr(torch.mps, "current_device"):
        # tensor_parallel queries torch.mps.current_device(), which may be missing on some builds.
        def _mps_current_device() -> int:  # pragma: no cover - hardware-specific path
            return 0

        torch.mps.current_device = _mps_current_device  # type: ignore[assignment]

    block = make_tensor_parallel(block, config, tensor_parallel_devices, output_device=output_device)

    if quant_type != QuantType.NONE:
        normalized_qt = _normalize_quant_type(quant_type)
        if normalized_qt != QuantType.NONE:
            device_types = {device.type for device in tensor_parallel_devices}
            if device_types != {"cuda"}:
                raise ValueError(
                    f"{quant_type.name} quantization requires GPU (CUDA/ROCm) devices. "
                    "Re-run with --device cuda or disable quantization via --quantization none. "
                    f"Detected device types: {sorted(device_types)}."
                )
            block = quantize_module(block, quant_type=quant_type)

    for shard, device in zip(block.module_shards, block.devices):
        shard.to(device)

    if adapters:
        from agentgrid.utils.peft import add_adapter_to_block, create_lora_adapter, load_peft

        create_lora_adapter(block)
        for adapter_name in adapters:
            adapter_config, adapter_state_dict = load_peft(
                adapter_name,
                block_idx=block_index,
                **kwargs,
            )
            add_adapter_to_block(block, block_index, adapter_name, adapter_config, adapter_state_dict)

    block.eval()

    return block


def _normalize_quant_type(quant_type: QuantType) -> QuantType:
    """Map deprecated aliases to their canonical torchao equivalents."""
    if quant_type == QuantType.INT8:
        return QuantType.INT8_WEIGHT_ONLY
    if quant_type == QuantType.NF4:
        return QuantType.INT4_WEIGHT_ONLY
    return quant_type


def quantize_module(model: nn.Module, *, quant_type: QuantType) -> nn.Module:
    """Apply torchao weight-only quantization to all Linear layers (except lm_head/score)."""
    quant_type = _normalize_quant_type(quant_type)

    try:
        from torchao.quantization import quantize_, int4_weight_only, int8_weight_only
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "torchao is required for quantization. "
            "Install it (e.g. `pip install agent-grid[gpu]` or `pip install agent-grid[rocm]`) "
            "or disable quantization with --quantization none. "
            f"Original error: {exc}"
        ) from exc

    def _filter_fn(module: nn.Module, fqn: str) -> bool:
        """Only quantize nn.Linear layers, skip lm_head and score."""
        if not isinstance(module, nn.Linear):
            return False
        name = fqn.rsplit(".", 1)[-1] if "." in fqn else fqn
        return name not in ("lm_head", "score")

    if quant_type == QuantType.INT8_WEIGHT_ONLY:
        quantize_(model, int8_weight_only(), filter_fn=_filter_fn)
    elif quant_type == QuantType.INT4_WEIGHT_ONLY:
        quantize_(model, int4_weight_only(group_size=128), filter_fn=_filter_fn)
    else:
        raise ValueError(f"Unsupported quant_type='{quant_type}'")

    return model


def make_tensor_parallel(
    block: nn.Module, model_config: PretrainedConfig, devices: Sequence[torch.device], output_device: torch.device
) -> nn.Module:
    if model_config.model_type == "bloom":
        tp_config = get_bloom_config(model_config, devices)
        del tp_config.state_rules[re.compile(".*word_embeddings.weight$")]
    else:
        tp_config = None

    expected_heads = 0
    for submodule in block.modules():
        if isinstance(submodule, model_config.attn_class):
            expected_heads += submodule.config.num_attention_heads

    tp_block = tp.TensorParallel(block, devices, tensor_parallel_config=tp_config, output_device=output_device, delay_init=True)
    total_heads = 0
    for tp_shard in tp_block.module_shards:
        for submodule in tp_shard.modules():
            if isinstance(submodule, model_config.attn_class):
                total_heads += submodule.config.num_attention_heads
    assert total_heads == expected_heads, (
        f"Number of attention heads mismatch: sharded block has {total_heads}, original block has {expected_heads}"
    )
    return tp_block


def check_device_balance(devices: Sequence[torch.device]):
    if not all(device.type == "cuda" for device in devices):
        logger.warning("Running tensor parallelism on non-GPU devices; proceed at your own risk")
        return
    unique_device_capabilities = set(map(torch.cuda.get_device_capability, devices))
    if len(unique_device_capabilities) > 1:
        logger.warning(
            f"Found GPUs with uneven capabilities: {unique_device_capabilities}. "
            f"Using GPUs with different performance will cause the server to wait for the slowest GPU."
        )

    memory_per_device = tuple(torch.cuda.get_device_properties(device).total_memory for device in devices)
    used_memory = min(memory_per_device) * len(memory_per_device)
    wasted_memory_rate = (sum(memory_per_device) - used_memory) / sum(memory_per_device)
    if wasted_memory_rate > 0.05:
        logger.warning(
            f"GPU devices have highly uneven memory, {wasted_memory_rate * 100:.2f}% memory is wasted. "
            f"Consider running high-memory GPUs in a separate server."
        )
