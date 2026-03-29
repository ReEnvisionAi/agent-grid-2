#
# Copyright (c) 2025 ReEnvision AI, LLC. All rights reserved.
#
# This software is the confidential and proprietary information of
# ReEnvision AI, LLC ("Confidential Information"). You shall not
# disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with ReEnvision AI, LLC.
#

from typing import Optional, Union

import torch
from accelerate import init_empty_weights
from transformers import PretrainedConfig, PreTrainedModel

from agentgrid.models.gpt_oss.block import WrappedGptOssBlock
from agentgrid.models.qwen2.block import WrappedQwen2Block
from agentgrid.models.qwen3_moe.block import WrappedQwen3MoeBlock
from agentgrid.models.qwen3.block import WrappedQwen3Block
from agentgrid.models.nemotron import WrappedNemotronBlock
from agentgrid.utils.convert_block import QuantType
from agentgrid.utils.misc import get_size_in_bytes


def resolve_block_dtype(config: PretrainedConfig, dtype: Union[str, torch.dtype]) -> torch.dtype:
    """If dtype is "auto", resolves it to float16 for cross-platform network safety. Returns `dtype` intact otherwise."""
    if dtype not in ("auto", None):
        return dtype
    # Default to float16 for cross-platform compatibility (bfloat16 crashes Apple Silicon nodes)
    return torch.float16


def get_block_size(
    config: PretrainedConfig,
    location: str,
    *,
    dtype: Optional[Union[str, torch.dtype]] = None,
    quant_type: QuantType = QuantType.NONE,
    eps: float = 0.01,  # eps accounts for ~1% of metainfo for tensor descriptions, quantization tables, etc.
    block_index: int = 0,
) -> int:
    if location == "memory":
        assert (
            dtype is not None and quant_type is not None
        ), 'get_block_size(..., location="memory") requires to specify dtype and quant_type for calculations'

    with init_empty_weights(include_buffers=False):
        block = get_model_block(config, block_index)
        n_params = sum(param.numel() for param in block.parameters())

    if location == "memory":
        if quant_type == QuantType.NONE:
            dtype = resolve_block_dtype(config, dtype)
            bytes_per_value = get_size_in_bytes(dtype)
        elif quant_type in (QuantType.INT8, QuantType.INT8_WEIGHT_ONLY):
            bytes_per_value = 1
        elif quant_type in (QuantType.NF4, QuantType.INT4_WEIGHT_ONLY):
            bytes_per_value = 4.25 / 8  # ~0.53 bytes per value for 4-bit weight-only quantization
        else:
            raise ValueError(f"Unsupported quant_type={quant_type}")
    elif location == "disk":
        dtype = resolve_block_dtype(config, "auto")
        bytes_per_value = get_size_in_bytes(dtype)

    return round(n_params * bytes_per_value * (1 + eps))


def get_model_block(config, layer_idx: int = 0):
    """
    The function to create a model block based on the block class
    kwargs argument **only** is necessary for specific classes, like Mixtral.
    They will not be passed to other block constructors.
    """
    if config.block_class in {
        WrappedQwen2Block,
        WrappedQwen3MoeBlock,
        WrappedNemotronBlock,
    }:
        return config.block_class(config, layer_idx)
    if config.block_class == WrappedQwen3Block or config.block_class == WrappedGptOssBlock:
        autoset = getattr(PreTrainedModel, "_autoset_attn_implementation", None)
        if autoset is not None:
            config = autoset(config)
        return config.block_class(config, layer_idx)
    return config.block_class(config)
