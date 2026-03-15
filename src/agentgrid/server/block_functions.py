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
This module implements server-side computations on served blocks: forward, backward and inference; used by handler
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple, Union

import torch
from hivemind.compression.serialization import deserialize_torch_tensor, serialize_torch_tensor
from hivemind.moe.expert_uid import ExpertUID
from hivemind.proto import runtime_pb2
from hivemind.utils.logging import get_logger
from hivemind.utils.nested import nested_flatten

from agentgrid.data_structures import Handle, InferenceMetadata
from agentgrid.server.backend import TransformerBackend
from agentgrid.server.task_pool import PrioritizedTaskPool
from agentgrid.server.task_prioritizer import TaskPrioritizerBase
from agentgrid.utils.convert_block import QuantType
from agentgrid.utils.misc import DUMMY, is_dummy
from agentgrid.utils.packaging import unpack_args_kwargs


def _prepare_prompts(prompts: torch.Tensor, requested_backends: Sequence[TransformerBackend]) -> List[torch.Tensor]:
    """Prepare prompts for use in forward and backward passes."""
    if prompts is None or is_dummy(prompts):
        return [DUMMY] * len(requested_backends)
    if prompts.shape[0] != len(requested_backends):
        raise ValueError(f"Received {prompts.shape[0]} prompts for {len(requested_backends)} backends")
    split_prompts = prompts.to(requested_backends[0].dtype).split(1, dim=0)
    prepared = []
    for p in split_prompts:
        p = p.squeeze(0)
        if p.ndim == 2:
            p = p.unsqueeze(0)  # Assume batch=1 if missing batch dim
        prepared.append(p)
    return prepared


# We prioritize short inference requests and make them use a *merged* inference pool,
# so they are processed without interruptions and extra overheads
MAX_SHORT_INFERENCE_TOKENS = 128
# Token threshold for 4-bit quantized inference merging (torchao int4_weight_only)
MAX_INT4_SHORT_INFERENCE_TOKENS = 128

logger = get_logger(__name__)


async def run_rpc_forward(
    *flat_tensors: torch.Tensor,
    requested_backends: Sequence[TransformerBackend],
    active_adapter: str = "",
    prioritizer: TaskPrioritizerBase,
    points: int = 0,
    args_structure: Any = None,
) -> torch.Tensor:
    """
    Run forward pass on deserialized inputs and prompts, used by rpc_forward and rpc_forward_stream

    :param flat_tensors: a list of tensors that includes first layer inputs, optional prompts and extra tensors
    :note: some input tensors can be missing, in which case they will be replaced with dummy tensors (see is_dummy)
    :param requested_backends: a sequence of transformer blocks in the same order as they appear in forward pass
    :returns: hidden states after the last layer [batch_size, seq_length, hid_size]
    """
    if args_structure is not None:
        unpacked_args, unpacked_kwargs = unpack_args_kwargs(flat_tensors, args_structure)
    else:
        unpacked_args, unpacked_kwargs = flat_tensors, {}
    hidden_states = unpacked_args[0]
    prompts = unpacked_args[1]
    attention_mask = unpacked_args[2]
    position_ids = unpacked_args[3]
    position_embeddings = unpacked_args[4]

    dtype = requested_backends[0].dtype
    # Cast incoming tensors to float16 for cross-platform network safety, then to backend dtype
    hidden_states = hidden_states.to(torch.float16).to(dtype)
    if hidden_states.ndim != 3:
        raise ValueError(f"Hidden states must be a 3D tensor, got shape {hidden_states.shape}")
    prompts = _prepare_prompts(prompts, requested_backends)

    # Run a chain of requested backends
    point_per_backend = points / len(requested_backends) if len(requested_backends) > 0 else 0
    for i, (backend, prompt) in enumerate(zip(requested_backends, prompts)):
        if not is_dummy(prompt):
            if prompt.ndim < 2:
                logger.warning(f"Skipping prompt addition due to invalid ndim {prompt.ndim}")
                continue
            prompt_len = prompt.shape[-2]
            try:
                hidden_states[:, :prompt_len] += prompt
            except RuntimeError as e:
                logger.error(f"Shape mismatch in prompt addition: hidden {hidden_states.shape}, prompt {prompt.shape}, error: {e}")

        if not isinstance(backend.inference_pool, PrioritizedTaskPool):
            raise ValueError("agentgrid supports only prioritized pools")
        # TODO: For better perf, make point_per_backend non-uniform based on profiled layer costs
        priority = prioritizer.prioritize(
            hidden_states, points=point_per_backend, backend=backend, type="forward"
        )
        (hidden_states,) = await backend.forward_pool.submit_task(
            hidden_states,
            attention_mask,
            position_ids,
            position_embeddings,
            *unpacked_kwargs.values(),
            active_adapter,
            priority=priority,
        )
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
            raise ValueError(f"Output from {type(backend)} must be a single 3D tensor")

    return hidden_states





async def iterate_rpc_inference(
    requested_uids: Sequence[ExpertUID],
    requested_backends: Sequence[TransformerBackend],
    active_adapter: Optional[str],
    input_iterator: AsyncIterator[Tuple[runtime_pb2.ExpertRequest, dict]],
    cache_handles: Sequence[Sequence[Handle]],
    *,
    max_length: int,
    prioritizer: TaskPrioritizerBase,
    points: int,
    quant_type: QuantType,
    args_structure: Any = None,
) -> AsyncIterator[Tuple[Sequence[runtime_pb2.Tensor], bool, Dict]]:
    if len(cache_handles) != len(requested_backends):
        raise ValueError(f"Cache handles length {len(cache_handles)} != backends {len(requested_backends)}")

    prefix_length = 0
    point_per_piece = points / max_length if max_length > 0 else 0.0

    async for request, step_metadata in input_iterator:
        if "start_from_position" in step_metadata:
            start_from_position = step_metadata["start_from_position"]
            if prefix_length < start_from_position:
                raise ValueError(f"prefix_length={prefix_length} < start_from_position={start_from_position}")
            prefix_length = start_from_position

        flat_tensors = tuple(deserialize_torch_tensor(tensor) for tensor in request.tensors)
        if args_structure is not None:
            unpacked_args, unpacked_kwargs = unpack_args_kwargs(flat_tensors, args_structure)
        else:
            unpacked_args, unpacked_kwargs = flat_tensors, {}

        hidden_states = unpacked_args[0]
        prompts = unpacked_args[1]
        attention_mask = unpacked_args[2]
        position_ids = unpacked_args[3]
        position_embeddings = unpacked_args[4]
        batch_size, length_increment, _ = hidden_states.shape

        # Cast incoming tensors to float16 for cross-platform network safety, then to backend dtype
        hidden_states = hidden_states.to(torch.float16).to(requested_backends[0].dtype)

        # parse deep prompts (optional argument)
        if prompts is None or is_dummy(prompts):
            prompts = [DUMMY] * len(requested_backends)
        else:
            prompts = _prepare_prompts(prompts, requested_backends)

        if prefix_length + length_increment > max_length:
            raise ValueError(
                f"Maximum length exceeded: prefix {prefix_length} + current {length_increment}"
                f" exceeds pre-allocated maximum {max_length}"
            )

        merge_max_tokens = MAX_INT4_SHORT_INFERENCE_TOKENS if quant_type in (QuantType.NF4, QuantType.INT4_WEIGHT_ONLY) else MAX_SHORT_INFERENCE_TOKENS
        can_merge_pools = batch_size * length_increment <= merge_max_tokens
        priority = prioritizer.prioritize(
            hidden_states,
            points=point_per_piece,
            requested_uids=requested_uids,
            type="inference",
        )

        # A client may pass a tensor with 0 tokens. This is a special case that occurs, e.g.
        # when user wants to pre-allocate cache or check that server *can* allocate that cache.
        if hidden_states.numel() > 0:
            if hidden_states.ndim != 3:
                raise ValueError(f"Hidden states must be a single 3D tensor, got {hidden_states.shape}")
            if can_merge_pools:
                inference_infos = tuple(
                    InferenceMetadata(uid, prefix_length, tuple(handles), active_adapter)
                    for uid, handles in zip(requested_uids, cache_handles)
                )
                (hidden_states,) = await requested_backends[0].inference_pool.submit_task(
                    hidden_states, attention_mask, position_ids, position_embeddings, inference_infos, *prompts, priority=priority
                )
            else:
                for backend, uid, handles, prompt in zip(requested_backends, requested_uids, cache_handles, prompts):
                    inference_infos = (InferenceMetadata(uid, prefix_length, tuple(handles), active_adapter),)
                    (hidden_states,) = await backend.inference_pool.submit_task(
                        hidden_states, attention_mask, position_ids, position_embeddings, inference_infos, prompt, priority=priority
                    )

        # serialize and send last layer outputs, enforcing float16 at the network boundary
        output_tensors = [
            serialize_torch_tensor(result.to(torch.float16), proto.compression, allow_inplace=True)
            for result, proto in zip((hidden_states,), nested_flatten(requested_backends[-1].outputs_schema))
        ]
        can_push = all(is_dummy(p) for p in prompts)
        yield output_tensors, can_push, step_metadata

        # prepare for next step
        prefix_length += length_increment