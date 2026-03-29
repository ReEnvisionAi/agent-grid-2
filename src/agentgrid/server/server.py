#
# Copyright (c) 2025-2026 ReEnvision AI, LLC. All rights reserved.
#
# This software is the confidential and proprietary information of
# ReEnvision AI, LLC ("Confidential Information"). You shall not
# disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with ReEnvision AI, LLC.
#

from __future__ import annotations

import gc
import math
import multiprocessing as mp
import os
import platform
import random
import sys
import threading
import time
from typing import Dict, List, Optional, Sequence, Union

import hivemind
import psutil
import torch
import torch.mps
from hivemind import DHT, MAX_DHT_TIME_DISCREPANCY_SECONDS, BatchTensorDescriptor, get_dht_time
from hivemind.moe.server.layers import add_custom_models_from_file
from hivemind.moe.server.runtime import Runtime
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils.logging import get_logger
from transformers import PretrainedConfig

import agentgrid
from agentgrid.constants import DTYPE_MAP, PUBLIC_INITIAL_PEERS
from agentgrid.data_structures import CHAIN_DELIMITER, UID_DELIMITER, ModelInfo, ServerInfo, ServerState, parse_uid
from agentgrid.server import block_selection
from agentgrid.server.backend import TransformerBackend, merge_inference_pools_inplace
from agentgrid.server.block_utils import get_block_size, resolve_block_dtype
from agentgrid.server.from_pretrained import load_pretrained_block
from agentgrid.server.handler import TransformerConnectionHandler
from agentgrid.server.memory_cache import MemoryCache
from agentgrid.server.reachability import ReachabilityProtocol, check_direct_reachability, validate_reachability
from agentgrid.server.throughput import get_dtype_name, get_device_name, get_server_throughput
from agentgrid.utils.auto_config import AutoDistributedConfig
from agentgrid.utils.convert_block import QuantType, check_device_balance, convert_block
from agentgrid.utils.dht import declare_active_modules, get_remote_module_infos
from agentgrid.utils.misc import get_size_in_bytes
from agentgrid.utils.ping import PingAggregator
from agentgrid.utils.random import sample_up_to
#from agentgrid.utils.version import get_compatible_model_repo

logger = get_logger(__name__)


class BlockLoadingOutOfMemoryError(RuntimeError):
    """Raised when the server runs out of GPU memory while loading transformer blocks."""

    def __init__(self, *, attempted_blocks: int, loaded_blocks: int, original_exception: BaseException):
        message = (
            f"Out of memory while loading {attempted_blocks} transformer blocks "
            f"(successfully loaded {loaded_blocks})."
        )
        super().__init__(message)
        self.attempted_blocks = attempted_blocks
        self.loaded_blocks = loaded_blocks
        self.original_exception = original_exception


def _is_out_of_memory_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    # Be more strict: only treat as OOM if it's clearly about CUDA memory allocation
    # Avoid false positives from messages about "allocatable memory" or other uses of "memory"
    if "out of memory" in message:
        # Check that it's actually about CUDA/GPU memory, not just a message about allocatable bytes
        return ("cuda" in message or "gpu" in message or "allocated" in message)
    return False


class Server:
    """
    Runs ModuleContainer, periodically checks that the network is balanced,
    restarts the ModuleContainer with other layers if the imbalance is significant
    """

    def __init__(
        self,
        *,
        initial_peers: List[str],
        dht_prefix: Optional[str],
        converted_model_name_or_path: str,
        public_name: Optional[str] = None,
        throughput: Union[float, str],
        num_blocks: Optional[int] = None,
        block_indices: Optional[str] = None,
        num_handlers: int = 8,
        inference_max_length: Optional[int] = None,
        min_batch_size: int = 1,
        max_batch_size: Optional[int] = None,
        max_chunk_size_bytes: int = 256 * 1024 * 1024,
        max_alloc_timeout: float = 600,
        attn_cache_tokens: Optional[int] = None,
        torch_dtype: str = "auto",
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_disk_space: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        compression=CompressionType.NONE,
        stats_report_interval: Optional[int] = None,
        custom_module_path=None,
        update_period: float = 60,
        expiration: Optional[float] = None,
        request_timeout: float = 3 * 60,
        session_timeout: float = 30 * 60,
        step_timeout: float = 5 * 60,
        prefetch_batches: int = 1,
        sender_threads: int = 1,
        balance_quality: float = 0.75,
        mean_balance_check_period: float = 120,
        mean_block_selection_delay: float = 5,
        token: Optional[Union[str, bool]] = None,
        quant_type: Optional[QuantType] = None,
        tensor_parallel_devices: Optional[Sequence[torch.device]] = None,
        skip_reachability_check: bool = False,
        reachable_via_relay: Optional[bool] = None,
        use_relay: bool = True,
        use_auto_relay: bool = True,
        adapters: Sequence[str] = (),
        warmup_tokens_interval: Optional[int] = None,
        compile_block: bool = False,
        **kwargs,
    ):
        """Create a server with one or more bloom blocks. See run_server.py for documentation."""

        # converted_model_name_or_path = get_compatible_model_repo(converted_model_name_or_path)
        self.converted_model_name_or_path = converted_model_name_or_path

        self.num_handlers = num_handlers
        self.compression = compression
        self.stats_report_interval, self.update_period = stats_report_interval, update_period
        self.prefetch_batches, self.sender_threads = prefetch_batches, sender_threads
        self.revision, self.token = revision, token

        if custom_module_path is not None:
            add_custom_models_from_file(custom_module_path)

        self.block_config = AutoDistributedConfig.from_pretrained(
            converted_model_name_or_path,
            token=token,
            revision=revision,
        )

        if dht_prefix is None:
            dht_prefix = self.block_config.dht_prefix
        assert UID_DELIMITER not in dht_prefix and CHAIN_DELIMITER not in dht_prefix, (
            f"DHT prefix should not contain '{UID_DELIMITER}' or '{CHAIN_DELIMITER}'. "
            f"Please specify another --dht_prefix manually when starting a server"
        )
        self.dht_prefix = dht_prefix

        if expiration is None:
            expiration = max(2 * update_period, MAX_DHT_TIME_DISCREPANCY_SECONDS)
        self.expiration = expiration

        self.request_timeout = request_timeout
        self.session_timeout, self.step_timeout = session_timeout, step_timeout

        self.module_uids = [
            f"{self.dht_prefix}{UID_DELIMITER}{block_index}"
            for block_index in range(self.block_config.num_hidden_layers)
        ]

        if reachable_via_relay is None:
            is_reachable = check_direct_reachability(initial_peers=initial_peers, use_relay=False, **kwargs)
            reachable_via_relay = is_reachable is False  # if can't check reachability (returns None), run a full peer
            logger.info(f"This server is accessible {'via relays' if reachable_via_relay else 'directly'}")
        self.dht = DHT(
            initial_peers=initial_peers,
            start=True,
            num_workers=self.block_config.num_hidden_layers,
            use_relay=use_relay,
            use_auto_relay=use_auto_relay,
            client_mode=reachable_via_relay,
            **kwargs,
        )
        self.reachability_protocol = ReachabilityProtocol.attach_to_dht(self.dht) if not reachable_via_relay else None

        visible_maddrs_str = [str(a) for a in self.dht.get_visible_maddrs()]
        if initial_peers == PUBLIC_INITIAL_PEERS:
            logger.info("Connecting to the public swarm")
        else:
            logger.info(f"Connecting to a private swarm, initial peers: {initial_peers}")
        logger.info(f"Running a server on {visible_maddrs_str}")
        self.should_validate_reachability = not skip_reachability_check and initial_peers == PUBLIC_INITIAL_PEERS

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                logger.warning(
                    "CUDA backend requested but not available; switching to Apple Metal (MPS)."
                )
                device = torch.device("mps")
            else:
                logger.warning(
                    "CUDA backend requested but this PyTorch build lacks CUDA support; falling back to CPU."
                )
                device = torch.device("cpu")
        if device.type == "cuda" and device.index is None:
            device = torch.device(device.type, index=0)
        self.device = device

        if self.device.type == "cuda":
            from agentgrid.utils.device_utils import is_rocm, get_gpu_name
            if is_rocm():
                logger.info("Using AMD ROCm (HIP) backend: %s", get_gpu_name(device.index or 0))
            else:
                logger.info("Using NVIDIA CUDA backend: %s", get_gpu_name(device.index or 0))
        elif self.device.type == "mps":
            logger.info("Using Apple Metal (MPS) backend")
            if hasattr(torch.backends, "mps"):
                try:
                    torch.backends.mps.matmul.allow_fp16_reduction(True)
                except AttributeError:
                    logger.debug("MPS backend does not expose matmul fp16 reduction toggle")
            torch.set_float32_matmul_precision("medium")
        elif self.device.type == "cpu":
            logger.info(
                "Using CPU backend; throughput will be significantly lower than GPU."
            )

        torch_dtype = resolve_block_dtype(self.block_config, DTYPE_MAP[torch_dtype])
        if device.type == "cpu" and torch_dtype == torch.float16:
            raise ValueError(
                f"Type float16 is not supported on CPU. Please use --torch_dtype float32 or --torch_dtype bfloat16"
            )
        if device.type == "mps" and torch_dtype == torch.bfloat16:
            logger.warning(f"Type bfloat16 is not supported on MPS, using float16 instead")
            torch_dtype = torch.float16
        self.torch_dtype = torch_dtype

        if tensor_parallel_devices is None:
            tensor_parallel_devices = (device,)
        self.tensor_parallel_devices = tuple(map(torch.device, tensor_parallel_devices))
        if any(dev.type == "cuda" for dev in self.tensor_parallel_devices) and not torch.cuda.is_available():
            raise ValueError(
                "CUDA tensor-parallel devices were specified, but this environment does not have CUDA support. "
                "Install a CUDA-enabled PyTorch build or remove --tensor_parallel_devices."
            )
        if len(self.tensor_parallel_devices) > 1:
            device_types = {dev.type for dev in self.tensor_parallel_devices}
            if device_types != {"cuda"}:
                raise ValueError(
                    "Tensor parallelism requires GPU (CUDA/ROCm) devices. "
                    f"Detected device types: {sorted(device_types)}"
                )
            logger.info(f"Model weights will be split between {', '.join(tensor_parallel_devices)}")
            check_device_balance(self.tensor_parallel_devices)

        tensor_parallel_device_types = {dev.type for dev in self.tensor_parallel_devices}
        user_requested_quant_type = quant_type is not None

        if quant_type is None:
            if device.type == "cuda" and tensor_parallel_device_types == {"cuda"}:
                quant_type = QuantType.INT4_WEIGHT_ONLY
            else:
                quant_type = QuantType.NONE
                if device.type != "cuda":
                    logger.info(
                        "Defaulting to --quantization none for %s devices; GPU (CUDA/ROCm) is required for quantization.",
                        device.type.upper(),
                    )
                elif tensor_parallel_device_types != {"cuda"}:
                    logger.info(
                        "Defaulting to --quantization none because tensor parallel devices include %s.",
                        sorted(tensor_parallel_device_types),
                    )
        else:
            if quant_type != QuantType.NONE and device.type != "cuda":
                message = (
                    f"{quant_type.name} quantization requires a GPU (CUDA/ROCm) device, but primary device is {device.type.upper()}. "
                    "Defaulting to --quantization none."
                )
                if user_requested_quant_type:
                    logger.warning(message)
                    quant_type = QuantType.NONE
                else:
                    raise ValueError(message)
            if quant_type != QuantType.NONE and tensor_parallel_device_types != {"cuda"}:
                message = (
                    f"{quant_type.name} quantization requires CUDA tensor-parallel devices, "
                    f"but detected {sorted(tensor_parallel_device_types)}. "
                    "Defaulting to --quantization none."
                )
                if user_requested_quant_type:
                    logger.warning(message)
                    quant_type = QuantType.NONE
                else:
                    raise ValueError(message)

        original_quant_type = quant_type
        if quant_type in (QuantType.NF4, QuantType.INT4_WEIGHT_ONLY) and getattr(self.block_config, "model_type", None) == "gpt_oss":
            logger.warning(
                "4-bit quantization is not supported for GPT-OSS blocks; falling back to float16 weights"
            )
            quant_type = QuantType.NONE
            # For GPT-OSS with failed quantization, we need to account for higher memory usage
            self.gpt_oss_quantization_fallback = True
            self.memory_overhead_multiplier = 4.0  # 4bit->float16 is ~4x memory
        else:
            self.gpt_oss_quantization_fallback = False
            self.memory_overhead_multiplier = 1.0

        self.quant_type = quant_type
        self.original_requested_quant_type = original_quant_type
        self.compile_block = compile_block

        logger.info(f"Model weights are loaded in {get_dtype_name(torch_dtype, quant_type)} format")
        if self.gpt_oss_quantization_fallback:
            logger.warning(f"GPT-OSS quantization fallback: Memory usage will be {self.memory_overhead_multiplier}x higher than originally expected")

        if hasattr(self.block_config, "block_configs") and self.block_config.block_configs is not None:
            num_key_value_groups = [
                self.block_config.num_attention_heads // b.attention.n_heads_in_group if b.attention.n_heads_in_group else None
                for b in self.block_config.block_configs
            ]
            is_multiquery_attn = any(g is not None and g > 1 for g in num_key_value_groups)
        elif hasattr(self.block_config, "num_key_value_heads") and isinstance(
            self.block_config.num_key_value_heads, list
        ):
            num_key_value_groups = [
                self.block_config.num_attention_heads // n_kv_h for n_kv_h in self.block_config.num_key_value_heads
            ]
            is_multiquery_attn = any(g > 1 for g in num_key_value_groups)
        else:
            num_key_value_groups = self.block_config.num_key_value_groups
            is_multiquery_attn = num_key_value_groups > 1

        if max_batch_size is None:
            max_batch_size = 8192 if is_multiquery_attn else 2048
        if inference_max_length is None:
            # Try to get max length from model config
            config_max_length = getattr(self.block_config, "max_position_embeddings", None)
            if config_max_length is not None and isinstance(config_max_length, int):
                inference_max_length = config_max_length
                logger.info(f"Model has 'max_position_embeddings' of {self.block_config.max_position_embeddings}")
            else:
                logger.info("Model does not have 'max_position_embeddings")
                logger.info(self.block_config)
                inference_max_length = 8192 if is_multiquery_attn else 2048
        self.min_batch_size, self.max_batch_size = min_batch_size, max_batch_size
        self.inference_max_length = inference_max_length
        self.max_chunk_size_bytes = max_chunk_size_bytes
        self.max_alloc_timeout = max_alloc_timeout

        # For attention cache in GPU or RAM
        if attn_cache_tokens is None:
            default_cache_tokens = 16384 if is_multiquery_attn else 4096
            # Ensure cache is at least as large as the max sequence length we support
            attn_cache_tokens = max(default_cache_tokens, inference_max_length)


        base_cache_values_per_block = 2 * self.block_config.hidden_size * attn_cache_tokens
        if isinstance(num_key_value_groups, list):
            cache_values_per_block = [base_cache_values_per_block // g if g else 0 for g in num_key_value_groups]
            avg_cache_values_per_block = sum(cache_values_per_block) / len(cache_values_per_block) if cache_values_per_block else 0
            self._cache_bytes_per_block = avg_cache_values_per_block * get_size_in_bytes(self.torch_dtype)
        else:
            cache_values_per_block = base_cache_values_per_block // num_key_value_groups
            self._cache_bytes_per_block = cache_values_per_block * get_size_in_bytes(self.torch_dtype)

        # Apply GPT-OSS quantization fallback memory overhead correction
        if self.gpt_oss_quantization_fallback:
            self._cache_bytes_per_block = int(self._cache_bytes_per_block * self.memory_overhead_multiplier)
            logger.info(f"Applied GPT-OSS quantization fallback correction: cache bytes per block increased to {self._cache_bytes_per_block // (1024**2):.0f} MB")

        # For disk cache
        self.cache_dir = cache_dir
        self.max_disk_space = max_disk_space
        self.adapters = adapters

        assert num_blocks is None or block_indices is None, "Please specify num_blocks or block_indices, not both"
        if num_blocks is None and block_indices is None:
            num_blocks = self._choose_num_blocks()
        if num_blocks is not None:
            num_blocks = min(num_blocks, self.block_config.num_hidden_layers)
        if block_indices is not None:
            try:
                start_block, end_block = [int(index.strip()) for index in block_indices.split(":")]
            except Exception as e:
                raise ValueError(f"Failed to parse `--block_indices {block_indices}`, must be start:end (e.g. 0:18)")
            block_indices = range(start_block, end_block)
            num_blocks = len(block_indices)
        self.strict_block_indices, self.num_blocks = block_indices, num_blocks

        gib = 1024**3
        self.attn_cache_bytes = self._cache_bytes_per_block * num_blocks
        logger.info(f"Attention cache for all blocks will consume up to {self.attn_cache_bytes / gib:.2f} GiB")

        assert isinstance(throughput, float) or throughput in ["auto", "eval", "dry_run"]
        if throughput in ["auto", "eval", "dry_run"]:
            force_eval = throughput in ["eval", "dry_run"]
            throughput_info = get_server_throughput(
                converted_model_name_or_path,
                self.block_config,
                device,
                torch_dtype,
                num_blocks=num_blocks,
                quant_type=quant_type,
                tensor_parallel_devices=self.tensor_parallel_devices,
                reachable_via_relay=reachable_via_relay,
                force_eval=force_eval,
                cache_dir=cache_dir,
            )
            if throughput == "dry_run":
                logger.info("Finished estimating throughput, exiting")
                sys.exit(0)
        else:
            throughput_info = {"throughput": throughput}

        operating_system = self._get_operating_system()
        video_card = self._get_video_card()
        self.server_info = ServerInfo(
            state=ServerState.JOINING,
            public_name=public_name,
            version=agentgrid.__version__,
            adapters=tuple(adapters),
            torch_dtype=str(torch_dtype).replace("torch.", ""),
            quant_type=quant_type.name.lower(),
            using_relay=reachable_via_relay,
            operating_system=operating_system,
            video_card=video_card,
            **throughput_info,
        )
        if operating_system or video_card:
            env_bits = []
            if operating_system:
                env_bits.append(f"OS: {operating_system}")
            if video_card:
                env_bits.append(f"GPU: {video_card}")
            logger.info("Server environment details - %s", "; ".join(env_bits))
        self._throughput_info = throughput_info
        self._reachable_via_relay = reachable_via_relay
        self.model_info = ModelInfo(num_blocks=self.block_config.num_hidden_layers)
        if not os.path.isdir(converted_model_name_or_path):
            self.model_info.repository = "https://huggingface.co/" + converted_model_name_or_path

        self.balance_quality = balance_quality
        self.mean_balance_check_period = mean_balance_check_period
        self.mean_block_selection_delay = mean_block_selection_delay
        self.warmup_tokens_interval = warmup_tokens_interval

        self.module_container = None
        self.stop = threading.Event()

    def _get_operating_system(self) -> Optional[str]:
        system = platform.system()
        if not system:
            return None
        release = platform.release()
        if release:
            return f"{system} {release}"
        return system

    def _get_video_card(self) -> Optional[str]:
        devices = getattr(self, "tensor_parallel_devices", ())
        if not devices:
            return None
        device_names = []
        for device in devices:
            try:
                device_names.append(get_device_name(device))
            except Exception:
                device_names.append(device.type.upper())
        if not device_names:
            return None

        counts: Dict[str, int] = {}
        order: List[str] = []
        for name in device_names:
            if name not in counts:
                counts[name] = 0
                order.append(name)
            counts[name] += 1

        parts = []
        for name in order:
            count = counts[name]
            parts.append(f"{count}x {name}" if count > 1 else name)
        return ", ".join(parts)

    def _choose_num_blocks(self) -> int:
        num_devices = len(self.tensor_parallel_devices) if self.tensor_parallel_devices else 1

        if num_devices > 1:
            if self.device.type != "cuda":
                raise ValueError(
                    "Tensor parallelism is supported only on CUDA devices. "
                    f"Primary device is {self.device.type.upper()}"
                )
            memory_per_device = tuple(
                torch.cuda.get_device_properties(device).total_memory for device in self.tensor_parallel_devices
            )
            total_memory = min(memory_per_device) * num_devices
            if max(memory_per_device) / min(memory_per_device) > 1.5:
                raise ValueError(
                    "GPU devices have highly uneven memory, which makes tensor parallelism inefficient. "
                    "Please launch individual servers on each GPU or set --num_blocks manually to "
                    "override this exception."
                )
        elif self.device.type == "cuda":
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
        else:
            if self.device.type in ("cpu", "mps"):
                logger.warning(
                    f"Auto-selecting number of transformer blocks for a {self.device.type.upper()} server. "
                    "Override with --num_blocks to fine-tune performance."
                )
            total_memory = psutil.virtual_memory().available

        gib = 1024**3
        autograd_memory = 0  # Removed backward pass, so autograd_memory is 0

        avg_block_size = self._get_avg_block_size_in_bytes()
        total_memory_per_block = avg_block_size + self._cache_bytes_per_block
        if self.adapters:
            # Delay import of agentgrid.utils.peft to keep it optional
            from agentgrid.utils.peft import estimate_adapter_memory_per_block

            total_memory_per_block += estimate_adapter_memory_per_block(
                self.block_config,
                self.torch_dtype,
                self.adapters,
                token=self.token,
                cache_dir=self.cache_dir,
                max_disk_space=self.max_disk_space,
            )

        num_blocks = math.floor((total_memory - autograd_memory) / total_memory_per_block)
        assert num_blocks >= 1, "Not enough memory to serve even one transformer block"

        num_blocks = min(num_blocks, self.block_config.num_hidden_layers)
        logger.info(
            f"Server will fill available {self.device.type.upper()} memory with {num_blocks} transformer blocks. "
            f"Specify a smaller --num_blocks to reserve headroom."
        )
        return num_blocks

    def _get_avg_block_size_in_bytes(self) -> float:
        if hasattr(self.block_config, "block_configs") and self.block_config.block_configs is not None:
            block_sizes = []
            for i, b in enumerate(self.block_config.block_configs):
                if b.attention.n_heads_in_group is not None:
                    block_sizes.append(
                        get_block_size(
                            self.block_config, "memory", dtype=self.torch_dtype, quant_type=self.quant_type, block_index=i
                        )
                    )
            return sum(block_sizes) / len(block_sizes) if block_sizes else 0
        elif hasattr(self.block_config, "num_key_value_heads") and isinstance(
            self.block_config.num_key_value_heads, list
        ):
            block_sizes = [
                get_block_size(
                    self.block_config, "memory", dtype=self.torch_dtype, quant_type=self.quant_type, block_index=i
                )
                for i in range(self.block_config.num_hidden_layers)
            ]
            return sum(block_sizes) / len(block_sizes) if block_sizes else 0
        else:
            return get_block_size(
                self.block_config, "memory", dtype=self.torch_dtype, quant_type=self.quant_type
            )

    def _update_throughput_info(self) -> None:
        throughput_info = get_server_throughput(
            self.converted_model_name_or_path,
            self.block_config,
            self.device,
            self.torch_dtype,
            num_blocks=self.num_blocks,
            quant_type=self.quant_type,
            tensor_parallel_devices=self.tensor_parallel_devices,
            reachable_via_relay=self._reachable_via_relay,
            cache_dir=self.cache_dir,
        )
        self._throughput_info = throughput_info
        self.server_info.throughput = throughput_info["throughput"]
        self.server_info.inference_rps = throughput_info.get("inference_rps")
        self.server_info.forward_rps = throughput_info.get("forward_rps")
        self.server_info.network_rps = throughput_info.get("network_rps")

    def run(self):
        while True:
            block_indices = self._choose_blocks()
            try:
                self.module_container = ModuleContainer.create(
                    dht=self.dht,
                    dht_prefix=self.dht_prefix,
                    converted_model_name_or_path=self.converted_model_name_or_path,
                    block_config=self.block_config,
                    attn_cache_bytes=self.attn_cache_bytes,
                    server_info=self.server_info,
                    model_info=self.model_info,
                    block_indices=block_indices,
                    num_handlers=self.num_handlers,
                    min_batch_size=self.min_batch_size,
                    max_batch_size=self.max_batch_size,
                    max_chunk_size_bytes=self.max_chunk_size_bytes,
                    max_alloc_timeout=self.max_alloc_timeout,
                    inference_max_length=self.inference_max_length,
                    torch_dtype=self.torch_dtype,
                    cache_dir=self.cache_dir,
                    max_disk_space=self.max_disk_space,
                    device=self.device,
                    compression=self.compression,
                    stats_report_interval=self.stats_report_interval,
                    update_period=self.update_period,
                    expiration=self.expiration,
                    request_timeout=self.request_timeout,
                    session_timeout=self.session_timeout,
                    step_timeout=self.step_timeout,
                    prefetch_batches=self.prefetch_batches,
                    sender_threads=self.sender_threads,
                    revision=self.revision,
                    token=self.token,
                    quant_type=self.quant_type,
                    tensor_parallel_devices=self.tensor_parallel_devices,
                    should_validate_reachability=self.should_validate_reachability,
                    warmup_tokens_interval=self.warmup_tokens_interval,
                    compile_block=self.compile_block,
                    gpt_oss_fallback=getattr(self, 'gpt_oss_quantization_fallback', False),
                    start=True,
                )
            except BlockLoadingOutOfMemoryError as exc:
                if exc.loaded_blocks <= 0:
                    logger.error(
                        "Out of memory before loading the first transformer block. Original error: %s",
                        exc.original_exception,
                    )
                    raise exc.original_exception

                previous_num_blocks = self.num_blocks
                self.num_blocks = exc.loaded_blocks
                self.attn_cache_bytes = self._cache_bytes_per_block * self.num_blocks
                self._update_throughput_info()

                gib = 1024**3
                logger.warning(
                    "Ran out of GPU memory while loading blocks (attempted %s, loaded %s). "
                    "Retrying with num_blocks=%s (attn cache %.2f GiB).",
                    previous_num_blocks,
                    exc.loaded_blocks,
                    self.num_blocks,
                    self.attn_cache_bytes / gib,
                )
                # Log original exception at WARNING level to ensure it's visible
                logger.warning("Original OOM exception: %s: %s", type(exc.original_exception).__name__, exc.original_exception)

                self._clean_memory_and_fds()
                continue

            try:
                self.module_container.ready.wait()

                while True:
                    timeout = random.random() * 2 * self.mean_balance_check_period
                    if self.stop.wait(timeout):
                        return

                    if not self.module_container.is_healthy():
                        logger.warning("One of subprocesses crashed, restarting the server")
                        break

                    if self._should_choose_other_blocks():
                        logger.info("Swarm is imbalanced, server will load other blocks")
                        break  # Stop serving this set of modules
            finally:
                self.module_container.shutdown()

            self._clean_memory_and_fds()

    def _clean_memory_and_fds(self):
        self.module_container = None
        gc.collect()  # In particular, this closes unused file descriptors

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

            allocated_vram = torch.cuda.memory_allocated(self.device)
            reserved_vram = torch.cuda.memory_reserved(self.device)
            gib = 1024**3
            logger.info(
                f"Cleaning up, left {allocated_vram / gib:.1f} GiB allocated memory, "
                f"{reserved_vram / gib:.1f} GiB reserved memory"
            )
        elif self.device.type == "mps":
            torch.mps.empty_cache()

    def _choose_blocks(self) -> List[int]:
        if self.strict_block_indices is not None:
            return self.strict_block_indices

        # If multiple servers (e.g., launched on the same machine by a script) get to this line at the same time,
        # this delay decreases the probability of a race condition while choosing the best blocks to serve.
        time.sleep(random.random() * 2 * self.mean_block_selection_delay)
        module_infos = get_remote_module_infos(self.dht, self.module_uids, latest=True)
        return block_selection.choose_best_blocks(self.num_blocks, module_infos)

    def _should_choose_other_blocks(self) -> bool:
        if self.strict_block_indices is not None:
            return False

        module_infos = get_remote_module_infos(self.dht, self.module_uids, latest=True)
        return block_selection.should_choose_other_blocks(self.dht.peer_id, module_infos, self.balance_quality)

    def shutdown(self, timeout: Optional[float] = 5):
        self.stop.set()
        if self.module_container is not None and self.module_container.is_alive():
            self.module_container.join(timeout)

        if self.reachability_protocol is not None:
            self.reachability_protocol.shutdown()
        self.dht.shutdown()
        self.dht.join()


class ModuleContainer(threading.Thread):
    """Serves a set of specific Bloom layers for inference, forward, and backward. Announces itself over the DHT."""

    # noinspection PyMethodOverriding
    @classmethod
    def create(
        cls,
        *,
        dht: DHT,
        dht_prefix: str,
        converted_model_name_or_path: str,
        block_config: PretrainedConfig,
        attn_cache_bytes: int,
        server_info: ServerInfo,
        model_info: ModelInfo,
        block_indices: List[int],
        min_batch_size: int,
        max_batch_size: int,
        max_chunk_size_bytes: int,
        max_alloc_timeout: float,
        torch_dtype: torch.dtype,
        cache_dir: str,
        max_disk_space: int,
        device: Union[str, torch.device],
        compression: CompressionType,
        update_period: float,
        expiration: Optional[float],
        revision: Optional[str],
        token: Optional[Union[str, bool]],
        quant_type: QuantType,
        tensor_parallel_devices: Sequence[torch.device],
        should_validate_reachability: bool,
        warmup_tokens_interval: Optional[int] = None,
        compile_block: bool = False,
        gpt_oss_fallback: bool = False,
        **kwargs,
    ) -> ModuleContainer:
        module_uids = [f"{dht_prefix}{UID_DELIMITER}{block_index}" for block_index in block_indices]
        # Apply additional safety buffer for GPT-OSS quantization fallback
        cache_bytes = attn_cache_bytes
        if gpt_oss_fallback:
            # Add 20% safety buffer for GPT-OSS due to quantization fallback unpredictability
            safety_buffer = int(cache_bytes * 0.2)
            cache_bytes += safety_buffer
            logger.info(f"Added GPT-OSS safety buffer: {safety_buffer // (1024**2):.0f} MB")

        memory_cache = MemoryCache(cache_bytes, max_alloc_timeout)

        server_info.state = ServerState.JOINING
        dht_announcer = ModuleAnnouncerThread(
            module_uids,
            dht,
            server_info,
            model_info,
            block_config=block_config,
            memory_cache=memory_cache,
            update_period=update_period,
            expiration=expiration,
            daemon=True,
        )
        dht_announcer.start()
        logger.info(f"Announced that blocks {block_indices} are joining")

        assert len(tensor_parallel_devices) >= 1 and all(isinstance(d, torch.device) for d in tensor_parallel_devices)

        blocks = {}
        try:
            for module_uid, block_index in zip(module_uids, block_indices):
                block = load_pretrained_block(
                    converted_model_name_or_path,
                    block_index,
                    config=block_config,
                    torch_dtype=torch_dtype,
                    revision=revision,
                    token=token,
                    cache_dir=cache_dir,
                    max_disk_space=max_disk_space,
                )
                block = convert_block(
                    block,
                    block_index,
                    block_config,
                    tensor_parallel_devices,
                    device,
                    quant_type,
                    adapters=server_info.adapters,
                    freeze=True,
                    token=token,
                    cache_dir=cache_dir,
                    max_disk_space=max_disk_space,
                )
                # Conditionally apply torch.compile for GPU devices
                if compile_block and torch.cuda.is_available() and device.type == "cuda":
                    from agentgrid.utils.device_utils import is_rocm
                    if is_rocm():
                        logger.info(
                            "AMD ROCm detected. Applying torch.compile(mode='reduce-overhead') to block %d...",
                            block_index,
                        )
                        block = torch.compile(block, mode="reduce-overhead")
                    else:
                        logger.info(
                            "CUDA detected. Applying torch.compile(mode='max-autotune') to block %d...",
                            block_index,
                        )
                        block = torch.compile(block, mode="max-autotune")
                elif compile_block and torch.backends.mps.is_available():
                    logger.info(
                        "Apple Silicon detected. Skipping torch.compile for block %d, using native MPS execution.",
                        block_index,
                    )

                blocks[module_uid] = TransformerBackend(
                    module_uid,
                    block,
                    config=block_config,
                    memory_cache=memory_cache,
                    backend_dtype=torch_dtype,
                    max_chunk_size_bytes=max_chunk_size_bytes,
                    warmup_tokens_interval=warmup_tokens_interval,
                    args_schema=(
                        BatchTensorDescriptor(
                            1, 2048, block_config.hidden_size, dtype=torch_dtype, compression=compression
                        ),
                    ),
                    kwargs_schema={},
                    outputs_schema=(
                        BatchTensorDescriptor(
                            1, 2048, block_config.hidden_size, dtype=torch_dtype, compression=compression
                        ),
                    ),
                    min_batch_size=min_batch_size,
                    max_batch_size=max_batch_size,
                )

            merge_inference_pools_inplace(blocks)

            if should_validate_reachability:
                validate_reachability(dht.peer_id)
        except Exception as exc:
            # Log the actual exception for debugging
            logger.error(f"Exception during block loading: {type(exc).__name__}: {exc}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.debug("Shutting down backends")
            for backend in blocks.values():
                backend.shutdown()

            dht_announcer.announce(ServerState.OFFLINE)
            logger.info(f"Announced that blocks {module_uids} are offline")

            if _is_out_of_memory_error(exc):
                raise BlockLoadingOutOfMemoryError(
                    attempted_blocks=len(block_indices),
                    loaded_blocks=len(blocks),
                    original_exception=exc,
                ) from exc
            raise

        return cls(
            dht,
            dht_prefix,
            blocks,
            dht_announcer=dht_announcer,
            server_info=server_info,
            update_period=update_period,
            expiration=expiration,
            **kwargs,
        )

    def __init__(
        self,
        dht: DHT,
        dht_prefix: str,
        module_backends: Dict[str, TransformerBackend],
        *,
        inference_max_length: int,
        num_handlers: int,
        dht_announcer: ModuleAnnouncerThread,
        server_info: ServerInfo,
        update_period: float,
        expiration: Optional[float] = None,
        request_timeout: float,
        session_timeout: float,
        step_timeout: float,
        start: bool,
        **kwargs,
    ):
        super().__init__()

        self.dht, self.module_backends = dht, module_backends
        self.server_info, self.update_period, self.expiration = server_info, update_period, expiration

        handler_event_queues = [mp.Queue() for _ in range(num_handlers)]
        self.conn_handlers = [
            TransformerConnectionHandler(
                dht,
                self.module_backends,
                adapters=server_info.adapters,
                dht_prefix=dht_prefix,
                handler_event_queues=handler_event_queues,
                handler_index=i,
                inference_max_length=inference_max_length,
                request_timeout=request_timeout,
                session_timeout=session_timeout,
                step_timeout=step_timeout,
                quant_type=QuantType[server_info.quant_type.upper()],
            )
            for i in range(num_handlers)
        ]

        self.runtime = RuntimeWithDeduplicatedPools(self.module_backends, device=None, **kwargs)
        # note: We set device=None in runtime to avoid moving all modules to device 0 in runtime.run(). tensor_parallel has already moved it as needed.

        dht_announcer.announce(ServerState.ONLINE)
        self.dht_announcer = dht_announcer

        if start:
            self.run_in_background(await_ready=True)

    def run(self):
        """
        Runs ModuleContainer in the current thread. Initializes dht if necessary, starts connection handlers,
        runs Runtime (self.runtime) to process incoming requests.
        """
        for handler in self.conn_handlers:
            handler.run_in_background()

        self.runtime.run()

    def run_in_background(self, await_ready=True, timeout=None):
        """
        Starts ModuleContainer in a background thread. if await_ready, this method will wait until the container
        is ready to process incoming requests or for :timeout: seconds max.
        """
        self.start()
        if await_ready and not self.ready.wait(timeout=timeout):
            raise TimeoutError("ModuleContainer didn't notify .ready in {timeout} seconds")

    @property
    def ready(self) -> mp.synchronize.Event:
        """
        An event (multiprocessing.Event) that is set when the container is ready to process requests.

        Example
        =======
        >>> container.start()
        >>> container.ready.wait(timeout=10)
        >>> print("Container ready" if container.ready.is_set() else "Container didn't start in 10 seconds")
        """
        return self.runtime.ready  # mp.Event that is true if self is ready to process batches

    def is_healthy(self) -> bool:
        return all(handler.is_alive() for handler in self.conn_handlers) and all(
            pool.is_alive() for pool in self.runtime.pools
        )

    def shutdown(self):
        """
        Gracefully terminate the container, process-safe.
        Please note that terminating container otherwise (e.g. by killing processes) may result in zombie processes.
        If you did already cause a zombie outbreak, your only option is to kill them with -9 (SIGKILL).
        """
        self.dht_announcer.announce(ServerState.OFFLINE)
        logger.info(f"Announced that blocks {list(self.module_backends.keys())} are offline")

        self.ready.clear()

        logger.debug("Shutting down connection handlers")
        for handler in self.conn_handlers:
            handler.shutdown()

        logger.debug(f"Shutting down pools")
        for pool in self.runtime.pools:
            if pool.is_alive():
                pool.shutdown()

        logger.debug(f"Shutting down runtime")
        self.runtime.shutdown()

        logger.debug("Shutting down backends")
        for backend in self.module_backends.values():
            backend.shutdown()

        logger.info("Module container shut down successfully")


class ModuleAnnouncerThread(threading.Thread):
    """Periodically announces that this container hosts the specified modules, visible to all DHT peers"""

    def __init__(
        self,
        module_uids: List[str],
        dht: DHT,
        server_info: ServerInfo,
        model_info: ModelInfo,
        *,
        block_config: PretrainedConfig,
        memory_cache: MemoryCache,
        update_period: float,
        expiration: float,
        max_pinged: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.module_uids = module_uids
        self.dht = dht
        self.server_info = server_info
        self.model_info = model_info
        self.memory_cache = memory_cache

        self.bytes_per_token = block_config.hidden_size * get_size_in_bytes(DTYPE_MAP[server_info.torch_dtype])
        if hasattr(block_config, "block_configs") and block_config.block_configs is not None:
            num_key_value_groups_list = []
            for b in block_config.block_configs:
                if b.attention.n_heads_in_group is not None:
                    num_key_value_groups_list.append(block_config.num_attention_heads // b.attention.n_heads_in_group)
            avg_num_key_value_groups = sum(num_key_value_groups_list) / len(num_key_value_groups_list) if num_key_value_groups_list else 1
            self.bytes_per_token /= avg_num_key_value_groups
        elif hasattr(block_config, "num_key_value_heads") and isinstance(block_config.num_key_value_heads, list):
            num_key_value_groups_list = [
                block_config.num_attention_heads // n_kv_h for n_kv_h in block_config.num_key_value_heads
            ]
            avg_num_key_value_groups = sum(num_key_value_groups_list) / len(num_key_value_groups_list)
            self.bytes_per_token /= avg_num_key_value_groups
        else:
            self.bytes_per_token //= block_config.num_key_value_groups

        self.update_period = update_period
        self.expiration = expiration
        self.trigger = threading.Event()

        self.dht_prefix = parse_uid(module_uids[0])[0]
        block_indices = [parse_uid(uid)[1] for uid in module_uids]
        self.server_info.start_block = min(block_indices)
        self.server_info.end_block = max(block_indices) + 1

        self.max_pinged = max_pinged
        self.next_uids = [
            f"{self.dht_prefix}{UID_DELIMITER}{i}"
            for i in range(self.server_info.start_block + 1, self.server_info.end_block + 1)
        ]
        self.ping_aggregator = PingAggregator(self.dht)

    def run(self) -> None:
        while True:
            start_time = time.perf_counter()

            self.server_info.cache_tokens_left = int(self.memory_cache.bytes_left // self.bytes_per_token)
            if self.server_info.state != ServerState.OFFLINE:
                self._ping_next_servers()
                self.server_info.next_pings = {
                    peer_id.to_base58(): rtt for peer_id, rtt in self.ping_aggregator.to_dict().items()
                }
            else:
                self.server_info.next_pings = None  # No need to ping if we're disconnecting

            declare_active_modules(
                self.dht,
                self.module_uids,
                self.server_info,
                expiration_time=get_dht_time() + self.expiration,
            )
            if self.server_info.state == ServerState.OFFLINE:
                break
            if not self.dht_prefix.startswith("_"):  # Not private
                self.dht.store(
                    key="_agentgrid.models",
                    subkey=self.dht_prefix,
                    value=self.model_info.to_dict(),
                    expiration_time=get_dht_time() + self.expiration,
                )

            delay = self.update_period - (time.perf_counter() - start_time)
            if delay < 0:
                logger.warning(
                    f"Declaring blocks to DHT takes more than --update_period, consider increasing it (currently {self.update_period})"
                )
            self.trigger.wait(max(delay, 0))
            self.trigger.clear()

    def announce(self, state: ServerState) -> None:
        self.server_info.state = state
        self.trigger.set()
        if state == ServerState.OFFLINE:
            self.join()

    def _ping_next_servers(self) -> Dict[hivemind.PeerID, float]:
        module_infos = get_remote_module_infos(self.dht, self.next_uids, latest=True)
        middle_servers = {peer_id for info in module_infos[:-1] for peer_id in info.servers}
        pinged_servers = set(sample_up_to(middle_servers, self.max_pinged))
        pinged_servers.discard(self.dht.peer_id)
        # Sample servers hosting the block after the last one (most likely continuations) separately
        pinged_servers |= set(sample_up_to(module_infos[-1].servers, self.max_pinged))
        self.ping_aggregator.ping(list(pinged_servers))


class RuntimeWithDeduplicatedPools(Runtime):
    """A version of hivemind.moe.server.runtime.Runtime that allows multiple backends to reuse a task pool"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pools = tuple(set(self.pools))
