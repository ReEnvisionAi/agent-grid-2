#
# Copyright (c) 2025-2026 ReEnvision AI, LLC. All rights reserved.
#
# This software is the confidential and proprietary information of
# ReEnvision AI, LLC ("Confidential Information"). You shall not
# disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with ReEnvision AI, LLC.
#

"""
A pytorch memory cache that can be allocated by ConnectionHandler (on cpu) and used over multiple calls to Runtime.

For now, the only purpose of this code is to ensure that allocated memory will be deleted properly.

"""
import asyncio
import contextlib
import ctypes
import gc
import multiprocessing as mp
import os
import time
from collections import OrderedDict
from typing import AsyncContextManager, Counter, Dict, List, Optional, Sequence, Set, Tuple

import async_timeout
import torch
from hivemind.utils import TensorDescriptor, enter_asynchronously, get_logger

from agentgrid.data_structures import Handle
from agentgrid.utils.asyncio import shield_and_wait
from agentgrid.utils.misc import get_size_in_bytes

logger = get_logger(__name__)


class MemoryCache:
    """A shared cache for storing tensors that persist across calls. Main use case: storing past attention KVs"""

    # LOCK ORDERING HIERARCHY (to prevent deadlocks):
    # 1. _lock_pools (lowest level - most data structures)
    # 2. _lock_metadata (for handle operations)
    # 3. _lock_acquire_memory (for memory allocation coordination)
    # 4. _ended_sessions_lock (highest level - session lifecycle)
    #
    # CRITICAL RULE: Never acquire a higher-level lock while holding a lower-level lock.
    # Always release lower locks before acquiring higher ones.

    def __init__(
        self,
        max_size_bytes: Optional[int],
        max_alloc_timeout: Optional[float] = None,
        model_weight_bytes: Optional[Dict[torch.device, int]] = None,
    ):
        self.max_size_bytes = max_size_bytes if max_size_bytes is not None else (2**64 - 1)
        self.max_alloc_timeout = max_alloc_timeout
        self._lock_metadata = mp.Lock()
        self._current_size = mp.Value(ctypes.c_int64, 0, lock=True)
        self._pooled_size_bytes = mp.Value(ctypes.c_int64, 0, lock=True)
        self._enqueued_size = mp.Value(ctypes.c_int64, 0, lock=True)
        self._handle_counter = mp.Value(ctypes.c_uint64, 0, lock=False)
        # Handle overflow detection: track when we've used 90% of the handle space
        self._handle_overflow_threshold = (2**64 - 1) // 10 * 9  # 90% of max uint64
        self._allocated_tensors: Dict[Handle, torch.Tensor] = {}
        self._session_handles: Dict[str, Set[Handle]] = {}
        # Track which handles came from pre-allocation (to avoid double-counting in _pooled_size_bytes)
        self._preallocated_handles: Set[Handle] = set()
        
        # New free pools structure: {device: {dtype: OrderedDict(numel: [tensors])}}
        self._free_pools: Dict[torch.device, Dict[torch.dtype, "OrderedDict[int, List[torch.Tensor]]"]] = {}
        
        # Global LRU Tracking
        # Tracks (device, dtype, numel) buckets in order of access
        self._lru_keys: "OrderedDict[Tuple[torch.device, torch.dtype, int], None]" = OrderedDict()

        # Smart Compaction Tracking
        # Only compact into sizes that have actually been requested by the runtime
        self._requested_sizes: "OrderedDict[int, None]" = OrderedDict()
        self._max_requested_sizes = 10000  # Keep at most 10k unique sizes

        self.runtime_pid = os.getpid()

        self._pipe_recv, self._pipe_send = mp.Pipe(duplex=False)  # any ConnectionHandler -> runtime (for allocation/free)
        self._lock_acquire_memory = mp.Lock()
        self._lock_pipe_processing = mp.Lock()  # Lock for coordinating pipe message processing
        # Lock for protecting free pools, allocated tensors, and related structures
        # This prevents race conditions between compaction, eviction, and normal operations
        self._lock_pools = mp.Lock()
        self._memory_freed_event = mp.Event()

        # Separate queue for async commands (evict, end_session) that need immediate processing
        # This allows a background thread to process these commands without interfering with allocation/free
        self._command_queue = mp.Queue()  # handlers -> runtime (for async commands)

        # Model weight tracking per device (provided externally or estimated)
        self._model_weight_bytes: Dict[torch.device, int] = model_weight_bytes or {}

        # Reserved memory for overhead (temporary buffers, fragmentation, allocator overhead)
        # Default to 20% of max_size_bytes or at least 500MB
        self._reserved_overhead_bytes = max(
            int(self.max_size_bytes * 0.20) if self.max_size_bytes != 2**64 - 1 else 500 * 1024 * 1024,
            500 * 1024 * 1024,  # At least 500MB
        )

        # Backpressure threshold - when to start rejecting requests
        self._backpressure_threshold = 0.85  # 85% of allocatable memory

        # Per-device CUDA memory limits (shared via Manager for access across forked processes)
        self._manager = mp.Manager()
        self._cuda_memory_limits = self._manager.dict()  # device -> (free, total)

        # Eviction request tracking for pipe-based architecture
        self._eviction_request_counter = mp.Value(ctypes.c_int64, 0, lock=False)
        self._pending_evictions = self._manager.dict()  # request_id -> {"bytes_requested": int, "timestamp": float}

        # Track sessions that have been or are being processed for idempotency
        # Using Manager.dict() for O(1) lookups and atomic operations
        self._ended_sessions = self._manager.dict()  # session_id -> timestamp when ended
        # Lock for protecting ended_sessions check-and-set operations
        self._ended_sessions_lock = mp.Lock()

        # Initialize CUDA memory pool optimizations
        self._initialize_cuda_memory_pools()

        # Compaction parameters
        self.COMPACTION_THRESHOLD = 10  # Compact if a bucket has more than this many tensors
        self._compaction_counter = mp.Value(ctypes.c_int64, 0, lock=True)
        self._compaction_calls_threshold = 100  # Base interval for compaction checks
        self._adaptive_compaction_min_interval = 50  # Minimum calls between compaction checks
        self._adaptive_compaction_max_interval = 500  # Maximum calls between compaction checks
        self._memory_pressure_threshold = 0.8  # Compact more aggressively when cache is 80% full
        self._last_compaction_found_work = False  # Track if last compaction actually did work

        # Memory monitoring
        self._allocation_count = 0
        self._eviction_count = 0
        self._compaction_count = 0
        self._last_monitoring_log = time.time()
        self._monitoring_interval = 60.0  # Log memory stats every 60 seconds

        # Optimized garbage collection parameters
        self._gc_counter = mp.Value(ctypes.c_int64, 0, lock=True)
        self._gc_interval = 1000  # Run GC every 1000 cache uses (was 100)
        self._gc_time_threshold = 0.001  # 1ms minimum time between GC calls
        self._last_gc_time = 0.0
        self._memory_pressure_gc_threshold = 0.85  # Trigger GC more aggressively under high memory pressure
        self._gc_min_objects = 100  # Only run GC if we expect to collect at least this many objects

        # Predictive allocation
        self._session_patterns: Dict[str, List[List[TensorDescriptor]]] = {}
        # LRU tracking for session patterns (most recently used at the end)
        self._session_patterns_lru: OrderedDict[str, None] = OrderedDict()
        self._max_session_patterns = 100  # Maximum number of patterns to keep per session
        self._max_total_sessions = 1000  # Maximum total sessions to track patterns for
        self._active_sessions: Dict[str, List[TensorDescriptor]] = {}
        self._pre_allocated_tensors: Dict[str, List[torch.Tensor]] = {}
        self._pre_allocation_timestamps: Dict[str, float] = {}
        self._pre_allocation_timeout = 300.0  # 5 minutes timeout for pre-allocated tensors
        self._active_session_timeouts: Dict[str, float] = {}
        self._session_timeout = 3600  # 1 hour

        # Background task for processing pipe messages
        self._pipe_processing_stop_event = mp.Event()
        self._pipe_processing_thread = None
        self._pipe_processing_started = False

        # Start background command processing in runtime process
        # This processes async commands (evict, end_session) from the command queue
        if self._is_runtime_process():
            self._start_background_command_processing()

        # Update reserved overhead based on actual model weights
        self._update_reserved_memory()

    @property
    def current_size_bytes(self) -> int:
        return self._current_size.value

    @current_size_bytes.setter
    def current_size_bytes(self, value: int):
        with self._current_size.get_lock():
            self._current_size.value = value

    @property
    def enqueued_size_bytes(self) -> int:
        return self._enqueued_size.value

    @enqueued_size_bytes.setter
    def enqueued_size_bytes(self, value: int):
        self._enqueued_size.value = value

    @property
    def bytes_left(self) -> int:
        with self._pooled_size_bytes.get_lock(), self._enqueued_size.get_lock():
            total_used = self.current_size_bytes + self._pooled_size_bytes.value + self.enqueued_size_bytes
            # Account for reserved overhead in the calculation
            return max(0, self.max_size_bytes - total_used - self._reserved_overhead_bytes)

    @property
    def handle_counter(self) -> int:
        return self._handle_counter.value

    @handle_counter.setter
    def handle_counter(self, value: int):
        self._handle_counter.value = value

    def __del__(self):
        """Cleanup when MemoryCache is destroyed."""
        try:
            if self._pipe_processing_started:
                self._stop_background_command_processing()
        except Exception:
            pass  # Ignore errors during cleanup

    def _initialize_cuda_memory_pools(self):
        """Initialize CUDA memory pool optimizations"""
        if torch.cuda.is_available():
            try:
                # Enable CUDA memory pool for more efficient memory allocation
                # This reduces fragmentation and improves allocation speed
                # Set to 85% to balance block loading with inference headroom
                for device_id in range(torch.cuda.device_count()):
                    device = torch.device(f'cuda:{device_id}')
                    with torch.cuda.device(device):
                        # Set memory fraction to 85% to prevent oversubscription during inference
                        # This leaves 15% headroom for temporary buffers and fragmentation
                        torch.cuda.set_per_process_memory_fraction(0.85, device_id)
                        logger.info(f"Initialized CUDA memory pool (85%) for device {device_id}")

                # Clear any existing cached memory
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA memory cache during initialization")

            except Exception as e:
                logger.warning(f"Failed to initialize CUDA memory pools: {e}")

    def _is_runtime_process(self) -> bool:
        """Check if we're in the runtime process (not a forked subprocess)."""
        return os.getpid() == self.runtime_pid

    def _log_prefix(self) -> str:
        """Return a prefix for log messages identifying the process type and PID."""
        pid = os.getpid()
        if self._is_runtime_process():
            return f"[Runtime PID:{pid}]"
        else:
            return f"[Handler PID:{pid}]"

    def _get_cuda_memory_info(self, device: torch.device) -> Tuple[int, int]:
        """Get actual CUDA memory info (free, total) for a device.

        In forked subprocesses, CUDA cannot be re-initialized, so we use cached values
        from the runtime process. Returns (0, 0) if no cached value is available, which
        will trigger safe backpressure behavior.
        """
        if device.type != "cuda":
            return (2**64 - 1, 2**64 - 1)

        # Use string key for shared dict (torch.device is not easily shareable)
        device_key = str(device)

        # In forked subprocesses, use cached values from shared dict
        if not self._is_runtime_process():
            if device_key in self._cuda_memory_limits:
                return self._cuda_memory_limits[device_key]
            # No cached value available - return 0 to trigger safe backpressure/failure
            # This is better than returning unlimited which could cause OOM
            logger.debug(f"No cached CUDA memory info for {device} in forked process")
            return (0, 0)

        # Only in runtime process - actually query CUDA and cache the result
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device.index or 0)
            self._cuda_memory_limits[device_key] = (free_bytes, total_bytes)
            return free_bytes, total_bytes
        except Exception as e:
            logger.warning(f"Failed to get CUDA memory info for {device}: {e}")
            return (0, 0)  # Return 0 on error to trigger safe backpressure

    def _update_reserved_memory(self):
        """Update reserved memory based on model weights and overhead."""
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                device = torch.device(f'cuda:{device_id}')
                free_cuda, total_cuda = self._get_cuda_memory_info(device)

                # If model weights not provided, estimate based on GPU memory
                if device not in self._model_weight_bytes:
                    # Rough estimate: assume model uses ~30% of GPU memory for large models
                    estimated_model_bytes = int(total_cuda * 0.30)
                    self._model_weight_bytes[device] = estimated_model_bytes
                    logger.info(
                        f"Estimated model weights at {estimated_model_bytes / 1024**3:.2f} GiB for {device}"
                    )

                logger.debug(
                    f"Device {device}: model={self._model_weight_bytes.get(device, 0) / 1024**3:.2f} GiB, "
                    f"reserved_overhead={self._reserved_overhead_bytes / 1024**3:.2f} GiB"
                )
        elif torch.backends.mps.is_available():
            # For MPS devices, we use unified memory so no need to estimate model weights
            # Model weights share system memory with the cache
            logger.debug("MPS device detected: using unified memory model")

    def _start_background_command_processing(self):
        """Start a background thread to process async commands from the command queue."""
        import threading

        def _command_processing_loop():
            """Background thread that continuously processes async commands."""
            logger.info(f"{self._log_prefix()} Starting background command processing thread")
            while not self._pipe_processing_stop_event.is_set():
                try:
                    # Process commands from the queue with a timeout
                    try:
                        command_info = self._command_queue.get(timeout=0.1)  # 100ms timeout
                        self._process_command(command_info)
                    except Exception:
                        # Queue.get() raises exception on timeout, which is expected
                        pass
                except Exception as e:
                    logger.error(f"{self._log_prefix()} Error in command processing: {e}")
                    import time
                    time.sleep(1.0)  # Wait before retrying on error
            logger.debug(f"{self._log_prefix()} Background command processing thread stopped")

        self._pipe_processing_thread = threading.Thread(
            target=_command_processing_loop,
            name=f"CommandProcessing_{os.getpid()}",
            daemon=False  # Enable graceful shutdown
        )
        self._pipe_processing_thread.start()
        self._pipe_processing_started = True

    def _stop_background_command_processing(self):
        """Stop the background command processing thread gracefully."""
        if not self._pipe_processing_started or self._pipe_processing_thread is None:
            return

        logger.debug(f"{self._log_prefix()} Stopping background command processing thread")
        self._pipe_processing_stop_event.set()

        # Drain remaining commands with timeout
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                command_info = self._command_queue.get(timeout=0.1)
                self._process_command(command_info)
            except Exception:
                break  # Queue empty

        # Wait for thread to finish
        self._pipe_processing_thread.join(timeout=2.0)
        if self._pipe_processing_thread.is_alive():
            logger.warning(f"{self._log_prefix()} Background thread did not stop gracefully")
        else:
            logger.debug(f"{self._log_prefix()} Background thread stopped successfully")
        self._pipe_processing_started = False

    def _process_command(self, command_info: dict):
        """Process a single command from the command queue."""
        if not self._is_runtime_process():
            return

        command = command_info.get("command")

        if command == "evict":
            self._handle_evict_command(command_info)
        elif command == "end_session":
            self._handle_end_session_command(command_info)
        else:
            logger.warning(f"{self._log_prefix()} Unknown command: {command}")

    def _handle_evict_command(self, command_info: dict):
        """Handle an eviction command from the pipe."""
        request_id = command_info.get("request_id")
        bytes_to_free = command_info.get("bytes_to_free", 0)

        logger.debug(
            f"{self._log_prefix()} [BG_THREAD] Received eviction request #{request_id}: "
            f"{bytes_to_free / 1024**2:.2f} MB"
        )

        if bytes_to_free > 0:
            start_time = time.time()
            self._evict_memory(bytes_to_free)
            elapsed = time.time() - start_time

            logger.debug(
                f"{self._log_prefix()} [BG_THREAD] Completed eviction request #{request_id}: "
                f"processed in {elapsed*1000:.1f}ms"
            )

        if request_id is not None and request_id in self._pending_evictions:
            del self._pending_evictions[request_id]

    def _handle_end_session_command(self, command_info: dict):
        """Handle an end_session command from the pipe."""
        session_id = command_info["session_id"]
        call_id = id(command_info)  # Unique identifier for this call

        logger.debug(
            f"{self._log_prefix()} [BG_THREAD] [SESSION_END_CALL_START] session_id={session_id}, "
            f"call_id={call_id}, ended_sessions_count={len(self._ended_sessions)}, "
            f"in_active_sessions={session_id in self._active_sessions}, "
            f"in_session_handles={session_id in self._session_handles}"
        )

        # Idempotency check: use atomic check-and-set with lock to prevent race condition
        # This prevents duplicate processing when multiple end_session commands are queued
        with self._ended_sessions_lock:
            if session_id in self._ended_sessions:
                logger.debug(
                    f"{self._log_prefix()} [BG_THREAD] [SESSION_END_ALREADY_PROCESSED] session_id={session_id}, skipping duplicate, "
                    f"call_id={call_id}"
                )
                return
            # Mark as processing immediately to prevent duplicates (atomic with the check above)
            self._ended_sessions[session_id] = time.time()

        logger.debug(f"{self._log_prefix()} [BG_THREAD] [SESSION_END_RECEIVED] session_id={session_id}, call_id={call_id}")

        # Clean up any active handles associated with this session to prevent leaks
        handles_freed_count = 0
        prealloc_freed_count = 0
        bytes_freed = 0
        handles_already_freed = 0

        # Use lock to protect access to all session structures, _allocated_tensors and _free_pools
        # Acquire _ended_sessions_lock first (higher level), then _lock_pools (lower level)
        # This follows the lock ordering hierarchy to prevent deadlocks
        with self._lock_pools:
            # Handle session pattern storage - now protected by _lock_pools
            if session_id in self._active_sessions:
                history = self._active_sessions.pop(session_id)
                patterns = self._session_patterns.setdefault(session_id, [])
                patterns.append(history)
                # Enforce per-session pattern limit (keep most recent)
                if len(patterns) > self._max_session_patterns:
                    patterns[:] = patterns[-self._max_session_patterns:]
                # Update LRU tracking for this session
                if session_id in self._session_patterns_lru:
                    self._session_patterns_lru.move_to_end(session_id)
                else:
                    self._session_patterns_lru[session_id] = None
                    # Enforce total sessions limit (evict least recently used)
                    while len(self._session_patterns_lru) > self._max_total_sessions:
                        oldest_session = next(iter(self._session_patterns_lru))
                        self._session_patterns_lru.pop(oldest_session)
                        self._session_patterns.pop(oldest_session, None)
                logger.debug(f"Ended session {session_id}, stored allocation pattern with {len(history)} steps.")

            if session_id in self._session_handles:
                session_handles = self._session_handles.pop(session_id)
                handles_to_free = [h for h in session_handles if h in self._allocated_tensors]

                if handles_to_free:
                    logger.debug(
                        f"{self._log_prefix()} [BG_THREAD] [SESSION_CACHE_CLEARING] session_id={session_id}, "
                        f"cleaning up {len(handles_to_free)} remaining handles"
                    )
                    for handle in handles_to_free:
                        # Handle may have been freed concurrently by pipe processing
                        # Use pop with default to safely handle this case
                        tensor = self._allocated_tensors.pop(handle, None)

                        # Check if this was a pre-allocated handle BEFORE checking if tensor is None
                        # This ensures we clean up _preallocated_handles even if the handle was already freed
                        is_preallocated = handle in self._preallocated_handles
                        if is_preallocated:
                            self._preallocated_handles.discard(handle)
                            prealloc_freed_count += 1

                        if tensor is None:
                            handles_already_freed += 1
                            continue

                        descr = TensorDescriptor.from_tensor(tensor)
                        numel = tensor.numel()

                        device_pool = self._free_pools.setdefault(descr.device, {})
                        dtype_pool = device_pool.setdefault(descr.dtype, OrderedDict())
                        numel_pool = dtype_pool.setdefault(numel, [])
                        numel_pool.append(tensor)
                        dtype_pool.move_to_end(numel)

                        self._lru_keys[(descr.device, descr.dtype, numel)] = None

                        if not is_preallocated:
                            tensor_size = numel * get_size_in_bytes(descr.dtype)
                            with self._pooled_size_bytes.get_lock():
                                self._pooled_size_bytes.value += tensor_size
                            bytes_freed += tensor_size

                        handles_freed_count += 1

                # Clean up any remaining handles from _preallocated_handles
                # This handles the case where handles were already freed (not in _allocated_tensors)
                # but their preallocated mark wasn't cleaned up
                # NOTE: We ONLY clean up the mark here, NOT the tensors (they were freed above)
                for handle in session_handles:
                    if handle in self._preallocated_handles:
                        self._preallocated_handles.discard(handle)
                        prealloc_freed_count += 1

                if handles_already_freed > 0:
                    logger.debug(
                        f"{self._log_prefix()} [BG_THREAD] {handles_already_freed} handles already freed by pipe processing"
                    )

                logger.debug(
                    f"{self._log_prefix()} [BG_THREAD] [SESSION_CACHE_CLEARED] session_id={session_id}, "
                    f"call_id={call_id}, freed {handles_freed_count} handles ({bytes_freed / 1024**2:.2f} MB), "
                    f"preallocated: {prealloc_freed_count}, already_freed: {handles_already_freed}"
                )

            # Clean up leftover pre-allocations
            prealloc_count = 0
            if session_id in self._pre_allocated_tensors:
                prealloc_count = len(self._pre_allocated_tensors[session_id])
                for tensor in self._pre_allocated_tensors.pop(session_id):
                    descr = TensorDescriptor.from_tensor(tensor)
                    device_pool = self._free_pools.setdefault(descr.device, {})
                    dtype_pool = device_pool.setdefault(descr.dtype, OrderedDict())
                    numel_pool = dtype_pool.setdefault(tensor.numel(), [])
                    numel_pool.append(tensor)
                    dtype_pool.move_to_end(tensor.numel())

                    self._lru_keys[(descr.device, descr.dtype, tensor.numel())] = None
                    self._lru_keys.move_to_end((descr.device, descr.dtype, tensor.numel()))

                    tensor_size = tensor.numel() * get_size_in_bytes(descr.dtype)
                    with self._pooled_size_bytes.get_lock():
                        self._pooled_size_bytes.value += tensor_size
                self._pre_allocation_timestamps.pop(session_id, None)
                if prealloc_count > 0:
                    logger.debug(
                        f"{self._log_prefix()} [BG_THREAD] [SESSION_PREALLOC_CLEARED] session_id={session_id}, "
                        f"freed {prealloc_count} pre-allocated tensors"
                    )

        logger.debug(
            f"{self._log_prefix()} [BG_THREAD] [SESSION_END_COMPLETE] session_id={session_id}, "
            f"total handles_freed={handles_freed_count}, "
            f"total_bytes_freed={bytes_freed / 1024**2:.2f} MB, "
            f"prealloc_freed={prealloc_count}"
        )

    def _get_allocatable_memory(self, device: torch.device) -> int:
        """
        Get the amount of memory that can actually be allocated for cache tensors.
        This accounts for model weights, reserved overhead, and actual device memory.
        For MPS devices, returns a large value since MPS has unified memory.
        """
        if device.type != "cuda":
            # For MPS/CPU, use a large value since they have unified/shared memory
            # and we can't accurately query allocatable memory
            return 2**64 - 1

        free_cuda, total_cuda = self._get_cuda_memory_info(device)
        model_bytes = self._model_weight_bytes.get(device, 0)

        # Allocatable = free_cuda - model_weights - overhead - buffer
        # The buffer is for concurrent allocations and fragmentation
        buffer = max(100 * 1024 * 1024, int(total_cuda * 0.05))  # At least 100MB or 5%
        allocatable = free_cuda - model_bytes - self._reserved_overhead_bytes - buffer

        return max(0, allocatable)

    def _should_apply_backpressure(self, device: torch.device, required_bytes: int) -> bool:
        """
        Check if backpressure should be applied based on actual CUDA memory.
        Returns True if we should reject or delay this allocation.
        """
        if device.type != "cuda":
            return False

        allocatable = self._get_allocatable_memory(device)
        if required_bytes > allocatable:
            return True

        # Also check if we're above the backpressure threshold
        free_cuda, total_cuda = self._get_cuda_memory_info(device)
        model_bytes = self._model_weight_bytes.get(device, 0)
        used_by_model_and_cache = total_cuda - free_cuda - model_bytes

        effective_total = total_cuda - model_bytes
        if effective_total > 0:
            usage_ratio = used_by_model_and_cache / effective_total
            if usage_ratio > self._backpressure_threshold:
                logger.warning(
                    f"Backpressure: device {device} at {usage_ratio * 100:.1f}% usage "
                    f"(threshold: {self._backpressure_threshold * 100:.1f}%)"
                )
                return True

        return False

    def force_garbage_collection(self, aggressive: bool = False) -> int:
        """
        Optimized force garbage collection with performance safeguards.

        :param aggressive: If True, bypasses performance optimizations for urgent cleanup
        :return: Number of objects collected
        """
        current_time = time.perf_counter()

        # Performance optimization: Skip GC if called too frequently
        if not aggressive and (current_time - self._last_gc_time) < self._gc_time_threshold:
            return 0

        try:
            # Check if we should run GC based on memory pressure
            if not aggressive:
                memory_pressure = self._get_memory_pressure()
                # Only run GC if under significant memory pressure
                if memory_pressure < self._memory_pressure_gc_threshold:
                    return 0

            # Collect Python garbage
            collected = gc.collect()

            # Only clear device caches if we collected objects or being aggressive
            if collected > 0 or aggressive:
                # Clear CUDA cache if available with device-specific clearing
                if torch.cuda.is_available():
                    # Clear cache for all CUDA devices
                    for device_id in range(torch.cuda.device_count()):
                        with torch.cuda.device(device_id):
                            torch.cuda.empty_cache()
                            # Reset peak memory stats to get accurate readings
                            torch.cuda.reset_peak_memory_stats(device_id)

                # Clear MPS cache if available (Apple Silicon)
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()

            self._last_gc_time = current_time

            if collected > 0:
                logger.debug(f"Garbage collection completed, collected {collected} objects")

            return collected
        except Exception as e:
            logger.warning(f"Force garbage collection failed: {e}")
            return 0

    def _should_run_gc(self) -> bool:
        """Determine if garbage collection should run based on heuristics"""
        # Check time-based frequency
        current_time = time.perf_counter()
        if (current_time - self._last_gc_time) < self._gc_time_threshold:
            return False

        # Check memory pressure
        memory_pressure = self._get_memory_pressure()
        if memory_pressure > self._memory_pressure_gc_threshold:
            return True

        # Check call frequency
        with self._gc_counter.get_lock():
            self._gc_counter.value += 1
            if self._gc_counter.value >= self._gc_interval:
                self._gc_counter.value = 0
                return True

        return False

    def ensure_memory_available(self, required_bytes: int, device: Optional[torch.device] = None):
        """
        Proactively evict memory to ensure enough space for allocation.
        Now includes actual CUDA memory checks, not just accounting.

        :param required_bytes: Amount of memory needed for allocation
        :param device: Optional device to check memory for
        """
        with self._pooled_size_bytes.get_lock(), self._enqueued_size.get_lock():
            total_used = self.current_size_bytes + self._pooled_size_bytes.value + self.enqueued_size_bytes

        # Check actual CUDA memory availability
        if device is not None and device.type == "cuda":
            free_cuda, total_cuda = self._get_cuda_memory_info(device)

            # Apply backpressure if CUDA memory is too low
            if self._should_apply_backpressure(device, required_bytes):
                allocatable = self._get_allocatable_memory(device)
                logger.warning(
                    f"CUDA backpressure for {device}: need {required_bytes / 1024**2:.2f} MB, "
                    f"but only ~{allocatable / 1024**2:.2f} MB allocatable (free CUDA: {free_cuda / 1024**2:.2f} MB)"
                )
                # Evict more aggressively when under backpressure
                bytes_to_evict = min(
                    self._pooled_size_bytes.value,
                    required_bytes + self._reserved_overhead_bytes,
                )
                if bytes_to_evict > 0:
                    logger.info(f"Backpressure evicting {bytes_to_evict / 1024**2:.2f} MB (via pipe)")
                    self._request_eviction(bytes_to_evict)
                    self.force_garbage_collection(aggressive=True)

        # Original accounting-based check
        if total_used + required_bytes > self.max_size_bytes:
            # Calculate how much memory we need to free
            memory_shortfall = (total_used + required_bytes) - self.max_size_bytes
            buffer = 1024 * 1024 * 100  # 100MB buffer for safety
            bytes_to_evict = min(self._pooled_size_bytes.value, memory_shortfall + buffer)

            if bytes_to_evict > 0:
                logger.info(f"Proactively evicting {bytes_to_evict / 1024**2:.2f} MB for {required_bytes / 1024**2:.2f} MB allocation (via pipe)")
                self._request_eviction(bytes_to_evict)

                # Only force garbage collection after eviction if significant memory was freed
                if bytes_to_evict > 100 * 1024 * 1024:  # Only if >100MB evicted
                    self.force_garbage_collection()

            # Check if we still don't have enough memory after eviction
            with self._pooled_size_bytes.get_lock(), self._enqueued_size.get_lock():
                total_used_after = self.current_size_bytes + self._pooled_size_bytes.value + self.enqueued_size_bytes

            if total_used_after + required_bytes > self.max_size_bytes:
                logger.warning(f"Still insufficient memory after eviction: need {required_bytes / 1024**2:.2f} MB, available {(self.max_size_bytes - total_used_after) / 1024**2:.2f} MB")

    def _request_eviction(self, bytes_to_free: int) -> None:
        """
        Send an eviction request to the runtime process via the command queue.
        This method should ONLY be called from ConnectionHandler processes.
        The runtime process will handle the actual eviction.
        """
        assert not self._is_runtime_process(), "_request_eviction must be called from ConnectionHandler, not runtime"

        # Generate unique request ID
        with self._eviction_request_counter.get_lock():
            request_id = int(self._eviction_request_counter.value)
            self._eviction_request_counter.value += 1

        # Record the request in shared state
        self._pending_evictions[request_id] = {
            "bytes_requested": bytes_to_free,
            "timestamp": time.time()
        }

        logger.info(
            f"{self._log_prefix()} Sending eviction request #{request_id}: "
            f"{bytes_to_free / 1024**2:.2f} MB via command queue"
        )

        # Send eviction request via command queue for immediate processing
        self._command_queue.put({
            "command": "evict",
            "bytes_to_free": bytes_to_free,
            "request_id": request_id
        })

        logger.debug(f"{self._log_prefix()} Eviction request #{request_id} sent to runtime")

    def end_session(self, session_id: str):
        """
        Signals the end of an allocation session, moving the session's allocation
        history to the stored patterns for future prediction.
        """
        assert os.getpid() != self.runtime_pid, "This method must be called from a ConnectionHandler"
        # Send to the command queue for immediate processing by background thread
        self._command_queue.put({"command": "end_session", "session_id": session_id})

    def set_model_weight_bytes(self, model_weight_bytes: Dict[torch.device, int]):
        """
        Set the model weight bytes for each device. This should be called after
        the model is loaded to improve memory allocation accuracy.

        :param model_weight_bytes: Dictionary mapping devices to the number of bytes
                                   occupied by model weights on that device.
        """
        assert os.getpid() == self.runtime_pid, "This method must be called from the runtime process"
        self._model_weight_bytes = model_weight_bytes.copy()
        logger.info(
            f"Updated model weight tracking: "
            + ", ".join(f"{d}: {b / 1024**3:.2f} GiB" for d, b in model_weight_bytes.items())
        )

    @contextlib.asynccontextmanager
    async def allocate_cache(
        self, *descriptors: TensorDescriptor, timeout: float, session_id: Optional[str] = None
    ) -> AsyncContextManager[Sequence[Handle]]:
        """
        Create a handle that is associated with buffers on unique device. If cache full, raises AllocationFailed.

        :param descriptors: one or more tensors tensor of this size, dtype, etc
        :param timeout: optional maximum time to wait for cache allocation; None (default) means no time limit
        :param session_id: optional identifier for tracking allocation patterns for prediction

        :note: if descriptors reside on different devices, it is expected that they are approximately balanced across devices;
          if not, it will count maximum tensor allocation across devices for the purposes of size limit

        :note: This function should be called by connection handlers, it can be called concurrently from multiple processes.
        Furthermore, it can be called concurrently with at most one use_cache call in runtime.
        """
        assert os.getpid() != self.runtime_pid, "must be called by a ConnectionHandler, not runtime"
        assert all(descr.device is not None for descr in descriptors), "please specify allocated devices"
        if self.max_alloc_timeout is not None:
            timeout = min(timeout, self.max_alloc_timeout)
        max_alloc_size = self.get_allocation_size(*descriptors)

        gib = 1024**3
        with self._pooled_size_bytes.get_lock():
            pooled_size = self._pooled_size_bytes.value
        cur_size, max_size = self.current_size_bytes + pooled_size, self.max_size_bytes
        # Prevent division by zero if max_size is somehow 0
        safe_max_size = max(1, max_size)
        friendly_max_size = f"{max_size / gib:.2f}" if max_size != 2**64 - 1 else "inf"
        logger.info(
            f"rpc_inference.wait_for_alloc(size={max_alloc_size / gib:.2f} GiB), "
            f"already used {cur_size / gib:.2f}/{friendly_max_size} GiB ({cur_size / safe_max_size * 100:.1f}%)"
        )

        alloc_task = asyncio.create_task(self._schedule_alloc(max_alloc_size, *descriptors, timeout=timeout, session_id=session_id))
        try:
            handles, pipe_send_succeeded = await shield_and_wait(alloc_task)
            logger.info(f"rpc_inference.alloc_done(size={max_alloc_size / gib:.2f} GiB)")
            yield handles
        finally:
            self._free(max_alloc_size, alloc_task, pipe_send_succeeded)
            # Only force garbage collection if under memory pressure (lightweight optimization)
            if self._get_memory_pressure() > 0.8:  # Only if >80% memory usage
                self.force_garbage_collection()

    @staticmethod
    def get_allocation_size(*descriptors: TensorDescriptor) -> int:
        """Return the memory size (bytes) to be allocated on a device. If there are many devices, return maximum"""
        if not descriptors:
            return 0

        alloc_size_by_device = Counter()
        for descr in descriptors:
            tensor_size = descr.numel() * get_size_in_bytes(descr.dtype)
            alloc_size_by_device[descr.device] += tensor_size
        return max(alloc_size_by_device.values())

    async def _schedule_alloc(
        self, alloc_size: int, *descriptors: TensorDescriptor, timeout: Optional[float], session_id: Optional[str] = None
    ) -> Tuple[Sequence[Handle], bool]:
        """
        This method should be called inside asyncio.shield() because:
            - hivemind.utils.enter_asynchronously() does not always release the lock on cancellation

        Returns: (handles, pipe_send_succeeded) - indicates if pipe send succeeded
        """
        try:
            # Proactively ensure memory is available before waiting for free memory
            if descriptors:
                # Use the device from the first descriptor (most important one)
                primary_device = descriptors[0].device
                self.ensure_memory_available(alloc_size, primary_device)

            async with self._wait_for_free_memory(alloc_size, timeout):
                with self._lock_metadata:
                    # Atomically read and increment handle_counter to prevent duplicate handles
                    # _lock_metadata protects the entire critical section
                    current_counter = self._handle_counter.value

                    # Check for potential overflow (approaching handle space limit)
                    if current_counter > self._handle_overflow_threshold:
                        # We're running out of handles - check if we can safely wrap around
                        # Only safe if no handles from the early range are still allocated
                        min_handle = min(self._allocated_tensors.keys()) if self._allocated_tensors else 0
                        if min_handle < (2**64 - 1) // 2:  # If there are handles from the first half
                            raise AllocationFailed(
                                f"Handle counter near overflow and {len(self._allocated_tensors)} "
                                f"handles still allocated. Cannot safely allocate new handles."
                            )
                        # Safe to wrap around - reset to 0
                        logger.warning(f"{self._log_prefix()} Handle counter wrapping around from {current_counter} to 0")
                        self._handle_counter.value = len(descriptors)
                        handles = tuple(i for i in range(len(descriptors)))
                    else:
                        handles = tuple(int(current_counter) + i for i in range(len(descriptors)))
                        # Increment using the same lock for atomicity
                        new_counter = current_counter + len(handles)
                        # Handle overflow by wrapping to 0 if we exceed max uint64
                        if new_counter >= 2**64:
                            new_counter = new_counter % (2**64)
                        self._handle_counter.value = new_counter

                    with self._current_size.get_lock():
                        self.current_size_bytes += alloc_size
                    self._allocation_count += 1
                    command_info = {"session_id": session_id} if session_id else None
                    self._pipe_send.send((handles, descriptors, command_info))
                    self._log_memory_stats()
                    return handles, True  # Return handles and success flag
        except TimeoutError:
            raise AllocationFailed(f"Could not allocate {alloc_size} (timeout={timeout})")

    @contextlib.asynccontextmanager
    async def _wait_for_free_memory(self, alloc_size: int, timeout: Optional[float]):
        start_time = time.perf_counter()
        loop = asyncio.get_event_loop()

        with self._enqueued_size.get_lock():
            self._enqueued_size.value += alloc_size
        allocated = False
        try:
            context_manager = async_timeout.timeout(timeout) if timeout != 0 else contextlib.AsyncExitStack()
            # contextlib.AsyncExitStack() is used as a null context here
            async with context_manager:
                with self._pooled_size_bytes.get_lock():
                    total_size = self.current_size_bytes + self._pooled_size_bytes.value + self.enqueued_size_bytes
                if timeout == 0 and total_size > self.max_size_bytes:
                    raise AllocationFailed(f"Could not allocate {alloc_size} bytes immediately: out of memory")

                async with enter_asynchronously(self._lock_acquire_memory):
                    with self._pooled_size_bytes.get_lock(), self._enqueued_size.get_lock():
                        current_total_size = self.current_size_bytes + self._pooled_size_bytes.value + self.enqueued_size_bytes
                    if current_total_size + alloc_size > self.max_size_bytes:
                        if timeout == 0:
                            raise AllocationFailed(f"Could not allocate {alloc_size} bytes immediately: out of memory")
                        elapsed_time = time.perf_counter() - start_time
                        remaining_timeout = max(0.0, timeout - elapsed_time) if timeout is not None else None
                        await loop.run_in_executor(None, self._wait_until_available, alloc_size, remaining_timeout)

                allocated = True
                with self._enqueued_size.get_lock():
                    self._enqueued_size.value -= alloc_size
                yield
        except asyncio.TimeoutError:
            raise AllocationFailed(f"Could not allocate {alloc_size} within {timeout} seconds")
        finally:
            if not allocated:
                with self._enqueued_size.get_lock():
                    self._enqueued_size.value -= alloc_size

    def _free(self, alloc_size: int, alloc_task: asyncio.Task, pipe_send_succeeded: bool):
        # Decrement current_size_bytes and free handles based on allocation state
        gib = 1024**3

        if alloc_task.exception() is not None:
            # Allocation failed - check if pipe send succeeded
            if pipe_send_succeeded:
                # Pipe send succeeded but task failed afterwards
                # Runtime has the handles but handler doesn't - we must send free request
                # The handles are in the task result even if there's an exception
                try:
                    result = alloc_task.result()
                    # Extract handles from tuple result (handles, pipe_send_succeeded)
                    if isinstance(result, tuple) and len(result) == 2:
                        handles, _ = result
                    elif isinstance(result, (tuple, list)):
                        handles = result
                    else:
                        handles = [result]
                    if handles is not None and any(h is not None for h in handles):
                        logger.debug(
                            f"{self._log_prefix()} [CACHE_FREE_AFTER_ERROR] sending handles to runtime for cleanup "
                            f"after task exception: {handles}"
                        )
                        with self._lock_metadata:
                            self._pipe_send.send((handles, None, None))
                            with self._current_size.get_lock():
                                self.current_size_bytes -= alloc_size
                        self._memory_freed_event.set()
                        logger.debug(
                            f"{self._log_prefix()} [CACHE_FREE_AFTER_ERROR] cleaned up {len(handles)} handles after exception, "
                            f"{alloc_size / gib:.3f} GiB"
                        )
                    else:
                        # Handles are None, just decrement accounting
                        logger.debug(f"{self._log_prefix()} [CACHE_FREE_AFTER_ERROR] handles are None, just decrementing accounting")
                        with self._current_size.get_lock():
                            self._current_size_bytes -= alloc_size
                except Exception as e:
                    logger.error(f"{self._log_prefix()} Failed to get handles for cleanup after exception: {e}")
                    # If we can't get handles but pipe send succeeded, the runtime has orphaned handles
                    # We still need to decrement accounting to prevent permanent leak
                    with self._current_size.get_lock():
                        self.current_size_bytes -= alloc_size
            else:
                # Pipe send failed - runtime doesn't have the handles, just decrement accounting
                logger.debug(f"{self._log_prefix()} [CACHE_FREE_SKIPPED] alloc_task had exception and pipe send failed")
                with self._current_size.get_lock():
                    self.current_size_bytes -= alloc_size
                logger.debug(
                    f"{self._log_prefix()} [CACHE_FREE_COMPLETE] corrected size accounting for failed allocation, "
                    f"decremented {alloc_size / gib:.3f} GiB, current_size={self.current_size_bytes / gib:.3f} GiB"
                )
            return

        # Allocation succeeded - extract handles from the result (which is now a tuple)
        result = alloc_task.result()
        if isinstance(result, tuple) and len(result) == 2:
            handles, _ = result  # Unpack (handles, pipe_send_succeeded)
        else:
            # Backward compatibility in case result format is unexpected
            handles = result if isinstance(result, (tuple, list)) else [result]

        logger.debug(
            f"{self._log_prefix()} [CACHE_FREE_REQUEST] freeing {len(handles)} handles, "
            f"{alloc_size / gib:.3f} GiB, handles={handles}"
        )

        with self._lock_metadata:
            self._pipe_send.send((handles, None, None))  # signal runtime to free these handles
            with self._current_size.get_lock():
                self.current_size_bytes -= alloc_size
        self._memory_freed_event.set()

        logger.debug(
            f"{self._log_prefix()} [CACHE_FREE_COMPLETE] freed {len(handles)} handles, "
            f"{alloc_size / gib:.3f} GiB, current_size={self.current_size_bytes / gib:.3f} GiB"
        )

    def _wait_until_available(self, allocated_size: int, timeout: Optional[float] = None):
        # note: this function should only be called inside _lock_acquire_memory!
        if allocated_size > self.max_size_bytes:
            raise AllocationFailed(
                f"Could not allocate {allocated_size} bytes, max cache size = {self.max_size_bytes} bytes"
            )
        timeout = timeout if timeout != float("inf") else None
        deadline = None if timeout is None else time.perf_counter() + timeout
        while True:
            with self._pooled_size_bytes.get_lock(), self._enqueued_size.get_lock():
                current_total_size = self.current_size_bytes + self._pooled_size_bytes.value + self.enqueued_size_bytes
            if current_total_size + allocated_size <= self.max_size_bytes:
                break

            remaining_time = None if timeout is None else deadline - time.perf_counter()
            if remaining_time is not None and remaining_time <= 0:
                # Always check memory condition even on timeout - event may have been set between check and wait
                with self._pooled_size_bytes.get_lock(), self._enqueued_size.get_lock():
                    current_total_size = self.current_size_bytes + self._pooled_size_bytes.value + self.enqueued_size_bytes
                if current_total_size + allocated_size <= self.max_size_bytes:
                    break  # Memory is available now, proceed
                raise AllocationFailed(
                    f"Server's attention cache is full, failed to allocate {allocated_size} bytes in {timeout} seconds"
                )

            if not self._memory_freed_event.wait(remaining_time):
                # On timeout or false return, always recheck the actual condition
                # This fixes the race condition where memory is freed between check and wait
                with self._pooled_size_bytes.get_lock(), self._enqueued_size.get_lock():
                    current_total_size = self.current_size_bytes + self._pooled_size_bytes.value + self.enqueued_size_bytes
                if current_total_size + allocated_size <= self.max_size_bytes:
                    break  # Memory became available, proceed
                if remaining_time is not None and remaining_time <= 0:
                    raise AllocationFailed(
                        f"Server's attention cache is full, failed to allocate {allocated_size} bytes in {timeout} seconds"
                    )
            self._memory_freed_event.clear()

    def _evict_memory(self, bytes_to_free: int):
        """Evicts tensors from free pools in Global LRU order until at least `bytes_to_free` are freed.

        NOTE: This should ONLY be called from the runtime process (via use_cache).
        When called from ConnectionHandler processes, it only affects the local copy
        of the pools and won't actually free memory in the runtime process.
        """
        assert self._is_runtime_process(), (
            f"_evict_memory must be called from runtime process, "
            f"but current PID is {os.getpid()} (runtime PID is {self.runtime_pid}). "
            f"Use _request_eviction() from ConnectionHandlers instead."
        )

        logger.debug(
            f"{self._log_prefix()} Starting eviction: need to free "
            f"{bytes_to_free / 1024**2:.2f} MB"
        )

        bytes_freed = 0

        # Use lock to protect free pools during eviction
        with self._lock_pools:
            # [IMPROVEMENT 1] Use global LRU keys to decide what to evict next
            # Iterate over a copy because we might modify the map
            for lru_key in list(self._lru_keys.keys()):
                if bytes_freed >= bytes_to_free:
                    break

                device, dtype, numel = lru_key

                # Check if this specific pool exists
                device_pools = self._free_pools.get(device)
                if not device_pools:
                    self._lru_keys.pop(lru_key, None)
                    continue

                dtype_pools = device_pools.get(dtype)
                if not dtype_pools:
                    self._lru_keys.pop(lru_key, None)
                    continue

                pool = dtype_pools.get(numel)
                if not pool:
                    self._lru_keys.pop(lru_key, None)
                    continue

                # Evict from this specific pool
                # We pop multiple if needed, but at least one to make progress on this LRU bucket
                while pool and bytes_freed < bytes_to_free:
                    tensor = pool.pop(0) # Pop from start (oldest in this bucket) or end?
                    # Actually, pool is appended to. So pool.pop(0) is oldest.
                    # But wait, we usually use pop() (newest) for reuse.
                    # For eviction, we want oldest. So pop(0).

                    tensor_size = tensor.numel() * get_size_in_bytes(tensor.dtype)
                    with self._pooled_size_bytes.get_lock():
                        self._pooled_size_bytes.value -= tensor_size
                    bytes_freed += tensor_size
                    del tensor  # Let python GC reclaim memory

                # If pool is now empty, remove it from structures
                if not pool:
                    del dtype_pools[numel]
                    self._lru_keys.pop(lru_key, None) # Remove from LRU tracking

        if bytes_freed > 0:
            self._memory_freed_event.set()
            self._eviction_count += 1

        if bytes_freed < bytes_to_free:
            logger.warning(
                f"{self._log_prefix()} Requested to evict {bytes_to_free / 1024**2:.2f} MB "
                f"but could only evict {bytes_freed / 1024**2:.2f} MB"
            )

        if bytes_freed > 0:
            logger.info(
                f"{self._log_prefix()} Evicted {bytes_freed / 1024**2:.2f} MB "
                f"(requested: {bytes_to_free / 1024**2:.2f} MB)"
            )

    def _needs_compaction(self) -> bool:
        """Check if any pools have enough tensors to warrant compaction."""
        for device_pools in self._free_pools.values():
            for dtype_pools in device_pools.values():
                for pool in dtype_pools.values():
                    if len(pool) >= self.COMPACTION_THRESHOLD:
                        return True
        return False

    def _calculate_fragmentation_score(self) -> float:
        """Calculate a fragmentation score based on pool distribution."""
        total_tensors = 0
        total_pools = 0

        for device_pools in self._free_pools.values():
            for dtype_pools in device_pools.values():
                for pool in dtype_pools.values():
                    if pool:  # Only count non-empty pools
                        total_tensors += len(pool)
                        total_pools += 1

        if total_pools == 0:
            return 0.0

        # Higher score means more fragmentation (many small pools)
        return total_pools / max(1, total_tensors)

    def _get_memory_pressure(self) -> float:
        """Get current memory pressure as a ratio of used/max memory."""
        with self._pooled_size_bytes.get_lock(), self._enqueued_size.get_lock():
            total_used = self.current_size_bytes + self._pooled_size_bytes.value + self.enqueued_size_bytes
        return total_used / self.max_size_bytes if self.max_size_bytes != 2**64 - 1 else 0.0

    def _calculate_adaptive_interval(self) -> int:
        """Calculate adaptive compaction interval based on memory pressure and fragmentation."""
        memory_pressure = self._get_memory_pressure()
        fragmentation_score = self._calculate_fragmentation_score()
        
        # Base interval adjustment factors
        pressure_factor = 1.0
        fragmentation_factor = 1.0
        recency_factor = 1.0
        
        # Adjust based on memory pressure - compact more often when under pressure
        if memory_pressure > self._memory_pressure_threshold:
            pressure_factor = 0.5  # Halve the interval (compact twice as often)
        elif memory_pressure > 0.6:
            pressure_factor = 0.75  # Compact 33% more often
        
        # Adjust based on fragmentation - more fragmented pools need more frequent compaction
        if fragmentation_score > 0.8:
            fragmentation_factor = 0.6  # Compact more often with high fragmentation
        elif fragmentation_score > 0.5:
            fragmentation_factor = 0.8
        
        # Adjust based on whether last compaction found work
        if not self._last_compaction_found_work:
            recency_factor = 1.5  # Increase interval if last compaction didn't find work
        
        # Calculate final interval
        adaptive_interval = int(
            self._compaction_calls_threshold * pressure_factor * fragmentation_factor * recency_factor
        )
        
        # Clamp to min/max bounds
        return max(
            self._adaptive_compaction_min_interval,
            min(self._adaptive_compaction_max_interval, adaptive_interval)
        )

    def _compact_memory_pools(self):
        """
        An adaptive compaction strategy that runs when pools actually need compaction.
        It finds buckets with many tensors and attempts to merge them into larger tensors.
        """
        if not self._needs_compaction():
            logger.debug("Skipping compaction - no pools need compaction")
            self._last_compaction_found_work = False
            return

        logger.debug("Running memory pool compaction...")
        compaction_work_done = False

        # Use lock to protect free pools during compaction
        with self._lock_pools:
            for device, device_pools in self._free_pools.items():
                for dtype, dtype_pools in device_pools.items():
                    # Find the bucket with the most tensors (candidate for compaction)
                    # We iterate over a copy of keys since we might modify the dictionary
                    for numel, pool in list(dtype_pools.items()):
                        if len(pool) >= self.COMPACTION_THRESHOLD:

                            # [IMPROVEMENT 2] Smart Compaction Check
                            # Check if the target merged size (2 * numel) is actually useful
                            target_size = 2 * numel
                            if target_size not in self._requested_sizes:
                                # Skip compacting this pool because the resulting tensors
                                # have never been requested by the model
                                continue

                            logger.debug(
                                f"Compacting pool on {device}:{dtype} with {len(pool)} tensors of size {numel}"
                            )
                            initial_pool_size = len(pool)

                            while len(pool) >= 2:
                                t1 = pool.pop()
                                t2 = pool.pop()
                                new_numel = t1.numel() + t2.numel()

                                try:
                                    # The accounting for _pooled_size_bytes does not change here,
                                    # as we are replacing two tensors with one of their combined size.
                                    new_tensor = torch.empty(new_numel, dtype=dtype, device=device)
                                except Exception as e:
                                    logger.warning(f"Compaction failed to allocate new tensor: {e}")
                                    # If allocation fails, put the tensors back and stop.
                                    pool.append(t1)
                                    pool.append(t2)
                                    break

                                # Add the new, larger tensor to its corresponding pool
                                new_pool = dtype_pools.setdefault(new_numel, [])
                                new_pool.append(new_tensor)
                                dtype_pools.move_to_end(new_numel)

                                # Update LRU for the new bucket
                                self._lru_keys[(device, dtype, new_numel)] = None

                                compaction_work_done = True

                            logger.debug(f"Compaction finished for pool. Size: {initial_pool_size} -> {len(pool)}")

        self._last_compaction_found_work = compaction_work_done
        if compaction_work_done:
            logger.debug("Memory pool compaction completed with work done")
        else:
            logger.debug("Memory pool compaction completed with no work needed")

    def _predict_and_preallocate(self, session_id: str):
        """Compares active session with past patterns and pre-allocates the next tensor if a match is found."""
        # Clean up expired pre-allocations (call outside lock to avoid holding lock during cleanup)
        self._cleanup_expired_preallocations()

        # Acquire lock before accessing any session structures
        with self._lock_pools:
            if session_id not in self._active_sessions or session_id in self._pre_allocated_tensors:
                return  # No history to predict from or already pre-allocated

            history = self._active_sessions[session_id]
            if not history:
                return

            # Update LRU when accessing session patterns
            if session_id in self._session_patterns_lru:
                self._session_patterns_lru.move_to_end(session_id)

            for pattern in self._session_patterns.get(session_id, []):
                if len(pattern) > len(history) and pattern[: len(history)] == history:
                    # Found a matching pattern, predict the next step
                    next_descriptor = pattern[len(history)]
                    logger.debug(f"Predictive allocation for session {session_id}: found pattern, next is {next_descriptor}")

                    # Attempt to find and reserve a tensor from free pools
                    numel = next_descriptor.numel()

                    device_pools = self._free_pools.get(next_descriptor.device)
                    if device_pools:
                        dtype_pools = device_pools.get(next_descriptor.dtype)
                        if dtype_pools and numel in dtype_pools and dtype_pools[numel]:
                            pre_allocated_tensor = dtype_pools[numel].pop()
                            # Decrement _pooled_size_bytes since tensor is being removed from free pool
                            tensor_size = pre_allocated_tensor.numel() * get_size_in_bytes(pre_allocated_tensor.dtype)
                            with self._pooled_size_bytes.get_lock():
                                self._pooled_size_bytes.value -= tensor_size
                            self._pre_allocated_tensors.setdefault(session_id, []).append(pre_allocated_tensor)
                            self._pre_allocation_timestamps[session_id] = time.time()
                            logger.debug(f"Pre-allocated tensor for session {session_id}")
                            # We found a prediction and pre-allocated, we are done for this step
                            return

    @contextlib.contextmanager
    def use_cache(self, *handles: Handle) -> Sequence[torch.Tensor]:
        assert os.getpid() == self.runtime_pid

        # Step 1: Adaptive periodic compaction
        with self._compaction_counter.get_lock():
            self._compaction_counter.value += 1
        adaptive_interval = self._calculate_adaptive_interval()

        with self._compaction_counter.get_lock():
            if self._compaction_counter.value >= adaptive_interval:
                self._compact_memory_pools()
                self._compaction_counter.value = 0
                self._compaction_count += 1

        # Periodically clean up stale sessions (every 500 calls)
        with self._compaction_counter.get_lock():
            if self._compaction_counter.value % 500 == 0:
                self._cleanup_stale_sessions()

        # Step 1.5: Optimized periodic garbage collection
        # Only run GC when needed based on memory pressure and frequency
        if self._should_run_gc():
            collected = self.force_garbage_collection()
            if collected > 0:
                logger.debug(f"Periodic GC collected {collected} objects")

        # Step 2: Evict memory if cache is over budget
        with self._pooled_size_bytes.get_lock(), self._enqueued_size.get_lock():
            total_size = self.current_size_bytes + self._pooled_size_bytes.value + self.enqueued_size_bytes

        if self.max_size_bytes == 2**64 - 1:
            should_evict = False
        else:
            # Calculate evict_threshold with overflow protection
            # When max_size_bytes is very large (like 2^64-1), adding 5% could overflow
            # Use a safe maximum value to prevent overflow
            max_safe_size = 2**63 - 1  # Maximum safe value for int64
            if self.max_size_bytes > max_safe_size // 105 * 100:
                # max_size_bytes is too large for safe calculation, cap the threshold
                evict_threshold = max_safe_size
            else:
                evict_threshold = min(self.max_size_bytes + int(self.max_size_bytes * 0.05), max_safe_size)
            should_evict = total_size > evict_threshold

        if should_evict:
            self._evict_memory(total_size - self.max_size_bytes)

        # Step 3: read creation/deletion requests from connection handlers
        # Note: Async commands (evict, end_session) are now handled via the command queue by a background thread
        # The pipe only carries allocation and free requests
        # Lock is acquired per-message, not for the entire loop, to avoid deadlock
        while self._pipe_recv.poll():
            message = self._pipe_recv.recv()
            recv_handles, recv_data, command_info = message if len(message) == 3 else (message[0], message[1], None)

            # Debug: log all message types for trace diagnostics
            msg_type = "unknown"
            if command_info:
                if command_info.get("session_id"):
                    msg_type = f"allocation for session {command_info.get('session_id')}"
            elif recv_data is not None:
                msg_type = f"allocation ({len(recv_data)} tensors)"
            elif recv_handles is not None:
                msg_type = f"free ({len(recv_handles)} handles)"

            logger.debug(f"{self._log_prefix()} Pipe message: {msg_type}")

            # PRIORITY 1: Handle allocation requests (recv_data is not None)
            if recv_data is not None:  # create new tensors
                assert len(recv_handles) == len(recv_data)
                session_id = command_info.get("session_id") if command_info else None

                # Acquire lock at start of allocation processing to protect session structures
                with self._lock_pools:
                    if session_id:
                        session_history = self._active_sessions.setdefault(session_id, [])
                        session_history.extend(recv_data)
                        # Track handles allocated for this session
                        self._session_handles.setdefault(session_id, set()).update(recv_handles)
                        # Update session timeout tracking
                        self._active_session_timeouts[session_id] = time.time()

                    for handle, descr in zip(recv_handles, recv_data):
                        # [IMPROVEMENT 2] Track requested sizes with LRU eviction
                        if descr.numel() in self._requested_sizes:
                            self._requested_sizes.move_to_end(descr.numel())
                        else:
                            self._requested_sizes[descr.numel()] = None
                            while len(self._requested_sizes) > self._max_requested_sizes:
                                self._requested_sizes.popitem(last=False)

                        reused_tensor = None
                        numel = descr.numel()

                        # Step 3a: Try to fulfill from pre-allocated tensors first
                        from_preallocated = False
                        if session_id and session_id in self._pre_allocated_tensors:
                            pre_allocated_pool = self._pre_allocated_tensors[session_id]
                            for i, tensor in enumerate(pre_allocated_pool):
                                if tensor.numel() == numel and tensor.dtype == descr.dtype and tensor.device == descr.device:
                                    reused_tensor = pre_allocated_pool.pop(i)
                                    from_preallocated = True  # Mark as pre-allocated
                                    logger.debug(f"Used pre-allocated tensor for session {session_id}")
                                    break
                            if not pre_allocated_pool:
                                del self._pre_allocated_tensors[session_id]
                                self._pre_allocation_timestamps.pop(session_id, None)

                        # Step 3b: If not found, try to fulfill from general free pools
                        if reused_tensor is None:
                            device_pools = self._free_pools.get(descr.device)
                            if device_pools:
                                dtype_pools = device_pools.get(descr.dtype)
                                if dtype_pools and numel in dtype_pools and dtype_pools[numel]:
                                    reused_tensor = dtype_pools[numel].pop()
                                    dtype_pools.move_to_end(numel)

                                    # [IMPROVEMENT 1] Update LRU (refresh usage)
                                    self._lru_keys.move_to_end((descr.device, descr.dtype, numel))

                        if reused_tensor is not None:
                            self._allocated_tensors[handle] = reused_tensor.view(descr.shape)
                            # Only decrement _pooled_size_bytes for tensors from general free pool
                            # Pre-allocated tensors already had _pooled_size_bytes decremented when pre-allocated
                            if not from_preallocated:
                                tensor_size = reused_tensor.numel() * get_size_in_bytes(reused_tensor.dtype)
                                with self._pooled_size_bytes.get_lock():
                                    self._pooled_size_bytes.value -= tensor_size
                            # Mark handle as pre-allocated for proper cleanup accounting
                            if from_preallocated:
                                self._preallocated_handles.add(handle)
                        else:
                            # Allocate new tensor
                            self._allocated_tensors[handle] = torch.empty(
                                descr.shape, dtype=descr.dtype, device=descr.device
                            )

                # Sanity check handles are allocated - check after all allocations
                with self._lock_pools:
                    missing_in_batch = [h for h in recv_handles if h not in self._allocated_tensors]
                if missing_in_batch:
                    raise RuntimeError(f"Failed to allocate handles: {missing_in_batch}")

                if session_id: # After fulfilling request, try to predict and pre-allocate the next one
                    self._predict_and_preallocate(session_id)

                continue

            # PRIORITY 2: Handle free requests (recv_handles is not None, recv_data is None)
            if recv_handles is not None:  # delete tensors by handle
                for handle in recv_handles:
                    # Handle may have been freed by background thread's end_session processing
                    # Use pop with default to safely handle this case
                    with self._lock_pools:
                        tensor = self._allocated_tensors.pop(handle, None)
                    if tensor is None:
                        logger.debug(
                            f"Handle {handle} already freed (likely by background thread end_session processing)"
                        )
                        continue

                    descr = TensorDescriptor.from_tensor(tensor)
                    numel = tensor.numel()
                    tensor_size = numel * get_size_in_bytes(descr.dtype)

                    with self._lock_pools:
                        device_pool = self._free_pools.setdefault(descr.device, {})
                        dtype_pool = device_pool.setdefault(descr.dtype, OrderedDict())
                        numel_pool = dtype_pool.setdefault(numel, [])
                        numel_pool.append(tensor)
                        dtype_pool.move_to_end(numel)  # Mark as recently used

                        # [IMPROVEMENT 1] Update LRU (refresh usage)
                        self._lru_keys[(descr.device, descr.dtype, numel)] = None
                        self._lru_keys.move_to_end((descr.device, descr.dtype, numel))

                        # Clear the preallocated mark if present - now protected by _lock_pools
                        self._preallocated_handles.discard(handle)

                    # Only increment _pooled_size_bytes for normal allocations
                    # Pre-allocated handles had _pooled_size_bytes decremented when pre-allocated,
                    # so incrementing here restores the balance
                    with self._pooled_size_bytes.get_lock():
                        self._pooled_size_bytes.value += tensor_size

                continue

        # Step 4: Yield tensors
        # Acquire lock to safely read _allocated_tensors
        with self._lock_pools:
            missing_handles = [h for h in handles if h not in self._allocated_tensors]
            if missing_handles:
                raise CacheHandleMissingError(
                    f"Handles {missing_handles} missing from cache. "
                    "This usually happens when the session times out or is closed by the client "
                    "while a request is being processed."
                )
            # Make a copy of tensors while holding lock
            tensors_to_yield = tuple(self._allocated_tensors[handle] for handle in handles)
        yield tensors_to_yield

    def force_memory_cleanup(self, target_free_bytes: Optional[int] = None) -> int:
        """
        Force aggressive memory cleanup and eviction.

        :param target_free_bytes: Target amount of memory to free, if None tries to free all pooled memory
        :return: Total bytes freed
        """
        total_freed = 0

        try:
            # Step 1: Force garbage collection
            collected = self.force_garbage_collection()
            logger.info(f"Force cleanup: collected {collected} Python objects")

            # Step 2: Evict pooled memory
            if target_free_bytes is None:
                # Try to free all pooled memory
                target_free_bytes = self._pooled_size_bytes.value

            if target_free_bytes > 0:
                self._evict_memory(target_free_bytes)
                total_freed += target_free_bytes
                logger.info(f"Force cleanup: evicted {target_free_bytes / 1024**2:.2f} MB from memory pools")

            # Step 3: Final garbage collection
            final_collected = self.force_garbage_collection()
            logger.info(f"Force cleanup: final GC collected {final_collected} objects")

        except Exception as e:
            logger.error(f"Force memory cleanup failed: {e}")

        return total_freed

    def recycle_tensors(self, tensors: Sequence[torch.Tensor]) -> None:
        """Return temporary tensors back to the free pool for future reuse."""
        assert os.getpid() == self.runtime_pid

        # Use lock when modifying free pools
        with self._lock_pools:
            for tensor in tensors:
                if tensor is None:
                    continue
                if not isinstance(tensor, torch.Tensor):
                    continue
                if tensor.device.type != "mps":
                    continue
                descr = TensorDescriptor.from_tensor(tensor)
                numel = tensor.numel()
                device_pool = self._free_pools.setdefault(descr.device, {})
                dtype_pool = device_pool.setdefault(descr.dtype, OrderedDict())
                numel_pool = dtype_pool.setdefault(numel, [])
                numel_pool.append(tensor)
                dtype_pool.move_to_end(numel)

                # [IMPROVEMENT 1] Update LRU
                self._lru_keys[(descr.device, descr.dtype, numel)] = None
                self._lru_keys.move_to_end((descr.device, descr.dtype, numel))

                tensor_size = numel * get_size_in_bytes(descr.dtype)
                with self._pooled_size_bytes.get_lock():
                    self._pooled_size_bytes.value += tensor_size

    def _cleanup_expired_preallocations(self):
        """Clean up pre-allocated tensors that have expired due to timeout."""
        # Use lock when accessing and modifying preallocation structures
        with self._lock_pools:
            current_time = time.time()
            expired_sessions = []

            # Iterate over _pre_allocation_timestamps - now protected by _lock_pools
            for session_id, timestamp in self._pre_allocation_timestamps.items():
                if current_time - timestamp > self._pre_allocation_timeout:
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                logger.warning(f"Cleaning up expired pre-allocations for session {session_id}")
                if session_id in self._pre_allocated_tensors:
                    for tensor in self._pre_allocated_tensors.pop(session_id):
                        # Return tensor to the free pool with proper accounting
                        descr = TensorDescriptor.from_tensor(tensor)
                        device_pool = self._free_pools.setdefault(descr.device, {})
                        dtype_pool = device_pool.setdefault(descr.dtype, OrderedDict())
                        numel_pool = dtype_pool.setdefault(tensor.numel(), [])
                        numel_pool.append(tensor)
                        dtype_pool.move_to_end(tensor.numel())

                        # Update LRU tracking
                        self._lru_keys[(descr.device, descr.dtype, tensor.numel())] = None
                        self._lru_keys.move_to_end((descr.device, descr.dtype, tensor.numel()))

                        # Increment _pooled_size_bytes since tensor is being added to free pool
                        tensor_size = tensor.numel() * get_size_in_bytes(descr.dtype)
                        with self._pooled_size_bytes.get_lock():
                            self._pooled_size_bytes.value += tensor_size

                    self._pre_allocation_timestamps.pop(session_id, None)
                    continue

        # Clean up _ended_sessions to prevent unbounded growth
        # This is done OUTSIDE _lock_pools to follow lock ordering hierarchy
        # (_ended_sessions_lock is a higher-level lock than _lock_pools)
        # Keep only the most recent 1000 ended sessions
        if len(self._ended_sessions) > 1000:
            with self._ended_sessions_lock:
                if len(self._ended_sessions) > 1000:
                        # Sort by timestamp and keep only the most recent 1000
                        sorted_sessions = sorted(self._ended_sessions.items(), key=lambda x: x[1])
                        # Remove oldest sessions, keep newest 1000
                        sessions_to_remove = [s_id for s_id, _ in sorted_sessions[:-1000]]
                        for s_id in sessions_to_remove:
                            self._ended_sessions.pop(s_id, None)

    def _cleanup_stale_sessions(self):
        """Remove sessions that have been inactive too long."""
        current_time = time.time()
        with self._lock_pools:
            stale_sessions = [
                session_id for session_id, timestamp in self._active_session_timeouts.items()
                if current_time - timestamp > self._session_timeout
            ]
            for session_id in stale_sessions:
                logger.warning(f"Cleaning up stale session {session_id}")
                self._active_sessions.pop(session_id, None)
                self._session_handles.pop(session_id, None)
                self._active_session_timeouts.pop(session_id, None)

    def _log_memory_stats(self):
        """Log memory statistics for monitoring purposes."""
        current_time = time.time()
        if current_time - self._last_monitoring_log >= self._monitoring_interval:
            gib = 1024**3
            with self._pooled_size_bytes.get_lock(), self._enqueued_size.get_lock():
                total_used = self.current_size_bytes + self._pooled_size_bytes.value + self.enqueued_size_bytes
                memory_pressure = total_used / self.max_size_bytes if self.max_size_bytes != 2**64 - 1 else 0.0

            logger.info(
                f"Memory Cache Stats - "
                f"Used: {total_used / gib:.2f} GiB ({memory_pressure * 100:.1f}%), "
                f"Available: {self.bytes_left / gib:.2f} GiB, "
                f"Allocations: {self._allocation_count}, "
                f"Evictions: {self._eviction_count}, "
                f"Compactions: {self._compaction_count}, "
                f"Active Sessions: {len(self._active_sessions)}, "
                f"Pre-allocated: {len(self._pre_allocated_tensors)}"
            )

            # Log warning if memory pressure is high
            if memory_pressure > 0.9:
                logger.warning(f"High memory pressure detected: {memory_pressure * 100:.1f}%")
            elif memory_pressure > 0.75:
                logger.info(f"Elevated memory pressure: {memory_pressure * 100:.1f}%")

            self._last_monitoring_log = current_time

    def _check_stalled_evictions(self, timeout_seconds: float = 5.0) -> None:
        """Log warnings for eviction requests that are taking too long.

        :param timeout_seconds: Threshold in seconds before considering a request stalled
        """
        current_time = time.time()
        stalled = []

        for request_id, info in list(self._pending_evictions.items()):
            age = current_time - info["timestamp"]
            if age > timeout_seconds:
                stalled.append((request_id, age, info["bytes_requested"]))

        if stalled:
            for request_id, age, bytes_req in stalled:
                logger.warning(
                    f"{self._log_prefix()} Stalled eviction request #{request_id}: "
                    f"{bytes_req / 1024**2:.2f} MB pending for {age:.1f}s"
                )


class AllocationFailed(Exception):
    pass


class CacheHandleMissingError(Exception):
    """Raised when attempting to access a cache handle that has been freed."""
    pass