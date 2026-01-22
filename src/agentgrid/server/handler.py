#
# Copyright (c) 2025 ReEnvision AI, LLC. All rights reserved.
#
# This software is the confidential and proprietary information of
# ReEnvision AI, LLC ("Confidential Information"). You shall not
# disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with ReEnvision AI, LLC.
#

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import sys
from enum import Enum
from itertools import chain
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from async_timeout import timeout
from hivemind import (
    DHT,
    MSGPackSerializer,
    P2PContext,
    PeerID,
    deserialize_tensor_stream,
    deserialize_torch_tensor,
    nested_flatten,
    nested_pack,
    serialize_torch_tensor,
)
from hivemind.moe.server.connection_handler import ConnectionHandler
from hivemind.p2p.p2p_daemon import DEFAULT_MAX_MSG_SIZE
from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import amap_in_executor, anext
from hivemind.utils.logging import get_logger
from hivemind.utils.streaming import split_for_streaming

import agentgrid
from agentgrid.data_structures import CHAIN_DELIMITER, UID_DELIMITER, Handle, ModuleUID
from agentgrid.server.backend import TransformerBackend
from agentgrid.server.block_functions import iterate_rpc_inference, run_rpc_forward
from agentgrid.server.task_prioritizer import DummyTaskPrioritizer, TaskPrioritizerBase
from agentgrid.utils.convert_block import QuantType

logger = get_logger(__name__)


# Fix pickling protobufs, see https://stackoverflow.com/a/74873028
sys.modules["runtime_pb2"] = runtime_pb2


CACHE_TOKENS_AVAILABLE = "cache_tokens_available"


class Event(Enum):
    NEW_SESSION = 0
    END_SESSION = 1
    PUSH = 2
    SHUTDOWN = 3


class TransformerConnectionHandler(ConnectionHandler):
    """Handles three request types: forward, backward and forward-incremental (inference)"""

    module_backends: Dict[ModuleUID, TransformerBackend]

    def __init__(
        self,
        dht: DHT,
        module_backends: Dict[str, TransformerBackend],
        *,
        adapters: Optional[Sequence[str]],
        dht_prefix: str,
        handler_event_queues: Sequence[mp.Queue],
        handler_index: int,
        inference_max_length: int,
        request_timeout: float,
        session_timeout: float,
        step_timeout: float,
        task_prioritizer: TaskPrioritizerBase = DummyTaskPrioritizer(),
        quant_type: QuantType,
    ):
        super().__init__(dht, module_backends)
        for module_backend in self.module_backends.values():
            assert isinstance(module_backend, TransformerBackend)
        self.dht_prefix = dht_prefix
        self.adapters = adapters
        self._handler_event_queues = handler_event_queues
        self._handler_index = handler_index
        self._own_event_queue = handler_event_queues[handler_index]
        self._listener_task: Optional[asyncio.Task] = None
        self._session_queues: Dict[str, asyncio.Queue] = {}
        self._session_handlers: Dict[str, int] = {}

        self.inference_max_length = inference_max_length
        self.request_timeout = request_timeout
        self.session_timeout, self.step_timeout = session_timeout, step_timeout
        self._prioritizer = task_prioritizer
        self.quant_type = quant_type

    async def add_p2p_handlers(self, *args, **kwargs) -> None:
        if self._listener_task is None:
            # Start listening to our own event queue before we accept any requests
            self._listener_task = asyncio.create_task(self._listen_to_event_queue())
        await super().add_p2p_handlers(*args, **kwargs)

    def shutdown(self):
        if self.is_alive():
            self._outer_pipe.send("_shutdown")
            self._own_event_queue.put((Event.SHUTDOWN, None, None))
            self.join(self.shutdown_timeout)
            if self.is_alive():
                logger.warning(f"{self.__class__.__name__} failed to shut down gracefully, sending SIGTERM")
                self.terminate()

    async def _gather_inputs(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> Tuple[str, List[torch.Tensor], Dict]:
        block_uid, metadata = None, None

        def _unpack(req: runtime_pb2.ExpertRequest) -> Iterable[runtime_pb2.Tensor]:
            nonlocal block_uid, metadata

            if block_uid is None:
                block_uid = req.uid
            elif block_uid != req.uid:
                raise ValueError("Block uids differ in one request")

            if metadata is None:
                metadata = MSGPackSerializer.loads(req.metadata) if req.metadata else {}

            return req.tensors

        tensors_stream = amap_in_executor(_unpack, requests)
        inputs = await deserialize_tensor_stream(tensors_stream)
        assert isinstance(block_uid, str) and isinstance(metadata, dict)
        return block_uid, inputs, metadata

    async def rpc_inference(
        self,
        requests: AsyncIterator[runtime_pb2.ExpertRequest],
        context: P2PContext,
    ) -> AsyncIterator[runtime_pb2.ExpertResponse]:
        """Compute a single step of inference using attention cache; update attention cache accordingly."""
        session_id = None
        try:
            request = await asyncio.wait_for(anext(requests), self.step_timeout)
        except asyncio.TimeoutError:
            self._log_request("rpc_inference.open", None, context, warning="timed out")
            return

        requested_uids = self._check_uids(request.uid)
        self._log_request("rpc_inference.open", requested_uids, context)
        try:
            metadata = MSGPackSerializer.loads(request.metadata) if request.metadata else {}
            requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)
            max_length = metadata.get("max_length")
            points = metadata.get("points", 0)
            session_id = metadata.get("session_id")
            logger.debug(f"[INFERENCE_REQUEST_RECEIVED] session_id={session_id}, handler_index={self._handler_index}, max_length={max_length}")
            alloc_timeout = float(metadata.get("alloc_timeout", 0.0))
            args_structure = metadata.get("args_structure")
            if not requested_uids:
                raise ValueError("User must specify at least one block for inference, but got none")
            assert isinstance(
                max_length, int
            ), f"rpc_inference metadata must contain int max_length, got {max_length}"
            assert isinstance(
                points, (float, int)
            ), f"rpc_inference should have number of points as a number or None, got {points}"
            if not 0 <= max_length <= self.inference_max_length:
                raise ValueError(
                    f"Cannot allocate KV cache for {max_length} tokens, max = {self.inference_max_length}"
                )

            batch_size = request.tensors[0].size[0] if request.tensors else 1

            try:
                cache_context = self._allocate_cache(
                    requested_backends,
                    batch_size=batch_size,
                    max_length=max_length,
                    timeout=alloc_timeout,
                    session_id=session_id,
                )
                async with cache_context as cache_handles:
                    background_task = None
                    async for output_tensors, can_push, step_metadata in iterate_rpc_inference(
                        requested_uids=requested_uids,
                        requested_backends=requested_backends,
                        active_adapter=self._get_active_adapter(metadata),
                        input_iterator=self._iterate_inference_steps(
                            request, requests, session_id, requested_uids, context
                        ),
                        cache_handles=cache_handles,
                        max_length=max_length,
                        prioritizer=self._prioritizer,
                        points=points,
                        quant_type=self.quant_type,
                        args_structure=args_structure,
                    ):
                        if can_push:
                            if background_task is None:
                                background_task = asyncio.create_task(
                                    self._push_outputs(request, output_tensors[0], step_metadata)
                                )
                        yield runtime_pb2.ExpertResponse(tensors=output_tensors)

            except Exception as e:
                # Fallback: try to force memory cleanup and retry with longer timeout
                if "out of memory" in str(e).lower() or "allocation" in str(e).lower():
                    logger.warning(f"Memory allocation failed, attempting cleanup: {e}")
                    try:
                        # Force cleanup on the first backend
                        first_backend = requested_backends[0]
                        bytes_freed = first_backend.memory_cache.force_memory_cleanup()
                        logger.info(f"Force cleanup freed {bytes_freed / 1024**2:.2f} MB")

                        # Retry allocation with a longer timeout
                        retry_timeout = max(alloc_timeout, 30.0)  # At least 30 seconds
                        cache_context = self._allocate_cache(
                            requested_backends,
                            batch_size=batch_size,
                            max_length=max_length,
                            timeout=retry_timeout,
                            session_id=session_id,
                        )
                        async with cache_context as cache_handles:
                            background_task = None
                            async for output_tensors, can_push, step_metadata in iterate_rpc_inference(
                                requested_uids=requested_uids,
                                requested_backends=requested_backends,
                                active_adapter=self._get_active_adapter(metadata),
                                input_iterator=self._iterate_inference_steps(
                                    request, requests, session_id, requested_uids, context
                                ),
                                cache_handles=cache_handles,
                                max_length=max_length,
                                prioritizer=self._prioritizer,
                                points=points,
                                quant_type=self.quant_type,
                                args_structure=args_structure,
                            ):
                                if can_push:
                                    if background_task is None:
                                        background_task = asyncio.create_task(
                                            self._push_outputs(request, output_tensors[0], step_metadata)
                                        )
                                yield runtime_pb2.ExpertResponse(tensors=output_tensors)
                    except Exception as retry_e:
                        logger.error(f"Retry after cleanup also failed: {retry_e}")
                        raise retry_e
                else:
                    raise e

        finally:
            # Signal the runtime that the session has ended.
            # This must happen after allocate_cache's finally block has run (freeing handles),
            # so that the runtime can properly clean up session state and store patterns.
            if session_id is not None:
                try:
                    # Call end_session on the first backend's memory cache
                    # This sends a command to the runtime via pipe
                    requested_backends = tuple(self.module_backends.get(uid) for uid in requested_uids or [])
                    if requested_backends:
                        logger.debug(f"[Handler {self._handler_index}] Calling end_session for session_id={session_id}")
                        requested_backends[0].memory_cache.end_session(session_id)
                except Exception as e:
                    # Log but don't fail the request if end_session fails
                    logger.warning(f"Failed to call end_session for session {session_id}: {e}")
            self._log_request("rpc_inference.close", requested_uids, context)

    @contextlib.contextmanager
    def _managed_session(self, session_id: str):
        assert session_id not in self._session_queues, f"session id {session_id} is not unique"
        logger.debug(f"[INFERENCE_SESSION_START] session_id={session_id}, handler_index={self._handler_index}")
        try:
            self._session_queues[session_id] = asyncio.Queue()
            self._session_handlers[session_id] = self._handler_index
            for other_index, other_queue in enumerate(self._handler_event_queues):
                if other_index != self._handler_index:
                    other_queue.put_nowait((Event.NEW_SESSION, session_id, self._handler_index))
            yield
        finally:
            logger.info(f"[INFERENCE_SESSION_STOP] session_id={session_id}, handler_index={self._handler_index}")
            logger.info(f"[CACHE_CLEARING_INITIATED] session_id={session_id}, handler_index={self._handler_index} - notifying all handlers to clear session state")
            self._session_queues.pop(session_id).put_nowait(None)  # put None so that the get task will not hang
            del self._session_handlers[session_id]
            # Note: END_SESSION events are no longer needed here - end_session() is now properly
            # called from rpc_inference's finally block, which communicates with the runtime
            # to clean up session state and store allocation patterns.
            for other_index, other_queue in enumerate(self._handler_event_queues):
                if other_index != self._handler_index:
                    other_queue.put_nowait((Event.END_SESSION, session_id, self._handler_index))
            logger.debug(f"[CACHE_CLEARING_NOTIFICATIONS_SENT] session_id={session_id}, notified {len(self._handler_event_queues) - 1} other handlers")

    def _put_into_session_queue(self, session_id: str, request: runtime_pb2.ExpertRequest):
        handler_index = self._session_handlers.get(session_id)
        if handler_index is None:
            logger.debug(f"Ignored rpc_push to unknown session ID: {session_id}")
        elif handler_index == self._handler_index:
            self._session_queues[session_id].put_nowait(request)
        else:
            self._handler_event_queues[handler_index].put_nowait((Event.PUSH, session_id, request))

    async def _get_from_session_queue(self, session_id: str) -> Optional[runtime_pb2.ExpertRequest]:
        assert self._session_handlers[session_id] == self._handler_index, "session belongs to another handler"
        return await self._session_queues[session_id].get()

    async def _listen_to_event_queue(self):
        loop = asyncio.get_event_loop()
        while True:
            try:
                event, session_id, payload = await loop.run_in_executor(None, self._own_event_queue.get)
                if event == Event.SHUTDOWN:
                    break
                elif event == Event.NEW_SESSION:
                    self._session_handlers[session_id] = payload  # index of the handler that owns that session
                elif event == Event.END_SESSION:
                    self._session_handlers.pop(session_id, None)
                elif event == Event.PUSH:
                    maybe_session_queue = self._session_queues.get(session_id)
                    if maybe_session_queue is not None:
                        maybe_session_queue.put_nowait(payload)
                else:
                    raise RuntimeError(f"Unexpected event: {event}")
            except Exception as e:
                logger.exception(e)

    async def _iterate_inference_steps(
        self,
        first_request: runtime_pb2.ExpertRequest,
        requests: AsyncIterator[runtime_pb2.ExpertRequest],
        session_id: Optional[str],
        requested_uids: Sequence[str],
        context: P2PContext,
    ) -> AsyncIterator[Tuple[runtime_pb2.ExpertRequest, dict]]:
        processed_step_ids = set()
        n_pushes = n_late_pushes = 0
        request = first_request
        anext_task = get_push_task = None
        try:
            with self._managed_session(session_id) if session_id is not None else contextlib.nullcontext():
                while request.tensors:  # iterate while user is willing to supply tensors
                    metadata = MSGPackSerializer.loads(request.metadata) if request.metadata else {}
                    step_id = metadata.get("step_id")

                    pushed = metadata.get("pushed")
                    if pushed:
                        n_pushes += 1
                        self._log_request("rpc_inference.push", requested_uids, context, debug=f"session received push")

                    if step_id is None or step_id not in processed_step_ids:
                        yield request, metadata
                        if step_id is not None:
                            processed_step_ids.add(step_id)
                    elif pushed:
                        n_late_pushes += 1
                        self._log_request(
                            "rpc_inference.push",
                            requested_uids,
                            context,
                            warning=f"arrived late {n_late_pushes / n_pushes * 100:.1f}% of the time",
                        )

                    # Wait for the next request, coming either from the `requests` iterator or `push_queue`
                    if anext_task is None:
                        anext_task = asyncio.create_task(anext(requests))
                    if get_push_task is None:
                        if session_id is not None:
                            get_push_task = asyncio.create_task(self._get_from_session_queue(session_id))
                        else:
                            get_push_task = asyncio.create_task(asyncio.Event().wait())  # Dummy never-ending task
                    done, _ = await asyncio.wait(
                        [anext_task, get_push_task], timeout=self.step_timeout, return_when=asyncio.FIRST_COMPLETED
                    )

                    if anext_task in done:
                        request = await anext_task
                        anext_task = None
                    elif get_push_task in done:
                        request = await get_push_task
                        get_push_task = None
                    else:
                        self._log_request("rpc_inference.step", requested_uids, context, warning="timed out")
                        anext_task.cancel()
                        get_push_task.cancel()
                        return
        except Exception:
            logger.warning("rpc_inference._iterate_inference_steps() exception:", exc_info=True)
            raise

    async def rpc_push(self, request: runtime_pb2.ExpertRequest, context: P2PContext) -> runtime_pb2.ExpertResponse:
        """Directly push activation tensors from one server to another"""

        requested_uids = self._check_uids(request.uid)
        metadata = MSGPackSerializer.loads(request.metadata)
        session_id = metadata["session_id"]
        self._log_request("rpc_push", requested_uids, context, debug=f"session_id={session_id}")
        self._put_into_session_queue(session_id, request)
        return runtime_pb2.ExpertResponse()

    async def _push_outputs(
        self, request: runtime_pb2.ExpertRequest, serialized_outputs: runtime_pb2.Tensor, metadata: dict
    ) -> None:
        try:
            next_servers = metadata.get("next_servers")
            if not next_servers:
                return

            next_peer_id, next_session_id, next_start, next_end = next_servers[0]
            next_peer_id = PeerID.from_base58(next_peer_id)
            next_uid = CHAIN_DELIMITER.join(f"{self.dht_prefix}{UID_DELIMITER}{i}" for i in range(next_start, next_end))

            # Sending hidden states serialized with output_schema to avoid double serialization
            next_tensors = [serialized_outputs] + request.tensors[1:]
            next_metadata = metadata.copy()
            next_metadata.update(session_id=next_session_id, next_servers=next_servers[1:], pushed=True)

            stub = self.get_stub(self._p2p, next_peer_id)
            await stub.rpc_push(
                runtime_pb2.ExpertRequest(
                    uid=next_uid,
                    tensors=next_tensors,
                    metadata=MSGPackSerializer.dumps(next_metadata),
                ),
                timeout=self.request_timeout,
            )
        except Exception:
            logger.debug(
                f"Failed to push outputs to peer_id={next_peer_id}, session_id={next_session_id}, blocks={next_start}:{next_end}:",
                exc_info=True,
            )

    async def rpc_forward(self, request: runtime_pb2.ExpertRequest, context: P2PContext) -> runtime_pb2.ExpertResponse:
        async with timeout(self.request_timeout):
            # Parse request and prepare backends
            flat_inputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
            requested_uids = self._check_uids(request.uid)
            self._log_request("rpc_forward", requested_uids, context)

            requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)
            metadata = MSGPackSerializer.loads(request.metadata) if request.metadata else {}
            active_adapter = self._get_active_adapter(metadata)
            points = metadata.get("points", 0)
            args_structure = metadata.get("args_structure")
            assert isinstance(
                points, (float, int)
            ), f"rpc_forward should have number of points as number or None, got {points}"

            hidden_states = await run_rpc_forward(
                *flat_inputs,
                requested_backends=requested_backends,
                prioritizer=self._prioritizer,
                active_adapter=active_adapter,
                points=points,
                args_structure=args_structure,
            )
            return runtime_pb2.ExpertResponse(
                tensors=self._serialize_outputs(hidden_states, requested_backends, metadata)
            )

    async def rpc_forward_stream(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertRequest]:
        async with timeout(self.request_timeout):
            # Parse requests and prepare backends
            uid_str, flat_inputs, metadata = await self._gather_inputs(requests, context)
            requested_uids = self._check_uids(uid_str)
            self._log_request("rpc_forward_stream", requested_uids, context)

            requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)
            active_adapter = self._get_active_adapter(metadata)
            points = metadata.get("points", 0)
            args_structure = metadata.get("args_structure")
            assert isinstance(
                points, (float, int)
            ), f"rpc_forward_stream should have number of points as number or None, got {points}"

            hidden_states = await run_rpc_forward(
                *flat_inputs,
                requested_backends=requested_backends,
                prioritizer=self._prioritizer,
                active_adapter=active_adapter,
                points=points,
                args_structure=args_structure,
            )

            # Split the serialized_output for streaming and respond to client
            for tensor in self._serialize_outputs(hidden_states, requested_backends, metadata):
                for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE):
                    yield runtime_pb2.ExpertResponse(tensors=[part])

    def _serialize_outputs(
        self,
        hidden_states: torch.Tensor,
        requested_backends: Sequence[TransformerBackend],
        metadata: Dict[str, Any],
    ) -> Sequence[runtime_pb2.Tensor]:
        """Serialize forward outputs using either outputs_schema or custom user-specified schema"""
        assert isinstance(hidden_states, torch.Tensor) and hidden_states.ndim == 3, "hidden_states must be a 3d tensor"
        outputs_schema = requested_backends[-1].outputs_schema

        if metadata.get("output_compression") is not None:
            assert isinstance(metadata["output_compression"], (list, tuple)), "output_compression must be a tuple/list"
            output_compression = tuple(metadata["output_compression"])
            assert all(isinstance(c, int) for c in output_compression), "output_compression must contain integers"
            assert len(output_compression) == 1, f"output_compression tuple should have 1 element"
        else:
            output_compression = tuple(tensor.compression for tensor in outputs_schema)

        return [
            serialize_torch_tensor(result.to(proto.dtype), compression, allow_inplace=True)
            for result, proto, compression in zip([hidden_states], outputs_schema, output_compression)
        ]

    

    def _get_active_adapter(self, metadata: dict) -> str:
        active_adapter = metadata.get("active_adapter", "")
        if active_adapter and (active_adapter not in self.adapters):
            raise KeyError(f"adapter {active_adapter} not found")
        return active_adapter

    

    def _check_uids(self, uids: str) -> Tuple[ModuleUID, ...]:
        """Check that the first request to rpc_inference is valid"""
        uids = (uids or "").split(CHAIN_DELIMITER)
        if not uids:
            raise RuntimeError("User did not provide any uids")
        for uid in uids:
            if uid not in self.module_backends:
                raise RuntimeError(f"Remote peer does not serve {uid}")
        return tuple(uids)

    @contextlib.asynccontextmanager
    async def _allocate_cache(
        self,
        backends: Sequence[TransformerBackend],
        *,
        batch_size: int,
        max_length: int,
        timeout: Optional[float],
        session_id: Optional[str] = None,
    ) -> Sequence[Sequence[Handle]]:
        """
        Allocate memory cache for all transformer blocks, return cache handle
        :returns: a list of {len(backends)} elements, where i-th element is a tuple of cache handles for i-th backend
        """
        descriptors = [backend.get_inference_cache_descriptors(batch_size, max_length) for backend in backends]
        async with backends[0].memory_cache.allocate_cache(
            *chain(*descriptors), timeout=timeout, session_id=session_id
        ) as handles:
            yield nested_pack(handles, descriptors)

    def _log_request(
        self,
        method: str,
        uids: Optional[Sequence[ModuleUID]],
        context: P2PContext,
        *,
        debug: Optional[str] = None,
        warning: Optional[str] = None,
    ) -> None:
        if uids is not None:
            friendly_uids = [uid.split(".")[-1] for uid in uids if "." in uid]
            friendly_uids = [int(uid) for uid in friendly_uids if uid.isdigit()]
            friendly_uids = f"{min(friendly_uids)}:{max(friendly_uids) + 1}" if friendly_uids else uids
        else:
            friendly_uids = "n/a"

        friendly_remote_id = "..." + str(context.remote_id)[-6:]

        message = f"{method}(blocks={friendly_uids}, remote_peer={friendly_remote_id})"
        if warning is not None:
            logger.warning(f"{message}: {warning}")
        elif debug is not None:
            logger.debug(f"{message}: {debug}")
        else:
            logger.info(message)

    async def rpc_info(self, request: runtime_pb2.ExpertUID, context: P2PContext) -> runtime_pb2.ExpertInfo:
        """Return metadata about stored block uids and current load"""

        backend = self.module_backends[request.uid] if request.uid else next(iter(self.module_backends.values()))
        result = {
            "version": agentgrid.__version__,
            "dht_client_mode": self.dht.client_mode,
            CACHE_TOKENS_AVAILABLE: backend.memory_cache.bytes_left // max(1, *backend.cache_bytes_per_token.values()),
        }

        if request.uid:
            block_info = self.module_backends[request.uid].get_info()
            common_keys = set(result.keys()) & set(block_info.keys())
            if common_keys:
                raise RuntimeError(f"The block's rpc_info has keys reserved for the server's rpc_info: {common_keys}")
            result.update(block_info)

        return runtime_pb2.ExpertInfo(serialized_info=MSGPackSerializer.dumps(result))
