# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BlockTransferEngine — vendor-agnostic block transfer engine.

No longer tied to XDMA or any specific FPGA model.  All data movement
is delegated to a ``DMABackend`` instance.

Usage::

    # Xilinx XDMA + bounce buffer (existing, no P2P required)
    backend = XDMABounceBufferBackend(xdma_handle, allocator)
    engine = BlockTransferEngine(backend, gpu_caches, store_stream, load_stream)

    # GPU DMA controller direct BAR write (P2P, zero bounce)
    backend = GPUDirectP2PBackend(allocator, bar_cuda_ptr, ...)
    engine = BlockTransferEngine(backend, gpu_caches, store_stream, load_stream)

    # Intel FPGA (vendor-specific driver)
    backend = IntelFPGAP2PBackend(allocator)
    engine = BlockTransferEngine(backend, gpu_caches, store_stream, load_stream)

    # Same launch API regardless of backend:
    engine.launch_copy(src_blocks, dst_blocks, is_store, event_idx, events_list)
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.fpga.backend_base import DMABackend

logger = init_logger(__name__)


class BlockTransferEngine:
    """Block transfer engine — vendor agnostic.

    Responsibilities:
      1. Background thread queuing and serialization (async
         ``launch_copy`` → background ``_copy_loop``).
      2. Block-by-block, layer-by-layer transfer strategy.
      3. CUDA event completion tracking.
      4. Backend lifecycle management.

    All data movement is delegated to the ``DMABackend`` instance,
    completely decoupling the transfer strategy from the physical
    transport (XDMA, P2P DMA, CXL.mem, ...).
    """

    def __init__(
        self,
        backend: DMABackend,
        gpu_caches: dict[str, torch.Tensor],
        store_stream: torch.cuda.Stream,
        load_stream: torch.cuda.Stream,
    ) -> None:
        self._backend = backend
        self._gpu_caches = gpu_caches
        self._store_stream = store_stream
        self._load_stream = load_stream

        # Backend may expose a bounce buffer property (XDMABounceBufferBackend
        # does); if absent we assume P2P-style direct transfer.
        self._bounce_buffer: torch.Tensor | None = (
            backend.bounce_buffer
            if hasattr(backend, "bounce_buffer")
            else None
        )

        # Background thread for serializing transfers.
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._shutdown: bool = False
        self._thread = threading.Thread(
            target=self._copy_loop, daemon=True,
        )

        # Event_idx → threading.Event for race-free completion signaling.
        # Set when the CUDA event has been recorded on the transfer stream.
        self._done_signals: dict[int, threading.Event] = {}
        self._done_lock = threading.Lock()

        self._thread.start()

        logger.info(
            "BlockTransferEngine: backend=%s, %d GPU cache tensor(s), "
            "bounce=%s",
            backend.name,
            len(gpu_caches),
            "yes" if self._bounce_buffer is not None else "no",
        )

    # -- Public API (unchanged from XDMACopyBackend) --------------------------

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        wait_event: torch.Event | None = None,
    ) -> None:
        """Enqueue a batch of block copies.

        Args:
            src_blocks: Block IDs on the **source** side.
            dst_blocks: Block IDs on the **destination** side.
            is_store: True → GPU → FPGA (store); False → FPGA → GPU (load).
            event_idx: Monotonic event index for completion tracking.
            events_list: Output list to which ``(event_idx, torch.Event)``
                         is appended after the transfer completes.
            wait_event: If given, the store stream waits for this event
                        before launching (ensures GPU compute has finished).
        """
        # Register a threading.Event so wait_for_copy() can wait without polling.
        with self._done_lock:
            self._done_signals[event_idx] = threading.Event()

        self._queue.put((
            src_blocks, dst_blocks,
            is_store, event_idx, events_list, wait_event,
        ))

    def wait_for_copy(self, event_idx: int, timeout: float | None = None
                      ) -> tuple[int, torch.Event]:
        """Block until the transfer with *event_idx* completes.

        Returns the ``(event_idx, torch.Event)`` tuple.

        Args:
            event_idx: event index passed to ``launch_copy``.
            timeout: seconds to wait before raising ``TimeoutError``.

        Raises:
            TimeoutError: if the transfer does not finish within *timeout*.
            KeyError: if *event_idx* is unknown or was already consumed.
        """
        with self._done_lock:
            signal = self._done_signals.get(event_idx)
        if signal is None:
            raise KeyError(
                f"No pending transfer with event_idx={event_idx}"
            )
        if not signal.wait(timeout=timeout):
            raise TimeoutError(
                f"Transfer {event_idx} did not complete within {timeout}s"
            )

        # The calling thread must have a CUDA context for event.synchronize().
        # PyTorch's event.synchronize() handles this internally.
        with self._done_lock:
            signal, cuda_event = self._done_signals.pop(event_idx)
        cuda_event.synchronize()
        return (event_idx, cuda_event)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._queue is not None:
            self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        # Wake any waiters that will never be signalled.
        with self._done_lock:
            for signal in self._done_signals.values():
                signal.set()
            self._done_signals.clear()
        self._backend.shutdown()

    # -- Background copy loop -------------------------------------------------

    def _copy_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            (src, dst, is_store, eid, events, wait) = item
            try:
                if is_store:
                    self._do_store(src, dst, wait)
                else:
                    self._do_load(src, dst)
            except Exception as e:
                logger.error("BlockTransferEngine transfer failed: %s", e)
                # Signal failure so wait_for_copy doesn't hang forever.
                with self._done_lock:
                    signal = self._done_signals.pop(eid, None)
                if signal:
                    signal.set()
                continue

            stream = self._store_stream if is_store else self._load_stream
            with torch.cuda.stream(stream):
                event = torch.Event()
                event.record(stream)
            events.append((eid, event))

            # Signal that the CUDA event is in the events list.
            # The waiting thread can now call event.synchronize().
            with self._done_lock:
                signal = self._done_signals.pop(eid, None)
            if signal:
                signal.set()

    # -- Store: GPU → FPGA ----------------------------------------------------

    def _do_store(
        self,
        gpu_block_ids: list[int],
        fpga_block_ids: list[int],
        wait_event: torch.Event | None = None,
    ) -> None:
        """Copy blocks from GPU KV cache to FPGA DRAM."""
        if wait_event is not None:
            self._store_stream.wait_event(wait_event)

        has_bounce = self._bounce_buffer is not None

        for gpu_bid, fpga_bid in zip(gpu_block_ids, fpga_block_ids):
            fpga_addr = self._backend.get_block_addr(fpga_bid)

            for _, gpu_tensor in self._gpu_caches.items():
                with torch.cuda.stream(self._store_stream):
                    gpu_src = gpu_tensor[gpu_bid].contiguous()
                    block_bytes = gpu_src.numel() * gpu_src.element_size()

                    if has_bounce:
                        # Step A: GPU → bounce buffer (CUDA stream).
                        bounce = self._bounce_buffer  # type: ignore[union-attr]
                        bounce_view = bounce[:block_bytes].view(
                            dtype=gpu_src.dtype).view(gpu_src.shape)
                        bounce_view.copy_(gpu_src, non_blocking=True)

                        # Step B: ensure bounce is ready, then write.
                        self._store_stream.synchronize()
                        self._backend.write(
                            bounce.data_ptr(), fpga_addr, block_bytes,
                        )
                    else:
                        # P2P path: single GPU DMA hop on the store stream.
                        self._backend.write(
                            gpu_src.data_ptr(), fpga_addr, block_bytes,
                            stream=self._store_stream.cuda_stream,
                        )

        logger.debug(
            "Store: %d GPU blocks → FPGA (backend=%s)",
            len(gpu_block_ids), self._backend.name,
        )

    # -- Load: FPGA → GPU ----------------------------------------------------

    def _do_load(
        self,
        fpga_block_ids: list[int],
        gpu_block_ids: list[int],
    ) -> None:
        """Copy blocks from FPGA DRAM to GPU KV cache."""
        has_bounce = self._bounce_buffer is not None

        for fpga_bid, gpu_bid in zip(fpga_block_ids, gpu_block_ids):
            fpga_addr = self._backend.get_block_addr(fpga_bid)

            for _, gpu_tensor in self._gpu_caches.items():
                block_bytes = (
                    gpu_tensor[gpu_bid].numel()
                    * gpu_tensor[gpu_bid].element_size()
                )

                if has_bounce:
                    # Step A: FPGA → bounce buffer via backend.
                    bounce = self._bounce_buffer  # type: ignore[union-attr]
                    self._backend.read(fpga_addr, bounce.data_ptr(), block_bytes)

                    # Step B: bounce buffer → GPU.
                    with torch.cuda.stream(self._load_stream):
                        bounce_view = bounce[:block_bytes].view(
                            dtype=gpu_tensor[gpu_bid].dtype).view(
                            gpu_tensor[gpu_bid].shape)
                        gpu_tensor[gpu_bid].copy_(bounce_view, non_blocking=True)
                else:
                    # P2P path: single GPU DMA hop on the load stream.
                    self._backend.read(
                        fpga_addr, gpu_tensor[gpu_bid].data_ptr(), block_bytes,
                        stream=self._load_stream.cuda_stream,
                    )

        logger.debug(
            "Load: %d FPGA blocks → GPU (backend=%s)",
            len(fpga_block_ids), self._backend.name,
        )
