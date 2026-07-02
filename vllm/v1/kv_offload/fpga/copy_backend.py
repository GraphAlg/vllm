# SPDX-License-Identifier: Apache-2.0
"""GPU ↔ FPGA DMA copy backend using XDMA driver.

Replaces ``DmaCopyBackend`` (which uses ``cuMemcpyBatchAsync`` for
GPU↔CPU transfers) with GPU↔FPGA transfers via the XDMA engine.

Transfer strategy
-----------------
XDMA can only transfer between **host memory** and FPGA, not directly
between GPU memory and FPGA.  We use a **pinned host bounce buffer**::

    Store (GPU → FPGA):
        GPU KV block ──cudaMemcpyAsync──→ bounce buffer (host pinned)
        bounce buffer ──────XDMA write───→ FPGA DRAM

    Load (FPGA → GPU):
        FPGA DRAM ──────XDMA read────────→ bounce buffer (host pinned)
        bounce buffer ──cudaMemcpyAsync──→ GPU KV block

GPUDirect RDMA alternative
--------------------------
If your GPU and FPGA support P2P DMA over PCIe, you can skip the
bounce buffer by registering GPU memory with the XDMA driver via
``nvidia-peermem`` or ``nvidia-p2p``.  See the ``GPUDirectBackend``
class at the bottom of this file for the sketch.
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.fpga.allocator import FPGABlockAllocator

if TYPE_CHECKING:
    from vllm.v1.kv_offload.fpga.xdma_driver import XDMAHandle, MockXDMAHandle

logger = init_logger(__name__)

# Maximum size of the bounce buffer (per-rank).  Each GPU KV cache layer's
# block size may differ; we use the largest across all layers.
_DEFAULT_BOUNCE_SIZE = 256 * 1024 * 1024  # 256 MB


class XDMACopyBackend:
    """GPU ↔ FPGA asynchronous copy backend.

    Works like ``DmaCopyBackend`` but replaces ``cuMemcpyBatchAsync``
    with XDMA host↔FPGA transfers.
    """

    def __init__(
        self,
        fpga_handle: XDMAHandle | MockXDMAHandle,
        fpga_allocator: FPGABlockAllocator,
        bounce_buffer_size: int = _DEFAULT_BOUNCE_SIZE,
    ) -> None:
        self._fpga = fpga_handle
        self._fpga_alloc = fpga_allocator

        # GPU KV caches (dict[str, Tensor]) registered by the worker.
        self._gpu_caches: dict[str, torch.Tensor] = {}

        # Pinned host bounce buffer for GPU ↔ FPGA staging.
        # Shape: (bounce_size,) int8
        self._bounce_buf = torch.zeros(
            bounce_buffer_size, dtype=torch.int8, device="cpu"
        )
        self._pin_tensor(self._bounce_buf)

        # CUDA streams for async copies (GPU ↔ bounce).
        low_pri, _ = torch.cuda.Stream.priority_range()
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None

        # Background thread for serializing XDMA operations.
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._shutdown: bool = False

        logger.info(
            "XDMACopyBackend: bounce buffer = %.2f MiB",
            bounce_buffer_size / (1024**2),
        )

    # -- Initialization -----------------------------------------------------

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        fpga_caches: dict[str, torch.Tensor],  # unused; kept for API compat
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        self._gpu_caches = gpu_caches
        self._load_stream = load_stream
        self._store_stream = store_stream

        # Validate bounce buffer is large enough for the largest layer.
        max_bpb = max(
            t.stride(0) * t.element_size() for t in gpu_caches.values()
        )
        if max_bpb > self._bounce_buf.numel():
            logger.warning(
                "Bounce buffer (%d bytes) smaller than block %d bytes; "
                "reallocating.",
                self._bounce_buf.numel(),
                max_bpb,
            )
            self._bounce_buf = torch.zeros(
                max_bpb, dtype=torch.int8, device="cpu"
            )
            self._pin_tensor(self._bounce_buf)

        # Start background thread.
        self._shutdown = False
        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._copy_loop,
            daemon=True,
        )
        self._thread.start()

    # -- Public API (called from FPGAOffloadWorker) -------------------------

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
        self._queue.put(
            (src_blocks, dst_blocks, is_store, event_idx, events_list, wait_event)
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._queue is not None:
            self._queue.put(None)  # signal thread to exit
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # -- Background copy loop -----------------------------------------------

    def _copy_loop(self) -> None:
        """Background thread: drain the queue and perform transfers."""
        while True:
            item = self._queue.get()
            if item is None:
                return
            (
                src_blocks,
                dst_blocks,
                is_store,
                event_idx,
                events_list,
                wait_event,
            ) = item
            try:
                if is_store:
                    self._do_store(src_blocks, dst_blocks, wait_event)
                else:
                    self._do_load(src_blocks, dst_blocks)
            except Exception as e:
                logger.error("XDMACopyBackend transfer failed: %s", e)
                raise

            # Record a CUDA event on the relevant stream so the worker's
            # polling loop can detect completion.
            stream = self._store_stream if is_store else self._load_stream
            assert stream is not None
            with torch.cuda.stream(stream):
                event = torch.Event()
                event.record(stream)
            events_list.append((event_idx, event))

    # -- Store: GPU → FPGA --------------------------------------------------

    def _do_store(
        self,
        gpu_block_ids: list[int],
        fpga_block_ids: list[int],
        wait_event: torch.Event | None = None,
    ) -> None:
        """Copy blocks from GPU KV cache to FPGA DRAM."""
        store_stream = self._store_stream
        assert store_stream is not None

        # Step 1: wait for GPU compute to finish (if needed).
        if wait_event is not None:
            store_stream.wait_event(wait_event)

        for gpu_bid, fpga_bid in zip(gpu_block_ids, fpga_block_ids):
            for layer_name, gpu_tensor in self._gpu_caches.items():
                with torch.cuda.stream(store_stream):
                    # (a) GPU → bounce buffer (on CUDA stream).
                    gpu_src = gpu_tensor[gpu_bid].contiguous()
                    block_bytes = gpu_src.numel() * gpu_src.element_size()
                    bounce_view = self._bounce_buf[:block_bytes].view(
                        gpu_src.shape
                    )
                    bounce_view.copy_(gpu_src, non_blocking=True)

                # (b) Synchronize CUDA stream so bounce buffer is ready.
                store_stream.synchronize()

                # (c) Bounce buffer → FPGA via XDMA (CPU-side).
                fpga_addr = self._fpga_alloc.get_block_offset(fpga_bid)
                self._fpga.write(
                    self._bounce_buf.data_ptr(), fpga_addr, block_bytes
                )

        logger.debug(
            "Store: %d GPU blocks → FPGA (event_idx=%s)",
            len(gpu_block_ids),
        )

    # -- Load: FPGA → GPU --------------------------------------------------

    def _do_load(
        self,
        fpga_block_ids: list[int],
        gpu_block_ids: list[int],
    ) -> None:
        """Copy blocks from FPGA DRAM to GPU KV cache."""
        load_stream = self._load_stream
        assert load_stream is not None

        for fpga_bid, gpu_bid in zip(fpga_block_ids, gpu_block_ids):
            for layer_name, gpu_tensor in self._gpu_caches.items():
                # (a) FPGA → bounce buffer via XDMA (CPU-side).
                block_bytes = (
                    gpu_tensor[gpu_bid].numel() * gpu_tensor[gpu_bid].element_size()
                )
                fpga_addr = self._fpga_alloc.get_block_offset(fpga_bid)
                self._fpga.read(
                    self._bounce_buf.data_ptr(), fpga_addr, block_bytes
                )

                # (b) Bounce buffer → GPU (on CUDA stream).
                with torch.cuda.stream(load_stream):
                    bounce_view = self._bounce_buf[:block_bytes].view(
                        gpu_tensor[gpu_bid].shape
                    )
                    gpu_tensor[gpu_bid].copy_(bounce_view, non_blocking=True)

        logger.debug(
            "Load: %d FPGA blocks → GPU (event_idx=%s)",
            len(fpga_block_ids),
        )

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _pin_tensor(tensor: torch.Tensor) -> None:
        """Pin CPU memory via cudaHostRegister."""
        err = torch.cuda.cudart().cudaHostRegister(
            tensor.data_ptr(), tensor.nbytes, 0
        )
        if err.value != 0:
            raise RuntimeError(f"cudaHostRegister failed: {err}")


# ---------------------------------------------------------------------------
# GPUDirect RDMA backend (sketch – requires nvidia-peermem or similar)
# ---------------------------------------------------------------------------
#
# If your GPU and FPGA support peer-to-peer DMA over PCIe, replace the
# bounce-buffer approach above with direct GPU ↔ FPGA transfers:
#
#   class GPUDirectXDMABackend:
#       """
#       GPU ↔ FPGA direct P2P DMA.
#
#       Prerequisites:
#       - nvidia-peermem kernel module loaded (or custom XDMA with GPUDirect)
#       - GPU and FPGA on the same PCIe root complex
#       - GPU memory registered via nvidia_p2p_get_pages()
#       """
#
#       def _register_gpu_memory(self, gpu_tensor: torch.Tensor):
#           """Register GPU memory physical pages with XDMA."""
#           # 1. ctypes bind to nvidia-p2p library
#           # 2. nvidia_p2p_get_pages(gpu_ptr) → phys_page_array
#           # 3. Pass phys_page_array to XDMA's dma_addr_register ioctl
#           pass
#
#       def launch_copy(self, ...):
#           if is_store:
#               # XDMA reads GPU physical pages directly → FPGA DRAM
#               for gpu_bid, fpga_bid in zip(...):
#                   gpu_pages = self._gpu_page_table[gpu_bid]
#                   self._xdma.sg_write(ch=0, sg_list=gpu_pages,
#                                       fpga_offset=fpga_offset)
#           else:
#               # XDMA writes FPGA DRAM → GPU physical pages directly
#               ...
