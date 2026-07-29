# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for FPGA KV cache offloading.

Supports pluggable DMA backends -- use ``VLLM_FPGA_BACKEND`` to select:

  ``xdma_bounce`` (default)
      Xilinx XDMA + host bounce buffer.  Requires ``/dev/xdma*`` devices.
  ``gpu_direct_p2p``
      GPU DMA controller writes FPGA BAR directly.  Zero bounce buffer.
      Requires driver to map FPGA BAR into CUDA address space.
  ``intel_fpga``
      Intel FPGA DMA (skeleton -- fill in vendor driver calls).

Examples::

    # XDMA (requires real or mock devices):
    VLLM_FPGA_MOCK=1 python start_with_fpga.py

    # P2P: no XDMA devices needed at all:
    export VLLM_FPGA_BACKEND=gpu_direct_p2p
    export VLLM_FPGA_BAR_ADDR=0x...
    python start_with_fpga.py
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.fpga.allocator import FPGABlockAllocator
from vllm.v1.kv_offload.fpga.copy_backend import BlockTransferEngine

if TYPE_CHECKING:
    from vllm.v1.kv_offload.fpga.backends.xdma_bounce_buffer import (
        XDMABounceBufferBackend,
    )

logger = init_logger(__name__)

# Recognised backend types -- extend when adding new backends.
_BACKEND_XDMA_BOUNCE = "xdma_bounce"
_BACKEND_GPU_P2P = "gpu_direct_p2p"
_BACKEND_INTEL = "intel_fpga"


def _create_backend_and_allocator(
    backend_name: str,
    xdma_prefix: str,
    fpga_capacity_bytes: int,
    total_bytes_per_block: int,
    store_stream: torch.cuda.Stream,
    load_stream: torch.cuda.Stream,
) -> tuple:
    """Factory: create (backend, fpga_allocator) from backend name.

    Only the XDMA path requires ``/dev/xdma*`` devices (or
    ``VLLM_FPGA_MOCK=1`` for mock).  P2P and Intel backends need
    no XDMA driver at all.
    """
    if backend_name == _BACKEND_XDMA_BOUNCE:
        # Delayed imports: only pull in XDMA when actually using it.
        from vllm.v1.kv_offload.fpga.xdma_driver import (
            MockXDMAHandle,
            XDMAHandle,
        )
        from vllm.v1.kv_offload.fpga.backends.xdma_bounce_buffer import (
            XDMABounceBufferBackend,
        )

        use_mock = os.environ.get("VLLM_FPGA_MOCK", "0") == "1"
        if use_mock:
            xdma = MockXDMAHandle(ddr_size=fpga_capacity_bytes)
            logger.info("FPGAWorker: using MockXDMAHandle")
        else:
            xdma = XDMAHandle()
            xdma.open(
                h2c_device=f"{xdma_prefix}_h2c_0",
                c2h_device=f"{xdma_prefix}_c2h_0",
            )

        allocator = FPGABlockAllocator(
            fpga=xdma,
            dram_size_bytes=fpga_capacity_bytes,
            block_size_bytes=total_bytes_per_block,
        )
        backend = XDMABounceBufferBackend(
            fpga=xdma,
            fpga_allocator=allocator,
        )
        return backend, allocator

    if backend_name == _BACKEND_GPU_P2P:
        from vllm.v1.kv_offload.fpga.backends.gpu_direct_p2p import (
            GPUDirectP2PBackend,
        )

        # BAR 路径和大小从环境变量读取，不需要手动传地址
        from vllm.v1.kv_offload.fpga.xdma_driver import MockXDMAHandle
        allocator = FPGABlockAllocator(
            fpga=MockXDMAHandle(ddr_size=fpga_capacity_bytes),
            dram_size_bytes=fpga_capacity_bytes,
            block_size_bytes=total_bytes_per_block,
        )

        device_id = int(os.environ.get("VLLM_FPGA_DEVICE_ID", "0"))
        backend = GPUDirectP2PBackend(
            fpga_allocator=allocator,
            device_id=device_id,
        )
        return backend, allocator

    if backend_name == _BACKEND_INTEL:
        from vllm.v1.kv_offload.fpga.backends.intel_fpga import (
            IntelFPGAP2PBackend,
        )

        from vllm.v1.kv_offload.fpga.xdma_driver import MockXDMAHandle
        allocator = FPGABlockAllocator(
            fpga=MockXDMAHandle(ddr_size=fpga_capacity_bytes),
            dram_size_bytes=fpga_capacity_bytes,
            block_size_bytes=total_bytes_per_block,
        )
        backend = IntelFPGAP2PBackend(fpga_allocator=allocator)
        return backend, allocator

    raise ValueError(
        f"Unknown FPGA backend '{backend_name}'. "
        f"Supported: {_BACKEND_XDMA_BOUNCE}, {_BACKEND_GPU_P2P}, "
        f"{_BACKEND_INTEL}"
    )


class FPGAOffloadingWorker(OffloadingWorker):
    """Worker-side FPGA offload handler.

    Manages GPU→FPGA and FPGA→GPU transfers via a pluggable
    ``DMABackend``.  Set ``VLLM_FPGA_BACKEND`` to select the transport
    (default: ``xdma_bounce``).
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        num_fpga_blocks: int,
        fpga_capacity_bytes: int,
        xdma_prefix: str = "/dev/xdma0",
    ) -> None:
        self._kv_caches = kv_caches
        self._block_size_factor = block_size_factor
        self._num_fpga_blocks = num_fpga_blocks
        self._fpga_capacity_bytes = fpga_capacity_bytes
        self._xdma_prefix = xdma_prefix

        # Resolved canonical tensors (int8, (num_blocks, page_size_bytes)).
        self._gpu_tensors: list[torch.Tensor] = []

        # FPGA block allocator + transfer engine (vendor agnostic).
        self._fpga_alloc: FPGABlockAllocator | None = None
        self._engine: BlockTransferEngine | None = None

        # Event tracking (mirrors SimpleCPUOffloadWorker pattern).
        self._load_events: list[tuple[int, torch.Event]] = []
        self._store_events: list[tuple[int, torch.Event]] = []
        self._load_hwm: int = -1
        self._store_hwm: int = -1

        # Pending vs completed tracking.
        self._pending_load_job_ids: set[int] = set()
        self._pending_store_job_ids: set[int] = set()
        self._completed_store_job_ids: dict[int, int] = {}

        # CUDA streams for async copies.
        low_pri, _ = torch.cuda.Stream.priority_range()
        self._store_stream = torch.cuda.Stream(priority=low_pri)
        self._load_stream = torch.cuda.Stream(priority=low_pri)

        # Store-compute sync event.
        self._store_compute_done = torch.Event()

        self._setup()

    def _setup(self) -> None:
        """Initialise FPGA hardware, allocator, and pluggable backend."""
        # Canonical tensors: each has shape (num_blocks, page_size_bytes), int8.
        gpu_cache_dict: dict[str, torch.Tensor] = {}
        for i, ct in enumerate(self._kv_caches.tensors):
            name = f"tensor_{i}"
            gpu_cache_dict[name] = ct.tensor

        # Compute block size.
        total_bytes_per_block = sum(
            ct.page_size_bytes for ct in self._kv_caches.tensors
        )

        # Backend selection via env var (default: xdma_bounce).
        backend_name = os.environ.get(
            "VLLM_FPGA_BACKEND", _BACKEND_XDMA_BOUNCE,
        )

        # Create backend + allocator (no XDMA unless backend needs it).
        backend, self._fpga_alloc = _create_backend_and_allocator(
            backend_name=backend_name,
            xdma_prefix=self._xdma_prefix,
            fpga_capacity_bytes=self._num_fpga_blocks * total_bytes_per_block,
            total_bytes_per_block=total_bytes_per_block,
            store_stream=self._store_stream,
            load_stream=self._load_stream,
        )

        # Pre-allocate FPGA blocks.
        self._fpga_alloc.alloc_batch(self._num_fpga_blocks)
        logger.info(
            "FPGAWorker: %d FPGA blocks reserved (%.2f MiB each), backend=%s",
            self._num_fpga_blocks,
            total_bytes_per_block / (1024**2),
            backend.name,
        )

        # Transfer engine.
        self._engine = BlockTransferEngine(
            backend=backend,
            gpu_caches=gpu_cache_dict,
            store_stream=self._store_stream,
            load_stream=self._load_stream,
        )
        self._gpu_tensors = [ct.tensor for ct in self._kv_caches.tensors]

    # -- OffloadingWorker interface ---------------------------------------

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        """Async GPU → FPGA."""
        gpu_block_ids: list[int] = src_spec.block_ids.tolist()
        fpga_block_ids: list[int] = dst_spec.block_ids.tolist()

        if len(gpu_block_ids) != len(fpga_block_ids):
            logger.error(
                "GPU block count (%d) != FPGA block count (%d)",
                len(gpu_block_ids), len(fpga_block_ids),
            )
            return False

        # Record that compute must finish before store.
        self._store_compute_done.record(torch.cuda.current_stream())

        assert self._engine is not None
        self._engine.launch_copy(
            src_blocks=gpu_block_ids,
            dst_blocks=fpga_block_ids,
            is_store=True,
            event_idx=job_id,
            events_list=self._store_events,
            wait_event=self._store_compute_done,
        )
        self._pending_store_job_ids.add(job_id)
        return True

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        """Async FPGA → GPU."""
        fpga_block_ids: list[int] = src_spec.block_ids.tolist()
        gpu_block_ids: list[int] = dst_spec.block_ids.tolist()

        if len(fpga_block_ids) != len(gpu_block_ids):
            logger.error(
                "FPGA block count (%d) != GPU block count (%d)",
                len(fpga_block_ids), len(gpu_block_ids),
            )
            return False

        assert self._engine is not None
        self._engine.launch_copy(
            src_blocks=fpga_block_ids,
            dst_blocks=gpu_block_ids,
            is_store=False,
            event_idx=job_id,
            events_list=self._load_events,
        )
        self._pending_load_job_ids.add(job_id)
        return True

    def get_finished(self) -> list[TransferResult]:
        """Return completed transfers since the last call."""
        results: list[TransferResult] = []

        if self._pending_load_job_ids:
            load_wm = self._poll_events(is_store=False)
            for jid in list(self._pending_load_job_ids):
                if jid <= load_wm:
                    self._pending_load_job_ids.discard(jid)
                    results.append(TransferResult(
                        job_id=jid, success=True, transfer_size=0,
                    ))

        if self._pending_store_job_ids:
            store_wm = self._poll_events(is_store=True)
            for jid in list(self._pending_store_job_ids):
                if jid <= store_wm:
                    self._pending_store_job_ids.discard(jid)
                    results.append(TransferResult(
                        job_id=jid, success=True, transfer_size=0,
                    ))

        return results

    def wait(self, job_ids: set[int]) -> None:
        """Block until the specified jobs complete."""
        for jid in job_ids:
            if jid in self._pending_load_job_ids:
                self._load_stream.synchronize()
            if jid in self._pending_store_job_ids:
                self._store_stream.synchronize()

    # -- Event polling ---------------------------------------------------

    def _poll_events(self, is_store: bool) -> int:
        """Non-blocking poll; return high-water mark of completed events."""
        events = self._store_events if is_store else self._load_events
        hwm = self._store_hwm if is_store else self._load_hwm
        while events:
            event_idx, event = events[0]
            if not event.query():
                break
            hwm = event_idx
            events.pop(0)
        if is_store:
            self._store_hwm = hwm
        else:
            self._load_hwm = hwm
        return hwm

    def shutdown(self) -> None:
        if self._engine is not None:
            self._engine.shutdown()
