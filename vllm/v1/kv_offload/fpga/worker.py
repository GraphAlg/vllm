# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for FPGA KV cache offloading."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.fpga.allocator import FPGABlockAllocator
from vllm.v1.kv_offload.fpga.xdma_driver import (
    MockXDMAHandle,
    XDMAHandle,
)
from vllm.v1.kv_offload.fpga.copy_backend import XDMACopyBackend
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)

# Default XDMA device prefix (h2c_0 / c2h_0 appended automatically).
_DEFAULT_XDMA_PREFIX = "/dev/xdma0"


class FPGAOffloadingWorker(OffloadingWorker):
    """Worker-side FPGA offload handler.

    Manages GPU→FPGA and FPGA→GPU transfers via XDMA, using a bounce
    buffer approach (GPU → pinned host → XDMA → FPGA DRAM, and vice
    versa).
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        num_fpga_blocks: int,
        fpga_capacity_bytes: int,
        xdma_prefix: str = _DEFAULT_XDMA_PREFIX,
    ) -> None:
        self._kv_caches = kv_caches
        self._block_size_factor = block_size_factor
        self._num_fpga_blocks = num_fpga_blocks
        self._fpga_capacity_bytes = fpga_capacity_bytes
        self._xdma_prefix = xdma_prefix

        # Resolved canonical tensors (int8, (num_blocks, page_size_bytes)).
        self._gpu_tensors: list[torch.Tensor] = []

        # FPGA block allocator + XDMA backend.
        self._fpga_alloc: FPGABlockAllocator | None = None
        self._backend: XDMACopyBackend | None = None

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
        """Initialise FPGA hardware, allocator, and copy backend."""
        # Canonical tensors: each has shape (num_blocks, page_size_bytes), int8.
        # The OffloadingConnectorWorker registers these as the "GPU side".
        gpu_cache_dict: dict[str, torch.Tensor] = {}
        for i, ct in enumerate(self._kv_caches.tensors):
            name = f"tensor_{i}"
            gpu_cache_dict[name] = ct.tensor

        # Compute block size.
        total_bytes_per_block = sum(
            ct.page_size_bytes for ct in self._kv_caches.tensors
        )

        # FPGA allocator.
        use_mock = os.environ.get("VLLM_FPGA_MOCK", "0") == "1"
        if use_mock:
            xdma = MockXDMAHandle(ddr_size=self._fpga_capacity_bytes)
            logger.info("FPGAWorker: using MockXDMAHandle")
        else:
            xdma = XDMAHandle()
            xdma.open(
                h2c_device=f"{self._xdma_prefix}_h2c_0",
                c2h_device=f"{self._xdma_prefix}_c2h_0",
            )

        self._fpga_alloc = FPGABlockAllocator(
            fpga=xdma,
            dram_size_bytes=self._num_fpga_blocks * total_bytes_per_block,
            block_size_bytes=total_bytes_per_block,
        )

        # Pre-allocate FPGA blocks.
        self._fpga_alloc.alloc_batch(self._num_fpga_blocks)
        logger.info(
            "FPGAWorker: %d FPGA blocks reserved (%.2f MiB each)",
            self._num_fpga_blocks,
            total_bytes_per_block / (1024**2),
        )

        # Copy backend.
        self._backend = XDMACopyBackend(
            fpga_handle=xdma,
            fpga_allocator=self._fpga_alloc,
        )
        self._backend.init(
            gpu_caches=gpu_cache_dict,
            fpga_caches={},
            device=gpu_cache_dict["tensor_0"].device,
            load_stream=self._load_stream,
            store_stream=self._store_stream,
        )

        self._gpu_tensors = [ct.tensor for ct in self._kv_caches.tensors]

    # -- OffloadingWorker interface ---------------------------------------

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        """Async GPU → FPGA.

        Args:
            job_id: Unique transfer identifier.
            src_spec: GPU block IDs (GPULoadStoreSpec).
            dst_spec: FPGA block IDs (FPGALoadStoreSpec).

        Returns:
            True if submitted, False on error.
        """
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

        # Enqueue via copy backend — backend's background thread records
        # a CUDA event in *events_list* when the copy on the store stream
        # completes.
        assert self._backend is not None
        self._backend.launch_copy(
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
        """Async FPGA → GPU.

        Args:
            job_id: Unique transfer identifier.
            src_spec: FPGA block IDs (FPGALoadStoreSpec).
            dst_spec: GPU block IDs (GPULoadStoreSpec).

        Returns:
            True if submitted, False on error.
        """
        fpga_block_ids: list[int] = src_spec.block_ids.tolist()
        gpu_block_ids: list[int] = dst_spec.block_ids.tolist()

        if len(fpga_block_ids) != len(gpu_block_ids):
            logger.error(
                "FPGA block count (%d) != GPU block count (%d)",
                len(fpga_block_ids), len(gpu_block_ids),
            )
            return False

        assert self._backend is not None
        self._backend.launch_copy(
            src_blocks=fpga_block_ids,
            dst_blocks=gpu_block_ids,
            is_store=False,
            event_idx=job_id,
            events_list=self._load_events,
        )

        self._pending_load_job_ids.add(job_id)
        return True

    def get_finished(self) -> list[TransferResult]:
        """Return completed transfers since the last call.

        Polls CUDA events recorded by the XDMACopyBackend background
        thread on each completed block copy.
        """
        results: list[TransferResult] = []

        # Check completed loads.
        if self._pending_load_job_ids:
            load_wm = self._poll_events(is_store=False)
            for jid in list(self._pending_load_job_ids):
                if jid <= load_wm:
                    self._pending_load_job_ids.discard(jid)
                    results.append(TransferResult(
                        job_id=jid, success=True, transfer_size=0,
                    ))

        # Check completed stores.
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
        if self._backend is not None:
            self._backend.shutdown()
