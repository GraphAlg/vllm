# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FPGA offloading block manager.

Manages the block allocation / eviction for FPGA DRAM.
The logic is **medium-agnostic** (same as CPUOffloadingManager) —
only the medium name and metric names differ.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_offload.base import OffloadingEvent, OffloadingManager
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.fpga.common import (
    FPGALoadStoreSpec,
    FPGAOffloadingMetrics,
)


class FPGAOffloadingManager(OffloadingManager):
    """
    FPGA DRAM block manager for KV cache offloading.

    This is a medium-agnostic block tracker.  It reuses the same
    pluggable CachePolicy (LRU / ARC) as ``CPUOffloadingManager``.

    The FPGA-specific concerns (XDMA transfers) live in
    ``FPGAOffloadingWorker``.
    """

    def __init__(
        self,
        num_blocks: int,
        cache_policy: Literal["lru", "arc"] = "lru",
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
    ) -> None:
        # Delegate to CPUOffloadingManager — all block management logic
        # is identical; only the medium name and metrics differ.
        self._cpu_mgr = CPUOffloadingManager(
            num_blocks=num_blocks,
            cache_policy=cache_policy,
            enable_events=enable_events,
            store_threshold=store_threshold,
            max_tracker_size=max_tracker_size,
        )
        self.medium: str = FPGALoadStoreSpec.medium()

    # -- Delegation to CPUOffloadingManager ---------------------------------

    def lookup(self, key, req_context):
        return self._cpu_mgr.lookup(key, req_context)

    def prepare_load(self, keys, req_context):
        return self._cpu_mgr.prepare_load(keys, req_context)

    def touch(self, keys, req_context):
        self._cpu_mgr.touch(keys, req_context)

    def complete_load(self, keys, req_context):
        self._cpu_mgr.complete_load(keys, req_context)

    def prepare_store(self, keys, req_context):
        return self._cpu_mgr.prepare_store(keys, req_context)

    def complete_store(self, keys, req_context, success=True):
        self._cpu_mgr.complete_store(keys, req_context, success)

    def on_new_request(self, req_context):
        return self._cpu_mgr.on_new_request(req_context)

    def on_request_finished(self, req_context):
        self._cpu_mgr.on_request_finished(req_context)

    def take_events(self) -> Iterable[OffloadingEvent]:
        return self._cpu_mgr.take_events()

    def on_schedule_end(self, context):
        self._cpu_mgr.on_schedule_end(context)

    def has_pending_work(self) -> bool:
        return self._cpu_mgr.has_pending_work()

    def reset_cache(self) -> None:
        self._cpu_mgr.reset_cache()

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = self._cpu_mgr.get_stats()
        if stats is not None:
            # Relabel medium-specific metrics.
            stats.medium = self.medium
        return stats

    def shutdown(self) -> None:
        self._cpu_mgr.shutdown()
