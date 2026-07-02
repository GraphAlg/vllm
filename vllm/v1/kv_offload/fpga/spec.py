# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FPGAOffloadingSpec — OffloadingSpec implementation for FPGA DRAM."""

from __future__ import annotations

from typing import Any

from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingGaugeMetadata,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.fpga.common import FPGAOffloadingMetrics
from vllm.v1.kv_offload.fpga.manager import FPGAOffloadingManager
from vllm.v1.kv_offload.fpga.worker import FPGAOffloadingWorker

logger = init_logger(__name__)


class FPGAOffloadingSpec(OffloadingSpec):
    """OffloadingSpec for FPGA DRAM (XDMA-based) KV cache offloading.

    Config via ``kv_connector_extra_config``:

        fpga_bytes_to_use: int   — Total FPGA DRAM per rank (default 16 GiB)
        xdma_prefix: str         — XDMA device prefix (default "/dev/xdma0")
        eviction_policy: str     — "lru" (default) or "arc"
    """

    BLOCK_SIZE_ALIGNMENT = 1

    @classmethod
    @override
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        return {
            FPGAOffloadingMetrics.FPGA_CACHE_USAGE_PERC: (
                OffloadingGaugeMetadata(
                    documentation=(
                        "Fraction of FPGA KV-cache space currently pinned "
                        "by active transfers."
                    ),
                )
            ),
        }

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)
        extra = self.extra_config or {}

        self.xdma_prefix: str = extra.get("xdma_prefix", "/dev/xdma0")
        self.eviction_policy: str = extra.get("eviction_policy", "lru")

        # Compute FPGA block count.
        fpga_bytes_to_use = int(extra.get("fpga_bytes_to_use", 16 * (1024**3)))
        world_size = vllm_config.parallel_config.world_size

        self.num_fpga_blocks = 0
        self.fpga_bytes_per_worker = 0
        self.kv_bytes_per_offloaded_block = 0

        if kv_cache_config is not None and kv_cache_config.num_blocks > 0:
            is_packed = any(
                t.block_stride for t in kv_cache_config.kv_cache_tensors
            )
            total_gpu_kv_bytes = (
                kv_cache_config.kv_cache_tensors[0].size
                if is_packed
                else sum(t.size for t in kv_cache_config.kv_cache_tensors)
            )
            kv_bytes_per_block = (
                total_gpu_kv_bytes // kv_cache_config.num_blocks
            ) * world_size
            kv_bytes_per_offloaded_block = (
                kv_bytes_per_block * self.block_size_factor
            )

            aligned = round_up(
                kv_bytes_per_offloaded_block, self.BLOCK_SIZE_ALIGNMENT
            )
            self.kv_bytes_per_offloaded_block = aligned
            self.fpga_bytes_per_worker = kv_bytes_per_offloaded_block // world_size

            # fpga_bytes_to_use is total across all ranks.
            per_rank = fpga_bytes_to_use // max(world_size, 1)
            self.num_fpga_blocks = max(1, per_rank // aligned)

        self._manager: FPGAOffloadingManager | None = None
        self._worker: FPGAOffloadingWorker | None = None

        logger.info(
            "FPGAOffloadingSpec: %d blocks, %.2f GiB, prefix=%s",
            self.num_fpga_blocks,
            (self.num_fpga_blocks * self.kv_bytes_per_offloaded_block) / (1024**3),
            self.xdma_prefix,
        )

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            store_threshold = int(
                (self.extra_config or {}).get("store_threshold", 0)
            )
            max_tracker = int(
                (self.extra_config or {}).get("max_tracker_size", 64_000)
            )
            self._manager = FPGAOffloadingManager(
                num_blocks=self.num_fpga_blocks,
                cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker,
            )
        return self._manager

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if not self._worker:
            if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
                raise RuntimeError(
                    "FPGA offloading requires CUDA-alike GPU platform"
                )
            self._worker = self.create_worker(kv_caches)
        assert self._worker is not None
        return self._worker

    def create_worker(self, kv_caches: CanonicalKVCaches) -> FPGAOffloadingWorker:
        """Create FPGAOffloadingWorker (public for testing)."""
        return FPGAOffloadingWorker(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            num_fpga_blocks=self.num_fpga_blocks,
            fpga_capacity_bytes=(
                self.num_fpga_blocks * self.kv_bytes_per_offloaded_block
            ),
            xdma_prefix=self.xdma_prefix,
        )
