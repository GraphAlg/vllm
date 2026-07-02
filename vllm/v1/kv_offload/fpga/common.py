# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FPGA offloading shared types: LoadStoreSpec and metrics keys."""

from typing_extensions import override

from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec


class FPGAOffloadingMetrics:
    STORES_SKIPPED = "vllm:kv_offload_fpga_stores_skipped"
    FPGA_CACHE_USAGE_PERC = "vllm:kv_offload_fpga_cache_usage_perc"


class FPGALoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing KV blocks to/from FPGA DRAM.

    The *block_ids* are logical block indices in FPGA DRAM space,
    translating to FPGA AXI addresses via ``FPGABlockAllocator``.
    """

    @staticmethod
    @override
    def medium() -> str:
        return "FPGA"
