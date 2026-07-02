# SPDX-License-Identifier: Apache-2.0
"""FPGA DRAM block allocator.

Manages a pool of fixed-size blocks in the FPGA's attached DRAM/HBM.
Block IDs are used by the scheduler to track which KV cache blocks
reside on the FPGA card.
"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.kv_offload.fpga.xdma_driver import XDMAHandle, MockXDMAHandle

logger = init_logger(__name__)

# Sentinel for an unallocated block.
_INVALID_BLOCK = -1


class FPGABlockAllocator:
    """Fixed-size block allocator for FPGA DRAM.

    The FPGA DRAM is partitioned into *num_blocks* equal chunks.
    Each chunk is identified by a block ID (0 .. num_blocks-1).

    Args:
        fpga: XDMA handle used to query/verify FPGA memory.
        dram_size_bytes: Total FPGA DRAM capacity in bytes.
        block_size_bytes: Size of each block in bytes.
    """

    def __init__(
        self,
        fpga: XDMAHandle | MockXDMAHandle,
        dram_size_bytes: int,
        block_size_bytes: int,
    ) -> None:
        assert block_size_bytes > 0
        assert dram_size_bytes >= block_size_bytes

        self._fpga = fpga
        self._block_size = block_size_bytes
        self._num_blocks = dram_size_bytes // block_size_bytes

        # Free-list: sorted list of available block IDs.
        self._free_list: list[int] = list(range(self._num_blocks))

        # Optional: track allocation for debugging.
        self._allocated: set[int] = set()

        logger.info(
            "FPGA DRAM: %s blocks of %s bytes (%s total)",
            self._num_blocks,
            self._block_size,
            self._format_bytes(self.capacity),
        )

    # -- Properties ---------------------------------------------------------

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def capacity(self) -> int:
        return self._num_blocks * self._block_size

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_list)

    @property
    def num_used_blocks(self) -> int:
        return self._num_blocks - len(self._free_list)

    # -- Allocation ---------------------------------------------------------

    def alloc(self) -> int:
        """Allocate one FPGA block and return its block ID.

        Raises RuntimeError if no free blocks remain.
        """
        if not self._free_list:
            raise RuntimeError("FPGA DRAM: out of memory")
        block_id = self._free_list.pop(0)
        self._allocated.add(block_id)
        logger.debug("FPGA alloc: block_id=%d", block_id)
        return block_id

    def alloc_batch(self, count: int) -> list[int]:
        """Allocate multiple blocks at once (more efficient)."""
        if len(self._free_list) < count:
            raise RuntimeError(
                f"FPGA DRAM: need {count} blocks, only {len(self._free_list)} free"
            )
        block_ids = self._free_list[:count]
        self._free_list = self._free_list[count:]
        self._allocated.update(block_ids)
        logger.debug("FPGA alloc batch: block_ids=%s", block_ids)
        return block_ids

    def free(self, block_id: int) -> None:
        """Return a block to the free pool."""
        if block_id not in self._allocated:
            logger.warning("FPGA free: block %d was not allocated", block_id)
            return
        self._allocated.discard(block_id)
        # Insert sorted to maintain free-list order.
        bisect.insort(self._free_list, block_id)

    def free_batch(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            if bid in self._allocated:
                self._allocated.discard(bid)
                bisect.insort(self._free_list, bid)

    def is_allocated(self, block_id: int) -> bool:
        return block_id in self._allocated

    # -- Address translation ------------------------------------------------

    def get_block_offset(self, block_id: int) -> int:
        """Return the byte offset of *block_id* in FPGA address space.

        This offset is used as the ``fpga_offset`` argument to
        ``XDMAHandle.write_channel / read_channel``.
        """
        assert 0 <= block_id < self._num_blocks, f"block_id {block_id} out of range"
        return block_id * self._block_size

    # -- Reset --------------------------------------------------------------

    def reset(self) -> None:
        """Free all blocks."""
        self._free_list = list(range(self._num_blocks))
        self._allocated.clear()
        logger.info("FPGA allocator: reset (all blocks freed)")

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _format_bytes(b: int) -> str:
        for unit in ("B", "KiB", "MiB", "GiB"):
            if b < 1024:
                return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} TiB"
