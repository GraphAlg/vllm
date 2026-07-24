# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU DMA controller direct FPGA BAR write (P2P, zero bounce buffer).

Store path:
  GPU HBM ──GPU DMA controller──→ FPGA BAR ──→ FPGA DRAM
  Zero CPU involvement, zero bounce buffer.

Load path:
  GPU HBM ←──GPU DMA controller── FPGA BAR ←── FPGA DRAM

Prerequisites:
  - GPU and FPGA are under the same PCIe root complex.
  - The driver module maps the FPGA BAR into a CUDA-accessible virtual address
    (e.g. via ``cuMemAddressReserve + cuMemMap``, or ``cudaHostRegister``
    of an mmaped BAR region).
  - ``bar_cuda_ptr`` is that mapped CUDA virtual address.

Usage:
  backend = GPUDirectP2PBackend(
      fpga_allocator=allocator,
      bar_cuda_ptr=driver.get_bar_cuda_addr(),
      block_size=total_bytes_per_block,
      store_stream=store_stream,
      load_stream=load_stream,
  )
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.fpga.backend_base import DMABackend

logger = init_logger(__name__)


class GPUDirectP2PBackend(DMABackend):
    """GPU DMA controller direct FPGA BAR write.

    The GPU DMA engine reads/writes the FPGA BAR as if it were regular
    GPU memory.  This eliminates the host bounce buffer entirely.
    """

    def __init__(
        self,
        fpga_allocator: FPGABlockAllocator,  # noqa: F821  # FPGA块分配器
        bar_cuda_ptr: int,  # BAR的CUDA指针地址
        block_size: int,  # 数据块大小
        store_stream: torch.cuda.Stream,  # 存储数据流
        load_stream: torch.cuda.Stream,  # 加载数据流
    ) -> None:
        super().__init__(fpga_allocator)
        self._bar_base = bar_cuda_ptr
        self._block_size = block_size
        self._store_stream = store_stream
        self._load_stream = load_stream

        logger.info(
            "GPUDirectP2PBackend: BAR base=0x%x, block_size=%d, "
            "store_stream=%s, load_stream=%s",
            bar_cuda_ptr, block_size,
            store_stream, load_stream,
        )

    @property
    def name(self) -> str:
        return "GPU_DIRECT_P2P"

    def write(self, src_ptr: int, dst_addr: int, size: int) -> None:
        """GPU DMA controller writes GPU data directly to FPGA BAR.

        ``torch.cuda.memcpy_async`` on the store stream issues a PCIe
        write from GPU memory to the BAR address.  No CPU bounce.
        """
        dst = self._bar_base + dst_addr
        with torch.cuda.stream(self._store_stream):
            torch.cuda.memcpy_async(
                dst=dst,
                src=src_ptr,
                count=size,
                stream=self._store_stream,
            )

    def read(self, src_addr: int, dst_ptr: int, size: int) -> None:
        """GPU DMA controller reads FPGA BAR data into GPU memory."""
        src = self._bar_base + src_addr
        with torch.cuda.stream(self._load_stream):
            torch.cuda.memcpy_async(
                dst=dst_ptr,
                src=src,
                count=size,
                stream=self._load_stream,
            )

    def store_stream(self) -> torch.cuda.Stream:
        return self._store_stream

    def load_stream(self) -> torch.cuda.Stream:
        return self._load_stream
