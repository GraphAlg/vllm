# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Xilinx XDMA + host bounce buffer backend.

Store path:   GPU HBM → cudaMemcpyAsync → bounce buffer → XDMA write → FPGA DRAM
Load path:    FPGA DRAM → XDMA read → bounce buffer → cudaMemcpyAsync → GPU HBM

This is the compatibility-fallback backend, requiring only stock XDMA v22
character devices (``/dev/xdma*``) and no GPU peer-registration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.fpga.backend_base import DMABackend

if TYPE_CHECKING:
    from vllm.v1.kv_offload.fpga.xdma_driver import XDMAHandle, MockXDMAHandle

logger = init_logger(__name__)

_DEFAULT_BOUNCE_SIZE = 256 * 1024 * 1024  # 256 MB


class XDMABounceBufferBackend(DMABackend):
    """Xilinx XDMA + host bounce buffer backend.

    XDMA can only transfer between **host memory** and FPGA DRAM, not
    directly between GPU memory and FPGA.  This backend uses a pinned
    bounce buffer on the host as intermediate staging.

    Use ``GPUDirectP2PBackend`` instead if your platform supports direct
    GPU↔FPGA P2P DMA.
    """

    def __init__(
        self,
        fpga: XDMAHandle | MockXDMAHandle,
        fpga_allocator: FPGABlockAllocator,  # noqa: F821
        bounce_buffer_size: int = _DEFAULT_BOUNCE_SIZE,
    ) -> None:
        super().__init__(fpga_allocator)
        self._fpga = fpga

        self._bounce_buf = torch.zeros(
            bounce_buffer_size, dtype=torch.int8, device="cpu",
        )
        self._pin_tensor(self._bounce_buf)

        logger.info(
            "XDMABounceBufferBackend: bounce buffer = %.2f MiB",
            bounce_buffer_size / (1024**2),
        )

    @property
    def name(self) -> str:
        return "XDMA_BOUNCE_BUFFER"

    def write(self, src_ptr: int, dst_addr: int, size: int) -> None:
        """Bounce buffer → FPGA via XDMA write.

        The caller (BlockTransferEngine) has already staged GPU data into
        ``self._bounce_buf`` before calling this method.
        """
        self._fpga.write(self._bounce_buf.data_ptr(), dst_addr, size)

    def read(self, src_addr: int, dst_ptr: int, size: int) -> None:
        """FPGA → bounce buffer via XDMA read.

        The caller (BlockTransferEngine) copies from ``self._bounce_buf``
        to GPU after this method returns.
        """
        self._fpga.read(self._bounce_buf.data_ptr(), src_addr, size)

    @property
    def bounce_buffer(self) -> torch.Tensor:
        return self._bounce_buf

    @staticmethod
    def _pin_tensor(tensor: torch.Tensor) -> None:
        err = torch.cuda.cudart().cudaHostRegister(
            tensor.data_ptr(), tensor.nbytes, 0,
        )
        if err.value != 0:
            raise RuntimeError(f"cudaHostRegister failed: {err}")

    def shutdown(self) -> None:
        super().shutdown()
        self._fpga.close()
        self._bounce_buf = None
