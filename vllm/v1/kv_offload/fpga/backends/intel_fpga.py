# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intel FPGA DMA backend (skeleton).

Fill in the driver-specific ``write`` / ``read`` implementation using
your Intel FPGA user-space driver library (e.g. ``libintel_fpga.so``,
``libopae-c.so``, or direct ``/dev/fpga*`` ioctls).

Once the driver calls are wired up, the backend works with
``BlockTransferEngine`` just like any other ``DMABackend``.
"""

from __future__ import annotations

import ctypes
from typing import Optional

from vllm.logger import init_logger
from vllm.v1.kv_offload.fpga.backend_base import DMABackend

logger = init_logger(__name__)


class IntelFPGAP2PBackend(DMABackend):
    """Intel FPGA DMA backend.

    Transfers data between GPU (or host) memory and Intel FPGA DRAM
    via the vendor DMA engine / user-space driver.

    **Implementation required**: fill in ``write()`` and ``read()``
    with the actual driver API calls for your Intel FPGA platform.
    """

    def __init__(
        self,
        fpga_allocator: FPGABlockAllocator,  # noqa: F821
        driver_lib: str = "libintel_fpga.so",
    ) -> None:
        super().__init__(fpga_allocator)

        # Load the Intel FPGA user-space driver library.
        try:
            self._lib = ctypes.CDLL(driver_lib)
        except OSError as e:
            logger.error("Failed to load Intel FPGA driver library %s: %s",
                         driver_lib, e)
            raise

        # TODO: Replace with actual driver initialization.
        #   fpga_handle = self._lib.intel_fpga_open()
        #   self._bar_addr = self._lib.intel_fpga_get_bar(fpga_handle)
        #   self._block_size = ...
        self._handle = None
        self._bar_addr = 0

        logger.info(
            "IntelFPGAP2PBackend: driver_lib=%s (skeleton — "
            "write()/read() not yet implemented)", driver_lib,
        )

    @property
    def name(self) -> str:
        return "INTEL_FPGA_P2P"

    # -- TODO: implement these two methods with your driver API ----------

    def write(self, src_ptr: int, dst_addr: int, size: int,
              stream: int = 0) -> None:
        """Host/GPU memory → Intel FPGA DRAM.
        ``stream`` is unused for Intel FPGA.
        """
        raise NotImplementedError(
            "IntelFPGAP2PBackend.write() — implement me using "
            "your Intel FPGA driver API"
        )

    def read(self, src_addr: int, dst_ptr: int, size: int,
             stream: int = 0) -> None:
        """Intel FPGA DRAM → Host/GPU memory.
        ``stream`` is unused for Intel FPGA.
        """
        raise NotImplementedError(
            "IntelFPGAP2PBackend.read() — implement me using "
            "your Intel FPGA driver API"
        )

    def shutdown(self) -> None:
        super().shutdown()
        if self._handle is not None:
            # self._lib.intel_fpga_close(self._handle)
            self._handle = None
