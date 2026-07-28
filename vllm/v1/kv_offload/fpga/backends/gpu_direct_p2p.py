# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU DMA controller direct FPGA BAR write via cuMemHostRegister(IOMEMORY).

Registered as ``VLLM_FPGA_BACKEND=gpu_direct_p2p``.

Data path (zero CPU bounce):

  Store: GPU HBM ──cudaMemcpyAsync(DeviceToDevice)──→ FPGA BAR ──→ FPGA DRAM
  Load:  GPU HBM ←──cudaMemcpyAsync(DeviceToDevice)── FPGA BAR ←── FPGA DRAM

Prerequisites
-------------
- FPGA PCIe BAR mapped and accessible under ``/sys/bus/pci/devices/*/resourceN``.
- CUDA driver library ``libcuda.so`` (shipped with NVIDIA driver).

Usage
-----
::

    export VLLM_FPGA_BACKEND=gpu_direct_p2p
    export VLLM_FPGA_BDF=0000:88:00.0
    export VLLM_FPGA_BAR_INDEX=2          # default 2
    export VLLM_FPGA_BAR_SIZE=$((4*1024**3))  # default 4 GB
    python ...
"""

from __future__ import annotations

import ctypes
import mmap
import os

import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.fpga.backend_base import DMABackend

logger = init_logger(__name__)

# ── CUDA Driver API constants and helpers ──────────────────────────────────

# ctypes type aliases matching CUDA driver types
CUresult = ctypes.c_int
CUdeviceptr = ctypes.c_uint64

CU_MEMHOSTREGISTER_IOMEMORY = 0x4  # I/O memory (PCIe BAR, etc.)
CU_MEMHOSTREGISTER_DEVICEMAP = 0x2  # also obtain a device-side pointer

_cuda = ctypes.CDLL("libcuda.so")


def _check_cu(ret: int) -> None:
    if ret != 0:
        buf = ctypes.create_string_buffer(256)
        _cuda.cuGetErrorString(ret, buf)
        raise RuntimeError(f"CUDA driver error ({ret}): {buf.value.decode()}")


# ── Backend ────────────────────────────────────────────────────────────────


class GPUDirectP2PBackend(DMABackend):
    """GPU DMA controller direct FPGA BAR write.

    Registers the FPGA BAR with the CUDA driver via
    ``cuMemHostRegister(CU_MEMHOSTREGISTER_IOMEMORY)`` so that the
    GPU DMA engine can directly read/write FPGA DRAM without any
    CPU bounce buffer.
    """

    def __init__(
        self,
        fpga_allocator: FPGABlockAllocator,  # noqa: F821
        bar_path: str | None = None,
        map_size: int = 0,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__(fpga_allocator)

        # ── 1. Resolve BAR path ──────────────────────────────────────────────
        if bar_path is None:
            bdf = os.environ.get("VLLM_FPGA_BDF", "0000:88:00.0")
            bar_idx = os.environ.get("VLLM_FPGA_BAR_INDEX", "2")
            bar_path = f"/sys/bus/pci/devices/{bdf}/resource{bar_idx}"
        self._bar_path = bar_path

        if map_size == 0:
            map_size = int(os.environ.get("VLLM_FPGA_BAR_SIZE", str(4 * 1024**3)))
        self._map_size = map_size

        # ── 2. mmap BAR ──────────────────────────────────────────────────────
        fd = os.open(bar_path, os.O_RDWR | os.O_SYNC)
        try:
            self._bar = mmap.mmap(
                fd, map_size, mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
            )
        finally:
            os.close(fd)

        bar_base = ctypes.addressof(ctypes.c_void_p.from_buffer(self._bar))

        # ── 3. Register with CUDA as I/O memory ──────────────────────────────
        _check_cu(_cuda.cuMemHostRegister(
            ctypes.c_void_p(bar_base),
            ctypes.c_size_t(map_size),
            ctypes.c_int(CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_DEVICEMAP),
        ))

        # ── 4. Get CUDA device pointer ───────────────────────────────────────
        d_bar = CUdeviceptr(0)
        _check_cu(_cuda.cuMemHostGetDevicePointer(
            ctypes.byref(d_bar),
            ctypes.c_void_p(bar_base),
            0,
        ))
        self._d_bar = int(d_bar)

        # ── 5. Stream (shared by all transfers) ──────────────────────────────
        self._stream = stream or torch.cuda.Stream()

        logger.info(
            "GPUDirectP2PBackend: bar=%s d_bar=0x%x map_size=%.1f GB",
            bar_path, self._d_bar, map_size / (1024**3),
        )

    # ── DMABackend interface ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "GPU_DIRECT_P2P"

    def write(self, src_ptr: int, dst_addr: int, size: int) -> None:
        """GPU DMA writes from GPU HBM into FPGA BAR."""
        with torch.cuda.stream(self._stream):
            torch.cuda.memcpy_async(
                dst=self._d_bar + dst_addr,
                src=src_ptr,
                count=size,
                stream=self._stream,
            )

    def read(self, src_addr: int, dst_ptr: int, size: int) -> None:
        """GPU DMA reads from FPGA BAR into GPU HBM."""
        with torch.cuda.stream(self._stream):
            torch.cuda.memcpy_async(
                dst=dst_ptr,
                src=self._d_bar + src_addr,
                count=size,
                stream=self._stream,
            )

    @property
    def bar_devptr(self) -> int:
        """BAR CUDA device pointer.  Can be used directly in cudaMemcpy."""
        return self._d_bar

    def shutdown(self) -> None:
        super().shutdown()
        if hasattr(self, "_bar") and self._bar is not None:
            bar_base = ctypes.addressof(ctypes.c_void_p.from_buffer(self._bar))
            _cuda.cuMemHostUnregister(ctypes.c_void_p(bar_base))
            self._bar.close()
            self._bar = None
