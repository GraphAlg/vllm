# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU DMA controller direct FPGA BAR write via cuMemHostRegister(IOMEMORY).

Follows the exact same pattern as ``gpu_dma_mix.cu``:

  1. cuCtxCreate → fresh CUDA context
  2. mmap FPGA BAR
  3. cuMemHostRegister(IOMEMORY | DEVICEMAP)
  4. cuMemHostGetDevicePointer → d_bar
  5. cudaMemcpyAsync(DeviceToDevice)  → zero-copy DMA
"""

from __future__ import annotations

import ctypes
import mmap
import os

from vllm.logger import init_logger
from vllm.v1.kv_offload.fpga.allocator import FPGABlockAllocator
from vllm.v1.kv_offload.fpga.backend_base import DMABackend

logger = init_logger(__name__)

# ── Hardware config ─────────────────────────────────────────────────────────
FPGA_PCI_BDF = "0000:35:00.0"
FPGA_BAR_INDEX = 4
FPGA_BAR_SIZE = 1 * 1024**3          # C 代码用 1GB

# ── CUDA Driver API ─────────────────────────────────────────────────────────
CUresult = ctypes.c_int
CUdeviceptr = ctypes.c_uint64

CU_MEMHOSTREGISTER_IOMEMORY = 0x4
CU_MEMHOSTREGISTER_DEVICEMAP = 0x2
CUDA_DEVICE_TO_DEVICE = 0  # cudaMemcpyDeviceToDevice

_cuda = ctypes.CDLL("libcuda.so")


def _check_cu(ret: int) -> None:
    if ret != 0:
        buf = ctypes.create_string_buffer(1024)
        _cuda.cuGetErrorString(ret, buf)
        raise RuntimeError(
            f"CUDA driver error ({ret}): {buf.value.decode('utf-8', errors='replace')}"
        )


# ── Backend ──────────────────────────────────────────────────────────────────


class GPUDirectP2PBackend(DMABackend):
    """GPU DMA · FPGA BAR via cuMemHostRegister(IOMEMORY).

    与 gpu_dma_mix.cu 完全相同的模式:
      - 创建独立 CUDA context
      - cuMemHostRegister 注册 I/O 内存
      - cudaMemcpyAsync(DeviceToDevice)
    """

    def __init__(
        self,
        fpga_allocator: FPGABlockAllocator,
        device_id: int = 0,
        map_size: int = 0,
    ) -> None:
        super().__init__(fpga_allocator)

        if map_size == 0:
            map_size = FPGA_BAR_SIZE

        # ── 0. Create independent CUDA context (matching gpu_dma_mix.cu) ───
        _check_cu(_cuda.cuInit(0))
        dev = ctypes.c_int(device_id)
        _check_cu(_cuda.cuDeviceGet(ctypes.byref(dev), dev))

        ctx = ctypes.c_void_p()
        _check_cu(_cuda.cuCtxCreate(ctypes.byref(ctx), 0, dev))
        self._ctx = ctx
        logger.info("GPUDirectP2PBackend: created CUDA context for device %d", device_id)

        # ── 1. mmap BAR ────────────────────────────────────────────────────
        bar_path = f"/sys/bus/pci/devices/{FPGA_PCI_BDF}/resource{FPGA_BAR_INDEX}"
        fd = os.open(bar_path, os.O_RDWR | os.O_SYNC)
        try:
            self._bar = mmap.mmap(fd, map_size, mmap.MAP_SHARED,
                                  mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)

        bar_base = ctypes.addressof(ctypes.c_char.from_buffer(self._bar))

        # ── 2. cuMemHostRegister(IOMEMORY) ────────────────────────────────
        _check_cu(_cuda.cuMemHostRegister(
            ctypes.c_void_p(bar_base),
            ctypes.c_size_t(map_size),
            ctypes.c_int(CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_DEVICEMAP),
        ))

        # ── 3. cuMemHostGetDevicePointer ──────────────────────────────────
        d_bar = CUdeviceptr(0)
        _check_cu(_cuda.cuMemHostGetDevicePointer(
            ctypes.byref(d_bar), ctypes.c_void_p(bar_base), 0))
        self._d_bar = int(d_bar)
        self._map_size = map_size

        logger.info(
            "GPUDirectP2PBackend: %s  ctx=0x%x  d_bar=0x%x  size=%.1f GB",
            bar_path, ctx.value, self._d_bar, map_size / (1024**3),
        )

    # ── DMABackend ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "GPU_DIRECT_P2P"

    def write(self, src_ptr: int, dst_addr: int, size: int) -> None:
        """cudaMemcpyAsync(DeviceToDevice) → FPGA BAR  (匹配 gpu_dma_mix.cu)."""
        _check_cu(_cuda.cuMemcpyAsync(
            ctypes.c_uint64(self._d_bar + dst_addr),
            ctypes.c_uint64(src_ptr),
            ctypes.c_size_t(size),
            ctypes.c_int(CUDA_DEVICE_TO_DEVICE),
            ctypes.c_void_p(None),  # default stream
        ))

    def read(self, src_addr: int, dst_ptr: int, size: int) -> None:
        """cudaMemcpyAsync(DeviceToDevice) from FPGA BAR."""
        _check_cu(_cuda.cuMemcpyAsync(
            ctypes.c_uint64(dst_ptr),
            ctypes.c_uint64(self._d_bar + src_addr),
            ctypes.c_size_t(size),
            ctypes.c_int(CUDA_DEVICE_TO_DEVICE),
            ctypes.c_void_p(None),
        ))

    @property
    def bar_devptr(self) -> int:
        return self._d_bar

    def shutdown(self) -> None:
        if hasattr(self, "_bar") and self._bar is not None:
            bar_base = ctypes.addressof(ctypes.c_char.from_buffer(self._bar))
            _cuda.cuMemHostUnregister(ctypes.c_void_p(bar_base))
            self._bar.close()
            self._bar = None
        if hasattr(self, "_ctx") and self._ctx is not None:
            try:
                _cuda.cuCtxDestroy(self._ctx)
            except Exception:
                pass
            self._ctx = None
        super().shutdown()
