# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU DMA controller direct FPGA BAR write via cuMemHostRegister(IOMEMORY).

Follows the reference C implementation pattern:
  1. Use the device primary CUDA context (compatible with PyTorch/vLLM)
  2. mmap FPGA BAR
  3. cuMemHostRegister(IOMEMORY | DEVICEMAP)
  4. cuMemHostGetDevicePointer → d_bar
  5. cuMemcpyAsync(DeviceToDevice)  → zero-copy DMA

NOTE: Uses the *primary* CUDA context (cuDevicePrimaryCtxRetain) rather than
creating a new context (cuCtxCreate) because PyTorch/vLLM already owns the
primary context and cuMemHostRegister(IOMEMORY) must be associated with it.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import threading

from vllm.logger import init_logger
from vllm.v1.kv_offload.fpga.allocator import FPGABlockAllocator
from vllm.v1.kv_offload.fpga.backend_base import DMABackend

logger = init_logger(__name__)

# ── Hardware config ─────────────────────────────────────────────────────────
FPGA_PCI_BDF = "0000:88:00.0"
FPGA_BAR_INDEX = 2
FPGA_BAR_SIZE = 1 * 1024**3          # 1 GB

# ── CUDA Driver API types ───────────────────────────────────────────────────
CUresult = ctypes.c_int
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUstream = ctypes.c_void_p
CUdeviceptr = ctypes.c_uint64

CUDA_SUCCESS = 0

CU_MEMHOSTREGISTER_IOMEMORY = 0x4
CU_MEMHOSTREGISTER_DEVICEMAP = 0x2

_cuda = ctypes.CDLL("libcuda.so.1")


def _resolve_symbol(*names: str):
    """Return the first CUDA Driver API symbol exported by the installed driver."""
    for name in names:
        try:
            return getattr(_cuda, name)
        except AttributeError:
            continue
    raise RuntimeError(
        "CUDA driver does not export any of: " + ", ".join(names)
    )


# Resolve versioned symbols (CUDA 12+ exports _v2 variants).
# NOTE: For cuMemHostRegister / cuMemHostUnregister we MUST resolve the
# _v2 variant explicitly — CUDA 13+ has separate function pointers for
# the legacy and v2 symbols, and the legacy variant rejects IOMEMORY.
_cuCtxSetCurrent = _resolve_symbol("cuCtxSetCurrent")
_cuMemHostRegister = _resolve_symbol(
    "cuMemHostRegister_v2",
    "cuMemHostRegister",
)
_cuMemHostUnregister = _resolve_symbol(
    "cuMemHostUnregister_v2",
    "cuMemHostUnregister",
)
_cuMemHostGetDevicePointer = _resolve_symbol(
    "cuMemHostGetDevicePointer_v2",
    "cuMemHostGetDevicePointer",
)
_cuMemcpyAsync = _resolve_symbol("cuMemcpyAsync")
_cuStreamSynchronize = _resolve_symbol("cuStreamSynchronize")


def _configure_cuda_api() -> None:
    """Declare exact argtypes/restype for every CUDA Driver API function."""

    _cuda.cuInit.argtypes = [ctypes.c_uint]
    _cuda.cuInit.restype = CUresult

    _cuda.cuDeviceGet.argtypes = [
        ctypes.POINTER(CUdevice),
        ctypes.c_int,
    ]
    _cuda.cuDeviceGet.restype = CUresult

    _cuda.cuDevicePrimaryCtxRetain.argtypes = [
        ctypes.POINTER(CUcontext),
        CUdevice,
    ]
    _cuda.cuDevicePrimaryCtxRetain.restype = CUresult

    _cuda.cuDevicePrimaryCtxRelease.argtypes = [CUdevice]
    _cuda.cuDevicePrimaryCtxRelease.restype = CUresult

    _cuCtxSetCurrent.argtypes = [CUcontext]
    _cuCtxSetCurrent.restype = CUresult

    _cuMemHostRegister.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    _cuMemHostRegister.restype = CUresult

    _cuMemHostUnregister.argtypes = [ctypes.c_void_p]
    _cuMemHostUnregister.restype = CUresult

    _cuMemHostGetDevicePointer.argtypes = [
        ctypes.POINTER(CUdeviceptr),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    _cuMemHostGetDevicePointer.restype = CUresult

    # cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream)
    # NOTE: The Driver API cuMemcpyAsync does NOT have a cudaMemcpyKind parameter.
    _cuMemcpyAsync.argtypes = [
        CUdeviceptr,
        CUdeviceptr,
        ctypes.c_size_t,
        CUstream,
    ]
    _cuMemcpyAsync.restype = CUresult

    _cuStreamSynchronize.argtypes = [CUstream]
    _cuStreamSynchronize.restype = CUresult

    _cuda.cuGetErrorName.argtypes = [
        CUresult,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    _cuda.cuGetErrorName.restype = CUresult

    _cuda.cuGetErrorString.argtypes = [
        CUresult,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    _cuda.cuGetErrorString.restype = CUresult


_configure_cuda_api()


def _check_cu(ret: int, operation: str = "CUDA operation") -> None:
    """Raise a readable exception for a CUDA Driver API result."""
    code = int(ret)
    if code == CUDA_SUCCESS:
        return

    name_ptr = ctypes.c_char_p()
    text_ptr = ctypes.c_char_p()
    _cuda.cuGetErrorName(CUresult(code), ctypes.byref(name_ptr))
    _cuda.cuGetErrorString(CUresult(code), ctypes.byref(text_ptr))

    name = (
        name_ptr.value.decode("utf-8", errors="replace")
        if name_ptr.value else "CUDA_ERROR_UNKNOWN"
    )
    text = (
        text_ptr.value.decode("utf-8", errors="replace")
        if text_ptr.value else "No description"
    )

    raise RuntimeError(
        f"{operation} failed: CUDA driver error {code} "
        f"({name}): {text}"
    )


# ── Backend ──────────────────────────────────────────────────────────────────


class GPUDirectP2PBackend(DMABackend):
    """GPU DMA · FPGA BAR via cuMemHostRegister(IOMEMORY).

    Mirrors the reference C implementation, but uses the primary CUDA context
    (cuDevicePrimaryCtxRetain) since PyTorch/vLLM already owns that context.
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

        self._lock = threading.RLock()
        self._device_id = int(device_id)
        self._map_size = int(map_size)
        self._bar: mmap.mmap | None = None
        self._bar_base = 0
        self._d_bar = 0
        self._ctx = CUcontext()
        self._device = CUdevice()
        self._primary_ctx_retained = False
        self._registered = False
        self._shutdown = False

        try:
            self._initialize_cuda()
            self._map_and_register_bar()
        except Exception:
            self._cleanup()
            raise

    def _initialize_cuda(self) -> None:
        _check_cu(_cuda.cuInit(0), "cuInit")

        _check_cu(
            _cuda.cuDeviceGet(
                ctypes.byref(self._device),
                ctypes.c_int(self._device_id),
            ),
            "cuDeviceGet",
        )

        # Retain the PRIMARY context (shared with PyTorch/vLLM).
        _check_cu(
            _cuda.cuDevicePrimaryCtxRetain(
                ctypes.byref(self._ctx),
                self._device,
            ),
            "cuDevicePrimaryCtxRetain",
        )
        self._primary_ctx_retained = True
        logger.info(
            "GPUDirectP2PBackend: retained primary CUDA context "
            "device=%d ctx=0x%x",
            self._device_id, self._ctx.value,
        )

    def _map_and_register_bar(self) -> None:
        bar_path = f"/sys/bus/pci/devices/{FPGA_PCI_BDF}/resource{FPGA_BAR_INDEX}"

        # mmap the BAR.
        fd = os.open(bar_path, os.O_RDWR | os.O_SYNC)
        try:
            self._bar = mmap.mmap(fd, self._map_size, mmap.MAP_SHARED,
                                  mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)

        # Get the address of the mmap'd BAR.
        self._bar_byte = ctypes.c_ubyte.from_buffer(self._bar, 0)
        self._bar_base = ctypes.addressof(self._bar_byte)

        # Ensure the context is current and register the BAR.
        _check_cu(
            _cuCtxSetCurrent(self._ctx),
            "cuCtxSetCurrent",
        )

        _check_cu(
            _cuMemHostRegister(
                ctypes.c_void_p(self._bar_base),
                ctypes.c_size_t(self._map_size),
                ctypes.c_uint(
                    CU_MEMHOSTREGISTER_IOMEMORY
                    | CU_MEMHOSTREGISTER_DEVICEMAP,
                ),
            ),
            "cuMemHostRegister",
        )
        self._registered = True

        d_bar = CUdeviceptr()
        _check_cu(
            _cuMemHostGetDevicePointer(
                ctypes.byref(d_bar),
                ctypes.c_void_p(self._bar_base),
                0,
            ),
            "cuMemHostGetDevicePointer",
        )

        self._d_bar = int(d_bar.value)
        logger.info(
            "GPUDirectP2PBackend: %s  ctx=0x%x  bar=0x%x  d_bar=0x%x  size=%.1f GB",
            bar_path, self._ctx.value, self._bar_base,
            self._d_bar, self._map_size / (1024**3),
        )

    # ── DMABackend ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "GPU_DIRECT_P2P"

    def write(self, src_ptr: int, dst_addr: int, size: int) -> None:
        """cuMemcpyAsync(DeviceToDevice) → FPGA BAR."""
        self._copy_and_wait(
            dst=self._d_bar + dst_addr,
            src=src_ptr,
            size=size,
            operation="write",
        )

    def read(self, src_addr: int, dst_ptr: int, size: int) -> None:
        """cuMemcpyAsync(DeviceToDevice) from FPGA BAR."""
        self._copy_and_wait(
            dst=dst_ptr,
            src=self._d_bar + src_addr,
            size=size,
            operation="read",
        )

    def _copy_and_wait(
        self, dst: int, src: int, size: int, operation: str,
    ) -> None:
        if size == 0:
            return

        with self._lock:
            _check_cu(
                _cuCtxSetCurrent(self._ctx),
                f"{operation}: cuCtxSetCurrent",
            )

            _check_cu(
                _cuMemcpyAsync(
                    CUdeviceptr(dst),
                    CUdeviceptr(src),
                    ctypes.c_size_t(size),
                    CUstream(None),
                ),
                f"{operation}: cuMemcpyAsync",
            )

            _check_cu(
                _cuStreamSynchronize(CUstream(None)),
                f"{operation}: cuStreamSynchronize",
            )

    @property
    def bar_devptr(self) -> int:
        return self._d_bar

    @property
    def bar_size(self) -> int:
        return self._map_size

    # ── Cleanup ─────────────────────────────────────────────────────────────

    def _cleanup(self) -> None:
        """Partial cleanup if init fails mid-way."""
        try:
            self._release_resources()
        except Exception:
            logger.exception("GPUDirectP2PBackend: cleanup after init failure")

    def _release_resources(self) -> None:
        unreg_error: BaseException | None = None

        if self._registered and self._bar_base:
            try:
                _check_cu(
                    _cuCtxSetCurrent(self._ctx),
                    "cuCtxSetCurrent (unregister)",
                )
                _cuMemHostUnregister(
                    ctypes.c_void_p(self._bar_base),
                )
            except BaseException as exc:
                unreg_error = exc
            finally:
                self._registered = False

        self._bar_base = 0
        self._d_bar = 0

        # Release the ctypes buffer reference BEFORE closing the mmap.
        self._bar_byte = None

        if self._bar is not None:
            try:
                self._bar.close()
            finally:
                self._bar = None

        if self._primary_ctx_retained:
            try:
                _cuda.cuDevicePrimaryCtxRelease(self._device)
            except Exception:
                pass
            finally:
                self._primary_ctx_retained = False
                self._ctx = CUcontext()

        if unreg_error is not None:
            raise unreg_error

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            try:
                self._release_resources()
            finally:
                super().shutdown()

        logger.info("GPUDirectP2PBackend: shutdown complete")
