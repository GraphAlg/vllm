# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU DMA controller direct FPGA BAR write via cuMemHostRegister(IOMEMORY).

Follows the reference C implementation pattern:
  1. cuCtxCreate → independent CUDA context
  2. mmap FPGA BAR  (via libc mmap, same as C code)
  3. cuMemHostRegister(IOMEMORY | DEVICEMAP)
  4. cuMemHostGetDevicePointer → d_bar
  5. cuMemcpyAsync(DeviceToDevice)  → zero-copy DMA
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

# libc for raw mmap (matching C code exactly — get a bare void* pointer).
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.mmap.argtypes = [
    ctypes.c_void_p,   # addr
    ctypes.c_size_t,   # length
    ctypes.c_int,      # prot
    ctypes.c_int,      # flags
    ctypes.c_int,      # fd
    ctypes.c_int64,    # offset (off_t on x86-64 Linux)
]
_libc.mmap.restype = ctypes.c_void_p

_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.munmap.restype = ctypes.c_int

libc_MAP_FAILED = ctypes.c_void_p(-1).value


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
_cuCtxCreate = _resolve_symbol("cuCtxCreate_v2", "cuCtxCreate")
_cuCtxDestroy = _resolve_symbol("cuCtxDestroy_v2", "cuCtxDestroy")
_cuCtxSetCurrent = _resolve_symbol("cuCtxSetCurrent")
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

    _cuCtxCreate.argtypes = [
        ctypes.POINTER(CUcontext),
        ctypes.c_uint,
        CUdevice,
    ]
    _cuCtxCreate.restype = CUresult

    _cuCtxDestroy.argtypes = [CUcontext]
    _cuCtxDestroy.restype = CUresult

    _cuCtxSetCurrent.argtypes = [CUcontext]
    _cuCtxSetCurrent.restype = CUresult

    _cuda.cuMemHostRegister.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    _cuda.cuMemHostRegister.restype = CUresult

    _cuda.cuMemHostUnregister.argtypes = [ctypes.c_void_p]
    _cuda.cuMemHostUnregister.restype = CUresult

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


def _resolve_map_size(
    bar_path: str,
    requested_map_size: int,
    default_size: int,
) -> int:
    """Resolve a BAR mapping size that is compatible with CUDA registration."""
    if requested_map_size == 0:
        requested_map_size = default_size

    if requested_map_size <= 0:
        raise ValueError(
            f"map_size must be positive, got {requested_map_size}"
        )

    if not os.path.exists(bar_path):
        raise FileNotFoundError(f"FPGA BAR resource does not exist: {bar_path}")

    resource_size = os.path.getsize(bar_path)
    if resource_size > 0:
        requested_map_size = min(requested_map_size, resource_size)

    page_size = mmap.PAGESIZE
    if requested_map_size % page_size != 0:
        requested_map_size -= requested_map_size % page_size

    if requested_map_size <= 0:
        raise ValueError(
            "Resolved BAR mapping size is too small for CUDA registration: "
            f"requested={requested_map_size}, page_size={page_size}"
        )

    return requested_map_size


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

    Mirrors the reference C implementation:
      - cuCtxCreate → independent CUDA context
      - libc mmap → raw void* pointer
      - cuMemHostRegister(IOMEMORY | DEVICEMAP)
      - cuMemcpyAsync(DeviceToDevice)  (Driver API, no kind parameter)
    """

    def __init__(
        self,
        fpga_allocator: FPGABlockAllocator,
        device_id: int = 0,
        map_size: int = 0,
    ) -> None:
        super().__init__(fpga_allocator)

        self._lock = threading.RLock()
        self._device_id = int(device_id)
        self._bar_path = (
            f"/sys/bus/pci/devices/{FPGA_PCI_BDF}/resource{FPGA_BAR_INDEX}"
        )
        self._map_size = int(
            _resolve_map_size(
                self._bar_path,
                int(map_size),
                FPGA_BAR_SIZE,
            )
        )
        self._bar_ptr = ctypes.c_void_p()   # raw void* from mmap (matches C code)
        self._d_bar = 0
        self._ctx = CUcontext()
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

        dev = CUdevice()
        _check_cu(
            _cuda.cuDeviceGet(
                ctypes.byref(dev),
                ctypes.c_int(self._device_id),
            ),
            "cuDeviceGet",
        )

        _check_cu(
            _cuCtxCreate(ctypes.byref(self._ctx), 0, dev),
            "cuCtxCreate",
        )
        logger.info(
            "GPUDirectP2PBackend: created CUDA context device=%d ctx=0x%x",
            self._device_id, self._ctx.value,
        )

    def _map_and_register_bar(self) -> None:
        bar_path = f"/sys/bus/pci/devices/{FPGA_PCI_BDF}/resource{FPGA_BAR_INDEX}"

        # Keep fd open during cuMemHostRegister (matches C code).
        fd = os.open(bar_path, os.O_RDWR | os.O_SYNC)
        try:
            # Use libc mmap directly to get a raw void*, matching C code exactly.
            result = _libc.mmap(
                None,
                self._map_size,
                mmap.PROT_READ | mmap.PROT_WRITE,
                mmap.MAP_SHARED,
                fd,
                0,  # offset
            )
            if result.value == libc_MAP_FAILED:
                err = ctypes.get_errno()
                raise OSError(err, f"mmap of {bar_path} failed: "
                                   f"{os.strerror(err)}")
            self._bar_ptr = result

            # cuCtxCreate already made the context current on this thread.
            # Use cuCtxSetCurrent (not PushCurrent) to mirror the C code.
            _check_cu(
                _cuCtxSetCurrent(self._ctx),
                "cuCtxSetCurrent",
            )

            _check_cu(
                _cuda.cuMemHostRegister(
                    self._bar_ptr,
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
                    self._bar_ptr,
                    0,
                ),
                "cuMemHostGetDevicePointer",
            )

        finally:
            os.close(fd)

        self._d_bar = int(d_bar.value)
        logger.info(
            "GPUDirectP2PBackend: %s  ctx=0x%x  bar_ptr=0x%x  d_bar=0x%x  size=%.1f GB",
            bar_path, self._ctx.value, self._bar_ptr.value,
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

        if self._registered and self._bar_ptr and self._bar_ptr.value:
            try:
                _check_cu(
                    _cuCtxSetCurrent(self._ctx),
                    "cuCtxSetCurrent (unregister)",
                )
                _cuda.cuMemHostUnregister(self._bar_ptr)
            except BaseException as exc:
                unreg_error = exc
            finally:
                self._registered = False

        self._d_bar = 0

        # munmap using the raw pointer (matching libc mmap).
        if self._bar_ptr and self._bar_ptr.value:
            _libc.munmap(self._bar_ptr, self._map_size)
            self._bar_ptr = ctypes.c_void_p()

        if self._ctx.value:
            try:
                _cuCtxDestroy(self._ctx)
            except Exception:
                pass
            finally:
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
