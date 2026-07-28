# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU direct DMA between CUDA memory and an FPGA PCIe BAR.

The FPGA BAR is mapped into the process and registered with CUDA using
CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_DEVICEMAP.

Important implementation details:

* Use the CUDA primary context so that CUDA pointers created by vLLM/PyTorch
  remain valid in this backend.
* Declare every CUDA Driver API ctypes prototype explicitly.
* Driver API ``cuMemcpyAsync`` has FOUR arguments. It does not take a
  ``cudaMemcpyKind`` argument.
* Push the retained CUDA context on every calling thread because CUDA contexts
  are thread-local.
* Keep a persistent ctypes view of the mmap for the entire registration
  lifetime.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import threading
from typing import Optional

from vllm.logger import init_logger
from vllm.v1.kv_offload.fpga.allocator import FPGABlockAllocator
from vllm.v1.kv_offload.fpga.backend_base import DMABackend

logger = init_logger(__name__)

# Hardware configuration.
FPGA_PCI_BDF = "0000:35:00.0"
FPGA_BAR_INDEX = 4
FPGA_BAR_SIZE = 1 * 1024**3

# CUDA Driver API scalar and opaque-handle types.
CUresult = ctypes.c_int
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUstream = ctypes.c_void_p
CUdeviceptr = ctypes.c_uint64

CUDA_SUCCESS = 0

CU_MEMHOSTREGISTER_DEVICEMAP = 0x02
CU_MEMHOSTREGISTER_IOMEMORY = 0x04

_cuda = ctypes.CDLL("libcuda.so.1")


def _resolve_symbol(*names: str):
    """Return the first CUDA symbol exported by the installed driver."""
    for name in names:
        try:
            return getattr(_cuda, name)
        except AttributeError:
            continue
    raise RuntimeError(
        "CUDA driver does not export any of these symbols: "
        + ", ".join(names)
    )


# Resolve versioned symbols where CUDA may expose either spelling.
_cuCtxCreate = _resolve_symbol("cuCtxCreate_v2", "cuCtxCreate")
_cuCtxDestroy = _resolve_symbol("cuCtxDestroy_v2", "cuCtxDestroy")
_cuCtxPushCurrent = _resolve_symbol(
    "cuCtxPushCurrent_v2", "cuCtxPushCurrent"
)
_cuCtxPopCurrent = _resolve_symbol(
    "cuCtxPopCurrent_v2", "cuCtxPopCurrent"
)
_cuMemHostGetDevicePointer = _resolve_symbol(
    "cuMemHostGetDevicePointer_v2",
    "cuMemHostGetDevicePointer",
)
_cuMemcpyAsync = _resolve_symbol("cuMemcpyAsync")
_cuStreamSynchronize = _resolve_symbol("cuStreamSynchronize")


def _configure_cuda_api() -> None:
    """Declare exact CUDA Driver API prototypes for ctypes."""

    _cuda.cuInit.argtypes = [ctypes.c_uint]
    _cuda.cuInit.restype = CUresult

    _cuda.cuDeviceGet.argtypes = [
        ctypes.POINTER(CUdevice),
        ctypes.c_int,
    ]
    _cuda.cuDeviceGet.restype = CUresult

    _cuda.cuDeviceGetName.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_int,
        CUdevice,
    ]
    _cuda.cuDeviceGetName.restype = CUresult

    _cuda.cuDevicePrimaryCtxRetain.argtypes = [
        ctypes.POINTER(CUcontext),
        CUdevice,
    ]
    _cuda.cuDevicePrimaryCtxRetain.restype = CUresult

    _cuda.cuDevicePrimaryCtxRelease.argtypes = [CUdevice]
    _cuda.cuDevicePrimaryCtxRelease.restype = CUresult

    _cuda.cuCtxGetCurrent.argtypes = [
        ctypes.POINTER(CUcontext),
    ]
    _cuda.cuCtxGetCurrent.restype = CUresult

    _cuCtxCreate.argtypes = [
        ctypes.POINTER(CUcontext),
        ctypes.c_uint,
        CUdevice,
    ]
    _cuCtxCreate.restype = CUresult

    _cuCtxDestroy.argtypes = [CUcontext]
    _cuCtxDestroy.restype = CUresult

    _cuCtxPushCurrent.argtypes = [CUcontext]
    _cuCtxPushCurrent.restype = CUresult

    _cuCtxPopCurrent.argtypes = [
        ctypes.POINTER(CUcontext),
    ]
    _cuCtxPopCurrent.restype = CUresult

    _cuda.cuMemHostRegister.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    _cuda.cuMemHostRegister.restype = CUresult

    _cuda.cuMemHostUnregister.argtypes = [
        ctypes.c_void_p,
    ]
    _cuda.cuMemHostUnregister.restype = CUresult

    _cuMemHostGetDevicePointer.argtypes = [
        ctypes.POINTER(CUdeviceptr),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    _cuMemHostGetDevicePointer.restype = CUresult

    # CUDA Driver API signature:
    # CUresult cuMemcpyAsync(
    #     CUdeviceptr dst,
    #     CUdeviceptr src,
    #     size_t ByteCount,
    #     CUstream hStream);
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

    name_ret = int(
        _cuda.cuGetErrorName(
            CUresult(code),
            ctypes.byref(name_ptr),
        )
    )
    text_ret = int(
        _cuda.cuGetErrorString(
            CUresult(code),
            ctypes.byref(text_ptr),
        )
    )

    name = (
        name_ptr.value.decode("utf-8", errors="replace")
        if name_ret == CUDA_SUCCESS and name_ptr.value
        else "CUDA_ERROR_UNKNOWN"
    )
    text = (
        text_ptr.value.decode("utf-8", errors="replace")
        if text_ret == CUDA_SUCCESS and text_ptr.value
        else "No CUDA error description"
    )

    raise RuntimeError(
        f"{operation} failed: CUDA driver error {code} "
        f"({name}): {text}"
    )


class _CudaContextScope:
    """Push a CUDA context for the current thread and restore the old one."""

    def __init__(self, context: CUcontext) -> None:
        self._context = context
        self._active = False

    def __enter__(self) -> "_CudaContextScope":
        _check_cu(
            _cuCtxPushCurrent(self._context),
            "cuCtxPushCurrent",
        )
        self._active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._active:
            return

        popped = CUcontext()
        ret = int(_cuCtxPopCurrent(ctypes.byref(popped)))
        self._active = False

        if ret != CUDA_SUCCESS and exc_type is None:
            _check_cu(ret, "cuCtxPopCurrent")


class GPUDirectP2PBackend(DMABackend):
    """CUDA DMA backend that exposes an FPGA BAR as a CUDA device pointer."""

    def __init__(
        self,
        fpga_allocator: FPGABlockAllocator,
        device_id: int = 0,
        map_size: int = 0,
    ) -> None:
        super().__init__(fpga_allocator)

        self._lock = threading.RLock()
        self._shutdown = False
        self._registered = False
        self._primary_ctx_retained = False

        self._device = CUdevice()
        self._ctx = CUcontext()
        self._bar: Optional[mmap.mmap] = None
        self._bar_view = None
        self._bar_base = 0
        self._d_bar = 0
        self._map_size = 0

        if map_size == 0:
            map_size = FPGA_BAR_SIZE
        if map_size <= 0:
            raise ValueError(
                f"map_size must be positive, got {map_size}"
            )
        if map_size % mmap.PAGESIZE != 0:
            raise ValueError(
                "map_size must be page aligned: "
                f"size={map_size}, page_size={mmap.PAGESIZE}"
            )

        self._map_size = int(map_size)
        self._device_id = int(device_id)
        self._bar_path = (
            f"/sys/bus/pci/devices/{FPGA_PCI_BDF}/"
            f"resource{FPGA_BAR_INDEX}"
        )

        try:
            self._initialize_cuda()
            self._map_and_register_bar()
        except Exception:
            self._cleanup_partial_initialization()
            raise

    def _initialize_cuda(self) -> None:
        _check_cu(_cuda.cuInit(0), "cuInit")

        # The output device handle must be separate from the input ordinal.
        _check_cu(
            _cuda.cuDeviceGet(
                ctypes.byref(self._device),
                ctypes.c_int(self._device_id),
            ),
            "cuDeviceGet",
        )

        name_buffer = ctypes.create_string_buffer(256)
        _check_cu(
            _cuda.cuDeviceGetName(
                name_buffer,
                len(name_buffer),
                self._device,
            ),
            "cuDeviceGetName",
        )

        # vLLM/PyTorch allocations normally live in the primary context.
        # Retaining that context avoids making their CUDA pointers invalid.
        _check_cu(
            _cuda.cuDevicePrimaryCtxRetain(
                ctypes.byref(self._ctx),
                self._device,
            ),
            "cuDevicePrimaryCtxRetain",
        )
        self._primary_ctx_retained = True

        if not self._ctx.value:
            raise RuntimeError(
                "cuDevicePrimaryCtxRetain returned a NULL context"
            )

        logger.info(
            "GPUDirectP2PBackend: retained primary CUDA context "
            "device=%d gpu=%s ctx=0x%x",
            self._device_id,
            name_buffer.value.decode("utf-8", errors="replace"),
            self._ctx.value,
        )

    def _map_and_register_bar(self) -> None:
        if not os.path.exists(self._bar_path):
            raise FileNotFoundError(
                f"FPGA BAR resource does not exist: {self._bar_path}"
            )

        resource_size = os.path.getsize(self._bar_path)
        if resource_size > 0 and self._map_size > resource_size:
            raise ValueError(
                "Requested BAR mapping exceeds resource size: "
                f"requested={self._map_size}, "
                f"resource={resource_size}, "
                f"path={self._bar_path}"
            )

        fd = os.open(
            self._bar_path,
            os.O_RDWR | os.O_SYNC,
        )
        try:
            self._bar = mmap.mmap(
                fd,
                self._map_size,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                offset=0,
            )
        finally:
            os.close(fd)

        if len(self._bar) != self._map_size:
            raise RuntimeError(
                "BAR mmap length mismatch: "
                f"mapped={len(self._bar)}, "
                f"expected={self._map_size}"
            )

        # Keep this view alive until after cuMemHostUnregister.
        self._bar_view = (
            ctypes.c_ubyte * self._map_size
        ).from_buffer(self._bar)
        self._bar_base = ctypes.addressof(self._bar_view)

        if self._bar_base % mmap.PAGESIZE != 0:
            raise RuntimeError(
                "BAR virtual address is not page aligned: "
                f"address=0x{self._bar_base:x}, "
                f"page_size={mmap.PAGESIZE}"
            )

        flags = (
            CU_MEMHOSTREGISTER_IOMEMORY
            | CU_MEMHOSTREGISTER_DEVICEMAP
        )

        with _CudaContextScope(self._ctx):
            _check_cu(
                _cuda.cuMemHostRegister(
                    ctypes.c_void_p(self._bar_base),
                    ctypes.c_size_t(self._map_size),
                    ctypes.c_uint(flags),
                ),
                "cuMemHostRegister(BAR4)",
            )
            self._registered = True

            device_pointer = CUdeviceptr()
            _check_cu(
                _cuMemHostGetDevicePointer(
                    ctypes.byref(device_pointer),
                    ctypes.c_void_p(self._bar_base),
                    ctypes.c_uint(0),
                ),
                "cuMemHostGetDevicePointer(BAR4)",
            )

        self._d_bar = int(device_pointer.value)
        if self._d_bar == 0:
            raise RuntimeError(
                "cuMemHostGetDevicePointer returned a NULL pointer"
            )

        logger.info(
            "GPUDirectP2PBackend: registered %s "
            "host=0x%x device=0x%x size=%.3f GiB flags=0x%x",
            self._bar_path,
            self._bar_base,
            self._d_bar,
            self._map_size / (1024**3),
            flags,
        )

    def _validate_transfer(
        self,
        bar_offset: int,
        cuda_ptr: int,
        size: int,
        operation: str,
    ) -> None:
        if self._shutdown:
            raise RuntimeError(
                f"{operation} called after backend shutdown"
            )
        if not self._registered or self._d_bar == 0:
            raise RuntimeError(
                f"{operation} called before BAR registration"
            )

        values = {
            "bar_offset": bar_offset,
            "cuda_ptr": cuda_ptr,
            "size": size,
        }
        for field, value in values.items():
            if not isinstance(value, int):
                raise TypeError(
                    f"{operation}: {field} must be int, "
                    f"got {type(value).__name__}"
                )

        if bar_offset < 0:
            raise ValueError(
                f"{operation}: BAR offset must be non-negative"
            )
        if cuda_ptr <= 0:
            raise ValueError(
                f"{operation}: CUDA pointer must be non-zero"
            )
        if size < 0:
            raise ValueError(
                f"{operation}: size must be non-negative"
            )
        if size == 0:
            return

        end = bar_offset + size
        if end < bar_offset or end > self._map_size:
            raise ValueError(
                f"{operation}: BAR range is outside mapping: "
                f"offset={bar_offset}, size={size}, "
                f"end={end}, map_size={self._map_size}"
            )

    def _copy_and_wait(
        self,
        dst: int,
        src: int,
        size: int,
        operation: str,
    ) -> None:
        if size == 0:
            return

        # The context stack is per-thread, so each vLLM worker thread must
        # push the context before issuing Driver API commands.
        with self._lock:
            with _CudaContextScope(self._ctx):
                default_stream = CUstream(None)

                _check_cu(
                    _cuMemcpyAsync(
                        CUdeviceptr(dst),
                        CUdeviceptr(src),
                        ctypes.c_size_t(size),
                        default_stream,
                    ),
                    f"{operation}: cuMemcpyAsync",
                )

                # Preserve the original backend's synchronous method
                # semantics. Remove this only after the caller has explicit
                # stream/event lifetime management.
                _check_cu(
                    _cuStreamSynchronize(default_stream),
                    f"{operation}: cuStreamSynchronize",
                )

    @property
    def name(self) -> str:
        return "GPU_DIRECT_P2P"

    def write(self, src_ptr: int, dst_addr: int, size: int) -> None:
        """Copy CUDA memory to FPGA BAR4."""
        self._validate_transfer(
            bar_offset=dst_addr,
            cuda_ptr=src_ptr,
            size=size,
            operation="write",
        )
        self._copy_and_wait(
            dst=self._d_bar + dst_addr,
            src=src_ptr,
            size=size,
            operation="write",
        )

    def read(self, src_addr: int, dst_ptr: int, size: int) -> None:
        """Copy FPGA BAR4 to CUDA memory."""
        self._validate_transfer(
            bar_offset=src_addr,
            cuda_ptr=dst_ptr,
            size=size,
            operation="read",
        )
        self._copy_and_wait(
            dst=dst_ptr,
            src=self._d_bar + src_addr,
            size=size,
            operation="read",
        )

    @property
    def bar_devptr(self) -> int:
        return self._d_bar

    @property
    def bar_size(self) -> int:
        return self._map_size

    def _cleanup_partial_initialization(self) -> None:
        try:
            self._release_resources()
        except Exception:
            logger.exception(
                "GPUDirectP2PBackend: cleanup after initialization "
                "failure also failed"
            )

    def _release_resources(self) -> None:
        unregister_error: Optional[BaseException] = None

        if self._registered and self._bar_base and self._ctx.value:
            try:
                with _CudaContextScope(self._ctx):
                    _check_cu(
                        _cuda.cuMemHostUnregister(
                            ctypes.c_void_p(self._bar_base)
                        ),
                        "cuMemHostUnregister(BAR4)",
                    )
            except BaseException as exc:
                unregister_error = exc
            finally:
                self._registered = False

        # A ctypes object exported from mmap must be released before close().
        self._bar_view = None
        self._bar_base = 0
        self._d_bar = 0

        if self._bar is not None:
            try:
                self._bar.close()
            finally:
                self._bar = None

        if self._primary_ctx_retained:
            try:
                _check_cu(
                    _cuda.cuDevicePrimaryCtxRelease(self._device),
                    "cuDevicePrimaryCtxRelease",
                )
            finally:
                self._primary_ctx_retained = False
                self._ctx = CUcontext()

        if unregister_error is not None:
            raise unregister_error

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
