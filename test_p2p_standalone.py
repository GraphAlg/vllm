#! /usr/bin/env python3
"""Standalone P2P test — matches gpu_dma_mix.cu exactly.

Usage:
    cd /home/iei && python3 /path/to/test_p2p_standalone.py
"""

import ctypes
import mmap
import os
import time

# ── CUDA driver API ──────────────────────────────────────────────────────────
_cuda = ctypes.CDLL("libcuda.so")
CUdeviceptr = ctypes.c_uint64
CU_MEMHOSTREGISTER_IOMEMORY = 0x4
CU_MEMHOSTREGISTER_DEVICEMAP = 0x2
CUDA_DEVICE_TO_DEVICE = 0


def check_cu(ret):
    if ret != 0:
        buf = ctypes.create_string_buffer(1024)
        _cuda.cuGetErrorString(ret, buf)
        raise RuntimeError(f"CUDA error ({ret}): {buf.value.decode('utf-8', errors='replace')}")


def now_sec():
    return time.monotonic()


# ── Config (matching gpu_dma_mix.cu) ──────────────────────────────────────────
BAR_PATH = "/sys/bus/pci/devices/0000:35:00.0/resource4"
MAP_SIZE = 1024 * 1024 * 1024       # 1 GB
TARGET_SECONDS = 5.0                 # 跑 5 秒
DEVICE_ID = 0

# ── 1. Init CUDA ─────────────────────────────────────────────────────────────
check_cu(_cuda.cuInit(0))
dev = ctypes.c_int(DEVICE_ID)
check_cu(_cuda.cuDeviceGet(ctypes.byref(dev), dev))

# Create context (matching gpu_dma_mix.cu)
ctx = ctypes.c_void_p()
check_cu(_cuda.cuCtxCreate(ctypes.byref(ctx), 0, dev))
print(f"CUDA context created for device {DEVICE_ID}")

# ── 2. mmap FPGA BAR ─────────────────────────────────────────────────────────
fd = os.open(BAR_PATH, os.O_RDWR | os.O_SYNC)
bar = mmap.mmap(fd, MAP_SIZE, mmap.MAP_SHARED)
os.close(fd)
bar_base = ctypes.addressof(ctypes.c_char.from_buffer(bar))
print(f"BAR mmaped at 0x{bar_base:x}")

# ── 3. Register I/O memory ──────────────────────────────────────────────────
check_cu(_cuda.cuMemHostRegister(
    ctypes.c_void_p(bar_base),
    ctypes.c_size_t(MAP_SIZE),
    ctypes.c_int(CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_DEVICEMAP),
))
print("cuMemHostRegister(IOMEMORY) OK")

# ── 4. Get device pointer ──────────────────────────────────────────────────
d_bar = CUdeviceptr(0)
check_cu(_cuda.cuMemHostGetDevicePointer(
    ctypes.byref(d_bar), ctypes.c_void_p(bar_base), 0))
d_bar_val = int(d_bar)
print(f"CUDA device pointer: 0x{d_bar_val:x}")

# ── 5. Allocate GPU HBM buffer ──────────────────────────────────────────────
d_gpu = CUdeviceptr(0)
check_cu(_cuda.cuMemAlloc(ctypes.byref(d_gpu), ctypes.c_size_t(MAP_SIZE)))
print(f"GPU HBM buffer: 0x{d_gpu.value:x}")

# ── 6. Fill GPU buffer with pattern ─────────────────────────────────────────
print("\nFilling GPU buffer with 0xAB pattern...")
dummy = (ctypes.c_uint8 * MAP_SIZE)()
ctypes.memset(dummy, 0xAB, MAP_SIZE)
check_cu(_cuda.cuMemcpyHtoD(d_gpu, dummy, ctypes.c_size_t(MAP_SIZE)))
print("  done")

# ── 7. Warmup ────────────────────────────────────────────────────────────────
print("\nWarmup...")
for _ in range(3):
    check_cu(_cuda.cuMemcpyAsync(
        CUdeviceptr(d_bar_val), d_gpu, ctypes.c_size_t(MAP_SIZE),
        ctypes.c_int(CUDA_DEVICE_TO_DEVICE), ctypes.c_void_p(None)))
    check_cu(_cuda.cuCtxSynchronize(ctx))
print("  done")

# ── 8. Bandwidth test: Store (GPU → FPGA) ────────────────────────────────────
print(f"\n=== Store bandwidth (GPU → FPGA) for {TARGET_SECONDS}s ===")
start = now_sec()
iters = 0
bw_sum = 0.0
bw_min = 1e18
bw_max = 0.0

while now_sec() < start + TARGET_SECONDS:
    t0 = now_sec()
    check_cu(_cuda.cuMemcpyAsync(
        CUdeviceptr(d_bar_val), d_gpu, ctypes.c_size_t(MAP_SIZE),
        ctypes.c_int(CUDA_DEVICE_TO_DEVICE), ctypes.c_void_p(None)))
    check_cu(_cuda.cuCtxSynchronize(ctx))
    elapsed = now_sec() - t0

    bw = MAP_SIZE / elapsed / 1e9
    bw_sum += bw
    bw_min = min(bw_min, bw)
    bw_max = max(bw_max, bw)
    iters += 1

print(f"  Iters: {iters}")
print(f"  Avg:   {bw_sum / iters:.2f} GB/s")
print(f"  Min:   {bw_min:.2f} GB/s")
print(f"  Max:   {bw_max:.2f} GB/s")

# ── 9. Verification: read back from FPGA ────────────────────────────────────
print("\n=== Verification ===")
d_read = CUdeviceptr(0)
check_cu(_cuda.cuMemAlloc(ctypes.byref(d_read), ctypes.c_size_t(4096)))

# Write pattern to FPGA
pattern = (ctypes.c_uint8 * 256)()
for i in range(256):
    pattern[i] = i & 0xFF
check_cu(_cuda.cuMemcpyHtoD(CUdeviceptr(d_bar_val), pattern, ctypes.c_size_t(256)))

# Read back from FPGA → GPU
check_cu(_cuda.cuMemcpyAsync(
    d_read, CUdeviceptr(d_bar_val), ctypes.c_size_t(256),
    ctypes.c_int(CUDA_DEVICE_TO_DEVICE), ctypes.c_void_p(None)))
check_cu(_cuda.cuCtxSynchronize(ctx))

# GPU → host
out = (ctypes.c_uint8 * 256)()
check_cu(_cuda.cuMemcpyDtoH(out, d_read, ctypes.c_size_t(256)))
ok = all(out[i] == (i & 0xFF) for i in range(256))
print(f"  Data integrity: {'✓ PASS' if ok else '✗ FAIL'}")

# ── Cleanup ──────────────────────────────────────────────────────────────────
check_cu(_cuda.cuMemFree(d_gpu))
check_cu(_cuda.cuMemFree(d_read))
check_cu(_cuda.cuMemHostUnregister(ctypes.c_void_p(bar_base)))
bar.close()
check_cu(_cuda.cuCtxDestroy(ctx))
print("\nDone.")
