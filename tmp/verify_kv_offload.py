#!/usr/bin/env python3
"""Verify whether KV cache blocks actually land in / come back from FPGA memory.

Runs vLLM's own FPGA offload machinery (FPGAOffloadingWorker → BlockTransferEngine
→ GPUDirectP2PBackend) on a small dummy KV cache, then reads the FPGA aperture
back and compares against the GPU data.

Run on the server:
    sudo fpga-env/bin/python tmp/verify_kv_offload.py
"""

import os

import torch

os.environ.setdefault("VLLM_FPGA_BACKEND", "gpu_direct_p2p")
# Deliberately do NOT set VLLM_FPGA_DEVICE_ID: it now defaults to the model's
# (torch's current) device.

from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCacheTensor,
    CanonicalKVCaches,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.fpga.common import FPGALoadStoreSpec
from vllm.v1.kv_offload.fpga.worker import FPGAOffloadingWorker

NUM_BLOCKS = 4
BLOCK_BYTES = 4096  # small block size is fine for a transfer test
PATTERN_MOD = 251  # keep values within int8 range


def build_dummy_kv() -> CanonicalKVCaches:
    gpu_tensor = torch.zeros(NUM_BLOCKS, BLOCK_BYTES, dtype=torch.int8, device="cuda")
    ct = CanonicalKVCacheTensor(tensor=gpu_tensor, page_size_bytes=BLOCK_BYTES)
    refs = [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=BLOCK_BYTES)]
    return CanonicalKVCaches(tensors=[ct], group_data_refs=[refs])


def main() -> None:
    print(f"model device = cuda:{torch.cuda.current_device()}")
    kv_caches = build_dummy_kv()
    gpu_tensor = kv_caches.tensors[0].tensor

    worker = FPGAOffloadingWorker(
        kv_caches=kv_caches,
        block_size_factor=1,
        num_fpga_blocks=NUM_BLOCKS,
        fpga_capacity_bytes=NUM_BLOCKS * BLOCK_BYTES,
        xdma_prefix="/dev/xdma0",
    )
    backend = worker._engine._backend  # noqa: SLF001
    fpga_block0_addr = backend.get_block_addr(0)

    # Fill GPU block 0 with a known pattern.
    gpu_tensor[0] = torch.arange(BLOCK_BYTES, dtype=torch.int64) % PATTERN_MOD

    # ── Store: GPU block 0 → FPGA block 0 ─────────────────────────────
    ok = worker.submit_store(
        job_id=0,
        src_spec=GPULoadStoreSpec([0], group_sizes=[1], block_indices=[0]),
        dst_spec=FPGALoadStoreSpec([0]),
    )
    assert ok, "submit_store returned False"
    worker.wait({0})
    finished = worker.get_finished()
    print(f"store finished jobs: {finished}")

    # ── Read FPGA block 0 back into a fresh GPU buffer ────────────────
    fpga_read = torch.zeros(BLOCK_BYTES, dtype=torch.int8, device="cuda")
    backend.read(
        fpga_block0_addr, fpga_read.data_ptr(), BLOCK_BYTES,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.current_stream().synchronize()

    store_match = torch.equal(fpga_read, gpu_tensor[0])
    if store_match:
        print("STORE VERIFY: PASS  (FPGA block 0 == GPU block 0)")
    else:
        mismatch = (fpga_read != gpu_tensor[0]).sum().item()
        print(f"STORE VERIFY: FAIL  ({mismatch}/{BLOCK_BYTES} bytes differ)")
        print(f"  GPU  block 0[:16]: {gpu_tensor[0][:16].tolist()}")
        print(f"  FPGA read [:16]:  {fpga_read[:16].tolist()}")
        # 0xff means the aperture never saw the write.
        all_ff = bool((fpga_read == -1).all().item())
        print(f"  FPGA read all 0xff (dead aperture)? {all_ff}")

    # ── Load: FPGA block 0 → GPU block 0 ──────────────────────────────
    gpu_tensor[0] = torch.zeros(BLOCK_BYTES, dtype=torch.int8, device="cuda")
    ok = worker.submit_load(
        job_id=1,
        src_spec=FPGALoadStoreSpec([0]),
        dst_spec=GPULoadStoreSpec([0], group_sizes=[1], block_indices=[0]),
    )
    assert ok, "submit_load returned False"
    worker.wait({1})
    torch.cuda.current_stream().synchronize()

    expected = (
        torch.arange(BLOCK_BYTES, dtype=torch.int64, device="cuda") % PATTERN_MOD
    ).to(torch.int8)
    load_match = torch.equal(gpu_tensor[0], expected)
    if load_match:
        print("LOAD VERIFY: PASS  (GPU block 0 restored from FPGA)")
    else:
        mismatch = (gpu_tensor[0] != expected).sum().item()
        print(f"LOAD VERIFY: FAIL  ({mismatch}/{BLOCK_BYTES} bytes differ)")
        print(f"  expected[:16]: {expected[:16].tolist()}")
        print(f"  GPU    [:16]:  {gpu_tensor[0][:16].tolist()}")

    worker.shutdown()
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
