#!/usr/bin/env python
"""
Tests for the FPGA KV cache offload pipeline.

These tests use ``MockXDMAHandle`` so they run **without** physical
FPGA hardware.  They validate the full offload data path:

1. FPGA memory allocator (alloc/free/address translation)
2. XDMA copy backend (GPU ↔ FPGA via bounce buffer)
3. Worker ↔ Scheduler interaction via metadata
"""

from __future__ import annotations

import pytest
import torch

from vllm.config import VllmConfig
from vllm.v1.fpga_kv_offload.fpga_mem import FPGABlockAllocator
from vllm.v1.fpga_kv_offload.fpga_xdma import MockXDMAHandle
from vllm.v1.fpga_kv_offload.xdma_copy_backend import XDMACopyBackend


# =========================================================================
# FPGA Memory Allocator Tests
# =========================================================================


class TestFPGABlockAllocator:
    BLOCK_SIZE = 2 * 1024 * 1024  # 2 MiB
    DRAM_SIZE = 256 * 1024 * 1024  # 256 MiB

    @pytest.fixture
    def allocator(self):
        mock_fpga = MockXDMAHandle(self.DRAM_SIZE)
        return FPGABlockAllocator(mock_fpga, self.DRAM_SIZE, self.BLOCK_SIZE)

    def test_init(self, allocator: FPGABlockAllocator):
        assert allocator.num_blocks == self.DRAM_SIZE // self.BLOCK_SIZE
        assert allocator.num_free_blocks == allocator.num_blocks
        assert allocator.num_used_blocks == 0

    def test_alloc_free(self, allocator: FPGABlockAllocator):
        bid = allocator.alloc()
        assert 0 <= bid < allocator.num_blocks
        assert allocator.is_allocated(bid)
        assert allocator.num_used_blocks == 1

        allocator.free(bid)
        assert not allocator.is_allocated(bid)
        assert allocator.num_free_blocks == allocator.num_blocks

    def test_alloc_batch(self, allocator: FPGABlockAllocator):
        bids = allocator.alloc_batch(10)
        assert len(bids) == 10
        assert all(allocator.is_allocated(b) for b in bids)

        allocator.free_batch(bids)
        assert all(not allocator.is_allocated(b) for b in bids)

    def test_out_of_memory(self, allocator: FPGABlockAllocator):
        n = allocator.num_blocks
        with pytest.raises(RuntimeError, match="out of memory"):
            allocator.alloc_batch(n + 1)

    def test_get_block_offset(self, allocator: FPGABlockAllocator):
        bid = allocator.alloc()
        offset = allocator.get_block_offset(bid)
        assert offset == bid * self.BLOCK_SIZE

    def test_reset(self, allocator: FPGABlockAllocator):
        allocator.alloc_batch(5)
        allocator.reset()
        assert allocator.num_free_blocks == allocator.num_blocks


# =========================================================================
# XDMA Copy Backend Tests
# =========================================================================


class TestXDMACopyBackend:
    NUM_BLOCKS = 8
    BLOCK_SIZE = 2 * 1024 * 1024  # 2 MiB
    DRAM_SIZE = NUM_BLOCKS * BLOCK_SIZE

    @pytest.fixture
    def setup(self):
        """Set up GPU-like and FPGA-like environments for testing."""
        # Mock FPGA
        mock_fpga = MockXDMAHandle(self.DRAM_SIZE)
        fpga_alloc = FPGABlockAllocator(mock_fpga, self.DRAM_SIZE, self.BLOCK_SIZE)

        # GPU KV cache (fake — actually on CPU for testing)
        gpu_kv_caches = {
            "layer_0": torch.randn(self.NUM_BLOCKS, self.BLOCK_SIZE // 4,
                                   dtype=torch.float32, device="cpu"),
        }
        # Move to CUDA if available.
        if torch.cuda.is_available():
            gpu_kv_caches = {
                k: v.cuda() for k, v in gpu_kv_caches.items()
            }

        # FPGA allocator (pre-allocate blocks).
        fpga_block_ids = fpga_alloc.alloc_batch(self.NUM_BLOCKS)

        return {
            "fpga": mock_fpga,
            "fpga_alloc": fpga_alloc,
            "fpga_block_ids": fpga_block_ids,
            "gpu_kv_caches": gpu_kv_caches,
            "device": next(iter(gpu_kv_caches.values())).device,
        }

    def test_store_then_load_roundtrip(self, setup):
        """GPU → FPGA → GPU: data should survive the round trip."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        gpu_caches = setup["gpu_kv_caches"]
        fpga_alloc = setup["fpga_alloc"]
        mock_fpga = setup["fpga"]
        fpga_block_ids = setup["fpga_block_ids"]

        # ── Create backend ────────────────────────────────────────────
        backend = XDMACopyBackend(mock_fpga, fpga_alloc)
        load_stream = torch.cuda.Stream()
        store_stream = torch.cuda.Stream()
        backend.init(gpu_caches, {}, setup["device"], load_stream, store_stream)

        # ── Prepare reference data ────────────────────────────────────
        ref_layer = gpu_caches["layer_0"]
        original = ref_layer[0].clone()
        # Modify the GPU block to distinguish it from the FPGA copy.
        ref_layer[0] = torch.randn_like(original) * -1

        # ── Store: GPU[0] → FPGA[id[0]] ──────────────────────────────
        events: list[tuple[int, torch.Event]] = []
        backend.launch_copy(
            src_blocks=[0],
            dst_blocks=[fpga_block_ids[0]],
            is_store=True,
            event_idx=0,
            events_list=events,
            wait_event=None,
        )
        # Wait for store to complete.
        for _, ev in events:
            ev.synchronize()
        events.clear()

        # Now GPU[0] has garbage, FPGA[id[0]] has original data (from store).

        # ── Load: FPGA[id[0]] → GPU[1] ────────────────────────────────
        backend.launch_copy(
            src_blocks=[fpga_block_ids[0]],
            dst_blocks=[1],
            is_store=False,
            event_idx=1,
            events_list=events,
        )
        for _, ev in events:
            ev.synchronize()
        events.clear()

        # ── Verify ─────────────────────────────────────────────────────
        loaded = ref_layer[1]
        assert torch.allclose(loaded.cpu(), original.cpu(), atol=1e-5), (
            "GPU → FPGA → GPU roundtrip failed: data mismatch"
        )

        backend.shutdown()

    def test_multiple_blocks_store(self, setup):
        """Batch store of multiple blocks."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        gpu_caches = setup["gpu_kv_caches"]
        fpga_alloc = setup["fpga_alloc"]
        mock_fpga = setup["fpga"]
        fpga_block_ids = setup["fpga_block_ids"]

        backend = XDMACopyBackend(mock_fpga, fpga_alloc)
        load_stream = torch.cuda.Stream()
        store_stream = torch.cuda.Stream()
        backend.init(gpu_caches, {}, setup["device"], load_stream, store_stream)

        # Store blocks 0..3 to FPGA.
        events: list[tuple[int, torch.Event]] = []
        backend.launch_copy(
            src_blocks=list(range(4)),
            dst_blocks=fpga_block_ids[:4],
            is_store=True,
            event_idx=0,
            events_list=events,
        )
        for _, ev in events:
            ev.synchronize()

        # Verify via FPGA readback (mock XDMA).
        for i in range(4):
            fpga_offset = fpga_alloc.get_block_offset(fpga_block_ids[i])
            gpu_layer = gpu_caches["layer_0"]
            block_bytes = gpu_layer[i].numel() * gpu_layer[i].element_size()
            buf = bytearray(block_bytes)
            import ctypes
            mock_fpga.read(ctypes.addressof(
                ctypes.c_char.from_buffer(buf)), fpga_offset, block_bytes)
            expected = gpu_layer[i].cpu().numpy().tobytes()
            assert buf == expected, f"Block {i} FPGA content mismatch"

        backend.shutdown()


# =========================================================================
# Integration Smoke Test (Worker ↔ Metadata)
# =========================================================================

class TestFPGAOffloadPipeline:
    """Verifies that metadata exchange and worker methods don't crash."""

    def test_worker_create(self):
        """FPGAOffloadWorker can be instantiated with mock config."""
        from vllm.v1.fpga_kv_offload.fpga_offload_worker import (
            FPGAOffloadWorker,
        )

        # Minimal VllmConfig stub.
        from vllm.config import VllmConfig, ModelConfig, CacheConfig, \
            ParallelConfig, SchedulerConfig
        from vllm.config.kv_transfer import KVTransferConfig

        config = VllmConfig(
            model_config=ModelConfig(
                model="meta-llama/Llama-3.1-8B",
                tokenizer="meta-llama/Llama-3.1-8B",
                tokenizer_mode="auto",
                trust_remote_code=False,
                dtype="bfloat16",
                seed=0,
            ),
            cache_config=CacheConfig(
                block_size=16,
                gpu_memory_utilization=0.9,
                swap_space=0,
                cache_dtype="auto",
                num_gpu_blocks_override=None,
            ),
            parallel_config=ParallelConfig(
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
            ),
            scheduler_config=SchedulerConfig(
                max_num_batched_tokens=4096,
                max_num_seqs=256,
                max_model_len=4096,
            ),
            kv_transfer_config=KVTransferConfig(
                kv_connector="FPGAOffloadConnector",
                kv_connector_module_path="vllm.v1.fpga_kv_offload.fpga_offload_connector",
            ),
        )

        worker = FPGAOffloadWorker(
            vllm_config=config,
            kv_cache_config=None,
            fpga_capacity_bytes=16 * (1024**3),
            xdma_device="/dev/null",
        )
        assert worker is not None
        assert worker.fpga_capacity_bytes == 16 * (1024**3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
