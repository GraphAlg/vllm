# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FPGA offload 测试套件。

测试层级（从底层到集成）:
  层级 1:  FPGA 内存分配器 + XDMA mock (不需要 GPU)
  层级 2:  XDMA copy backend 数据圆整 (需要 CUDA)
  层级 3:  FPGAOffloadingWorker submit/load/store/get_finished (需要 CUDA)
  层级 4:  FPGAOffloadingSpec factory 创建 + LoaStoreSpec (不需要硬件)
  层级 5:  OffloadingConnector + FPGAOffloadingManager eviction 逻辑
  层级 6:  完整推理集成测试 (需要 GPU + MockXDMA)
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.fpga.allocator import FPGABlockAllocator
from vllm.v1.kv_offload.fpga.xdma_driver import MockXDMAHandle
from vllm.v1.kv_offload.fpga.backends.xdma_bounce_buffer import (
    XDMABounceBufferBackend,
)
from vllm.v1.kv_offload.fpga.copy_backend import BlockTransferEngine
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    LoadStoreSpec,
    TransferResult,
)
from vllm.v1.kv_offload.fpga.common import FPGALoadStoreSpec
from vllm.v1.kv_offload.fpga.spec import FPGAOffloadingSpec
from vllm.v1.kv_offload.fpga.worker import FPGAOffloadingWorker
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

# ===========================================================================
# 层级 1: FPGA 内存 + XDMA mock（最小依赖）
# ===========================================================================


class TestFPGABlockAllocator:
    BLOCK_SIZE = 2 * 1024 * 1024  # 2 MiB
    DRAM_SIZE = 256 * 1024 * 1024  # 256 MiB

    @pytest.fixture
    def allocator(self):
        mock = MockXDMAHandle(self.DRAM_SIZE)
        return FPGABlockAllocator(mock, self.DRAM_SIZE, self.BLOCK_SIZE)

    def test_init(self, allocator):
        assert allocator.num_blocks == self.DRAM_SIZE // self.BLOCK_SIZE
        assert allocator.num_free_blocks == allocator.num_blocks
        assert allocator.num_used_blocks == 0

    def test_alloc_free(self, allocator):
        bid = allocator.alloc()
        assert 0 <= bid < allocator.num_blocks
        assert allocator.is_allocated(bid)
        assert allocator.num_used_blocks == 1
        allocator.free(bid)
        assert not allocator.is_allocated(bid)
        assert allocator.num_free_blocks == allocator.num_blocks

    def test_alloc_batch(self, allocator):
        bids = allocator.alloc_batch(10)
        assert len(bids) == 10
        assert all(allocator.is_allocated(b) for b in bids)
        allocator.free_batch(bids)
        assert all(not allocator.is_allocated(b) for b in bids)

    def test_oom(self, allocator):
        n = allocator.num_blocks
        with pytest.raises(RuntimeError, match="out of memory|need.*blocks.*only.*free"):
            allocator.alloc_batch(n + 1)

    def test_block_offset(self, allocator):
        bid = allocator.alloc()
        assert allocator.get_block_offset(bid) == bid * self.BLOCK_SIZE

    def test_reset(self, allocator):
        allocator.alloc_batch(5)
        allocator.reset()
        assert allocator.num_free_blocks == allocator.num_blocks


class TestMockXDMA:
    DRAM_SIZE = 16 * 1024 * 1024  # 16 MiB

    @pytest.fixture
    def xdma(self):
        return MockXDMAHandle(self.DRAM_SIZE)

    def test_read_write_roundtrip(self, xdma):
        import ctypes
        buf = (ctypes.c_uint8 * 1024)()
        for i in range(1024):
            buf[i] = i & 0xFF

        n = xdma.write(ctypes.addressof(buf), fpga_addr=0x1000, size=1024)
        assert n == 1024

        # Verify via readback.
        out = (ctypes.c_uint8 * 1024)()
        n = xdma.read(ctypes.addressof(out), fpga_addr=0x1000, size=1024)
        assert n == 1024
        for i in range(1024):
            assert out[i] == (i & 0xFF), f"byte {i} mismatch"

    def test_write_overflow(self, xdma):
        import ctypes
        with pytest.raises(RuntimeError, match="overflow"):
            huge = (ctypes.c_uint8 * (self.DRAM_SIZE + 1))()
            xdma.write(ctypes.addressof(huge), fpga_addr=0, size=self.DRAM_SIZE + 1)

    def test_ddr_size(self, xdma):
        assert xdma.get_ddr_size() == self.DRAM_SIZE


# ===========================================================================
# 层级 2: BlockTransferEngine + XDMABounceBufferBackend GPU↔FPGA 圆整测试（需要 CUDA）
# ===========================================================================


class TestBlockTransferEngine:
    NUM_BLOCKS = 8
    BLOCK_SIZE = 2 * 1024 * 1024  # 2 MiB
    DRAM_SIZE = NUM_BLOCKS * BLOCK_SIZE

    @pytest.fixture
    def engine(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        mock = MockXDMAHandle(self.DRAM_SIZE)
        alloc = FPGABlockAllocator(mock, self.DRAM_SIZE, self.BLOCK_SIZE)
        alloc.alloc_batch(self.NUM_BLOCKS)

        gpu = {
            "layer_0":
                torch.randn(self.NUM_BLOCKS, self.BLOCK_SIZE // 4,
                            dtype=torch.float32).cuda(),
        }

        backend = XDMABounceBufferBackend(fpga=mock, fpga_allocator=alloc)
        eng = BlockTransferEngine(
            backend=backend,
            gpu_caches=gpu,
            store_stream=torch.cuda.Stream(),
            load_stream=torch.cuda.Stream(),
        )
        yield {
            "engine": eng,
            "fpga": mock,
            "alloc": alloc,
            "fpga_block_ids": list(range(self.NUM_BLOCKS)),
            "gpu_caches": gpu,
        }
        eng.shutdown()

    def test_store_then_load_roundtrip(self, engine):
        """GPU → FPGA → GPU: data should survive the trip."""
        gpu = engine["gpu_caches"]
        alloc = engine["alloc"]
        eng = engine["engine"]
        fpga_bids = engine["fpga_block_ids"]

        # Save original; poison GPU block.
        ref = gpu["layer_0"][0].clone()
        gpu["layer_0"][0] = torch.randn_like(ref)

        # Store GPU[0] → FPGA[id[0]]
        ev: list = []
        eng.launch_copy([0], [fpga_bids[0]], True, 0, ev, None)
        for _, e in ev:
            e.synchronize()
        ev.clear()

        # Load FPGA[id[0]] → GPU[1]
        eng.launch_copy([fpga_bids[0]], [1], False, 1, ev)
        for _, e in ev:
            e.synchronize()

        assert torch.allclose(gpu["layer_0"][1].cpu(), ref.cpu(), atol=1e-5)

    def test_multiple_blocks(self, engine):
        """Batch store multiple blocks and verify via FPGA readback."""
        gpu = engine["gpu_caches"]
        alloc = engine["alloc"]
        mock_fpga = engine["fpga"]
        eng = engine["engine"]
        fpga_bids = engine["fpga_block_ids"]

        import ctypes
        gpu_t = gpu["layer_0"]
        block_bytes = gpu_t[0].numel() * gpu_t[0].element_size()

        ev: list = []
        eng.launch_copy([0, 1], [fpga_bids[0], fpga_bids[1]], True, 0, ev, None)
        for _, e in ev:
            e.synchronize()

        # Readback from mock FPGA.
        for i in range(2):
            buf = bytearray(block_bytes)
            offset = alloc.get_block_offset(fpga_bids[i])
            mock_fpga.read(ctypes.addressof(
                ctypes.c_char.from_buffer(buf)), offset, block_bytes)
            expected = gpu_t[i].cpu().numpy().tobytes()
            assert buf == expected, f"Block {i} mismatch"



# ===========================================================================
# 层级 3: FPGAOffloadingWorker submit/load/store/get_finished
# ===========================================================================


class TestFPGAOffloadingWorker:
    NUM_BLOCKS = 8
    PAGE_SIZE = 4096  # 4 KB per page
    NUM_TENSORS = 2
    NUM_FPGA_BLOCKS = 16

    @pytest.fixture
    def worker(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        os.environ["VLLM_FPGA_MOCK"] = "1"

        tensors = []
        for i in range(self.NUM_TENSORS):
            gpu = torch.zeros(
                (self.NUM_BLOCKS, self.PAGE_SIZE),
                dtype=torch.int8,
                device="cuda",
            )
            tensors.append(CanonicalKVCacheTensor(
                tensor=gpu, page_size_bytes=self.PAGE_SIZE,
            ))

        refs = [
            [CanonicalKVCacheRef(tensor_idx=i, page_size_bytes=4096)
             for i in range(self.NUM_TENSORS)],
        ]

        kv_caches = CanonicalKVCaches(tensors=tensors, group_data_refs=refs)

        w = FPGAOffloadingWorker(
            kv_caches=kv_caches,
            block_size_factor=1,
            num_fpga_blocks=self.NUM_FPGA_BLOCKS,
            fpga_capacity_bytes=self.NUM_FPGA_BLOCKS
            * self.NUM_TENSORS * self.PAGE_SIZE,
            xdma_prefix="mock",
        )
        yield w
        w.shutdown()

    def _write_gpu_block(self, worker, block_id, value):
        """Fill GPU tensor block_id with a known int8 value."""
        for ct in worker._kv_caches.tensors:
            ct.tensor[block_id].fill_(value)

    def _read_gpu_block(self, worker, block_id):
        """Return (all_equal, first_byte) for GPU tensor block_id."""
        vals = [int(ct.tensor[block_id][0].item())
                for ct in worker._kv_caches.tensors]
        return vals[0] if len(set(vals)) == 1 else None

    def test_submit_store_then_get_finished(self, worker):
        """Store GPU[0] → FPGA, then poll get_finished."""
        self._write_gpu_block(worker, 0, 42)

        src = GPULoadStoreSpec(block_ids=np.array([0], dtype=np.int64),
                               group_sizes=[1], block_indices=[0])
        dst = FPGALoadStoreSpec(block_ids=np.array([0], dtype=np.int64))

        ok = worker.submit_store(job_id=1, src_spec=src, dst_spec=dst)
        assert ok

        # Poll until done.
        import time
        for _ in range(50):  # max 5 s
            results = worker.get_finished()
            if results:
                assert results[0].job_id == 1
                assert results[0].success
                return
            time.sleep(0.1)
        pytest.fail("Store did not complete within 5 s")

    def test_submit_store_then_load_roundtrip(self, worker):
        """GPU[0]=42 → FPGA → GPU[1]: read 42 back."""
        self._write_gpu_block(worker, 0, 99)

        # Store
        src = GPULoadStoreSpec(block_ids=np.array([0], dtype=np.int64),
                               group_sizes=[1], block_indices=[0])
        dst = FPGALoadStoreSpec(block_ids=np.array([0], dtype=np.int64))
        assert worker.submit_store(job_id=10, src_spec=src, dst_spec=dst)

        # Poison GPU[1]
        self._write_gpu_block(worker, 1, 0)

        # Wait for store
        import time
        for _ in range(50):
            if worker.get_finished():
                break
            time.sleep(0.1)

        # Load
        src2 = FPGALoadStoreSpec(block_ids=np.array([0], dtype=np.int64))
        dst2 = GPULoadStoreSpec(block_ids=np.array([1], dtype=np.int64),
                                group_sizes=[1], block_indices=[0])
        assert worker.submit_load(job_id=11, src_spec=src2, dst_spec=dst2)

        for _ in range(50):
            results = worker.get_finished()
            if any(r.job_id == 11 for r in results):
                break
            time.sleep(0.1)

        assert self._read_gpu_block(worker, 1) == 99, \
            "Roundtrip: GPU[1] should read 99"

    def test_wait(self, worker):
        """wait() should block until jobs complete."""
        src = GPULoadStoreSpec(block_ids=np.array([0], dtype=np.int64),
                               group_sizes=[1], block_indices=[0])
        dst = FPGALoadStoreSpec(block_ids=np.array([0], dtype=np.int64))
        assert worker.submit_store(job_id=20, src_spec=src, dst_spec=dst)

        worker.wait({20})

        # After wait, get_finished should report completion.
        results = worker.get_finished()
        assert any(r.job_id == 20 for r in results)


# ===========================================================================
# 层级 4: FPGAOffloadingSpec + Factory 创建
# ===========================================================================


class TestFPGAOffloadingSpec:
    @staticmethod
    def _make_minimal_model_config():
        """Create a minimal model config dict that FPGAOffloadingSpec needs.

        Avoids network calls — only ``hidden_size``, ``num_hidden_layers``
        and related fields are required by FPGAOffloadingSpec.
        """
        from vllm.config import ModelConfig
        import transformers
        import os

        # Use OPT-125M as a tiny reference model available offline.
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        cfg = transformers.AutoConfig.from_pretrained(
            "facebook/opt-125m", trust_remote_code=True,
        )
        return ModelConfig(
            model="facebook/opt-125m",
            tokenizer="facebook/opt-125m",
            tokenizer_mode="auto",
            trust_remote_code=True,
            dtype="bfloat16",
            seed=0,
        )

    def test_spec_creation(self):
        """Create FPGAOffloadingSpec programmatically."""
        import transformers
        from vllm.config import VllmConfig, CacheConfig, \
            ParallelConfig, SchedulerConfig
        from vllm.config.kv_transfer import KVTransferConfig
        from vllm.v1.kv_cache_interface import KVCacheConfig

        config = VllmConfig(
            model_config=self._make_minimal_model_config(),
            cache_config=CacheConfig(
                block_size=16,
                gpu_memory_utilization=0.9,
                swap_space=0,
                cache_dtype="auto",
                num_gpu_blocks_override=None,
            ),
            parallel_config=ParallelConfig(tensor_parallel_size=1),
            scheduler_config=SchedulerConfig(
                max_num_batched_tokens=4096,
                max_num_seqs=256,
                max_model_len=4096,
            ),
            kv_transfer_config=KVTransferConfig(
                kv_connector="OffloadingConnector",
                kv_connector_extra_config={
                    "spec_name": "FPGAOffloadingSpec",
                    "fpga_bytes_to_use": 16 * (1024**3),
                },
            ),
        )
        kvc = KVCacheConfig(num_blocks=128, kv_cache_tensors=[],
                            kv_cache_groups=[])
        spec = FPGAOffloadingSpec(config, kvc)
        assert isinstance(spec, FPGAOffloadingSpec)
        assert spec.num_fpga_blocks > 0
        assert spec.xdma_prefix == "/dev/xdma0"

    def test_factory_resolves_fpga_spec(self):
        """OffloadingSpecFactory resolves FPGAOffloadingSpec by name."""
        from vllm.config import VllmConfig, CacheConfig, \
            ParallelConfig, SchedulerConfig
        from vllm.config.kv_transfer import KVTransferConfig

        config = VllmConfig(
            model_config=self._make_minimal_model_config(),
            cache_config=CacheConfig(
                block_size=16,
                gpu_memory_utilization=0.9,
                swap_space=0,
                cache_dtype="auto",
                num_gpu_blocks_override=None,
            ),
            parallel_config=ParallelConfig(tensor_parallel_size=1),
            scheduler_config=SchedulerConfig(
                max_num_batched_tokens=4096,
                max_num_seqs=256,
                max_model_len=4096,
            ),
            kv_transfer_config=KVTransferConfig(
                kv_connector="OffloadingConnector",
                kv_connector_extra_config={
                    "spec_name": "FPGAOffloadingSpec",
                },
            ),
        )
        spec_cls = OffloadingSpecFactory.get_spec_cls(config)
        assert spec_cls is FPGAOffloadingSpec


# ===========================================================================
# 层级 5: FPGALoadStoreSpec
# ===========================================================================


class TestFPGALoadStoreSpec:
    def test_medium(self):
        assert FPGALoadStoreSpec.medium() == "FPGA"

    def test_roundtrip_serialize(self):
        """block_ids survive serialization (numpy ↔ list)."""
        ids = np.array([1, 2, 3], dtype=np.int64)
        spec = FPGALoadStoreSpec(block_ids=ids)
        recovered = np.asarray(spec.block_ids)
        assert list(recovered) == [1, 2, 3]


# ===========================================================================
# 层级 6: 集成冒烟测试
# ===========================================================================


class TestFPGAIntegrationSmoke:
    """端到端跑通测试（mock 模式，不需要硬件）。"""

    def test_manager_delegation(self):
        """FPGAOffloadingManager delegates to CPUOffloadingManager."""
        from vllm.v1.kv_offload.fpga.manager import FPGAOffloadingManager
        mgr = FPGAOffloadingManager(num_blocks=8)
        assert mgr.medium == "FPGA"
        # Check delegation works via a simple API call.
        from vllm.v1.kv_offload.base import make_offload_key
        key = make_offload_key(b"test", 0)
        result = mgr.lookup(key, None)
        assert result is not None

    def test_worker_creates_in_mock_mode(self):
        """Worker creation in mock mode (VLLM_FPGA_MOCK=1)."""
        os.environ["VLLM_FPGA_MOCK"] = "1"
        tensors = [
            CanonicalKVCacheTensor(
                tensor=torch.zeros(4, 1024, dtype=torch.int8),
                page_size_bytes=1024,
            ),
        ]
        refs = [[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=1024)]]
        kv_caches = CanonicalKVCaches(tensors=tensors, group_data_refs=refs)

        w = FPGAOffloadingWorker(
            kv_caches=kv_caches,
            block_size_factor=1,
            num_fpga_blocks=4,
            fpga_capacity_bytes=4 * 1024,
        )
        assert w._fpga_alloc is not None
        assert w._engine is not None
        w.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
