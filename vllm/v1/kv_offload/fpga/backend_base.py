"""DMA 传输后端抽象层 — vendor agnostic。

定义 DMABackend 接口，所有 FPGA 传输后端 (XDMA/P2P/CXL.mem) 均实现此接口。
BlockTransferEngine 通过此接口与具体的 FPGA 硬件解耦。
"""

from abc import ABC, abstractmethod

import torch

from vllm.v1.kv_offload.fpga.allocator import FPGABlockAllocator


class DMABackend(ABC):
    """DMA 传输后端抽象。

    一个 backend = 一种物理传输方式。
    实现此接口后即可被 BlockTransferEngine 使用。
    具体 FPGA 型号 (Xilinx/Intel/...) 和传输方式 (XDMA/P2P/CXL)
    被完全封装在实现类中。
    """

    def __init__(self, fpga_allocator: FPGABlockAllocator):
        self._fpga_alloc = fpga_allocator

    @abstractmethod
    def write(self, src_ptr: int, dst_addr: int, size: int) -> None:
        """Host/GPU → FPGA 数据传输。

        Args:
            src_ptr: 源数据的内存地址 (CPU 虚拟地址 或 GPU 设备指针)
            dst_addr: FPGA 侧目标地址 (由 get_block_addr 得到)
            size: 传输字节数
        """
        ...

    @abstractmethod
    def read(self, src_addr: int, dst_ptr: int, size: int) -> None:
        """FPGA → Host/GPU 数据传输。

        Args:
            src_addr: FPGA 侧源地址
            dst_ptr: 目标内存地址
            size: 传输字节数
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """backend 名称，用于日志和调试。"""
        ...

    def get_block_addr(self, block_id: int) -> int:
        """block_id → FPGA 侧地址。

        默认实现使用 FPGABlockAllocator 的地址映射。
        如果 backend 需要特殊映射，可覆盖此方法。
        """
        return self._fpga_alloc.get_block_offset(block_id)

    def shutdown(self) -> None:
        """释放 backend 资源。默认无操作。"""
        pass
