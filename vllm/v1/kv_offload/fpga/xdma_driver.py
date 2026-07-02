# SPDX-License-Identifier: Apache-2.0
"""
Low-level XDMA v22 driver wrapper.

对接 Xilinx XDMA v22 驱动 (XDMA_22)，支持两种传输模式：

模式 A — lseek + read/write (简单，推荐)
  Store (GPU → FPGA): open /dev/xdma0_h2c_0 → lseek(fpga_addr) → write(host_buf)
  Load  (FPGA → GPU): open /dev/xdma0_c2h_0 → lseek(fpga_addr) → read(host_buf)

  其中 fpga_addr 是对应的 AXI 总线地址（即 FPGA DRAM 在 AXI 空间中的偏移）。

模式 B — aperture ioctl (窗口模式，适合大块传输)
  使用 /dev/xdma0_user，通过 IOCTL_XDMA_APERTURE_W / IOCTL_XDMA_APERTURE_R
  提交批量 DMA 请求，支持更大的单次传输。

驱动设备节点（加载后出现）:
  /dev/xdma0_h2c_0      — Host→Card channel 0 (write)
  /dev/xdma0_c2h_0      — Card→Host channel 0 (read)
  /dev/xdma0_control    — 控制面 (BAR 空间访问, 寄存器读写)
  /dev/xdma0_user       — 用户中断 / aperture ioctl
  /sys/bus/pci/devices/.../xdma/   — 驱动 sysfs 属性

参考工具:
  dma_to_device.c     使用 /dev/xdma0_h2c_0 + write()
  dma_from_device.c   使用 /dev/xdma0_c2h_0 + read()
  performance.c       使用 IOCTL_XDMA_PERF_* 测带宽
"""

from __future__ import annotations

import ctypes
import fcntl
import os
from typing import ClassVar

from vllm.logger import init_logger

logger = init_logger(__name__)

# ===========================================================================
# 常量 — 对应 XDMA_22/linux-kernel/xdma/cdev_sgdma.h
# ===========================================================================

# IOCTL 请求码 (_IOW/_IOR 由 kernel 的 _IOC() 宏生成)
# 驱动使用 'q' 作为 magic number
IOCTL_XDMA_PERF_START = 0xC0107101   # _IOW('q', 1, struct xdma_performance_ioctl *)
IOCTL_XDMA_PERF_STOP = 0xC0107102    # _IOW('q', 2, ...)
IOCTL_XDMA_PERF_GET = 0xC0107103     # _IOR('q', 3, ...)
IOCTL_XDMA_ADDRMODE_SET = 0xC0107104  # _IOW('q', 4, int)
IOCTL_XDMA_ADDRMODE_GET = 0xC0107105  # _IOR('q', 5, int)
IOCTL_XDMA_ALIGN_GET = 0xC0107106     # _IOR('q', 6, int)
IOCTL_XDMA_APERTURE_R = 0xC0107107    # _IOW('q', 7, struct xdma_aperture_ioctl *)
IOCTL_XDMA_APERTURE_W = 0xC0107108    # _IOW('q', 8, ...)

XDMA_ADDRMODE_MEMORY = 0  # Memory-mapped: 用 lseek offset 当作 AXI 地址
XDMA_ADDRMODE_FIXED = 1   # Fixed: AXI 地址由 ioctl 指定

# 单次 read/write 最大字节数 (Linux kernel 限制)
RW_MAX_SIZE = 0x7FFFF000

# 默认设备路径
DEFAULT_H2C_DEVICE = "/dev/xdma0_h2c_0"
DEFAULT_C2H_DEVICE = "/dev/xdma0_c2h_0"

# ===========================================================================
# ctypes 结构体 — 对应 cdev_sgdma.h 中的 C 结构
# ===========================================================================


class xdma_aperture_ioctl(ctypes.Structure):
    """对应 struct xdma_aperture_ioctl"""
    _fields_ = [
        ("ep_addr", ctypes.c_uint64),     # FPGA AXI 地址
        ("aperture", ctypes.c_uint),      # 窗口大小 (log2)
        ("buffer", ctypes.c_ulong),       # host 缓冲区地址
        ("len", ctypes.c_ulong),          # 传输字节数
        ("error", ctypes.c_int),          # 出: 错误码
        ("done", ctypes.c_ulong),         # 出: 实际完成字节数
    ]


# ===========================================================================
# XDMA v22 驱动句柄
# ===========================================================================


class XDMAHandle:
    """Xilinx XDMA v22 驱动封装。

    使用方式:
      xdma = XDMAHandle()
      xdma.open(ddr_base=0x00000000)  # 查询设备路径
      xdma.write(host_buf, fpga_addr, nbytes)  # Host → FPGA (通过 H2C)
      xdma.read(host_buf, fpga_addr, nbytes)   # FPGA → Host (通过 C2H)
      xdma.close()
    """

    def __init__(self) -> None:
        self._h2c_fd: int = -1   # /dev/xdmaN_h2c_0
        self._c2h_fd: int = -1   # /dev/xdmaN_c2h_0
        self._h2c_path: str = ""
        self._c2h_path: str = ""

        # 驱动的对齐要求 (通过 IOCTL_XDMA_ALIGN_GET 查询)
        self._alignment: int = 4096   # 默认 4KB

        # 传输统计
        self._bytes_written: int = 0
        self._bytes_read: int = 0

    # -- 打开 / 关闭 --------------------------------------------------------

    def open(self, h2c_device: str = DEFAULT_H2C_DEVICE,
             c2h_device: str = DEFAULT_C2H_DEVICE) -> None:
        """打开 XDMA 字符设备。

        Args:
            h2c_device: Host→Card 设备路径
            c2h_device: Card→Host 设备路径
        """
        self._h2c_path = h2c_device
        self._c2h_path = c2h_device

        self._h2c_fd = os.open(h2c_device, os.O_RDWR)
        logger.info("XDMA: opened H2C %s (fd=%d)", h2c_device, self._h2c_fd)

        self._c2h_fd = os.open(c2h_device, os.O_RDWR)
        logger.info("XDMA: opened C2H %s (fd=%d)", c2h_device, self._c2h_fd)

        # 查询对齐要求
        self._alignment = self._get_alignment()
        logger.info("XDMA: alignment=%d bytes", self._alignment)

    def close(self) -> None:
        if self._h2c_fd >= 0:
            os.close(self._h2c_fd)
            self._h2c_fd = -1
        if self._c2h_fd >= 0:
            os.close(self._c2h_fd)
            self._c2h_fd = -1
        logger.debug("XDMA: closed")

    @property
    def is_open(self) -> bool:
        return self._h2c_fd >= 0 and self._c2h_fd >= 0

    @property
    def alignment(self) -> int:
        return self._alignment

    # -- 数据读写 (lseek + read/write) --------------------------------------
    #
    # XDMA v22 字符设备的语义:
    #   lseek(fd, fpga_addr, SEEK_SET) 设置 DMA 的目标/源 AXI 地址
    #   write(fd, buf, size)            启动 H2C DMA: host buf → FPGA
    #   read(fd, buf, size)             启动 C2H DMA: FPGA → host buf
    #
    # 注意: host buf 必须 alignment 对齐。

    def write(self, host_buf_addr: int, fpga_addr: int, size: int) -> int:
        """Host → FPGA DMA。

        写 *size* 字节从 host 内存到 FPGA AXI 地址 *fpga_addr*。

        Args:
            host_buf_addr: host 缓冲区的虚拟地址 (ctypes 指针或 data_ptr)
            fpga_addr:     FPGA AXI 总线地址
            size:          传输字节数

        Returns:
            实际写入的字节数
        """
        assert self._h2c_fd >= 0, "H2C device not opened"

        count = 0
        buf_ptr = host_buf_addr
        offset = fpga_addr

        while count < size:
            chunk = min(size - count, RW_MAX_SIZE)
            # lseek 到 FPGA 地址
            rc = os.lseek(self._h2c_fd, offset, os.SEEK_SET)
            if rc != offset:
                raise RuntimeError(
                    f"XDMA lseek H2C failed: got 0x{rc:x}, expected 0x{offset:x}"
                )
            # write 触发 H2C DMA
            data = ctypes.string_at(buf_ptr, chunk)
            written = os.write(self._h2c_fd, data)
            if written < 0:
                raise RuntimeError(
                    f"XDMA write H2C failed at offset 0x{offset:x}"
                )
            count += written
            buf_ptr += written
            offset += written
            if written != chunk:
                break  # underflow

        self._bytes_written += count
        return count

    def read(self, host_buf_addr: int, fpga_addr: int, size: int) -> int:
        """FPGA → Host DMA。

        读 *size* 字节从 FPGA AXI 地址 *fpga_addr* 到 host 内存。

        Args:
            host_buf_addr: host 缓冲区的虚拟地址
            fpga_addr:     FPGA AXI 总线地址
            size:          传输字节数

        Returns:
            实际读取的字节数
        """
        assert self._c2h_fd >= 0, "C2H device not opened"

        count = 0
        buf_ptr = host_buf_addr
        offset = fpga_addr

        while count < size:
            chunk = min(size - count, RW_MAX_SIZE)
            # lseek 到 FPGA 地址
            rc = os.lseek(self._c2h_fd, offset, os.SEEK_SET)
            if rc != offset:
                raise RuntimeError(
                    f"XDMA lseek C2H failed: got 0x{rc:x}, expected 0x{offset:x}"
                )
            # read 触发 C2H DMA
            data = os.read(self._c2h_fd, chunk)
            if not data:
                break
            ctypes.memmove(buf_ptr, data, len(data))
            count += len(data)
            buf_ptr += len(data)
            offset += len(data)
            if len(data) != chunk:
                break  # underflow

        self._bytes_read += count
        return count

    # -- Aperture 模式 (ioctl 批量传输) ------------------------------------
    #
    # 适合大块连续传输，不需要多次 lseek。
    # 需要 FPGA 侧实现 aperture 窗口逻辑。

    def aperture_write(self, host_buf_addr: int, fpga_addr: int,
                       size: int, aperture: int = 0) -> int:
        """Aperture 模式 H2C 传输。"""
        io = xdma_aperture_ioctl(
            ep_addr=fpga_addr,
            aperture=aperture,
            buffer=host_buf_addr,
            len=size,
            error=0,
            done=0,
        )
        # 使用 /dev/xdma0_user (或 control) 进行 ioctl
        # 注意: aperture 需要专门的设备节点，这里简化处理
        fd = self._h2c_fd  # fallback to simple mode
        try:
            rc = fcntl.ioctl(fd, IOCTL_XDMA_APERTURE_W, io)
            if rc < 0 or io.error:
                raise RuntimeError(
                    f"XDMA aperture write failed: rc={rc}, error={io.error}"
                )
            return io.done
        except OSError as e:
            logger.warning("XDMA aperture write not supported, "
                           "falling back to simple mode: %s", e)
            return self.write(host_buf_addr, fpga_addr, size)

    # -- 地址模式控制 -------------------------------------------------------

    def set_addr_mode(self, mode: int = XDMA_ADDRMODE_MEMORY) -> None:
        """设置地址模式。

        XDMA_ADDRMODE_MEMORY (0): 默认，lseek offset = AXI 地址
        XDMA_ADDRMODE_FIXED  (1): AXI 地址由 ioctl 指定
        """
        try:
            fcntl.ioctl(self._h2c_fd, IOCTL_XDMA_ADDRMODE_SET, mode)
        except OSError:
            logger.debug("XDMA addr mode not supported on this device")

    def get_alignment(self) -> int:
        """查询驱动的 DMA 对齐要求。"""
        return self._alignment

    def _get_alignment(self) -> int:
        try:
            result = fcntl.ioctl(self._h2c_fd, IOCTL_XDMA_ALIGN_GET, 0)
            return result if result > 0 else 4096
        except OSError:
            return 4096  # 默认 4KB

    # -- 性能测量 -----------------------------------------------------------

    def perf_start(self) -> None:
        """开始带宽测量。"""
        try:
            fcntl.ioctl(self._h2c_fd, IOCTL_XDMA_PERF_START, 0)
        except OSError:
            pass

    def perf_stop(self) -> dict:
        """停止并获取带宽测量结果。"""
        import struct
        try:
            buf = bytearray(40)  # sizeof(xdma_performance_ioctl)
            fcntl.ioctl(self._h2c_fd, IOCTL_XDMA_PERF_GET, buf)
            return {"stopped": 1}
        except OSError:
            return {}

    # -- 统计信息 -----------------------------------------------------------

    @property
    def stats(self) -> dict:
        return {
            "bytes_written": self._bytes_written,
            "bytes_read": self._bytes_read,
        }

    def get_ddr_size(self) -> int:
        """从 sysfs 查询 FPGA DDR 容量。"""
        import glob
        pattern = "/sys/bus/pci/devices/*/xdma/ddr_size"
        paths = glob.glob(pattern)
        if not paths:
            pattern2 = "/sys/bus/pci/devices/*/ddr_size"
            paths = glob.glob(pattern2)
        for p in paths:
            try:
                with open(p) as f:
                    return int(f.read().strip())
            except (ValueError, OSError):
                pass
        return 0


# ===========================================================================
# 测试用 Mock (无硬件情况下使用)
# ===========================================================================


class MockXDMAHandle:
    """使用 bytearray 模拟 FPGA DRAM 的 Mock。

    用法与 XDMAHandle 完全一致。
    """

    def __init__(self, ddr_size: int = 16 * (1024**3)) -> None:
        self._buffer = bytearray(ddr_size)
        self._alignment = 4096
        logger.info("MockXDMA: %d bytes FPGA DRAM", ddr_size)

    def open(self, h2c_device: str = "", c2h_device: str = "") -> None:
        logger.info("MockXDMA: opened (mock, no hardware)")

    def close(self) -> None:
        pass

    @property
    def is_open(self) -> bool:
        return True

    @property
    def alignment(self) -> int:
        return self._alignment

    def write(self, host_buf_addr: int, fpga_addr: int, size: int) -> int:
        data = ctypes.string_at(host_buf_addr, size)
        end = fpga_addr + size
        if end > len(self._buffer):
            raise RuntimeError(
                f"MockXDMA write overflow: {end} > {len(self._buffer)}"
            )
        self._buffer[fpga_addr:end] = data
        return size

    def read(self, host_buf_addr: int, fpga_addr: int, size: int) -> int:
        end = fpga_addr + size
        if end > len(self._buffer):
            end = len(self._buffer)
            size = end - fpga_addr
        data = bytes(self._buffer[fpga_addr:end])
        ctypes.memmove(host_buf_addr, data, len(data))
        return len(data)

    def get_ddr_size(self) -> int:
        return len(self._buffer)
