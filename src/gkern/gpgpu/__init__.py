"""gpgpu 子包：在纯 Python 上模拟国产 GPGPU 的执行模型。

模块划分：
- ``device``   设备/网格/线程层级（grid -> block -> warp -> lane）
- ``warp``     SIMT 执行单元：分支发散、活跃掩码、lane 内并行
- ``memory``   访存体系：全局内存 + 共享内存 + bank 冲突统计
- ``kernel``   用户侧装饰器与启动入口
"""

from .kernel import KernelMeta, LaunchResult, WarpContext, cdiv, kernel, launch
from .device import Device, DeviceInfo
from .warp import WarpTrace, DivergenceReport
from .memory import GlobalMemory, SharedMemory, MemoryStats

__all__ = [
    "kernel",
    "KernelMeta",
    "LaunchResult",
    "WarpContext",
    "cdiv",
    "launch",
    "Device",
    "DeviceInfo",
    "WarpTrace",
    "DivergenceReport",
    "GlobalMemory",
    "SharedMemory",
    "MemoryStats",
]
