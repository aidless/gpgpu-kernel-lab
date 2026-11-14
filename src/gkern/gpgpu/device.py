"""设备与线程层级建模。

真实 GPGPU 的执行层级是 grid -> block(CTA) -> warp -> lane(thread)。本模块把这个
层级做成可枚举的对象，使「一个 warp 里 32 个 lane 同时执行同一条指令」这件事在
Python 里能被真正展开，而不是被当成一次标量调用。

设计要点：
1. 所有层级都是**不可变数据 + 显式索引**，便于在报告里精确定位到
   ``block=3, warp=1, lane=7`` 这样的坐标；
2. 设备规格（SM 数、warp 大小、共享内存容量、寄存器文件大小）集中在
   :class:`DeviceInfo`，算子与编译器只读它，不硬编码 32/64 这类魔数；
3. 设备对象本身不执行任何计算，只负责「把 grid 切成 block、把 block 切成 warp」，
   执行语义在 :mod:`gkern.gpgpu.warp` 与 :mod:`gkern.gpgpu.kernel`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Sequence


@dataclass(frozen=True)
class DeviceInfo:
    """设备规格。默认值取自一个典型的国产通用 GPU 加速卡（SIMT 结构）。"""

    name: str = "generic-domestic-gpgpu"
    sm_count: int = 64                 # 流式多处理器个数
    warp_size: int = 32                # 一个 warp 的 lane 数
    max_warps_per_sm: int = 32         # 每个 SM 最多驻留的 warp
    shared_memory_per_block: int = 48 * 1024   # 每 block 可用共享内存（字节）
    registers_per_sm: int = 64 * 1024  # 每 SM 寄存器文件（32bit 个）
    l2_bytes: int = 4 * 1024 * 1024
    memory_clock_ghz: float = 1.6
    bus_width_bits: int = 256
    fp32_cores_per_sm: int = 128       # 每 SM 的 FP32 单元数
    clock_ghz: float = 1.6             # 核心频率

    @classmethod
    def tesla_t4(cls) -> "DeviceInfo":
        """NVIDIA T4 的等效参数（用于把代价模型与云端实测放在同一把尺上）。

        T4 的公开规格：40 个 SM、Tensor Core 版 Turing、显存 16 GB（可用 15 GB 左右）、
        显存带宽 320 GB/s。这里把 ``memory_clock_ghz`` 与 ``bus_width_bits`` 取成
        「等效值」——本模型只用到两者的乘积，所以选 ``5.0 GHz × 256 bit`` 使
        ``theoretical_bandwidth_gbps`` 恰好等于 320.0，而不是去还原物理时钟。

        FP32 峰值按本模型公式 ``sm_count × fp32_cores_per_sm × 2 × clock_ghz`` 算得
        8.141 TFLOPS，与 T4 公开的 8.1 TFLOPS 相符（T4 每个 SM 64 个 FP32 单元、
        主频 1.59 GHz），roofline 交叉点因此是 25.44 FLOP/B——比假设国产卡的 256.0
        低一个数量级。这正是「同一份 kernel 在不同卡上受限类型可能相反」的来源。
        """
        return cls(
            name="nvidia-tesla-t4",
            sm_count=40,
            max_warps_per_sm=48,
            shared_memory_per_block=48 * 1024,
            registers_per_sm=64 * 1024,
            l2_bytes=4 * 1024 * 1024,
            memory_clock_ghz=5.0,
            bus_width_bits=256,
            fp32_cores_per_sm=64,
            clock_ghz=1.59,
        )

    @property
    def threads_per_block_limit(self) -> int:
        """一个 block 的线程数上限（由每 SM warp 上限推出）。"""
        return self.max_warps_per_sm * self.warp_size

    @property
    def peak_flops_tflops(self) -> float:
        """FP32 峰值算力（TFLOPS）= SM 数 × 每 SM FP32 单元 × 2(FMA) × 频率。"""
        return (self.sm_count * self.fp32_cores_per_sm * 2 * self.clock_ghz) / 1000.0

    @property
    def theoretical_bandwidth_gbps(self) -> float:
        """理论显存带宽 = 时钟 × 2(DDR) × 位宽/8。"""
        return self.memory_clock_ghz * 2.0 * (self.bus_width_bits / 8.0)


@dataclass(frozen=True)
class BlockCoord:
    """block 在 grid 中的三维坐标。"""

    x: int
    y: int = 0
    z: int = 0

    def __str__(self) -> str:  # pragma: no cover - 仅用于日志
        return f"block({self.x},{self.y},{self.z})"


@dataclass(frozen=True)
class ThreadCoord:
    """线程在 block 内的坐标，并派生出它属于哪个 warp、哪个 lane。"""

    block: BlockCoord
    tid: int          # block 内线性线程号
    warp_size: int = 32

    @property
    def warp_in_block(self) -> int:
        return self.tid // self.warp_size

    @property
    def lane(self) -> int:
        return self.tid % self.warp_size

    def __str__(self) -> str:  # pragma: no cover - 仅用于日志
        return f"{self.block}/warp{self.warp_in_block}/lane{self.lane}"


class Device:
    """设备：负责把 grid 切成 block，把 block 切成 warp。

    ``Device`` 只做划分，不做执行。它存在的意义是让「发射一个 kernel」这件事
    有一个明确的对象边界，方便后续接入真实后端（PrivateUse1 / 自定义 C++ 扩展）。
    """

    def __init__(self, info: DeviceInfo | None = None) -> None:
        self.info = info or DeviceInfo()

    # ---- 划分 -----------------------------------------------------------
    def iter_blocks(self, grid: Sequence[int]) -> Iterator[BlockCoord]:
        """按 CUDA 的行优先顺序枚举 grid 里的所有 block。"""
        gx, gy, gz = _as_triple(grid)
        for z in range(gz):
            for y in range(gy):
                for x in range(gx):
                    yield BlockCoord(x, y, z)

    def iter_threads(
        self, block: BlockCoord, block_dim: Sequence[int]
    ) -> Iterator[ThreadCoord]:
        """枚举一个 block 内的全部线程。"""
        bx, by, bz = _as_triple(block_dim)
        total = bx * by * bz
        for tid in range(total):
            yield ThreadCoord(block=block, tid=tid, warp_size=self.info.warp_size)

    def warps_per_block(self, block_dim: Sequence[int]) -> int:
        bx, by, bz = _as_triple(block_dim)
        total = bx * by * bz
        return (total + self.info.warp_size - 1) // self.info.warp_size

    def occupancy(self, block_dim: Sequence[int]) -> float:
        """占用率估计：驻留 warp / 每 SM warp 上限。"""
        warps = self.warps_per_block(block_dim)
        return min(1.0, warps / float(self.info.max_warps_per_sm))

    def validate(self, grid: Sequence[int], block_dim: Sequence[int]) -> List[str]:
        """发射前的合法性检查，返回问题列表（空列表表示可以发射）。"""
        problems: List[str] = []
        bx, by, bz = _as_triple(block_dim)
        total = bx * by * bz
        if total <= 0:
            problems.append(f"block_dim 非法：{block_dim}")
        if total > self.info.threads_per_block_limit:
            problems.append(
                f"block 线程数 {total} 超过上限 {self.info.threads_per_block_limit}"
            )
        for axis, val in zip("xyz", _as_triple(grid)):
            if val <= 0:
                problems.append(f"grid.{axis} 非法：{val}")
        return problems


def _as_triple(dims: Sequence[int]) -> tuple[int, int, int]:
    """把 1/2/3 维尺寸补成三元组。"""
    if isinstance(dims, int):
        dims = (dims,)
    seq = tuple(int(d) for d in dims)
    if len(seq) == 1:
        return seq[0], 1, 1
    if len(seq) == 2:
        return seq[0], seq[1], 1
    if len(seq) == 3:
        return seq[0], seq[1], seq[2]
    raise ValueError(f"维度必须是 1~3 维，收到 {len(seq)} 维")
