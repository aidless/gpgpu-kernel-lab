"""波量化（wave quantization）：把 N 个工作单元铺到 C 个执行槽上，最后剩什么。

GPU 上的「波」不是一个调度单位，而是一个**硬件事实**：一个 SM 同时最多驻留若干
block（由寄存器/共享内存/线程数决定），grid 里超出的 block 只能排队等下一轮。
于是「一共能同时跑多少个 block」= `sm_count × blocks_per_sm`，这就是本模块的
`capacity`；而 `units` 是这次的网格里有多少个 block。

100 个 block 铺到 40 个槽上：40 + 40 + 20 —— 最后一波有一半 SM 在空转，
整体利用率 100/120 ≈ 0.83。矩阵乘里这个效应最刺眼的地方在**输出很小、K 很深**
的形状：128×128 的 tile 做一个 256×256 的输出只有 4 个 block，铺 40 个 SM 上
就是 0.1 的利用率，40 个槽里 36 个闲着。

本模块只算这笔账，不算时间；时间在 :mod:`gkern.schedule.gemm_decomp` 里按
「计算时间 ÷ 利用率」折算。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["WaveStats", "wave_stats"]


@dataclass(frozen=True)
class WaveStats:
    """一次网格发射的波次账。

    - ``waves``：需要几轮才能把 ``units`` 个单元都跑完
    - ``last_wave_units``：最后一轮真正在跑的单元数
    - ``idle_slots_last_wave``：最后一轮空着的槽位数
    - ``efficiency``：``units / (waves × capacity)``，即平均槽位占用率
    """

    units: int
    capacity: int
    waves: int
    last_wave_units: int
    idle_slots_last_wave: int

    @property
    def efficiency(self) -> float:
        return self.units / float(self.waves * self.capacity)

    @property
    def wasted_fraction(self) -> float:
        return self.idle_slots_last_wave / float(self.waves * self.capacity)

    def as_dict(self) -> dict:
        return {
            "units": self.units,
            "capacity": self.capacity,
            "waves": self.waves,
            "last_wave_units": self.last_wave_units,
            "idle_slots_last_wave": self.idle_slots_last_wave,
            "efficiency": round(self.efficiency, 6),
            "wasted_fraction": round(self.wasted_fraction, 6),
        }


def wave_stats(units: int, capacity: int) -> WaveStats:
    """把 ``units`` 个单元铺到 ``capacity`` 个槽上，算出波次与最后一波的浪费。

    ``capacity`` 是「同时能驻留多少个 block」= ``sm_count × blocks_per_sm``。
    两者都必须为正：单元数为 0 说明调度问题本身没定义，容量为 0 说明硬件假设错了，
    两种都是调用方的 bug，直接抛错而不是静默返回 0 利用率。
    """
    if units <= 0:
        raise ValueError(f"units 必须为正，收到 {units}")
    if capacity <= 0:
        raise ValueError(f"capacity 必须为正，收到 {capacity}")

    waves = -(-units // capacity)          # ceil
    last = units - (waves - 1) * capacity  # 最后一波实际单元数
    return WaveStats(
        units=units,
        capacity=capacity,
        waves=waves,
        last_wave_units=last,
        idle_slots_last_wave=capacity - last,
    )
