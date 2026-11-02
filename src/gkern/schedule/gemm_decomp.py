"""GEMM 的四种工作分解：naive / split-K / stream-K / persistent。

同一个矩阵乘，硬件上有多种「把工作切开再铺到 SM 上」的方式，它们的差别不是
数学，而是**最后一波有多少 SM 在空转**，以及**为了填满这些 SM 要多写多少字节**。
本模块把这两笔账都算成整数，让选择有据可依而不是靠感觉。

记：输出切成 ``tiles = ceil(M/bm) × ceil(N/bn)`` 个 tile，每个 tile 要做
``k_iters = K / bk`` 次 k 方向的累加。于是「工作单元」有两种粒度：

- **以 tile 为单位**（naive / split-K / persistent）：一共 ``tiles``（或
  ``tiles × splits``）个单元，每个单元内部还有 ``k_iters`` 次循环。
- **以 k-iter 为单位**（stream-K）：一共 ``tiles × k_iters`` 个单元，粒度细 33 倍，
  于是 tail 从「几十个 tile」缩到「几十个 iter」。

四种策略的真实差别（也是本模块要证明的）：

| 策略 | 单元数 | tail 改善 | 额外流量 |
|---|---|---|---|
| naive | tiles | — | 0 |
| split-K(σ) | tiles×σ | 有，但 σ 只能取整数 | 每个 tile 每个 split 一份部分和（写+读） |
| stream-K | tiles×k_iters | 最好（粒度最细） | 只有跨 worker 边界的部分和（capacity−1 份） |
| persistent | tiles | **没有** | 0 |

persistent 常被误认为能治 tail —— 它治的是**块调度开销**（每波重新派发 block），
而 tail 是并行度用不满，换谁派发都一样。本模型里它与 naive 的波指标完全相同，
这是诚实的边界，不是没实现。

两笔账都要算的原因：split-K 的额外流量在宽扁大矩阵上会被计算时间完全盖住
（所以它在那儿是免费的），而在 K 很深、输出很小的形状上，stream-K 同时赢在
利用率与额外流量上 —— **结论随形状翻转**，这正是本模块存在的意义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..gpgpu.device import DeviceInfo
from .wave import WaveStats, wave_stats

__all__ = ["GemmShape", "StrategyRow", "StreamKUnit", "compare_strategies",
           "recommend", "streamk_units"]

BYTES_PER_ELEMENT = 4  # 本模型只算 fp32 累加器与 fp32 矩阵元素


@dataclass(frozen=True)
class GemmShape:
    """一个矩阵乘的几何：``(M,K) @ (K,N)``，切成 ``bm × bn`` 的输出 tile、``bk`` 的 k 步长。

    ``K % bk != 0`` 直接拒绝：本仓库的 IR 是把 k 循环展开成线性指令的，
    尾部不足一步的 tile 需要额外掩码处理，语义会与「完整 bk 步」不同 ——
    与其在调度模型里含糊算一个近似值，不如要求调用方给出整除的形状。
    """

    M: int
    N: int
    K: int
    bm: int
    bn: int
    bk: int

    def __post_init__(self) -> None:
        for name in ("M", "N", "K", "bm", "bn", "bk"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} 必须为正，收到 {value}")
        if self.K % self.bk != 0:
            raise ValueError(
                f"K={self.K} 必须是 bk={self.bk} 的整数倍（本 IR 展开 k 循环，不支持残余步）"
            )

    @property
    def tiles_m(self) -> int:
        return -(-self.M // self.bm)

    @property
    def tiles_n(self) -> int:
        return -(-self.N // self.bn)

    @property
    def tiles(self) -> int:
        """输出 tile 总数 = 以 tile 为粒度的单元数（naive / persistent）。"""
        return self.tiles_m * self.tiles_n

    @property
    def k_iters(self) -> int:
        return self.K // self.bk

    @property
    def iter_units(self) -> int:
        """以 k-iter 为粒度的单元总数（stream-K 用）。"""
        return self.tiles * self.k_iters

    @property
    def flops(self) -> int:
        return 2 * self.M * self.N * self.K

    @property
    def base_bytes(self) -> int:
        """A + B + C 各读/写一次的理想流量（不计任何部分和）。"""
        return (self.M * self.K + self.K * self.N + self.M * self.N) * BYTES_PER_ELEMENT

    def as_dict(self) -> dict:
        return {
            "M": self.M, "N": self.N, "K": self.K,
            "bm": self.bm, "bn": self.bn, "bk": self.bk,
            "tiles": self.tiles, "k_iters": self.k_iters,
            "iter_units": self.iter_units,
            "flops": self.flops,
            "base_bytes": self.base_bytes,
        }


@dataclass(frozen=True)
class StrategyRow:
    """一种分解策略的账：单元数、波次、额外流量、折算后的预测时间。"""

    strategy: str
    units: int
    capacity: int
    waves: int
    efficiency: float
    tail_units: int
    partial_cuts: int
    extra_bytes: int
    total_bytes: int
    launches: int
    predicted_ms: float
    note: str

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "units": self.units,
            "capacity": self.capacity,
            "waves": self.waves,
            "efficiency": round(self.efficiency, 6),
            "tail_units": self.tail_units,
            "partial_cuts": self.partial_cuts,
            "extra_bytes": self.extra_bytes,
            "total_bytes": self.total_bytes,
            "launches": self.launches,
            "predicted_ms": round(self.predicted_ms, 6),
            "note": self.note,
        }


@dataclass(frozen=True)
class StreamKUnit:
    """stream-K 划分出来的一段工作：某个 worker 负责某个 tile 的某段 k 区间。

    ``k_start`` / ``k_end`` 是 **tile 内**的迭代下标（0 ≤ start < end ≤ k_iters）。
    一段区间落在 tile 边界之内意味着这个 tile 被多个 worker 先后累加，
    于是要落一份部分和——这正是 stream-K 的全部代价所在。
    """

    worker: int
    tile: int
    k_start: int
    k_end: int


def streamk_units(shape: GemmShape, device: DeviceInfo,
                  blocks_per_sm: int = 1) -> list[StreamKUnit]:
    """把 ``iter_units`` 次 k-迭代连续切给 ``capacity`` 个 worker，返回具体分段。

    切法是「尽量均分」：``base, rem = divmod(total, workers)``，前 ``rem`` 个 worker
    各多拿一次迭代。越界不能靠猜——``sum(k_end - k_start)`` 必须等于 ``iter_units``，
    每个 tile 的区间必须首尾相接（`tests/test_schedule.py` 两条断言钉住）。
    """
    if blocks_per_sm <= 0:
        raise ValueError(f"blocks_per_sm 必须为正，收到 {blocks_per_sm}")

    capacity = device.sm_count * blocks_per_sm
    total = shape.iter_units
    workers = min(capacity, total)
    base, rem = divmod(total, workers)

    units: list[StreamKUnit] = []
    cursor = 0
    for worker in range(workers):
        span = base + (1 if worker < rem else 0)
        end = cursor + span
        while cursor < end:
            tile = cursor // shape.k_iters
            tile_base = tile * shape.k_iters
            seg_end = min(end, tile_base + shape.k_iters)
            units.append(StreamKUnit(worker=worker, tile=tile,
                                     k_start=cursor - tile_base,
                                     k_end=seg_end - tile_base))
            cursor = seg_end
    return units


def streamk_partials(shape: GemmShape, device: DeviceInfo, blocks_per_sm: int = 1) -> int:
    """数一数真实划分里要落几份部分和：每个 tile 上「多出来的 worker」各一份。

    不能直接用 ``capacity - 1``：只有**落在 tile 内部的** worker 边界才需要部分和，
    对齐到 tile 边界的切分不需要。所以这个数必须从 :func:`streamk_units` 数出来，
    而不是用闭式公式估——公式和划分各算各的，迟早会漂。
    """
    owners: dict[int, set[int]] = {}
    for unit in streamk_units(shape, device, blocks_per_sm=blocks_per_sm):
        owners.setdefault(unit.tile, set()).add(unit.worker)
    return sum(len(workers) - 1 for workers in owners.values())


def _predicted_ms(shape: GemmShape, device: DeviceInfo, efficiency: float,
                  total_bytes: int) -> float:
    """把利用率折进计算时间，再与访存时间取大者。

    tail 期间 SM 空转 → 有效算力按 ``efficiency`` 打折，这是本模型唯一的
    「tail 很贵」的机制；反过来，额外流量只在访存侧加时间，不会减少计算。
    """
    compute_ms = shape.flops / (device.peak_flops_tflops * 1e12) * 1e3
    memory_ms = total_bytes / (device.theoretical_bandwidth_gbps * 1e9) * 1e3
    return max(compute_ms / efficiency, memory_ms)


def _row(shape: GemmShape, device: DeviceInfo, strategy: str, stats: WaveStats,
         extra_bytes: int, launches: int, partial_cuts: int, note: str) -> StrategyRow:
    total = shape.base_bytes + extra_bytes
    return StrategyRow(
        strategy=strategy,
        units=stats.units,
        capacity=stats.capacity,
        waves=stats.waves,
        efficiency=stats.efficiency,
        tail_units=stats.units - (stats.waves - 1) * stats.capacity
        if stats.waves > 1 else stats.units % stats.capacity,
        partial_cuts=partial_cuts,
        extra_bytes=extra_bytes,
        total_bytes=total,
        launches=launches,
        predicted_ms=_predicted_ms(shape, device, stats.efficiency, total),
        note=note,
    )


def compare_strategies(shape: GemmShape, device: DeviceInfo,
                       splits: int = 2, blocks_per_sm: int = 1) -> Dict[str, dict]:
    """四种分解并排：返回 ``{策略名: 账本字典}``。

    ``blocks_per_sm`` 是每个 SM 同时驻留几个 block（寄存器/共享内存受限时常常就是 1），
    它决定 ``capacity = sm_count × blocks_per_sm``，也就是「一波能铺多少单元」。
    ``splits`` 只影响 split-K。
    """
    if splits <= 0:
        raise ValueError(f"splits 必须为正，收到 {splits}")
    if blocks_per_sm <= 0:
        raise ValueError(f"blocks_per_sm 必须为正，收到 {blocks_per_sm}")

    capacity = device.sm_count * blocks_per_sm
    rows: Dict[str, StrategyRow] = {}

    # ① naive：一个 tile 一个 block，k 方向自己循环
    rows["naive"] = _row(
        shape, device, "naive", wave_stats(shape.tiles, capacity),
        extra_bytes=0, launches=1, partial_cuts=0,
        note="最朴素：grid = tile 数，最后一波用不满就空着",
    )

    # ② split-K：把 K 切成 splits 段，每段独立算一部分和，最后再归约
    split_units = shape.tiles * splits
    partial_bytes = shape.tiles * splits * shape.bm * shape.bn * BYTES_PER_ELEMENT
    rows["split-k"] = _row(
        shape, device, "split-k", wave_stats(split_units, capacity),
        extra_bytes=2 * partial_bytes,            # 写部分和 + 归约时读回
        launches=2,                               # GEMM + 归约两个 kernel
        partial_cuts=split_units,
        note=f"K 切成 {splits} 段：单元数 ×{splits}，每个 split 落一份部分和",
    )

    # ③ stream-K：以 k-iter 为单元，把迭代区间连续切给每个 worker
    iter_stats = wave_stats(shape.iter_units, capacity)
    # 部分和份数从真实划分里数出来（只有落在 tile 内部的边界才要落），不是 capacity-1
    cuts = streamk_partials(shape, device, blocks_per_sm=blocks_per_sm)
    rows["stream-k"] = _row(
        shape, device, "stream-k", iter_stats,
        extra_bytes=2 * cuts * shape.bm * shape.bn * BYTES_PER_ELEMENT,
        launches=2,                               # 主 kernel + fixup
        partial_cuts=cuts,
        note="以 k-iter 为单元连续切分：tail 最小，只有边界需要部分和",
    )

    # ④ persistent：常驻 block 循环取任务，波指标与 naive 相同
    naive_stats = wave_stats(shape.tiles, capacity)
    rows["persistent"] = _row(
        shape, device, "persistent", naive_stats,
        extra_bytes=0, launches=1, partial_cuts=0,
        note="常驻块循环取任务：省的是块调度开销，tail 与 naive 完全一样",
    )

    return {name: row.as_dict() for name, row in rows.items()}


def recommend(shape: GemmShape, device: DeviceInfo, splits: int = 2,
              blocks_per_sm: int = 1) -> str:
    """按 ``predicted_ms`` 选一个策略名。平局时按声明顺序（naive 优先）。

    「预测时间」是**模型时间**，不是实测时间：它只含「计算 ÷ 利用率」与
    「字节 ÷ 带宽」两项，不含发射延迟、占用率爬坡、L2 命中。见
    `docs/ARCHITECTURE.md` §8.2 的已知缺口清单。
    """
    rows = compare_strategies(shape, device, splits=splits, blocks_per_sm=blocks_per_sm)
    order = ["naive", "split-k", "stream-k", "persistent"]
    return min(order, key=lambda name: (rows[name]["predicted_ms"], order.index(name)))
