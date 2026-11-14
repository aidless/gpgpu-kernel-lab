"""SIMT 执行单元：warp、活跃掩码、分支发散。

GPU 不是「32 个独立线程」在跑，而是**一条指令同时驱动 32 个 lane**。当同一个
warp 里的 lane 走进不同的分支，硬件不能并行执行两条路径，只能先跑其中一条
（把不满足条件的 lane 掩掉），再跑另一条。这就是分支发散（divergence），
它浪费的是**执行槽**而不是时间片。

本模块把这件事做成可测量的：

- 每条指令记一次 ``instruction``，消耗 32 个 lane 槽；
- 真正干活的是活跃 lane，记入 ``active_lane_cycles``；
- 走两次分支则指令数翻倍 → 发散效率自然下降。

因此 ``divergence_efficiency = 活跃lane槽 / 总lane槽`` 是这台模拟器的核心指标，
也是面试里「你的 kernel 有多少 warp 效率」这类问题的可计算版本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class WarpTrace:
    """一个 warp 的执行轨迹。"""

    warp_id: str
    warp_size: int = 32
    instructions: int = 0          # 发出的指令条数（含发散后重复发出的）
    active_lane_cycles: int = 0    # 活跃 lane 槽总数
    diverged_branches: int = 0     # 发生发散的 branch 次数
    branch_count: int = 0          # branch 总数

    @property
    def issued_lane_slots(self) -> int:
        return self.instructions * self.warp_size

    @property
    def efficiency(self) -> float:
        """warp 执行效率 = 有效 lane 槽 / 发射 lane 槽。"""
        if self.issued_lane_slots == 0:
            return 1.0
        return self.active_lane_cycles / float(self.issued_lane_slots)

    @property
    def wasted_lane_slots(self) -> int:
        return self.issued_lane_slots - self.active_lane_cycles


@dataclass
class DivergenceReport:
    """整个 kernel 的发散汇总。"""

    warps: int = 0
    instructions: int = 0
    active_lane_cycles: int = 0
    diverged_branches: int = 0
    branch_count: int = 0

    def add(self, trace: WarpTrace) -> None:
        self.warps += 1
        self.instructions += trace.instructions
        self.active_lane_cycles += trace.active_lane_cycles
        self.diverged_branches += trace.diverged_branches
        self.branch_count += trace.branch_count

    @property
    def efficiency(self) -> float:
        slots = self.instructions * 32
        if slots == 0:
            return 1.0
        return self.active_lane_cycles / float(slots)

    @property
    def divergence_rate(self) -> float:
        if self.branch_count == 0:
            return 0.0
        return self.diverged_branches / float(self.branch_count)

    def as_dict(self) -> dict:
        return {
            "warps": self.warps,
            "instructions": self.instructions,
            "active_lane_cycles": self.active_lane_cycles,
            "efficiency": round(self.efficiency, 4),
            "branch_count": self.branch_count,
            "diverged_branches": self.diverged_branches,
            "divergence_rate": round(self.divergence_rate, 4),
        }


class Warp:
    """一个 warp 的执行上下文。

    lane 数据是 numpy 数组（长度 = warp_size），因此「同一条指令作用在 32 个 lane
    上」是真的一次向量化执行，而不是 32 次标量调用。
    """

    def __init__(self, warp_id: str, size: int = 32) -> None:
        self.warp_id = warp_id
        self.size = size
        self.mask = np.ones(size, dtype=bool)
        self.trace = WarpTrace(warp_id=warp_id, warp_size=size)

    # ---- 指令 -----------------------------------------------------------
    def op(self, fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None) -> None:
        """记一条指令。有 fn 时在活跃 lane 上执行它。"""
        self.trace.instructions += 1
        self.trace.active_lane_cycles += int(self.mask.sum())

    def elementwise(self, fn: Callable[[np.ndarray], np.ndarray], values: np.ndarray) -> np.ndarray:
        """逐 lane 的一元运算：一条指令、32 个 lane 同时算。"""
        self.op()
        out = np.asarray(values).copy()
        out[self.mask] = fn(np.asarray(values)[self.mask])
        return out

    def binary(self, fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
               lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        self.op()
        out = np.asarray(lhs).copy()
        out[self.mask] = fn(np.asarray(lhs)[self.mask], np.asarray(rhs)[self.mask])
        return out

    # ---- 控制流 ---------------------------------------------------------
    def branch(self, predicate: np.ndarray,
               then_body: Callable[[], None],
               else_body: Optional[Callable[[], None]] = None) -> None:
        """SIMT 分支：两侧都串行执行，各自掩码。

        这正是 GPU 上分支的真实代价：当两侧都有活跃 lane 时，
        指令条数按路径数增加，效率下降。
        """
        pred = np.asarray(predicate, dtype=bool) & self.mask
        base_mask = self.mask
        self.trace.branch_count += 1

        then_mask = base_mask & pred
        else_mask = base_mask & ~pred
        has_then = bool(then_mask.any())
        has_else = bool(else_mask.any()) and else_body is not None

        if has_then and has_else:
            self.trace.diverged_branches += 1

        if has_then:
            self.mask = then_mask
            then_body()
        if has_else:
            self.mask = else_mask
            else_body()
        # 没有 else 分支时，pred 为假的 lane 直接跳过（掩码不变）
        self.mask = base_mask

    # ---- 收尾 -----------------------------------------------------------
    def finish(self) -> WarpTrace:
        return self.trace
