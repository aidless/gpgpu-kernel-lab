"""调度层：波量化与 GEMM 工作分解。

这一层**不跑 kernel**，只回答「工作怎么摊到硬件上，代价是多少」——
与 :mod:`gkern.compiler.costmodel` 的分工是：

- costmodel 算「这个 kernel 必须搬多少字节、多少 FLOP」→ 单次执行的下界；
- schedule 算「这些工作怎么铺到 SM 上」→ 最后一波浪费多少并行度、
  拆细之后又多写多少字节。

两者都不含 cache 命中与发射开销，合成一句话：**能给出可比的下界，不能预测墙钟**。
"""

from __future__ import annotations

from .gemm_decomp import (GemmShape, StrategyRow, StreamKUnit, compare_strategies,
                          recommend, streamk_partials, streamk_units)
from .wave import WaveStats, wave_stats

__all__ = [
    "GemmShape", "StrategyRow", "StreamKUnit", "compare_strategies", "recommend",
    "streamk_units", "streamk_partials",
    "WaveStats", "wave_stats",
]
