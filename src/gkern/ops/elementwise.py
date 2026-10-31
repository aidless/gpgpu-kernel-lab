"""逐元素算子：向量加法与「缩放 + GELU」融合算子。

这两个 kernel 是整个仓库的「hello world」，但它们各自说明一件真事：

- ``vector_add``：访存合并。当 BLOCK 与 warp 对齐、且下标连续时，
  一个 warp 的 32 个 lane 只触发少数几次 128B 事务；
  一旦把下标改成跨步（stride>1），事务数会成倍放大——这一点在
  ``benchmarks/coalescing_demo.py`` 里用同一套设备模型对照出来。
- ``fused_scale_gelu``：框架里 ``y = gelu(x * scale)`` 常常是两三个 kernel，
  中间结果写回全局内存。融合后只读一次、写一次，访存直接减半。
"""

from __future__ import annotations

import numpy as np

from ..compiler import jit, constexpr
from ..compiler import dsl as tl


@jit
def vector_add(X, Y, OUT, n, BLOCK: constexpr):
    """out[i] = x[i] + y[i]"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X, offs, mask)
    y = tl.load(Y, offs, mask)
    tl.store(OUT, offs, x + y, mask)


@jit
def scale_gelu(X, OUT, scale, n, BLOCK: constexpr):
    """out[i] = gelu(x[i] * scale)，tanh 近似（与 PyTorch 默认的 exact 版对齐误差 < 1e-3）。

    这个 kernel 内部把「乘标量 + 三次方 + tanh + 乘加」压在一次 tile 计算里，
    中间结果不落全局内存——这正是算子融合在单 kernel 内的形态。
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X, offs, mask) * scale
    # GELU(tanh 近似)：0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))
    inner = (x + 0.044715 * x * x * x) * 0.7978845608028654
    y = 0.5 * x * (1.0 + tl.tanh(inner))
    tl.store(OUT, offs, y, mask)


def reference_scale_gelu(x: np.ndarray, scale: float) -> np.ndarray:
    """numpy 参考实现（tanh 近似），用于对拍。"""
    x = np.asarray(x, dtype=np.float32) * np.float32(scale)
    inner = 0.7978845608028654 * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def reference_vector_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray(x) + np.asarray(y)
