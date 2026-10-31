"""行级 softmax 与 LayerNorm。

这两个算子是「归约类 kernel」的典型代表：它们不是逐元素独立的，而是要先扫一遍
整行求出 max / sum / mean，再扫一遍做归一化。真实实现里的关键权衡是：

- 一行装得下一个 block 的寄存器/共享内存 → 单遍读入、寄存器内归约（本实现）；
- 一行装不下 → 需要 split-K 两遍扫描或 online softmax（见 docs/ARCHITECTURE.md）。

本实现走第一种，并且把 ``other=-inf`` 的掩码语义写进 load，使尾部不参与归约——
这是真实 kernel 里最容易写错、也最容易在尾块上出数值 bug 的地方。
"""

from __future__ import annotations

import numpy as np

from ..compiler import jit, constexpr
from ..compiler import dsl as tl


@jit
def softmax_rows(X, Y, M, N, BLOCK_N: constexpr):
    """每一行做数值稳定 softmax（减 max 防溢出）。"""
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    offs = pid * N + cols
    x = tl.load(X, offs, mask, other=float("-inf"))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    tl.store(Y, offs, e / s, mask)


@jit
def layernorm_rows(X, W, B, Y, M, N, eps, BLOCK_N: constexpr):
    """每行做 LayerNorm：y = (x - mean) / sqrt(var + eps) * w + b。"""
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    offs = pid * N + cols
    x = tl.load(X, offs, mask)
    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    # 尾部 lane 是被掩掉的（load 得到 0），它们的 (0-mean)^2 会污染方差，
    # 所以这里必须用 where 把无效 lane 的贡献显式置零——这是归约类 kernel 的经典坑。
    var = tl.sum(tl.where(mask, xc * xc, 0.0), axis=0) / N
    norm = xc * tl.rsqrt(var + eps)
    w = tl.load(W, cols, mask, other=0.0)
    b = tl.load(B, cols, mask, other=0.0)
    tl.store(Y, offs, norm * w + b, mask)


def reference_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def reference_layernorm(x: np.ndarray, w: np.ndarray, b: np.ndarray,
                        eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return ((x - mu) / np.sqrt(var + eps) * w + b).astype(np.float32)
