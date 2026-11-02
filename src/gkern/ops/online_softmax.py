"""在线（分块）softmax：一行装不下时，怎么把输入只读一遍。

两遍扫描的做法很直白：第一遍求 max 与总和，第二遍重新读输入算 exp(x-m)/s。
代价是**输入被读两次**——行很长时这一笔在 HBM 上就是实打实的两倍字节。

在线做法把第一遍的中间结果保住，靠的是 softmax 对整行平移不变：

    对每个块 b:  m_b  = max(x_b)                     # 块内最大值
                 n_b  = exp(x_b - m_b)               # 按块内基准的分子
                 m_new = max(m_old, m_b)
                 s    = s * exp(m_old - m_new) + sum(n_b) * exp(m_b - m_new)
    最后:        out_b = n_b * exp(m_b - m_final) / s

**修正因子 exp(m_b - m_final) 是事后乘的**，所以输入只需要读一遍。这两条等式
（老的 sum 按新 max 重缩放、每块的分子按自己当时的基准记）就是 FlashAttention
那一脉在线 softmax 的全部数学内容；写错的典型症状是每行和不为 1。

本模块同时给出两遍扫描版本用于对比字节数——两者数值一致，差别只在访存：

| 版本 | 输入读 | 输出写 |
|---|---:|---:|
| `online_softmax_rows` | 1 次 | 1 次 |
| `recompute_softmax_rows` | 2 次 | 1 次 |

这也是本仓库「字节账算得清、墙钟算不准」的正面例子：省下的是确定的 N×4 字节，
能不能变成更快的时间取决于这块数据有没有被 L2 接住（见 `docs/ARCHITECTURE.md` §8.2）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from ..compiler import codegen, compile_kernel, costmodel
from ..compiler import dsl as tl
from ..compiler.dsl import constexpr, jit
from ..gpgpu.device import DeviceInfo
from .softmax import reference_softmax

__all__ = [
    "OnlineSoftmaxResult",
    "online_softmax_rows",
    "recompute_softmax_rows",
    "run_online_softmax",
    "run_recompute_softmax",
    "compare_online_vs_recompute",
]


@jit
def online_softmax_rows(X, Y, M: constexpr, N: constexpr, BLOCK: constexpr):
    """分块在线 softmax：输入只读一遍，每块的分子与块内最大值留在寄存器。

    块数在 trace 期展开成线性 IR（本 IR 没有循环结构，循环都被展开），
    所以这里量的是访存账与归约结构，不是指令数。
    """
    nblk = (N + BLOCK - 1) // BLOCK
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)

    running_max = tl.full((1,), float("-inf"))
    running_sum = tl.full((1,), 0.0)
    kept: List[Tuple[Any, Any, Any, Any]] = []  # (cols, mask, numer, block_max)

    for b in range(nblk):
        cols = b * BLOCK + lane
        mask = cols < N
        x = tl.load(X, pid * N + cols, mask, other=float("-inf"))
        block_max = tl.max(x, axis=0)          # 整块被掩掉时是 -inf
        numer = tl.exp(x - block_max)          # 整块被掩掉时是 0
        new_max = tl.maximum(running_max, block_max)
        # 旧的 sum 换到新基准；被整块掩掉时 alpha = exp(-inf - m) = 0，不产生 NaN
        alpha = tl.exp(running_max - new_max)
        beta = tl.exp(block_max - new_max)
        running_sum = running_sum * alpha + tl.sum(numer, axis=0) * beta
        running_max = new_max
        kept.append((cols, mask, numer, block_max))

    final_sum = running_sum
    for (cols, mask, numer, block_max) in kept:
        # 把每块「按自己基准」的分子换回全局基准，再统一归一
        out = numer * tl.exp(block_max - running_max) / final_sum
        tl.store(Y, pid * N + cols, out, mask)


@jit
def recompute_softmax_rows(X, Y, M: constexpr, N: constexpr, BLOCK: constexpr):
    """两遍扫描版本：第一遍只求 max/sum，第二遍重新读输入——输入被读两次。"""
    nblk = (N + BLOCK - 1) // BLOCK
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)

    running_max = tl.full((1,), float("-inf"))
    running_sum = tl.full((1,), 0.0)
    for b in range(nblk):
        cols = b * BLOCK + lane
        mask = cols < N
        x = tl.load(X, pid * N + cols, mask, other=float("-inf"))
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(running_max, block_max)
        alpha = tl.exp(running_max - new_max)
        beta = tl.exp(block_max - new_max)
        running_sum = running_sum * alpha + tl.sum(tl.exp(x - block_max), axis=0) * beta
        running_max = new_max

    for b in range(nblk):
        cols = b * BLOCK + lane
        mask = cols < N
        x = tl.load(X, pid * N + cols, mask, other=float("-inf"))  # 第二遍重读
        tl.store(Y, pid * N + cols, tl.exp(x - running_max) / running_sum, mask)


@dataclass
class OnlineSoftmaxResult:
    output: np.ndarray
    cost: costmodel.Cost
    program: Any
    blocks: int

    def as_dict(self) -> dict:
        return {"blocks": self.blocks, **self.cost.as_dict()}


def _run(fn, x: np.ndarray, block: int, device: DeviceInfo | None) -> OnlineSoftmaxResult:
    rows, cols = x.shape
    if block <= 0:
        raise ValueError(f"BLOCK 必须为正，收到 {block}")
    out = np.zeros_like(x)
    buffers: Dict[str, Any] = {"X": x, "Y": out}
    prog = compile_kernel(fn, buffers, {"M": rows}, {"N": cols, "BLOCK": block}, (rows,))
    codegen.run(prog, buffers)
    cost = costmodel.analyze(prog, device or DeviceInfo())
    return OnlineSoftmaxResult(out, cost, prog, (cols + block - 1) // block)


def run_online_softmax(x: np.ndarray, block: int = 256,
                       device: DeviceInfo | None = None) -> OnlineSoftmaxResult:
    """在线版本：输入读一遍。"""
    return _run(online_softmax_rows, np.ascontiguousarray(x, dtype=np.float32), block, device)


def run_recompute_softmax(x: np.ndarray, block: int = 256,
                          device: DeviceInfo | None = None) -> OnlineSoftmaxResult:
    """两遍扫描版本：输入读两遍（对照用）。"""
    return _run(recompute_softmax_rows, np.ascontiguousarray(x, dtype=np.float32), block, device)


def compare_online_vs_recompute(shape=(8, 1024), block: int = 256) -> list[dict]:
    """并排两条路径：数值是否一致、字节差多少。benchmark 与文档都用它。"""
    rng = np.random.default_rng(7)
    x = (rng.standard_normal(shape) * 3.0).astype(np.float32)
    a = run_online_softmax(x, block)
    b = run_recompute_softmax(x, block)
    ref = reference_softmax(x)
    same = bool(np.allclose(a.output, b.output, atol=1e-6))
    return [
        {
            "variant": "online",
            "blocks": a.blocks,
            "bytes_total": a.cost.bytes_total,
            "max_error": float(np.abs(a.output - ref).max()),
            "equal_to_recompute": same,
        },
        {
            "variant": "recompute",
            "blocks": b.blocks,
            "bytes_total": b.cost.bytes_total,
            "max_error": float(np.abs(b.output - ref).max()),
            "equal_to_recompute": same,
        },
    ]
