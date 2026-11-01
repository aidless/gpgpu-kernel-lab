"""混合精度的可测量演示：同一个 softmax，两种累加精度。

为什么单开一个模块，而不是给 `softmax_rows` 加个开关：

- 真实实现里「累加精度」不是一个布尔量，而是**在哪些步骤舍入**的问题。
  这里把它写成 dsL 里的显式 `tl.round_bf16(...)` 调用，让每一次降精度都
  能在 IR 里被看见、被计数（代价模型会为每个舍入记 1 次 ALU 操作）。
- 于是「精度」与「代价」被拆成两件可分别测量的事：精度损失是数值问题，
  带宽与算力是性能问题。两条路径的字节与 FLOP 几乎相同——这正是 bf16
  在真实硬件上有吸引力的原因，也是「默认用 fp32 累加」需要理由的原因。

bf16 的 8 位尾数意味着每个舍入点引入 ~2^-9 的相对误差；行 softmax 的
归一化分母把这些误差累积成可测量的绝对误差。`tests/test_mixed_precision.py`
把「比 fp32 差 100 倍以上」写成断言，防止哪天实现偷偷没降精度。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

import gkern.compiler.dsl as tl
from gkern.compiler import compile_kernel
from gkern.compiler.codegen import run
from gkern.compiler.costmodel import Cost, analyze
from gkern.compiler.dsl import jit
from gkern.gpgpu.device import DeviceInfo
from gkern.ops.softmax import softmax_rows

__all__ = [
    "PrecisionResult",
    "softmax_rows_bf16acc",
    "run_softmax_precision",
    "precision_summary",
    "reference_softmax_float64",
]

_ACCUMULATE_CHOICES = ("fp32", "bf16")


@dataclass
class PrecisionResult:
    output: np.ndarray
    cost: Cost
    accumulate: str
    max_error: float
    mean_error: float
    row_sum_max_dev: float


@jit
def softmax_rows_bf16acc(X, Y, M: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    """行 softmax，但在四个位置把中间结果舍入到 bfloat16 精度。

    四个舍入点对应真实实现里 bf16 累加的四个天然断点：输入量化、行最大值、
    指数结果、归约和。全部保留 float32 存储，只是每一步都套一层 bf16 舍入，
    这样「降精度」的代价可以被单独归因（而不是整条路径一起糊掉）。
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X, row * N + cols, mask, other=-float("inf"))
    x = tl.round_bf16(x)
    m = tl.max(x, 0)
    m = tl.round_bf16(m)
    e = tl.exp(x - m)
    e = tl.round_bf16(e)
    s = tl.sum(e, 0)
    s = tl.round_bf16(s)
    out = e / s
    tl.store(Y, row * N + cols, out, mask)


def reference_softmax_float64(x: np.ndarray) -> np.ndarray:
    """float64 参考解：误差分母，不参与比较。"""
    a = np.asarray(x, dtype=np.float64)
    m = a.max(axis=-1, keepdims=True)
    e = np.exp(a - m)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def run_softmax_precision(x: np.ndarray, accumulate: str = "fp32",
                          device: DeviceInfo | None = None) -> PrecisionResult:
    """跑指定的累加精度路径，并给出相对 float64 参考的误差。"""
    if accumulate not in _ACCUMULATE_CHOICES:
        raise ValueError(f"accumulate 只能是 {_ACCUMULATE_CHOICES}，收到 {accumulate!r}")

    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"输入必须是二维 (M, N)，收到 shape={x.shape}")

    rows, cols = x.shape
    block = int(2 ** np.ceil(np.log2(cols)))
    y = np.zeros_like(x)
    dev = device or DeviceInfo()
    fn = softmax_rows if accumulate == "fp32" else softmax_rows_bf16acc

    buffers: Dict[str, Any] = {"X": x, "Y": y}
    scalars = {"M": rows, "N": cols}
    constexprs = {"BLOCK_N": block}
    program = compile_kernel(fn, buffers, scalars, constexprs, (rows,))
    out = run(program, buffers)["Y"]
    cost = analyze(program, dev)

    ref = reference_softmax_float64(x)
    err = np.abs(out.astype(np.float64) - ref.astype(np.float64))
    return PrecisionResult(
        output=out,
        cost=cost,
        accumulate=accumulate,
        max_error=float(err.max()) if err.size else 0.0,
        mean_error=float(err.mean()) if err.size else 0.0,
        row_sum_max_dev=float(np.abs(out.sum(axis=-1) - 1.0).max()) if err.size else 0.0,
    )


def precision_summary(x: np.ndarray) -> list[dict]:
    """两条路径并排的对照行，给 benchmark 与文档用。"""
    rows = []
    for accumulate in _ACCUMULATE_CHOICES:
        r = run_softmax_precision(x, accumulate=accumulate)
        rows.append({
            "accumulate": accumulate,
            "max_error": r.max_error,
            "mean_error": r.mean_error,
            "row_sum_max_dev": r.row_sum_max_dev,
            "bytes": r.cost.bytes_total,
            "flops": r.cost.flops,
            "bound": r.cost.bound,
        })
    return rows
