"""算子库：每个算子都是 DSL kernel + numpy 参考实现 + 便捷 runner。

runner 的返回值统一带「程序 IR + 代价估计」，这样 benchmark 里能把
「跑了多久」和「理论下界」放在一起看，而不是只报一个孤零零的耗时。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..compiler import codegen, compile_kernel, costmodel
from ..compiler import dsl as tl
from . import elementwise, matmul, smem, softmax

__all__ = [
    "elementwise", "softmax", "matmul",
    "run_vector_add", "run_scale_gelu", "run_softmax", "run_layernorm",
    "run_matmul", "run_fused_mlp", "run_smem_column", "OpResult",
]


def run_smem_column(pitch: int = 32, block: int = 32) -> "smem.SmemColumnResult":
    """共享内存按列读的 bank 冲突实验（见 :mod:`gkern.ops.smem`）。"""
    return smem.run_smem_column(pitch=pitch, block=block)


@dataclass
class OpResult:
    """一次算子执行的产物：数值结果 + IR + 代价估计。"""

    output: np.ndarray
    program: object
    cost: costmodel.Cost

    def summary(self) -> dict:
        return {"kernel": self.program.name, **self.cost.as_dict()}


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _require_2d(op: str, arr: np.ndarray) -> tuple[int, int]:
    """校验并返回二维矩阵的 (行, 列)。

    一维/三维输入若不在这里拦下，`m, n = arr.shape` 只会抛出一个含糊的解包错误，
    甚至喂给 kernel 后按拍平的地址空间静默算错——所以必须在 runner 入口显式报错。
    """
    if arr.ndim != 2:
        raise ValueError(f"{op} 要求二维输入，收到 shape={arr.shape}")
    return arr.shape


def run_vector_add(x: np.ndarray, y: np.ndarray, block: int = 256) -> OpResult:
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    if x.shape != y.shape:
        raise ValueError(f"vector_add 要求 X/Y 形状一致，收到 {x.shape} 与 {y.shape}")
    out = np.empty_like(x)
    n = x.size
    grid = (_cdiv(n, block),)
    bufs = {"X": x.reshape(-1), "Y": y.reshape(-1), "OUT": out.reshape(-1)}
    prog = compile_kernel(elementwise.vector_add, bufs, {"n": n}, {"BLOCK": block}, grid)
    codegen.run(prog, bufs)
    return OpResult(out, prog, costmodel.analyze(prog))


def run_scale_gelu(x: np.ndarray, scale: float = 1.0, block: int = 256) -> OpResult:
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty_like(x)
    n = x.size
    grid = (_cdiv(n, block),)
    bufs = {"X": x.reshape(-1), "OUT": out.reshape(-1)}
    prog = compile_kernel(elementwise.scale_gelu, bufs, {"n": n, "scale": scale},
                          {"BLOCK": block}, grid)
    codegen.run(prog, bufs)
    return OpResult(out, prog, costmodel.analyze(prog))


def run_softmax(x: np.ndarray, block_n: Optional[int] = None) -> OpResult:
    x = np.ascontiguousarray(x, dtype=np.float32)
    m, n = _require_2d("softmax", x)
    bn = block_n or 1 << max(0, (n - 1).bit_length())
    if bn < n:
        # 单遍归约要求整行装进一个 BLOCK_N；BLOCK_N < N 时只会算前 BLOCK_N 列，
        # 其余列留在未初始化的输出缓冲里，是典型的「静默算错」——直接拒绝。
        raise ValueError(f"softmax 的 BLOCK_N={bn} 必须 ≥ N={n}（一行必须装进一个 block）")
    out = np.empty_like(x)
    grid = (m,)
    bufs = {"X": x, "Y": out}
    prog = compile_kernel(softmax.softmax_rows, bufs, {"M": m, "N": n},
                          {"BLOCK_N": bn}, grid)
    codegen.run(prog, bufs)
    return OpResult(out, prog, costmodel.analyze(prog))


def run_layernorm(x: np.ndarray, w: np.ndarray, b: np.ndarray,
                  eps: float = 1e-5, block_n: Optional[int] = None) -> OpResult:
    x = np.ascontiguousarray(x, dtype=np.float32)
    w = np.ascontiguousarray(w, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    m, n = _require_2d("layernorm", x)
    if w.shape != (n,):
        raise ValueError(f"layernorm 权重 W 形状必须为 ({n},)，收到 {w.shape}")
    if b.shape != (n,):
        raise ValueError(f"layernorm 偏置 B 形状必须为 ({n},)，收到 {b.shape}")
    bn = block_n or 1 << max(0, (n - 1).bit_length())
    if bn < n:
        raise ValueError(f"layernorm 的 BLOCK_N={bn} 必须 ≥ N={n}（一行必须装进一个 block）")
    out = np.empty_like(x)
    bufs = {"X": x, "W": w, "B": b, "Y": out}
    prog = compile_kernel(softmax.layernorm_rows, bufs, {"M": m, "N": n, "eps": eps},
                          {"BLOCK_N": bn}, (m,))
    codegen.run(prog, bufs)
    return OpResult(out, prog, costmodel.analyze(prog))


def run_matmul(a: np.ndarray, b: np.ndarray, bm: int = 32, bn: int = 32,
               bk: int = 16, use_smem: bool = False, pad: int = 0) -> OpResult:
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    m, k = _require_2d("matmul 的 A", a)
    k2, n = _require_2d("matmul 的 B", b)
    if k != k2:
        raise ValueError(f"matmul 维度不匹配：{a.shape} x {b.shape}")
    if k % bk != 0:
        raise ValueError(f"K={k} 必须是 BK={bk} 的整数倍（IR 无循环，K 为编译期常量）")
    out = np.zeros((m, n), dtype=np.float32)
    grid = (_cdiv(m, bm), _cdiv(n, bn))
    bufs = {"A": a, "B": b, "C": out}
    if use_smem:
        prog = compile_kernel(matmul.matmul_smem, bufs, {"M": m, "N": n},
                              {"BM": bm, "BN": bn, "BK": bk, "K": k, "PAD": pad}, grid)
    else:
        prog = compile_kernel(matmul.matmul_tiled, bufs, {"M": m, "N": n},
                              {"BM": bm, "BN": bn, "BK": bk, "K": k}, grid)
    codegen.run(prog, bufs)
    return OpResult(out, prog, costmodel.analyze(prog))


def run_fused_mlp(a: np.ndarray, w: np.ndarray, bias: np.ndarray,
                  bm: int = 32, bn: int = 32, bk: int = 16,
                  fuse: bool = True) -> OpResult:
    """``gelu(a @ w + bias)``：可选是否把 epilogue 融进 matmul kernel。"""
    from ..compiler.passes import fuse_programs

    a = np.ascontiguousarray(a, dtype=np.float32)
    w = np.ascontiguousarray(w, dtype=np.float32)
    bias = np.ascontiguousarray(bias, dtype=np.float32)
    m, k = _require_2d("fused_mlp 的 A", a)
    k2, n = _require_2d("fused_mlp 的 W", w)
    if k != k2:
        raise ValueError(f"fused_mlp 权重形状不匹配：A{a.shape} @ W{w.shape}（内维 {k} != {k2}）")
    if bias.shape != (n,):
        raise ValueError(f"fused_mlp 偏置 bias 形状必须为 ({n},)，收到 {bias.shape}")
    if k % bk != 0:
        raise ValueError(f"K={k} 必须是 BK={bk} 的整数倍")

    tmp = np.zeros((m, n), dtype=np.float32)
    out = np.zeros((m, n), dtype=np.float32)
    grid = (_cdiv(m, bm), _cdiv(n, bn))

    mm = compile_kernel(matmul.matmul_tiled, {"A": a, "B": w, "C": tmp},
                        {"M": m, "N": n}, {"BM": bm, "BN": bn, "BK": bk, "K": k}, grid)
    ep = compile_kernel(matmul.bias_gelu_epilogue, {"C": tmp, "BIAS": bias, "OUT": out},
                        {"M": m, "N": n}, {"BM": bm, "BN": bn}, grid)

    if fuse:
        merged = fuse_programs(mm, ep, "C", name="mlp_fused")
        codegen.run(merged, {"A": a, "B": w, "BIAS": bias, "OUT": out})
        return OpResult(out, merged, costmodel.analyze(merged))

    codegen.run(mm, {"A": a, "B": w, "C": tmp})
    codegen.run(ep, {"C": tmp, "BIAS": bias, "OUT": out})
    total = costmodel.analyze(mm)
    ep_cost = costmodel.analyze(ep)
    total.bytes_read += ep_cost.bytes_read
    total.bytes_written += ep_cost.bytes_written
    total.flops += ep_cost.flops
    return OpResult(out, ep, total)
