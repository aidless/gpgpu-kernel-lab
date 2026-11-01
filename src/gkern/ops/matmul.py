"""分块矩阵乘（tiled matmul）与融合的 MLP 尾巴。

为什么矩阵乘要分块
------------------
朴素写法每个输出元素要读 K 个 A 元素和 K 个 B 元素，访存是 O(MNK)，算力是 O(MNK)，
算术强度只有常数级 —— 完全被带宽打死。分块（tiling）后，一个 BM×BN 的 tile 只需
读 (BM×K + K×BN) 个元素，就能算出 BM×BN 个结果，算术强度提升约 BM×BN/(BM+BN) 倍。
这是所有 GEMM 优化的共同起点，也是本仓库自动调优器真正在调的东西。

IR 的诚实限制
-------------
本 IR 是线性指令表，没有循环结构，所以 K 必须是**编译期常量**（循环被展开）。
真实 Triton 会把 K 循环留在 IR 里交给 LLVM/PTX 后端。这个限制写在 README 的
「已知边界」里，不藏着。
"""

from __future__ import annotations

import numpy as np

from ..compiler import jit, constexpr
from ..compiler import dsl as tl


@jit
def matmul_tiled(A, B, C, M, N, BM: constexpr, BN: constexpr, BK: constexpr,
                 K: constexpr):
    """C[M,N] = A[M,K] @ B[K,N]，按 BM×BN 输出 tile、BK 步长累加。"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)          # (BM,)
    rn = pid_n * BN + tl.arange(0, BN)          # (BN,)
    rm2 = tl.expand_dims(rm, 1)                  # (BM,1)
    rn2 = tl.expand_dims(rn, 0)                  # (1,BN)

    acc = tl.zeros((BM, BN))
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)               # (BK,)
        rk_r = tl.expand_dims(rk, 0)             # (1,BK)
        rk_c = tl.expand_dims(rk, 1)             # (BK,1)
        a_offs = rm2 * K + rk_r                  # (BM,BK)
        a_mask = (rm2 < M) & (rk_r < K)
        b_offs = rk_c * N + rn2                  # (BK,BN)
        b_mask = (rk_c < K) & (rn2 < N)
        a = tl.load(A, a_offs, a_mask)
        b = tl.load(B, b_offs, b_mask)
        acc = tl.dot(a, b) + acc

    c_offs = rm2 * N + rn2
    c_mask = (rm2 < M) & (rn2 < N)
    tl.store(C, c_offs, acc, c_mask)


@jit
def matmul_smem(A, B, C, M, N, BM: constexpr, BN: constexpr, BK: constexpr,
                K: constexpr, PAD: constexpr = 0):
    """和 :func:`matmul_tiled` 做同样的事，但 A/B 面板先落共享内存。

    这次改动**不减少** HBM 流量（每个 block 仍要把自己的 A/B 面板从全局读一遍），
    它改变的是三件别的事：

    - 面板进片上内存后，乘加从共享内存取数，不再反复走 global → L1 的路；
    - ``PAD`` 给行宽加 padding，避开按列访问时的 bank 冲突；
    - 共享内存容量（48 KiB/block）因此成为分块尺寸的真实约束。

    「加了 smem 就该变成 compute bound」是常见误解：``tests/test_smem.py``
    第一条断言就是把这件事钉住——两者的全局字节数必须**完全相等**。
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    stride_a = BK + PAD
    stride_b = BN + PAD
    a_smem = tl.smem_alloc("a_panel", (BM, stride_a))
    b_smem = tl.smem_alloc("b_panel", (BK, stride_b))

    bi = tl.expand_dims(tl.arange(0, BM), 1)          # (BM,1)
    bj = tl.expand_dims(tl.arange(0, BK), 0)          # (1,BK)
    a_idx = bi * stride_a + bj                        # (BM,BK)
    ki = tl.expand_dims(tl.arange(0, BK), 1)          # (BK,1)
    nj = tl.expand_dims(tl.arange(0, BN), 0)          # (1,BN)
    b_idx = ki * stride_b + nj                        # (BK,BN)

    rows2 = tl.expand_dims(pid_m * BM + tl.arange(0, BM), 1)   # (BM,1)
    cols2 = tl.expand_dims(pid_n * BN + tl.arange(0, BN), 0)   # (1,BN)

    acc = tl.zeros((BM, BN))
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        kk_r = tl.expand_dims(kk, 0)                  # (1,BK)
        kk_c = tl.expand_dims(kk, 1)                  # (BK,1)
        a = tl.load(A, rows2 * K + kk_r, (rows2 < M) & (kk_r < K))
        b = tl.load(B, kk_c * N + cols2, (kk_c < K) & (cols2 < N))
        tl.smem_store(a_smem, a_idx, a)
        tl.smem_store(b_smem, b_idx, b)
        acc = tl.dot(tl.smem_load(a_smem, a_idx), tl.smem_load(b_smem, b_idx)) + acc

    tl.store(C, rows2 * N + cols2, acc, (rows2 < M) & (cols2 < N))


@jit
def bias_gelu_epilogue(C, BIAS, OUT, M, N, BM: constexpr, BN: constexpr):
    """在 matmul 的同一套 2D tile 网格上做 ``gelu(c + bias)``。

    它和 ``matmul_tiled`` 的 grid 完全一致，因此可以被
    :func:`gkern.compiler.passes.fuse_programs` 融合成一个 kernel：
    中间张量 C 不再落全局内存。
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm2 = tl.expand_dims(pid_m * BM + tl.arange(0, BM), 1)
    rn = pid_n * BN + tl.arange(0, BN)
    rn2 = tl.expand_dims(rn, 0)

    offs = rm2 * N + rn2
    mask = (rm2 < M) & (rn2 < N)
    c = tl.load(C, offs, mask)
    bias = tl.load(BIAS, rn, rn < N)
    x = c + bias
    inner = (x + 0.044715 * x * x * x) * 0.7978845608028654
    y = 0.5 * x * (1.0 + tl.tanh(inner))
    tl.store(OUT, offs, y, mask)


def reference_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)).astype(np.float32)


def reference_bias_gelu(c: np.ndarray, bias: np.ndarray) -> np.ndarray:
    x = np.asarray(c, dtype=np.float32) + np.asarray(bias, dtype=np.float32)
    inner = 0.7978845608028654 * (x + 0.044715 * x ** 3)
    return (0.5 * x * (1.0 + np.tanh(inner))).astype(np.float32)
