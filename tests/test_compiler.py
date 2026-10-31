"""编译栈的测试：DSL → IR → 执行 / 代价 / 融合。

这里最关键的一条是 :func:`test_fusion_saves_exactly_two_intermediate_writes`：
它把「融合省了多少访存」变成一个可复算的等式，而不是一句宣传语。
"""

from __future__ import annotations

import numpy as np
import pytest

from gkern.compiler import codegen, compile_kernel, costmodel
from gkern.compiler import dsl as tl
from gkern.compiler.passes import can_fuse, fuse_programs
from gkern.ops import matmul, run_fused_mlp, run_scale_gelu, run_softmax, run_vector_add
from gkern.ops.elementwise import reference_scale_gelu


def test_vector_add_matches_numpy_including_tail():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000).astype(np.float32)
    y = rng.standard_normal(1000).astype(np.float32)
    res = run_vector_add(x, y)
    np.testing.assert_allclose(res.output, x + y, rtol=1e-6)
    # 1000 个元素 / BLOCK=256 → 4 个 program，最后一个只用了 232 个 lane
    assert res.cost.programs == 4


def test_scalar_float_is_not_truncated():
    """回归：早期版本用 int() 绑定标量参数，把 scale=1.7 截成了 1。"""
    x = np.linspace(-2, 2, 64, dtype=np.float32)
    got = run_scale_gelu(x, scale=1.7).output
    want = reference_scale_gelu(x, 1.7)
    truncated = reference_scale_gelu(x, 1.0)
    np.testing.assert_allclose(got, want, atol=1e-6)
    assert not np.allclose(got, truncated, atol=1e-6), "标量被截断成 1 时这条必须红"


def test_softmax_rows_normalize_and_are_masked():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 37)).astype(np.float32)
    out = run_softmax(x).output
    np.testing.assert_allclose(out.sum(axis=1), np.ones(8), rtol=1e-6)
    from gkern.ops.softmax import reference_softmax
    np.testing.assert_allclose(out, reference_softmax(x), atol=1e-6)


def test_layernorm_zero_mean_unit_var():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 37)).astype(np.float32)
    w = np.ones(37, dtype=np.float32)
    b = np.zeros(37, dtype=np.float32)
    from gkern.ops import run_layernorm
    out = run_layernorm(x, w, b, eps=0.0).output
    np.testing.assert_allclose(out.mean(axis=1), np.zeros(4), atol=1e-5)
    np.testing.assert_allclose(out.std(axis=1), np.ones(4), atol=1e-4)


def test_matmul_rejects_k_not_multiple_of_bk():
    a = np.zeros((8, 10), dtype=np.float32)
    b = np.zeros((10, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="BK"):
        matmul.matmul_tiled  # 存在性
        from gkern.ops import run_matmul
        run_matmul(a, b, bm=8, bn=8, bk=16)


def test_fusion_saves_exactly_two_intermediate_writes():
    """融合省掉的访存 = 中间张量的一次写 + 一次读 = 2×|C|，逐字节可复算。"""
    rng = np.random.default_rng(3)
    m, k, n = 64, 32, 48
    a = rng.standard_normal((m, k)).astype(np.float32)
    w = rng.standard_normal((k, n)).astype(np.float32)
    bias = rng.standard_normal(n).astype(np.float32)

    fused = run_fused_mlp(a, w, bias, bm=32, bn=16, bk=16, fuse=True)
    unfused = run_fused_mlp(a, w, bias, bm=32, bn=16, bk=16, fuse=False)

    np.testing.assert_allclose(fused.output, unfused.output, atol=1e-5)
    saved = unfused.cost.bytes_total - fused.cost.bytes_total
    assert saved == 2 * m * n * 4 == 24576
    # 融合后不再需要外部提供中间缓冲区 C
    assert "C" not in fused.program.inputs


def test_fusion_gate_can_fail():
    """门必须能红：给一对不该融合的程序，can_fuse 必须返回 False。"""
    rng = np.random.default_rng(4)
    m, k, n = 32, 16, 16
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    tmp = np.zeros((m, n), dtype=np.float32)
    out = np.zeros((m, n), dtype=np.float32)

    mm = compile_kernel(matmul.matmul_tiled, {"A": a, "B": b, "C": tmp},
                        {"M": m, "N": n}, {"BM": 32, "BN": 16, "BK": 16, "K": k},
                        (1, 1))
    ep = compile_kernel(matmul.bias_gelu_epilogue, {"C": tmp, "BIAS": np.zeros(n, np.float32),
                                                    "OUT": out},
                        {"M": m, "N": n}, {"BM": 32, "BN": 16}, (1, 1))

    ok, reason = can_fuse(mm, ep, "C")
    assert ok and reason == ""
    ok2, reason2 = can_fuse(mm, ep, "NOT_A_BUFFER")
    assert not ok2 and reason2, "不存在的中间缓冲区必须被拒绝"
    # grid 不同（2x2 vs 1x1）也不能融合
    ep_big = compile_kernel(matmul.bias_gelu_epilogue, {"C": tmp, "BIAS": np.zeros(n, np.float32),
                                                         "OUT": out},
                            {"M": m, "N": n}, {"BM": 32, "BN": 16}, (2, 2))
    ok3, reason3 = can_fuse(mm, ep_big, "C")
    assert not ok3 and "grid" in reason3


def test_costmodel_arithmetic_intensity_grows_with_tile_size():
    """分块越大，同一份 A/B tile 被复用的次数越多 → 算术强度越高。

    这条是**实测口径**，不是教科书口号：本代价模型按「每个 program 各自从全局
    内存搬 A/B tile」计数（不建模 L2/共享内存复用），所以 tiled matmul 的算术
    强度会随 tile 变大而上升，但仍远低于 roofline 交叉点 256 FLOP/B——
    这也是 README「模型边界」一节明写的局限：真实 GEMM 靠共享内存把 bytes
    降到 2n²，本模型做不到，所以它给出的 matmul 时间是**悲观下界**。
    """
    n = 256
    a = np.zeros((n, n), dtype=np.float32)
    b = np.zeros((n, n), dtype=np.float32)
    from gkern.ops import run_matmul

    intensities = []
    for tile in (16, 32, 64, 128):
        cost = run_matmul(a, b, bm=tile, bn=tile, bk=16).cost
        intensities.append(cost.arithmetic_intensity)
        assert cost.bound == "memory", "本模型下 tiled matmul 一律访存受限"
    assert intensities == sorted(intensities), "tile 变大，算术强度必须单调上升"
    assert intensities[0] < 5 and intensities[-1] > 25

    # 纯逐元素算子的算术强度必然远低于 1 FLOP/B（读改写 3 次访存换 1 次加法）
    x = np.zeros(4096, dtype=np.float32)
    add = run_vector_add(x, x)
    assert add.cost.bound == "memory"
    assert add.cost.arithmetic_intensity < 1.0


def test_executor_rejects_missing_input_buffer():
    rng = np.random.default_rng(6)
    x = rng.standard_normal(64).astype(np.float32)
    out = np.zeros(64, dtype=np.float32)
    prog = compile_kernel(
        lambda X, Y, OUT, n, BLOCK: tl.store(
            OUT, tl.arange(0, BLOCK),
            tl.load(X, tl.arange(0, BLOCK)) + tl.load(Y, tl.arange(0, BLOCK)),
            tl.arange(0, BLOCK) < n),
        {"X": x, "Y": x, "OUT": out}, {"n": 64}, {"BLOCK": 64}, (1,))
    with pytest.raises(KeyError, match="Y"):
        codegen.run(prog, {"X": x, "OUT": out})
