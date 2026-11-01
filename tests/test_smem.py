"""共享内存分块与「复用上界」的验收（RED 先行：实现前这些用例必须全红）。

这里回答三个问题，每个都对应一条可复算的断言：

1. 共享内存分块**不改变**全局访存字节——它省的是 L1/global 往返与 bank 冲突，
   不是 HBM 流量。这条反直觉但正确，是理解「为什么加了 smem 还是一样慢」的关键。
2. 让 roofline 从 memory 翻成 compute 的**不是 smem 分块**，而是「面板能被 L2 复用」
   加上「问题规模够大」：理想复用下方阵 GEMM 的算术强度 = N/6，
   因此交叉点 256 FLOP/B 需要 N >= 1536。
3. 共享内存的 bank 冲突是真的能被测出来的：行宽 32 个 float 时按列读 → 32 路冲突，
   行宽加 1 个 padding（33）→ 0 冲突。
"""

from __future__ import annotations

import numpy as np
import pytest

from gkern.compiler import costmodel
from gkern.compiler.ir import SmemStoreInstr
from gkern.gpgpu.device import Device
from gkern.ops import run_matmul, run_smem_column
from gkern.ops.matmul import reference_matmul


def _matmul_program(n: int, bm: int = 32, bn: int = 32, bk: int = 16,
                    use_smem: bool = False):
    """只 trace 不执行地拿到 Program（大尺寸靠这个，避免跑几亿次浮点）。"""
    from gkern.compiler import compile_kernel
    from gkern.ops import matmul as mm

    a = np.zeros((n, n), dtype=np.float32)
    b = np.zeros((n, n), dtype=np.float32)
    c = np.zeros((n, n), dtype=np.float32)
    fn = mm.matmul_smem if use_smem else mm.matmul_tiled
    grid = ((n + bm - 1) // bm, (n + bn - 1) // bn)
    constexprs = {"BM": bm, "BN": bn, "BK": bk, "K": n}
    if use_smem:
        constexprs["PAD"] = 0
    return compile_kernel(fn, {"A": a, "B": b, "C": c}, {"M": n, "N": n}, constexprs, grid)


# ---------------------------------------------------------------------------
# 1. smem 分块不改变全局字节
# ---------------------------------------------------------------------------

def test_smem_tiling_does_not_change_global_bytes():
    n = 64
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n)).astype(np.float32)
    b = rng.standard_normal((n, n)).astype(np.float32)

    plain = run_matmul(a, b, bm=32, bn=32, bk=16)
    fused = run_matmul(a, b, bm=32, bn=32, bk=16, use_smem=True)

    assert fused.cost.bytes_total == plain.cost.bytes_total
    assert fused.cost.flops == plain.cost.flops
    # 确认它真的走了共享内存，而不是被优化掉
    assert any(isinstance(x, SmemStoreInstr) for x in fused.program.instrs)
    assert plain.program.smem == {}


def test_smem_matmul_matches_reference():
    n, k = 64, 32
    rng = np.random.default_rng(1)
    a = rng.standard_normal((n, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    out = run_matmul(a, b, bm=32, bn=32, bk=16, use_smem=True).output
    assert np.allclose(out, reference_matmul(a, b), atol=1e-4)


# ---------------------------------------------------------------------------
# 2. 复用上界：翻成 compute 的条件是「问题够大」，不是「用了 smem」
# ---------------------------------------------------------------------------

def test_reuse_bound_flips_to_compute_only_at_large_n():
    dev = Device()
    # 理想复用（每个输入只从 HBM 读一次）的渐近 AI = N/6，交叉点 256 → N = 1536
    assert int(6 * costmodel.crossover_intensity(dev.info)) == 1536

    small = costmodel.reuse_bound(_matmul_program(1024, bm=64, bn=64), dev.info)
    large = costmodel.reuse_bound(_matmul_program(2048, bm=64, bn=64), dev.info)

    assert small.bound == "memory"
    assert large.bound == "compute"
    # AI = N/6 + N/192：后一项是累加指令本身（K/BK 次 BM×BN 加法）的浮点开销，
    # 模型把它算进去了，所以比理想值略高——这条断言把「差多少、为什么差」也钉住。
    assert small.arithmetic_intensity == pytest.approx(1024 / 6 + 1024 / 192, rel=1e-9)
    assert large.arithmetic_intensity == pytest.approx(2048 / 6 + 2048 / 192, rel=1e-9)


def test_reuse_bound_is_strictly_below_global_lower_bound():
    prog = _matmul_program(512, bm=64, bn=64)
    lower = costmodel.analyze(prog)
    upper = costmodel.reuse_bound(prog)
    assert upper.bytes_total < lower.bytes_total
    assert upper.flops == lower.flops
    assert upper.reuse is True and lower.reuse is False
    # 下界与上界必须落在同一个算力/带宽口径上
    assert upper.device == lower.device


def test_tile_growth_alone_cannot_cross_the_crossover():
    """把 tile 从 16 加到 128，AI 只从 4.3 涨到 29.6——离 256 还差一个数量级。

    这是「路线图里「加共享内存分块就能翻成 compute」这个说法不成立」的量化依据。
    """
    ai = []
    for tile in (16, 32, 64, 128):
        prog = _matmul_program(512, bm=tile, bn=tile)
        ai.append(costmodel.analyze(prog).arithmetic_intensity)
        assert costmodel.analyze(prog).bound == "memory"
    assert ai == sorted(ai)
    assert ai[-1] < 40.0


# ---------------------------------------------------------------------------
# 3. bank 冲突：padding 真的有用
# ---------------------------------------------------------------------------

def test_smem_column_read_has_32_way_bank_conflict_without_padding():
    r = run_smem_column(pitch=32, block=32)
    # 按列读那一次：32 个 lane 全落 bank 0 → 32 × (32-1) = 992 路额外轮次
    assert r.stats.shared_bank_conflicts == 992
    # 访问总数里含那次无冲突的整块写（32×32 = 1024）
    assert r.stats.shared_accesses == 1056
    assert r.stats.shared_conflict_rate == pytest.approx(992 / 1056)


def test_smem_column_read_has_no_conflict_with_one_padding():
    r = run_smem_column(pitch=33, block=32)
    assert r.stats.shared_bank_conflicts == 0
    assert r.stats.shared_conflict_rate == pytest.approx(0.0)


def test_smem_column_read_is_numerically_correct():
    r = run_smem_column(pitch=32, block=32)
    assert np.array_equal(r.output, np.arange(32, dtype=np.float32) * 32.0)
