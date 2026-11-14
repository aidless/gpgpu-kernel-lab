"""编译栈的测试：DSL → IR → 执行 / 代价 / 融合。

这里最关键的一条是 :func:`test_fusion_saves_exactly_two_intermediate_writes`：
它把「融合省了多少访存」变成一个可复算的等式，而不是一句宣传语。
"""

from __future__ import annotations

import numpy as np
import pytest

from gkern.compiler import codegen, compile_kernel
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


# ---------------------------------------------------------------------------
# 融合闸的四个危险分支（审阅 F1/F2 的回归）
#
# 背景：早期 can_fuse 只查「grid 相等 / buffer 是 A 的输出 / B 读过它」三条，
# 结果会把下面四类程序判成可融合，其中第一类（地址偏移）会产生**静默的错误
# 数值**——融合前后结果不同但不报错，这是最难发现的一类缺陷。
# ---------------------------------------------------------------------------


def _prog(fn, buffers, scalars, constexprs, grid=(1,)):
    return compile_kernel(fn, buffers, scalars, constexprs, grid)


def _fusion_pair(producer_body, consumer_body, n=8, extra_buffers=None):
    """构造一对 1D producer/consumer 程序，供闸的拒绝用例复用。"""
    x = np.arange(n, dtype=np.float32)
    c = np.zeros(n, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    bufs = {"X": x, "C": c, "OUT": out}
    bufs.update(extra_buffers or {})
    a = _prog(producer_body, bufs, {"n": n}, {"BLOCK": n}, (1,))
    b = _prog(consumer_body, bufs, {"n": n}, {"BLOCK": n}, (1,))
    return a, b, bufs


def test_fusion_gate_rejects_halo_offsets():
    """A 写 C[i]、B 读 C[i+1]（halo/旋转）绝不能融合——否则是静默错结果。"""
    def producer(X, C, n, BLOCK):
        offs = tl.arange(0, BLOCK)
        tl.store(C, offs, 2.0 * tl.load(X, offs, offs < n), offs < n)

    def consumer(C, OUT, n, BLOCK):
        offs = tl.arange(0, BLOCK)
        tl.store(OUT, offs, tl.load(C, offs + 1, offs < n - 1) + 1.0, offs < n)

    a, b, bufs = _fusion_pair(producer, consumer)
    ok, reason = can_fuse(a, b, "C")
    assert not ok, "地址偏移的 producer/consumer 必须被拒绝"
    assert "偏移" in reason or "offsets" in reason
    with pytest.raises(ValueError, match="无法融合"):
        fuse_programs(a, b, "C")

    # 反证：真的融合会给出与未融合不同的数值（所以这道闸不是装饰）
    unfused = codegen.run(b, {**codegen.run(a, bufs), "OUT": np.zeros(8, np.float32)})
    assert not np.allclose(unfused["OUT"][:7], np.array([1, 3, 5, 7, 9, 11, 13], np.float32))


def test_fusion_gate_rejects_shape_mismatch():
    """地址结构相同、但写入值 shape 与读取 tile shape 不同时必须拒绝。

    这里刻意让两边 offsets 结构一致（都是 ``arange(0, BLOCK)``），把
    shape 判据单独逼出来：写的是 (1,BLOCK) 的 2D 值，读的是 (BLOCK,)，
    早期版本会把它们判成可融合，融合后 codegen 报
    ``ValueError: input operand has more dimensions than allowed by the axis remapping``。
    """
    def producer(X, C, n, BLOCK):
        offs = tl.arange(0, BLOCK)
        val = tl.expand_dims(tl.load(X, offs, offs < n), 0)
        tl.store(C, offs, val, offs < n)

    def consumer(C, OUT, n, BLOCK):
        offs = tl.arange(0, BLOCK)
        tl.store(OUT, offs, tl.load(C, offs, offs < n) + 1.0, offs < n)

    a, b, bufs = _fusion_pair(producer, consumer)
    ok, reason = can_fuse(a, b, "C")
    assert not ok, "shape 不一致必须被拒绝"
    assert "shape" in reason or "形状" in reason


def test_fusion_gate_rejects_dtype_mismatch():
    """写入 float32、读取 float64 时必须拒绝（dtype 不同意味着语义不同）。"""
    def producer(X, C, n, BLOCK):
        offs = tl.arange(0, BLOCK)
        tl.store(C, offs, tl.load(X, offs, offs < n), offs < n)

    x = np.arange(8, dtype=np.float32)
    c32 = np.zeros(8, dtype=np.float32)
    c64 = np.zeros(8, dtype=np.float64)
    out = np.zeros(8, dtype=np.float32)
    a = _prog(producer, {"X": x, "C": c32}, {"n": 8}, {"BLOCK": 8}, (1,))

    def consumer64(C, OUT, n, BLOCK):
        offs = tl.arange(0, BLOCK)
        tl.store(OUT, offs, tl.load(C, offs, offs < n) + 1.0, offs < n)

    b = _prog(consumer64, {"C": c64, "OUT": out}, {"n": 8}, {"BLOCK": 8}, (1,))
    ok, reason = can_fuse(a, b, "C")
    assert not ok, "dtype 不一致必须被拒绝"
    assert "dtype" in reason or "类型" in reason


def test_fusion_gate_rejects_read_modify_write():
    """A 就地读写 C（C 既是输入又是输出）时，融合后 C 被 pop 掉会 KeyError。"""
    def producer(C, n, BLOCK):
        offs = tl.arange(0, BLOCK)
        tl.store(C, offs, tl.load(C, offs, offs < n) + 1.0, offs < n)

    def consumer(C, OUT, n, BLOCK):
        offs = tl.arange(0, BLOCK)
        tl.store(OUT, offs, tl.load(C, offs, offs < n) + 1.0, offs < n)

    a, b, bufs = _fusion_pair(producer, consumer)
    ok, reason = can_fuse(a, b, "C")
    assert not ok, "A 自己读 C 时必须拒绝（否则 pop 掉输入就 KeyError）"
    assert "就地" in reason or "in-place" in reason or "输入" in reason


def test_fusion_gate_rejects_double_store():
    """A 对 C 写两次时，融合只取最后一次 store，前一次被静默丢弃。"""
    def producer(X, C, n, BLOCK):
        offs = tl.arange(0, BLOCK)
        tl.store(C, offs, tl.load(X, offs, offs < n), offs < n)
        tl.store(C, offs, 3.0 * tl.load(X, offs, offs < n), offs < n)

    def consumer(C, OUT, n, BLOCK):
        offs = tl.arange(0, BLOCK)
        tl.store(OUT, offs, tl.load(C, offs, offs < n) + 1.0, offs < n)

    a, b, bufs = _fusion_pair(producer, consumer)
    ok, reason = can_fuse(a, b, "C")
    assert not ok, "多次写同一缓冲区必须被拒绝"
    assert "一次" in reason or "多次" in reason


def test_fusion_gate_rejects_lane_mixing_consumer():
    """B 对 C 做跨 lane 归约（shape 变化）时，被掩掉的 lane 语义不再成立。"""
    def producer(X, C, n, BLOCK):
        offs = tl.arange(0, BLOCK)
        tl.store(C, offs, tl.load(X, offs, offs < n), offs < n)

    def consumer(C, OUT, n, BLOCK):
        offs = tl.arange(0, BLOCK)
        v = tl.load(C, offs, offs < n)
        s = tl.sum(v, 0, True)
        tl.store(OUT, tl.arange(0, 1), s)

    a, b, bufs = _fusion_pair(producer, consumer, extra_buffers={"OUT": np.zeros(1, np.float32)})
    ok, reason = can_fuse(a, b, "C")
    assert not ok, "跨 lane 归约会改变形状，必须拒绝"
    assert "形状" in reason or "shape" in reason

def test_fusion_gate_rejects_extra_unmasked_store():
    """B 把读到的值写到一个**无掩码**的输出时，被掩掉的 lane 会漏出脏数据。

    这是「只检查了 Tile 的 shape、忘了检查 store 的掩码」会漏掉的一类：
    producer 用 mask 写 C，consumer 读回后既按 mask 写 OUT1，又**不加掩码**
    写 OUT2——融合后 OUT2 的越界 lane 拿到的不是 0.0 而是 producer 算出的脏值。
    """
    def producer(X, C, n, BLOCK):
        offs = tl.arange(0, BLOCK)
        tl.store(C, offs, 2.0 * tl.load(X, offs), offs < n)

    def consumer(C, OUT, OUT2, n, BLOCK):
        offs = tl.arange(0, BLOCK)
        v = tl.load(C, offs, offs < n)
        tl.store(OUT, offs, v + 1.0, offs < n)   # 掩码与生产者一致
        tl.store(OUT2, offs, v + 1.0)            # 无掩码：会漏出脏数据

    a, b, bufs = _fusion_pair(producer, consumer,
                              extra_buffers={"OUT2": np.zeros(8, np.float32)})
    ok, reason = can_fuse(a, b, "C")
    assert not ok, "B 里存在无掩码/异掩码 store 时必须拒绝"
    assert "掩码" in reason
