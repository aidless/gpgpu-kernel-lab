"""六个 NumPy DSL 算子的正确性回归。

覆盖两类东西：

1. **正常路径**（数值必须对拍 numpy 参考实现）：
   - 1000 元素的尾部块（1000 = 3×256 + 232，最后一个 block 只用了部分 lane）；
   - 8×37 的行级归约（softmax / LayerNorm，一行装进一个 BLOCK_N）；
   - 64×32 @ 32×48 的 matmul 与 fused_mlp，验证融合省下的字节数。

2. **非法输入必须显式报错，而不是静默算错**：
   - 空输入（应返回空结果，不崩）；
   - shape 不匹配（vector_add 的 x/y、LayerNorm 的 w/b）；
   - softmax 的 BLOCK_N < N（单遍归约要求一行装进一个 block）；
   - matmul / fused_mlp 的 K % BK != 0、内维不匹配、weight/bias 形状错误。

最后一条 test_fusion_byte_is_static_tile_estimate 专门说明：`2*M*N*4` 这个
「融合省了多少字节」是**静态 tile 估计**——代价模型按每块的 tile 面积计费，
不扣掉尾部块里被 mask 掉的 lane。因此只有当分块正好整除 M/N 时它才精确等于
2×|C|；出现尾块时按 tile 面积算出的值会偏大。这不是 bug，是模型的诚实边界。
"""

from __future__ import annotations

import numpy as np
import pytest

from gkern.ops import (
    run_fused_mlp,
    run_layernorm,
    run_matmul,
    run_scale_gelu,
    run_softmax,
    run_vector_add,
)
from gkern.ops.elementwise import reference_scale_gelu, reference_vector_add
from gkern.ops.matmul import reference_bias_gelu, reference_matmul
from gkern.ops.softmax import reference_layernorm, reference_softmax


# ---------------------------------------------------------------------------
# 1. 正常路径
# ---------------------------------------------------------------------------


def test_vector_add_1000_elements_with_tail_block():
    """1000 元素：1000 / 256 → 4 个 program，最后一块只覆盖 232 个 lane。"""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000).astype(np.float32)
    y = rng.standard_normal(1000).astype(np.float32)
    res = run_vector_add(x, y, block=256)
    np.testing.assert_allclose(res.output, reference_vector_add(x, y), rtol=1e-6)
    # 1000 = 3*256 + 232，所以必须是 4 个 block（尾部那一块是回归重点）
    assert 1000 % 256 == 232
    assert res.cost.programs == 4


def test_scale_gelu_empty_input_returns_empty():
    """空输入是合法输入：应返回同形状空数组，而不是抛异常或算出 nan。"""
    x = np.zeros(0, dtype=np.float32)
    res = run_scale_gelu(x, scale=1.7)
    assert res.output.shape == (0,)
    assert res.output.size == 0


def test_softmax_8x37_rows_reduce_to_one():
    """8×37：BLOCK_N 默认取到 ≥37 的 2 次幂（64），尾部 27 个 lane 被 mask 掉。"""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 37)).astype(np.float32)
    out = run_softmax(x).output
    np.testing.assert_allclose(out, reference_softmax(x), atol=1e-6)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(8, dtype=np.float32), rtol=1e-6)


def test_layernorm_8x37_rows_zero_mean_unit_var():
    """8×37 LayerNorm：eps=0、w=1、b=0 时每行均值 0、标准差 1。"""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 37)).astype(np.float32)
    w = np.ones(37, dtype=np.float32)
    b = np.zeros(37, dtype=np.float32)
    out = run_layernorm(x, w, b, eps=0.0).output
    np.testing.assert_allclose(out, reference_layernorm(x, w, b, eps=0.0), atol=1e-5)
    np.testing.assert_allclose(out.mean(axis=1), np.zeros(8, dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(out.std(axis=1), np.ones(8, dtype=np.float32), atol=1e-4)


def test_matmul_64x32_at_32x48_matches_numpy():
    """64×32 @ 32×48 的 tiled matmul 必须逐元素对拍 numpy。"""
    rng = np.random.default_rng(3)
    a = rng.standard_normal((64, 32)).astype(np.float32)
    b = rng.standard_normal((32, 48)).astype(np.float32)
    res = run_matmul(a, b, bm=32, bn=16, bk=16)
    np.testing.assert_allclose(res.output, reference_matmul(a, b), atol=1e-5)


def test_fused_mlp_64x32_at_32x48_fusion_saves_two_intermediate_copies():
    """融合省下的字节 = 中间张量 C 的一次写 + 一次读 = 2×64×48×4 = 24576。

    这里用 bm=32、bn=16（都能整除 64/48），所以静态 tile 估计恰好等于 2×|C|。
    """
    rng = np.random.default_rng(4)
    m, k, n = 64, 32, 48
    a = rng.standard_normal((m, k)).astype(np.float32)
    w = rng.standard_normal((k, n)).astype(np.float32)
    bias = rng.standard_normal(n).astype(np.float32)

    fused = run_fused_mlp(a, w, bias, bm=32, bn=16, bk=16, fuse=True)
    unfused = run_fused_mlp(a, w, bias, bm=32, bn=16, bk=16, fuse=False)

    # 融合前后数值必须一致，且都等于 gelu(a @ w + bias)
    np.testing.assert_allclose(fused.output, unfused.output, atol=1e-5)
    np.testing.assert_allclose(fused.output, reference_bias_gelu(a @ w, bias), atol=1e-5)

    saved = unfused.cost.bytes_total - fused.cost.bytes_total
    assert saved == 2 * m * n * 4 == 24576
    # 融合后中间缓冲区 C 不再需要外部提供
    assert "C" not in fused.program.inputs


def test_fusion_byte_is_static_tile_estimate():
    """说明 saved 是「静态 tile 估计」：代价模型按每块 tile 面积计费，不扣尾部 mask。

    bn=32 无法整除 n=48，尾块里被 mask 掉的 lane 仍按 tile 面积计入，
    所以 saved 变成 2 × (grid 块数) × BM × BN × 4，比真实的 2×|C| 偏大。
    只有分块正好整除时它才精确——这条测试把这个边界钉死，防止有人把它当恒定公式。
    """
    rng = np.random.default_rng(5)
    m, k, n = 64, 32, 48
    a = rng.standard_normal((m, k)).astype(np.float32)
    w = rng.standard_normal((k, n)).astype(np.float32)
    bias = rng.standard_normal(n).astype(np.float32)

    bm, bn = 32, 32           # n=48 不能被 32 整除 → 有尾块
    fused = run_fused_mlp(a, w, bias, bm=bm, bn=bn, bk=16, fuse=True)
    unfused = run_fused_mlp(a, w, bias, bm=bm, bn=bn, bk=16, fuse=False)
    saved = unfused.cost.bytes_total - fused.cost.bytes_total

    blocks = int(np.prod(fused.program.grid))
    # 静态 tile 估计：按完整 tile 面积算，不看 mask
    assert saved == 2 * blocks * bm * bn * 4
    # 因此与「真实 2×|C|」不相等——这正是「静态估计」的含义
    assert saved != 2 * m * n * 4


# ---------------------------------------------------------------------------
# 2. 空输入与 shape 校验（这些在修复前会静默算错，必须先红）
# ---------------------------------------------------------------------------


def test_vector_add_empty_input_is_identity():
    res = run_vector_add(np.zeros(0, np.float32), np.zeros(0, np.float32))
    assert res.output.shape == (0,)
    assert res.cost.programs == 0


def test_matmul_empty_rows_returns_empty():
    a = np.zeros((0, 16), dtype=np.float32)
    b = np.zeros((16, 8), dtype=np.float32)
    res = run_matmul(a, b, bm=32, bn=16, bk=16)
    assert res.output.shape == (0, 8)


def test_vector_add_rejects_shape_mismatch():
    x = np.zeros(4, dtype=np.float32)
    y = np.zeros(5, dtype=np.float32)
    with pytest.raises(ValueError, match="形状一致"):
        run_vector_add(x, y)


def test_vector_add_rejects_same_size_but_different_shape():
    """(2,3) 与 (3,2) 元素数相同，但形状不同——绝不能把拍平后相加当成正确。"""
    x = np.zeros((2, 3), dtype=np.float32)
    y = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="形状一致"):
        run_vector_add(x, y)


def test_softmax_rejects_block_n_less_than_n():
    """BLOCK_N < N 时只算前 BLOCK_N 列，其余列是未初始化的垃圾——必须报错。"""
    x = np.zeros((8, 37), dtype=np.float32)
    with pytest.raises(ValueError, match="BLOCK_N"):
        run_softmax(x, block_n=16)


def test_softmax_rejects_non_2d_input():
    with pytest.raises(ValueError, match="二维"):
        run_softmax(np.zeros(37, dtype=np.float32))


def test_layernorm_rejects_non_2d_input():
    with pytest.raises(ValueError, match="二维"):
        run_layernorm(np.zeros(37, np.float32), np.ones(37, np.float32), np.zeros(37, np.float32))


def test_layernorm_rejects_weight_shape_mismatch():
    x = np.zeros((4, 37), dtype=np.float32)
    with pytest.raises(ValueError, match="权重 W"):
        run_layernorm(x, np.ones(20, np.float32), np.zeros(37, np.float32))


def test_layernorm_rejects_bias_shape_mismatch():
    x = np.zeros((4, 37), dtype=np.float32)
    with pytest.raises(ValueError, match="偏置 B"):
        run_layernorm(x, np.ones(37, np.float32), np.zeros(20, np.float32))


def test_matmul_rejects_k_not_multiple_of_bk():
    a = np.zeros((8, 10), dtype=np.float32)
    b = np.zeros((10, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="BK"):
        run_matmul(a, b, bm=8, bn=8, bk=16)


def test_matmul_rejects_inner_dimension_mismatch():
    a = np.zeros((8, 16), dtype=np.float32)
    b = np.zeros((10, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="维度不匹配"):
        run_matmul(a, b, bm=8, bn=8, bk=16)


def test_matmul_rejects_non_2d_input():
    with pytest.raises(ValueError, match="二维"):
        run_matmul(np.zeros(16, np.float32), np.zeros((16, 8), np.float32))


def test_fused_mlp_rejects_weight_inner_dimension_mismatch():
    """w 的行数必须等于 a 的列数（K），否则 matmul 会静默读错内存。"""
    a = np.zeros((4, 16), dtype=np.float32)
    w = np.zeros((8, 8), dtype=np.float32)   # 8 != 16
    bias = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError, match="权重形状"):
        run_fused_mlp(a, w, bias)


def test_fused_mlp_rejects_bias_shape_mismatch():
    a = np.zeros((4, 16), dtype=np.float32)
    w = np.zeros((16, 8), dtype=np.float32)
    bias = np.zeros(3, dtype=np.float32)    # 应为 (8,)
    with pytest.raises(ValueError, match="bias"):
        run_fused_mlp(a, w, bias)


def test_fused_mlp_rejects_k_not_multiple_of_bk():
    a = np.zeros((4, 10), dtype=np.float32)
    w = np.zeros((10, 8), dtype=np.float32)
    bias = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError, match="BK"):
        run_fused_mlp(a, w, bias, bk=16)


def test_fused_mlp_rejects_non_2d_weight():
    a = np.zeros((4, 16), dtype=np.float32)
    w = np.zeros(16, dtype=np.float32)
    bias = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError, match="二维"):
        run_fused_mlp(a, w, bias)


# ---------------------------------------------------------------------------
# 3. 其余算子的空输入与 1000 元素正常路径
# ---------------------------------------------------------------------------


def test_scale_gelu_1000_elements_matches_reference():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(1000).astype(np.float32)
    res = run_scale_gelu(x, scale=1.7, block=256)
    np.testing.assert_allclose(res.output, reference_scale_gelu(x, 1.7), atol=1e-6)
    assert res.cost.programs == 4


def test_softmax_empty_rows_returns_empty():
    res = run_softmax(np.zeros((0, 37), dtype=np.float32))
    assert res.output.shape == (0, 37)


def test_layernorm_empty_rows_returns_empty():
    res = run_layernorm(np.zeros((0, 37), np.float32),
                        np.ones(37, np.float32), np.zeros(37, np.float32))
    assert res.output.shape == (0, 37)


def test_fused_mlp_empty_rows_returns_empty():
    res = run_fused_mlp(np.zeros((0, 16), np.float32),
                        np.zeros((16, 8), np.float32), np.zeros(8, np.float32))
    assert res.output.shape == (0, 8)
