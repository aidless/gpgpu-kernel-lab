"""混合精度：把「累加器精度」做成可执行、可测量的对象。

真实 GPU 上 bf16 训练/推理的性能优势有一半来自精度，另一半来自带宽；
但「精度」这件事常常被当成参数表里的一行，没人量它到底损失多少。
这里把它变成三条可测量的断言：

1. ``round_bf16`` 与 ``torch.bfloat16`` 的舍入**逐值相同**（含 tie-to-even、Inf/NaN）；
2. bf16 累加的 softmax 误差**比 fp32 累加大几个数量级**（不是「差不多」）；
3. 两条路径的**代价完全相同**——所以精度是免费的选择，但必须明确选。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gkern.compiler.codegen import _bf16_round
from gkern.ops.mixed_precision import run_softmax_precision, precision_summary


# ------------------------------------------------------------ 舍入语义

def test_bf16_round_matches_torch_cast() -> None:
    rng = np.random.default_rng(20261101)
    x = np.concatenate([
        rng.standard_normal(20_000).astype(np.float32),
        (rng.standard_normal(2_000) * 1e-30).astype(np.float32),
        (rng.standard_normal(2_000) * 1e30).astype(np.float32),
        np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32),
        np.nextafter(np.float32(1.0), np.float32(2.0)).repeat(5),
    ]).astype(np.float32)

    want = torch.from_numpy(x).to(torch.bfloat16).float().numpy()
    got = _bf16_round(x)

    assert got.dtype == np.float32
    nan = np.isnan(want)
    assert np.array_equal(nan, np.isnan(got))
    assert np.array_equal(got[~nan].view(np.uint32), want[~nan].view(np.uint32)), (
        "round_bf16 与 torch.bfloat16 的位模式不一致"
    )


def test_bf16_round_ties_to_even() -> None:
    """两个恰好落在 bfloat16 中点的值，都舍到「尾数为偶」的那侧。

    1.0 附近的 bfloat16 间距是 2^-7，所以中点是 1 + 2^-8（舍到 1.0）和
    1 + 3·2^-8（舍到 1 + 2·2^-7 = 1.015625，因为它的尾数是偶数）。
    """
    tie_down = np.float32(1.0) + np.float32(2.0 ** -8)     # 中点在 1.0 与 1+2^-7 之间
    tie_up = np.float32(1.0) + np.float32(3 * 2.0 ** -8)   # 中点在 1+2^-7 与 1+2·2^-7 之间
    assert _bf16_round(np.array([tie_down], np.float32))[0] == np.float32(1.0)
    assert _bf16_round(np.array([tie_up], np.float32))[0] == np.float32(1.0 + 2 * 2 ** -7)
    # 与 torch 的结论一致
    t = torch.tensor([tie_down, tie_up]).to(torch.bfloat16).float().numpy()
    assert np.array_equal(_bf16_round(np.array([tie_down, tie_up], np.float32)), t)


def test_bf16_round_keeps_nan_and_inf() -> None:
    x = np.array([np.nan, np.inf, -np.inf], np.float32)
    got = _bf16_round(x)
    assert np.isnan(got[0])
    assert got[1] == np.inf and got[2] == -np.inf


# ------------------------------------------------------------ 累加误差

def test_bf16_accumulation_error_is_orders_of_magnitude_worse() -> None:
    rng = np.random.default_rng(11)
    x = (rng.standard_normal((256, 1024)) * 3.0).astype(np.float32)

    fp32 = run_softmax_precision(x, accumulate="fp32")
    bf16 = run_softmax_precision(x, accumulate="bf16")

    assert fp32.max_error < 1e-6, f"fp32 路径误差过大：{fp32.max_error}"
    assert bf16.max_error > 1e-3, f"bf16 路径误差小得可疑：{bf16.max_error}"
    assert bf16.max_error / fp32.max_error > 100, (
        f"bf16/fp32 误差比只有 {bf16.max_error / fp32.max_error:.1f}×，"
        "说明 bf16 路径其实没有真的降精度"
    )


def test_bf16_rows_still_normalize() -> None:
    rng = np.random.default_rng(13)
    x = (rng.standard_normal((64, 512)) * 2.0).astype(np.float32)
    out = run_softmax_precision(x, accumulate="bf16").output
    assert np.allclose(out.sum(axis=-1), 1.0, atol=1e-2)
    assert (out >= 0).all()


def test_precision_does_not_change_cost() -> None:
    """两条路径的访存字节与 FLOP 完全相同：精度是免费的，但必须显式选。"""
    rng = np.random.default_rng(17)
    x = (rng.standard_normal((128, 256))).astype(np.float32)
    fp32 = run_softmax_precision(x, accumulate="fp32")
    bf16 = run_softmax_precision(x, accumulate="bf16")

    assert bf16.cost.bytes_total == fp32.cost.bytes_total
    # bf16 路径多了 4 处舍入：输入 tile 与指数结果 tile 是整行（各 numel=cols×rows），
    # 行最大值与归约和是标量 tile（shape=(1,)，每 program 一个）。记账必须按 IR 里
    # 真实存在的 tile 算，而不是按「我以为有几次舍入」算。
    rows, cols = x.shape
    extra = bf16.cost.flops - fp32.cost.flops
    expected = 2 * x.size + 2 * rows
    assert extra == expected, f"舍入的 FLOP 记账不对：{extra} vs {expected}"


def test_precision_summary_shape() -> None:
    rng = np.random.default_rng(19)
    x = (rng.standard_normal((32, 128))).astype(np.float32)
    rows = precision_summary(x)
    assert [r["accumulate"] for r in rows] == ["fp32", "bf16"]
    assert rows[0]["max_error"] < rows[1]["max_error"]
    assert rows[0]["bytes"] == rows[1]["bytes"]


def test_unknown_accumulate_is_rejected() -> None:
    with pytest.raises(ValueError, match="accumulate"):
        run_softmax_precision(np.zeros((4, 8), np.float32), accumulate="fp16")
