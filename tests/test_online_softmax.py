"""在线 softmax 的断言：数值必须和两遍扫描一致，字节必须少一整遍输入。"""

from __future__ import annotations

import numpy as np
import pytest

from gkern.ops.online_softmax import (compare_online_vs_recompute,
                                      run_online_softmax,
                                      run_recompute_softmax)
from gkern.ops.softmax import reference_softmax


def _x(rows: int, cols: int, seed: int = 0, scale: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((rows, cols)) * scale).astype(np.float32)


def test_online_matches_reference_when_row_fits_whole_blocks() -> None:
    x = _x(8, 1024)
    r = run_online_softmax(x, block=256)
    assert r.blocks == 4
    assert np.allclose(r.output, reference_softmax(x), atol=1e-6)


def test_online_matches_reference_with_tail_block() -> None:
    """1000 列、块 256 → 最后一块只有 232 列有效，掩码错一位就会在行和上露出来。"""
    x = _x(5, 1000)
    r = run_online_softmax(x, block=256)
    assert r.blocks == 4
    assert np.allclose(r.output, reference_softmax(x), atol=1e-6)
    assert np.allclose(r.output.sum(axis=-1), 1.0, atol=1e-6)


def test_online_and_recompute_agree() -> None:
    x = _x(6, 999, seed=11)
    a = run_online_softmax(x, block=128)
    b = run_recompute_softmax(x, block=128)
    assert np.allclose(a.output, b.output, atol=1e-6)


def test_online_reads_input_once_recompute_reads_it_twice() -> None:
    """字节账：online = 1 读 + 1 写；recompute = 2 读 + 1 写。差值恰是一整遍输入。"""
    rows, cols = 8, 1024
    x = _x(rows, cols)
    a = run_online_softmax(x, block=256)
    b = run_recompute_softmax(x, block=256)
    one_pass = rows * cols * 4
    assert a.cost.bytes_total == 2 * one_pass, a.cost.bytes_total
    assert b.cost.bytes_total == 3 * one_pass, b.cost.bytes_total
    assert b.cost.bytes_total - a.cost.bytes_total == one_pass


def test_online_is_numerically_stable_on_large_magnitudes() -> None:
    """整行 +10000 时，不减 max 的实现会 exp 溢出成 inf；这里必须仍然等于参考解。"""
    x = _x(4, 512, seed=3)
    x = (x + np.float32(10000.0)).astype(np.float32)
    r = run_online_softmax(x, block=128)
    assert np.isfinite(r.output).all()
    assert np.allclose(r.output, reference_softmax(x), atol=1e-6)


def test_block_must_be_positive() -> None:
    with pytest.raises(ValueError):
        run_online_softmax(_x(2, 64), block=0)


def test_compare_helper_reports_byte_difference() -> None:
    rows = compare_online_vs_recompute((4, 512), block=128)
    assert [r["variant"] for r in rows] == ["online", "recompute"]
    # online 读一遍 + 写一遍 = 2 遍；两遍扫描读两遍 + 写一遍 = 3 遍
    assert rows[0]["bytes_total"] * 3 == rows[1]["bytes_total"] * 2
    # 差出来的必须**恰好是一整遍输入**：4 行 × 512 列 × 4 字节
    assert rows[1]["bytes_total"] - rows[0]["bytes_total"] == 4 * 512 * 4
    assert all(r["equal_to_recompute"] for r in rows)
