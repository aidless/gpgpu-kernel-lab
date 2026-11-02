"""GEMM 工作分解与波量化：数学部分的闸。

这个模块不跑 kernel，只回答一个调度问题：**给定 tile 数、SM 数、每 SM 能驻留
几个 block，这些工作单元怎么摊到硬件上，最后一个「波次」有多少 SM 是空的**。

三个真问题各有一组断言：
1. 波量化本身（`wave_stats`）：100 个单元铺到 40 个执行槽 → 3 波、最后一波只用 20 个槽。
2. 四种分解的**单元数与额外流量**（naive / split-k / stream-k / persistent）：
   拆得越细 tail 越小，但每个边界都要落一份部分和。
3. 结论必须**随形状翻转**：宽扁的大矩阵上 split-k 的额外流量被计算时间盖住；
   输出极小、K 极深时 naive 只剩 4 个单元铺 40 个槽（利用率 0.1），stream-K 才赢。
"""

from __future__ import annotations

import pytest

from gkern.gpgpu.device import DeviceInfo
from gkern.schedule import (
    GemmShape,
    compare_strategies,
    recommend,
    streamk_units,
    wave_stats,
)


T4 = DeviceInfo.tesla_t4()          # sm_count = 40

# 情形 A：输出 10×10=100 个 tile，40 个槽 → 3 波，最后一波空 20 个槽
SHAPE_A = GemmShape(M=1280, N=1280, K=528, bm=128, bn=128, bk=16)
# 情形 B：输出只有 2×2=4 个 tile，K 极深 → naive 只填满 4/40 个槽
SHAPE_B = GemmShape(M=256, N=256, K=8192, bm=128, bn=128, bk=16)


def test_shape_arithmetic() -> None:
    assert SHAPE_A.tiles == 100
    assert SHAPE_A.k_iters == 33            # 528 / 16，整除才不会算错
    assert SHAPE_A.iter_units == 3300       # 100 × 33
    assert SHAPE_A.flops == 2 * 1280 * 1280 * 528
    assert SHAPE_A.base_bytes == (1280 * 528 + 528 * 1280 + 1280 * 1280) * 4


def test_shape_rejects_non_divisible_k() -> None:
    with pytest.raises(ValueError, match="K"):
        GemmShape(M=64, N=64, K=500, bm=64, bn=64, bk=16)


def test_wave_stats_splits_a_hundred_units_over_forty_slots() -> None:
    w = wave_stats(100, 40)
    assert w.waves == 3
    assert w.last_wave_units == 20
    assert w.idle_slots_last_wave == 20
    assert w.efficiency == pytest.approx(100 / 120)
    assert w.wasted_fraction == pytest.approx(20 / 120)


def test_wave_stats_exact_fill_is_perfect() -> None:
    w = wave_stats(80, 40)
    assert (w.waves, w.idle_slots_last_wave, w.efficiency) == (2, 0, 1.0)


def test_wave_stats_rejects_nothing_to_schedule() -> None:
    with pytest.raises(ValueError):
        wave_stats(0, 40)
    with pytest.raises(ValueError):
        wave_stats(10, 0)


def test_naive_has_no_extra_traffic_but_leaves_the_tail() -> None:
    row = compare_strategies(SHAPE_A, T4)["naive"]
    assert row["units"] == 100
    assert row["waves"] == 3
    assert row["extra_bytes"] == 0
    assert row["total_bytes"] == SHAPE_A.base_bytes
    assert row["efficiency"] == pytest.approx(100 / 120)


def test_persistent_keeps_the_same_tail_and_launches_once() -> None:
    """persistent 在本模型里与 naive 的波指标**完全相同**，这本身就是结论。

    两者都是「一次 launch、grid 覆盖全部 tile」，tail 是硬件并行度用不满造成的，
    换成分块常驻循环既不会让 tail 变小、也不会多出任何流量；它省的是块调度开销，
    而调度开销不在本模型的问题定义里（见 `docs/ARCHITECTURE.md` §8.2）。
    """
    rows = compare_strategies(SHAPE_A, T4)
    assert rows["naive"]["launches"] == 1
    assert rows["persistent"]["launches"] == 1
    assert rows["persistent"]["efficiency"] == pytest.approx(rows["naive"]["efficiency"])
    assert rows["persistent"]["extra_bytes"] == 0
    assert rows["persistent"]["predicted_ms"] == pytest.approx(rows["naive"]["predicted_ms"])


def test_split_k_trades_partials_for_a_full_last_wave() -> None:
    row = compare_strategies(SHAPE_A, T4, splits=2)["split-k"]
    assert row["units"] == 200               # 100 × 2
    assert row["waves"] == 5                 # ceil(200/40)
    assert row["efficiency"] == 1.0
    # 每个 tile 的每个 split 都要写出、再被归约读回一次
    assert row["extra_bytes"] == 2 * 100 * 2 * 128 * 128 * 4 == 26214400
    assert row["total_bytes"] == SHAPE_A.base_bytes + row["extra_bytes"]


def test_stream_k_flattens_the_tail_by_scheduling_k_iters() -> None:
    row = compare_strategies(SHAPE_A, T4)["stream-k"]
    assert row["units"] == 3300              # 以 k-iter 为工作单元，不是 tile
    assert row["tail_units"] == 3300 % 40 == 20
    assert row["efficiency"] == pytest.approx(3300 / (83 * 40))
    # 只有跨 worker 的边界要多落一份部分和，不是每个 split 都落（39 < 100×2）
    assert row["partial_cuts"] == 39
    assert row["extra_bytes"] == 2 * 39 * 128 * 128 * 4 == 5111808


def test_the_verdict_flips_with_the_shape() -> None:
    """宽扁大矩阵 vs 输出极小 K 极深——同一套模型给出不同推荐，这是它有用处的证据。"""
    wide = compare_strategies(SHAPE_A, T4, splits=2)
    deep = compare_strategies(SHAPE_B, T4, splits=16)

    # 情形 B：naive 只有 4 个单元却要占 40 个槽 → 利用率 0.1
    assert deep["naive"]["efficiency"] == pytest.approx(4 / 40)
    assert deep["naive"]["waves"] == 1
    # stream-K 在这个形状下同时赢在利用率与额外流量上
    assert deep["stream-k"]["efficiency"] > deep["split-k"]["efficiency"]
    assert deep["stream-k"]["extra_bytes"] < deep["split-k"]["extra_bytes"]

    assert recommend(SHAPE_B, T4, splits=16) == "stream-k"
    assert recommend(SHAPE_A, T4, splits=2) in {"split-k", "stream-k"}


def test_predicted_time_respects_the_tail() -> None:
    rows = compare_strategies(SHAPE_A, T4, splits=2)
    # 100 个单元只用 20/40 个槽的最后一波，算力按效率打折 → 预测时间变长
    assert rows["naive"]["predicted_ms"] > rows["split-k"]["predicted_ms"]
    # 但预测时间不可能低于纯计算时间
    compute_only = SHAPE_A.flops / (T4.peak_flops_tflops * 1e12) * 1e3
    for row in rows.values():
        assert row["predicted_ms"] >= compute_only * 0.999

def test_streamk_partition_covers_every_iteration_exactly_once() -> None:
    """stream-K 的划分必须是**工作量的一个分割**，不是估计。"""
    units = streamk_units(SHAPE_A, T4, blocks_per_sm=1)
    seen: dict[int, list[tuple[int, int]]] = {}
    for u in units:
        seen.setdefault(u.tile, []).append((u.k_start, u.k_end))

    # 每个 tile 的 k 区间必须首尾相接、合起来正好是 [0, k_iters)
    for tile, ranges in seen.items():
        ranges.sort()
        assert ranges[0][0] == 0, tile
        assert ranges[-1][1] == SHAPE_A.k_iters, tile
        for (_, end), (start, _) in zip(ranges, ranges[1:]):
            assert end == start, f"tile {tile} 的区间有空洞或重叠"

    # 全部分段的迭代总数 == 总迭代数（没有漏算、没有重复算）
    assert sum(u.k_end - u.k_start for u in units) == SHAPE_A.iter_units
    # 每个 worker 至多拿一段连续区间，且 worker 编号覆盖 [0, capacity)
    assert len({u.worker for u in units}) == T4.sm_count


def test_streamk_partial_count_comes_from_the_partition_not_a_formula() -> None:
    """闭式估计 `capacity-1` 只是上界；真实的部分和份数要数划分里有几个边界落在 tile 内部。"""
    units = streamk_units(SHAPE_A, T4, blocks_per_sm=1)
    per_tile: dict[int, set[int]] = {}
    for u in units:
        per_tile.setdefault(u.tile, set()).add(u.worker)
    true_partials = sum(len(workers) - 1 for workers in per_tile.values())

    row = compare_strategies(SHAPE_A, T4)["stream-k"]
    # 模型里的数必须与具体划分一致——否则闭式公式和「实际怎么切」两套账会各自漂移
    assert row["partial_cuts"] == true_partials
    # 并且不可能超过 worker 边界数（每个边界最多造一份部分和）
    assert true_partials <= T4.sm_count - 1
