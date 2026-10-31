"""SIMT 模型与设备规格的测试。

这些断言全部盯**可复算的量**：峰值算力、带宽、事务数、发散率。
任何一个数字改动了，测试必须红——否则它就不是门禁，是装饰。
"""

from __future__ import annotations

import numpy as np
import pytest

from gkern.gpgpu.device import BlockCoord, Device, DeviceInfo, ThreadCoord
from gkern.gpgpu.memory import BANK_COUNT, TRANSACTION_BYTES, count_transactions
from gkern.gpgpu.samples import (BLOCK, relu_branchy, run_sample,
                                 vec_add_coalesced, vec_add_strided)


def test_peak_flops_matches_formula():
    info = DeviceInfo()
    expected = info.sm_count * info.fp32_cores_per_sm * 2 * info.clock_ghz / 1000.0
    assert info.peak_flops_tflops == pytest.approx(expected)
    # 默认配置：64 SM × 128 核 × 2 × 1.6GHz = 26.2 TFLOPS
    assert info.peak_flops_tflops == pytest.approx(26.2144)


def test_theoretical_bandwidth_matches_formula():
    info = DeviceInfo()
    # 1.6GHz × 2(DDR) × 256bit/8 = 102.4 GB/s
    assert info.theoretical_bandwidth_gbps == pytest.approx(102.4)


def test_roofline_crossover_is_bandwidth_over_flops():
    """算术强度超过「峰值算力/峰值带宽」之后才轮到算力受限。"""
    info = DeviceInfo()
    crossover = info.peak_flops_tflops * 1e12 / (info.theoretical_bandwidth_gbps * 1e9)
    assert crossover == pytest.approx(256.0, rel=1e-6)   # 26.2144e12 / 102.4e9


def test_thread_coord_derives_warp_and_lane():
    tc = ThreadCoord(block=BlockCoord(0), tid=37)
    assert tc.warp_in_block == 1
    assert tc.lane == 5


def test_device_rejects_oversized_block():
    dev = Device()
    problems = dev.validate((1,), (dev.info.threads_per_block_limit + 1,))
    assert problems, "超限 block 必须被拒绝"
    assert dev.validate((1,), (256,)) == []


def test_occupancy_is_capped_at_one():
    dev = Device()
    assert dev.occupancy((256,)) == pytest.approx(8 / 32)
    assert dev.occupancy((4096,)) == pytest.approx(1.0)


def test_count_transactions_groups_by_128b_segments():
    # 32 个相邻 float32 = 128B = 1 个事务
    offsets = np.arange(32, dtype=np.int64) * 4
    assert count_transactions(offsets) == 1
    # 跨到下一个 128B 段就变 2 个
    assert count_transactions(np.array([0, TRANSACTION_BYTES])) == 2
    # 全部落在一个 128B 段内的任意散布仍只算 1 个
    assert count_transactions(np.array([0, 4, 64, 124])) == 1


def test_coalesced_access_is_fully_efficient():
    res, a, b = run_sample(vec_add_coalesced, n=1024)
    np.testing.assert_allclose(res.outputs["out"], a + b, rtol=1e-6)
    # 1024 个元素 / 32 lane = 32 个 warp；每次 load 每 warp 1 个事务
    assert res.memory.global_load_transactions == 64     # a 和 b 各 32
    assert res.memory.global_store_transactions == 32
    assert res.memory.coalescing_efficiency == pytest.approx(1.0)


def test_strided_access_explodes_transaction_count():
    res, a, b = run_sample(vec_add_strided, n=1024)
    # 同样的数据量，事务数变成 32 倍：每 warp 每 load 触碰 32 个不同 128B 段
    assert res.memory.global_load_transactions == 64 * 32
    assert res.memory.coalescing_efficiency < 0.05
    # 数据量没变，所以「请求字节」和合并版完全一样——差的全是浪费
    assert res.memory.global_bytes_requested == 1024 * 4 * 3


def test_branch_on_mixed_sign_diverges():
    res, a, _ = run_sample(relu_branchy, n=1024, seed=1)
    # 正态分布必有正有负 → 每个 warp 都发散
    assert res.report.divergence_rate == pytest.approx(1.0)
    assert res.report.efficiency < 0.75
    np.testing.assert_allclose(res.outputs["out"], np.maximum(a, 0.0), atol=1e-6)


def test_branch_on_uniform_sign_does_not_diverge():
    """全正输入：if 分支永远成立，else 从不执行 → 零发散、效率接近 1。"""
    from gkern.gpgpu.kernel import launch

    n = 1024
    a = np.abs(np.random.default_rng(2).standard_normal(n).astype(np.float32)) + 1.0
    out = np.zeros(n, dtype=np.float32)
    res = launch(relu_branchy, {"a": a, "out": out},
                 grid=(n // BLOCK,), block=(BLOCK,), extra=(n,))
    assert res.report.divergence_rate == 0.0
    assert res.report.efficiency > 0.99


def test_shared_memory_bank_conflict_degree():
    """bank 冲突：32 个 lane 打同一个 bank 时，一次访问被拆成 32 次。"""
    from gkern.gpgpu.memory import SharedMemory, MemoryStats

    stats = MemoryStats()
    smem = SharedMemory(elements=64, stats=stats)
    idx = np.arange(32, dtype=np.int64)
    smem.store(idx, np.arange(32, dtype=np.float32), mask=np.ones(32, bool))
    assert stats.shared_bank_conflicts == 0, "连续地址不应有冲突"

    stats2 = MemoryStats()
    smem2 = SharedMemory(elements=1024, stats=stats2)
    stride_idx = np.arange(32, dtype=np.int64) * BANK_COUNT
    smem2.store(stride_idx, np.ones(32, dtype=np.float32), mask=np.ones(32, bool))
    # 口径：每个活跃 lane 贡献 (冲突度-1) 个被串行化的 lane 槽，
    # 所以 32 路冲突 = 32 × 31 = 992 个额外槽（不是「31 次」这种含糊说法）。
    assert stats2.shared_bank_conflicts == 32 * 31 == 992
    assert stats2.shared_conflict_rate == pytest.approx(31.0)
