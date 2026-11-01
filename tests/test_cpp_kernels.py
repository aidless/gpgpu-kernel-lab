"""C++ 参考实现与 numpy 参考实现的对拍。

这里测的是**同一语义的两份实现**：

- Python/numpy 侧：`gkern.gpgpu.memory.count_transactions` /
  `bank_conflict_degree` / `warp_bank_conflicts`（教学模型，nights 在 trace 里被调用）
- C++/pybind11 侧：`gkern_cpp`（被打包成可选扩展，用来回答「这份语义在 C++ 里
  怎么写、以及快多少」）

对拍的价值不在「结果一样」这句结论，而在**跨语言语义差**上：Python 的 `//`
是向下取整，C++ 的 `/` 是向零截断。负偏移量在两者下会分到不同段，所以本文件
专门用负偏移钉住这条边界（见 `test_negative_offsets_match_python_floor_semantics`）。

模块缺失时整文件跳过——扩展是可选构建项（`pip install -e ".[cpp]"`），
但 `scripts/verify_cpp.py` 会在**要求它必须存在**的场合把它当硬门禁。
"""

from __future__ import annotations

import numpy as np
import pytest

import gkern.gpgpu.memory as mem

gkern_cpp = pytest.importorskip(
    "gkern_cpp",
    reason="C++ 扩展未构建；先 pip install -e \".[cpp]\"",
)


# ---------------------------------------------------------------- 事务计数

@pytest.mark.parametrize("offsets", [
    np.array([], dtype=np.int64),
    np.array([0], dtype=np.int64),
    np.array([0, 4, 8, 12], dtype=np.int64),          # 同一 128B 段
    np.array([0, 127, 128], dtype=np.int64),          # 跨越边界：两段
    np.array([127, 128, 255, 256], dtype=np.int64),   # 两段
    np.arange(0, 1024, 4, dtype=np.int64),            # 每 32 个元素一段
    np.arange(0, 8192, 128, dtype=np.int64),          # 一元素一段
])
def test_count_transactions_matches_numpy(offsets: np.ndarray) -> None:
    expected = mem.count_transactions(offsets)
    got = gkern_cpp.count_transactions(offsets)
    assert got == expected


def test_count_transactions_randomized_matches_numpy() -> None:
    rng = np.random.default_rng(20261031)
    for _ in range(64):
        n = int(rng.integers(0, 200))
        offsets = rng.integers(0, 100_000, size=n).astype(np.int64)
        assert gkern_cpp.count_transactions(offsets) == mem.count_transactions(offsets)


def test_negative_offsets_match_python_floor_semantics() -> None:
    """Python 的 // 向下取整：-1 // 128 == -1，而 C++ 的 -1 / 128 == 0。

    真实的字节偏移不会是负的，但「两份实现语义必须逐值相同」这条不能靠
    「输入总是合法」来保证——所以专门把这条差异钉在测试里。
    """
    offsets = np.array([-1, -128, -129, 0], dtype=np.int64)
    assert gkern_cpp.count_transactions(offsets) == mem.count_transactions(offsets)
    # 向下取整：-1//128 == -1、-128//128 == -1、-129//128 == -2、0//128 == 0 → 3 段。
    # 若按向零截断：-1/128 == 0、-128/128 == -1、-129/128 == -1、0/128 == 0 → 只有 2 段。
    assert mem.count_transactions(offsets) == 3


# ---------------------------------------------------------------- bank 冲突

@pytest.mark.parametrize("n", [0, 1, 31, 32, 33, 64, 65, 128])
def test_bank_conflict_degree_matches_numpy(n: int) -> None:
    offsets = np.arange(n, dtype=np.int64) * 4
    assert gkern_cpp.bank_conflict_degree(offsets) == mem.bank_conflict_degree(offsets)


def test_bank_conflict_degree_stride_32_is_worst_case() -> None:
    """行宽 32 个 float 的按列读：32 个 lane 全撞同一个 bank。"""
    offsets = np.arange(32, dtype=np.int64) * 32 * 4
    assert gkern_cpp.bank_conflict_degree(offsets) == 32
    assert mem.bank_conflict_degree(offsets) == 32

    offsets33 = np.arange(32, dtype=np.int64) * 33 * 4
    assert gkern_cpp.bank_conflict_degree(offsets33) == 1
    assert mem.bank_conflict_degree(offsets33) == 1


def test_bank_conflict_degree_randomized_matches_numpy() -> None:
    rng = np.random.default_rng(7)
    for _ in range(64):
        n = int(rng.integers(0, 300))
        offsets = (rng.integers(0, 512, size=n) * 4).astype(np.int64)
        assert gkern_cpp.bank_conflict_degree(offsets) == mem.bank_conflict_degree(offsets)


def test_warp_bank_conflicts_32_vs_33_pitch() -> None:
    """把 32×32 的行主序缓冲按列读：pitch=32 每个 lane 都撞同一个 bank，pitch=33 全避开。

    - 单个 warp（32 lane 读同一列）：pitch=32 → 32×31 = **992** 额外轮次，
      这正是 README 坑 #9 引用的那个数；
    - 整块 32 warp 全读一遍：32×992 = **31744**；
    - pitch=33 两种情形都是 0。

    C++ 侧复算一遍，防止「只有 Python 那条路径是对的」。
    """
    lanes = np.arange(32, dtype=np.int64)
    one_warp_32 = (lanes * 32)[None, :].reshape(-1)          # 32 lane，同一列
    one_warp_33 = (lanes * 33)[None, :].reshape(-1)
    tile_32 = (lanes[None, :] * 32 + lanes[:, None]).reshape(-1)   # 32 warp × 32 lane
    tile_33 = (lanes[None, :] * 33 + lanes[:, None]).reshape(-1)

    assert gkern_cpp.warp_bank_conflicts(one_warp_32) == 992
    assert gkern_cpp.warp_bank_conflicts(tile_32) == 31744
    assert gkern_cpp.warp_bank_conflicts(one_warp_33) == 0
    assert gkern_cpp.warp_bank_conflicts(tile_33) == 0
    for arr in (one_warp_32, one_warp_33, tile_32, tile_33):
        assert gkern_cpp.warp_bank_conflicts(arr) == mem.warp_bank_conflicts(arr)


def test_warp_bank_conflicts_randomized_matches_numpy() -> None:
    rng = np.random.default_rng(99)
    for _ in range(32):
        n = int(rng.integers(0, 200))
        indices = rng.integers(0, 2048, size=n).astype(np.int64)
        for item_bytes in (4, 8):
            assert (gkern_cpp.warp_bank_conflicts(indices, item_bytes)
                    == mem.warp_bank_conflicts(indices, item_bytes))


def test_warp_bank_conflicts_accepts_non_multiple_of_32() -> None:
    """尾块不足 32 lane 时，冲突按实际 lane 数计，不能补到 32。"""
    indices = np.arange(0, 40, dtype=np.int64) * 32   # 40 个元素 = 32 + 8
    assert gkern_cpp.warp_bank_conflicts(indices) == mem.warp_bank_conflicts(indices)
