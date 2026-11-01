"""访存体系模拟：全局内存与共享内存。

这一层要回答的是两个真实 GPU 上最贵的性能问题：

1. **全局访存是否合并（coalesced）**：一个 warp 的 32 个 lane 同时读 ``x[i]``，
   如果地址连续，硬件把它们拼成少数几个 128B 事务；如果地址散开（跨步访存），
   事务数会放大到接近 32 倍。这里按 128B 段计数事务数，直接给出放大倍数。
2. **共享内存 bank 冲突**：共享内存被切成 32 个 bank（4B 宽）。同一个 warp
   的多个 lane 命中同一个 bank 的不同地址时，硬件必须串行化。这里按
   ``bank = (byte_offset // 4) % 32`` 分组，冲突度 = 单 bank 最大访问数。

两者都用 numpy 的整型数组承载真实数据，所以「事务数」不是拍脑袋估的，
而是从实际索引集合上算出来的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

TRANSACTION_BYTES = 128      # 一次全局内存事务覆盖的字节数
BANK_COUNT = 32              # 共享内存 bank 数
BANK_WIDTH_BYTES = 4         # 每个 bank 的位宽（32bit）


@dataclass
class MemoryStats:
    """一次内核执行内的访存统计。"""

    global_load_transactions: int = 0
    global_store_transactions: int = 0
    global_bytes_requested: int = 0
    shared_bank_conflicts: int = 0
    shared_accesses: int = 0
    smem_bytes: int = 0

    @property
    def coalescing_efficiency(self) -> float:
        """理想事务数 / 实际事务数。理想 = 请求字节 / 128B 上取整。"""
        if not self.global_bytes_requested:
            return 1.0
        ideal = max(1, -(-self.global_bytes_requested // TRANSACTION_BYTES))
        actual = self.global_load_transactions + self.global_store_transactions
        if actual == 0:
            return 1.0
        return min(1.0, ideal / actual)

    @property
    def shared_conflict_rate(self) -> float:
        if not self.shared_accesses:
            return 0.0
        return self.shared_bank_conflicts / float(self.shared_accesses)

    def as_dict(self) -> dict:
        return {
            "global_load_transactions": self.global_load_transactions,
            "global_store_transactions": self.global_store_transactions,
            "global_bytes_requested": self.global_bytes_requested,
            "coalescing_efficiency": round(self.coalescing_efficiency, 4),
            "shared_accesses": self.shared_accesses,
            "shared_bank_conflicts": self.shared_bank_conflicts,
            "shared_conflict_rate": round(self.shared_conflict_rate, 4),
            "smem_bytes": self.smem_bytes,
        }


def count_transactions(byte_offsets: np.ndarray) -> int:
    """把一组字节地址折成 128B 事务数（不同段的个数）。"""
    if byte_offsets.size == 0:
        return 0
    segments = np.unique(np.asarray(byte_offsets, dtype=np.int64) // TRANSACTION_BYTES)
    return int(segments.size)


def bank_conflict_degree(byte_offsets: np.ndarray, bank_count: int = BANK_COUNT,
                         bank_width: int = BANK_WIDTH_BYTES) -> int:
    """返回单 bank 最大命中数（1 表示无冲突）。"""
    if byte_offsets.size == 0:
        return 0
    banks = (np.asarray(byte_offsets, dtype=np.int64) // bank_width) % bank_count
    counts = np.bincount(banks, minlength=bank_count)
    return int(counts.max())


class GlobalMemory:
    """一块（或多块）全局内存。用 numpy 数组做真实承载。"""

    def __init__(self, stats: Optional[MemoryStats] = None) -> None:
        self._buffers: dict[str, np.ndarray] = {}
        self.stats = stats if stats is not None else MemoryStats()

    # ---- 分配 -----------------------------------------------------------
    def alloc(self, name: str, arr: np.ndarray) -> np.ndarray:
        """登记一块缓冲区（不拷贝，直接引用）。"""
        self._buffers[name] = arr
        return arr

    def buffer(self, name: str) -> np.ndarray:
        if name not in self._buffers:
            raise KeyError(f"未登记的缓冲区：{name}")
        return self._buffers[name]

    def names(self) -> list[str]:
        return sorted(self._buffers)

    # ---- 访存（带统计） -------------------------------------------------
    def load(self, buf: np.ndarray, indices: np.ndarray, mask: np.ndarray,
             item_bytes: int = 4, out_of_range: float = 0.0) -> np.ndarray:
        """按 lane 掩码读取，并累计事务数与请求字节。"""
        indices = np.asarray(indices, dtype=np.int64)
        mask = np.asarray(mask, dtype=bool)
        active_idx = indices[mask]
        self.stats.global_bytes_requested += int(active_idx.size * item_bytes)
        self.stats.global_load_transactions += count_transactions(active_idx * item_bytes)

        out = np.full(indices.shape, out_of_range, dtype=np.result_type(buf, np.float32))
        valid = mask & (indices >= 0) & (indices < buf.shape[0])
        out[valid] = buf[indices[valid]]
        return out

    def store(self, buf: np.ndarray, indices: np.ndarray, values: np.ndarray,
              mask: np.ndarray, item_bytes: int = 4) -> None:
        """按 lane 掩码写回，并累计事务数。"""
        indices = np.asarray(indices, dtype=np.int64)
        mask = np.asarray(mask, dtype=bool)
        active_idx = indices[mask]
        self.stats.global_bytes_requested += int(active_idx.size * item_bytes)
        self.stats.global_store_transactions += count_transactions(active_idx * item_bytes)

        valid = mask & (indices >= 0) & (indices < buf.shape[0])
        buf[indices[valid]] = values[valid]


def warp_bank_conflicts(indices: np.ndarray, item_bytes: int = 4,
                        bank_count: int = BANK_COUNT,
                        warp_size: int = 32) -> int:
    """按 **warp 粒度** 统计共享内存访问的额外轮次（冲突数）。

    为什么必须按 32 个 lane 分组：一个 warp 才是硬件真正一起发射的单位。
    整块 tile 一次性算 ``bank_conflict_degree`` 会得出「1024 次访问全撞在
    32 个 bank 上」这种不存在的结论（实测把一次无冲突的整块写算成 31 倍冲突）。
    """
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    extra = 0
    for start in range(0, indices.size, warp_size):
        group = indices[start:start + warp_size]
        degree = bank_conflict_degree(group * item_bytes, bank_count, item_bytes)
        if degree > 1:
            extra += int(group.size) * (degree - 1)
    return extra


class SharedMemory:
    """每 block 一块共享内存，带 bank 冲突统计。

    共享内存里的数据是一维的（以元素为单位），bank 由元素下标推出，
    这样 ``smem[lane * stride]`` 这种经典的冲突写法能被如实测出来。
    """

    def __init__(self, elements: int, stats: Optional[MemoryStats] = None,
                 item_bytes: int = 4, bank_count: int = BANK_COUNT) -> None:
        self.data = np.zeros(elements, dtype=np.float32)
        self.stats = stats if stats is not None else MemoryStats()
        self.item_bytes = item_bytes
        self.bank_count = bank_count
        self.stats.smem_bytes += elements * item_bytes

    def load(self, indices: np.ndarray, mask: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        mask = np.asarray(mask, dtype=bool)
        active = indices[mask]
        self.stats.shared_accesses += int(active.size)
        self.stats.shared_bank_conflicts += warp_bank_conflicts(
            active, self.item_bytes, self.bank_count)
        out = np.zeros(indices.shape, dtype=np.float32)
        valid = mask & (indices >= 0) & (indices < self.data.shape[0])
        out[valid] = self.data[indices[valid]]
        return out

    def store(self, indices: np.ndarray, values: np.ndarray, mask: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        mask = np.asarray(mask, dtype=bool)
        active = indices[mask]
        self.stats.shared_accesses += int(active.size)
        self.stats.shared_bank_conflicts += warp_bank_conflicts(
            active, self.item_bytes, self.bank_count)
        valid = mask & (indices >= 0) & (indices < self.data.shape[0])
        self.data[indices[valid]] = values[valid]
