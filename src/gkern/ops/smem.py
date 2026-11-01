"""共享内存的 bank 冲突演示：同一个算子，行宽差 1 个 float，冲突差 31 路。

GPU 的共享内存被切成 32 个 bank（这里按 4 字节宽、32 bank 建模）。一个 warp 的
32 个 lane 若同时访问**同一个 bank 的不同地址**，硬件只能串行化——这就是 bank 冲突。

经典踩法：行主序二维面板按列读。行宽 32 个 float 时，``[0, 32, 64, ...]`` 全部落在
bank 0，32 路冲突；把行宽改成 33（末尾多 1 个 padding），下标记变成 ``[0, 33, 66, ...]``，
32 个 lane 各占一个 bank，冲突消失。

这不是「理论上会慢」：``shared_bank_conflicts`` 与 ``shared_conflict_rate``
都是从执行器里数出来的，断言就写在这两个数上。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..compiler import codegen, compile_kernel
from ..compiler import dsl as tl
from ..compiler.dsl import constexpr
from ..compiler.ir import Program
from ..gpgpu.memory import MemoryStats


def smem_column_kernel(OUT, BLOCK: constexpr, PITCH: constexpr):
    """把 BLOCK×BLOCK 面板按行宽 ``PITCH`` 写进共享内存，再按列读出。"""
    tile = tl.smem_alloc("tile", (BLOCK, PITCH))
    ri = tl.expand_dims(tl.arange(0, BLOCK), 1)          # (BLOCK,1) 行号
    cj = tl.expand_dims(tl.arange(0, BLOCK), 0)          # (1,BLOCK) 列号
    write_idx = ri * PITCH + cj                          # (BLOCK,BLOCK) 行主序、行宽 PITCH
    payload = ri * BLOCK + cj                            # 填一个可验证的图案
    tl.smem_store(tile, write_idx, tl.cast(payload, np.float32))
    # 每个 lane 读同一列的不同行：下标步长恰好是 PITCH
    col_idx = tl.arange(0, BLOCK) * PITCH
    tl.store(OUT, tl.arange(0, BLOCK), tl.smem_load(tile, col_idx))


@dataclass
class SmemColumnResult:
    output: np.ndarray
    stats: MemoryStats
    program: Program


def run_smem_column(pitch: int = 32, block: int = 32) -> SmemColumnResult:
    """跑一次按列读实验。``pitch == block`` 时是 32 路冲突，``pitch == block + 1`` 时无冲突。"""
    if block != 32:
        raise ValueError("BLOCK 必须等于 warp 宽度 32，否则冲突路数不可比")
    out = np.zeros((block,), dtype=np.float32)
    program = compile_kernel(smem_column_kernel, {"OUT": out}, None,
                             {"BLOCK": block, "PITCH": pitch}, (1,))
    outputs, stats = codegen.run_with_stats(program, {"OUT": out})
    return SmemColumnResult(outputs["OUT"], stats, program)
