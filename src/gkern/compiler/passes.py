"""IR 优化 pass：算子融合与分块自动调优。

这两个 pass 是「图算优化」在最小尺度上的可运行版本，也是本仓库唯一敢说
「有真实收益」的地方——收益都能被 :mod:`gkern.compiler.costmodel` 用 IR 本身
算出来，不依赖任何拍脑袋的系数。

1. **算子融合**（:func:`fuse_programs`）
   典型链路 ``matmul -> bias -> gelu`` 在框架里是三个 kernel，中间结果要写回
   全局内存再读出来，访存是 2×中间张量。融合后中间结果留在寄存器/共享内存里，
   访存直接省掉这两笔。这里实现的是最保守也最安全的一种融合：**生产者-消费者
   一对一**（B 只读 A 写出的那块缓冲区，且 A 不再被其他 kernel 读）。

2. **分块自动调优**（:func:`autotune`）
   BLOCK 太小 → 事务不合并、warp 利用率低；太大 → 寄存器压力、尾部浪费。
   调优器枚举候选配置，逐个追踪成 IR，用代价模型挑 ``predicted_ms`` 最小的那个。
   真实 Triton 的 autotune 靠实测；这里没有 GPU，所以用模型——但模型与设备
   模拟器的一致性有测试兜底（见 tests/test_costmodel.py）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import numpy as np

from . import costmodel
from .ir import Program, StoreInstr, Tile
from ..gpgpu.device import DeviceInfo


# ---------------------------------------------------------------------------
# 融合
# ---------------------------------------------------------------------------


def producers(program: Program) -> Dict[str, int]:
    """缓冲区名 -> 写出它的 store 指令的 value tile id。"""
    out: Dict[str, int] = {}
    for node in program.instrs:
        if isinstance(node, StoreInstr):
            out[node.buffer] = node.value
    return out


def can_fuse(a: Program, b: Program, buffer: str) -> tuple[bool, str]:
    """判断 B 能否融合进 A（返回 ``(可否, 拒绝原因)``）。

    把判据单独拿出来是为了让「门能红」：测试可以拿一对**不该融合**的程序
    调它，断言它确实拒绝——如果它永远返回 True，那融合就是装饰。
    """
    if tuple(a.grid) != tuple(b.grid):
        return False, f"grid 不一致：{a.grid} vs {b.grid}"
    if buffer not in producers(a):
        return False, f"{buffer} 不是 {a.name} 的输出"
    if not any(isinstance(n, Tile) and n.op == "load" and n.attrs.get("buffer") == buffer
               for n in b.instrs):
        return False, f"{b.name} 没有读取 {buffer}"
    return True, ""


def fuse_programs(a: Program, b: Program, buffer: str,
                  name: Optional[str] = None) -> Program:
    """把 B 融合进 A：B 对 ``buffer`` 的读直接被 A 的计算结果替换。

    约束（不满足直接报错，不静默降级）：
    - 两个 program 的 grid 必须一致；
    - ``buffer`` 必须由 A 写出、且被 B 读取。
    """
    ok, reason = can_fuse(a, b, buffer)
    if not ok:
        raise ValueError(f"无法融合：{reason}")

    merged = Program(
        name=name or f"{a.name}+{b.name}",
        inputs=dict(a.inputs),
        constexprs={**a.constexprs, **b.constexprs},
        grid=tuple(a.grid),
        num_warps=max(a.num_warps, b.num_warps),
    )

    # 1) 拷 A，跳过对 buffer 的那次 store（中间结果不再落全局内存）
    remap: Dict[int, int] = {}
    for node in a.instrs:
        if isinstance(node, StoreInstr):
            if node.buffer == buffer:
                continue
            merged.emit_store(node.buffer,
                              _tile_of(merged, remap, node.offsets),
                              _tile_of(merged, remap, node.value),
                              None if node.mask is None else _tile_of(merged, remap, node.mask))
        else:
            remap[node.id] = _copy_tile(merged, node, remap).id

    fused_value = remap[producers(a)[buffer]]

    # 2) 拷 B，把对 buffer 的 load 换成 A 的计算结果
    for node in b.instrs:
        if isinstance(node, Tile):
            if node.op == "load" and node.attrs.get("buffer") == buffer:
                remap[node.id] = fused_value
            else:
                remap[node.id] = _copy_tile(merged, node, remap).id
        else:
            merged.emit_store(node.buffer,
                              _tile_of(merged, remap, node.offsets),
                              _tile_of(merged, remap, node.value),
                              None if node.mask is None else _tile_of(merged, remap, node.mask))

    merged.inputs.pop(buffer, None)   # 中间缓冲区不再需要外部提供
    for name_, spec in b.inputs.items():
        if name_ == buffer:
            continue
        merged.inputs.setdefault(name_, spec)
    merged.outputs = [o for o in dict.fromkeys(a.outputs + b.outputs) if o != buffer]
    return merged


def _copy_tile(program: Program, node: Tile, remap: Dict[int, int]) -> Tile:
    args = tuple(remap[a] for a in node.args)
    return program.emit(node.op, node.shape, node.dtype, args, **dict(node.attrs))


def _tile_of(program: Program, remap: Dict[int, int], tid: int) -> Tile:
    return program.tile(remap[tid])


# ---------------------------------------------------------------------------
# 自动调优
# ---------------------------------------------------------------------------


@dataclass
class TuneResult:
    """一次调优的结论。"""

    config: Dict[str, Any]
    program: Program
    cost: costmodel.Cost
    trials: list[dict]

    def as_dict(self) -> dict:
        return {
            "best_config": self.config,
            "predicted_ms": round(self.cost.predicted_ms, 6),
            "bound": self.cost.bound,
            "trials": self.trials,
        }


def autotune(fn: Callable, buffers: Dict[str, np.ndarray], scalars: Dict[str, int],
             grid_fn: Callable[[Dict[str, Any]], Sequence[int]],
             configs: Iterable[Dict[str, Any]],
             device: DeviceInfo | None = None) -> TuneResult:
    """枚举分块配置，用代价模型选最优。

    每个候选都真的被追踪成 IR 并分析，所以 ``trials`` 里的数字是算出来的，
    可以直接贴进报告。
    """
    from .dsl import trace

    dev = device or DeviceInfo()
    trials: list[dict] = []
    best: Optional[tuple[float, Dict[str, Any], Program, costmodel.Cost]] = None

    for cfg in configs:
        grid = tuple(int(g) for g in grid_fn(cfg))
        program = trace(fn, buffers, scalars, cfg, grid)
        cost = costmodel.analyze(program, dev)
        row = {
            "config": dict(cfg),
            "grid": list(grid),
            "predicted_ms": round(cost.predicted_ms, 6),
            "bound": cost.bound,
            "bytes_total": cost.bytes_total,
            "flops": cost.flops,
        }
        trials.append(row)
        if best is None or cost.predicted_ms < best[0]:
            best = (cost.predicted_ms, dict(cfg), program, cost)

    if best is None:
        raise ValueError("配置列表为空，无法调优")
    return TuneResult(config=best[1], program=best[2], cost=best[3], trials=trials)
