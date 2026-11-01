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
   模拟器的一致性有测试兜底（见 tests/test_compiler.py 与 tests/test_gpgpu_model.py）。
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


def _sig(program: Program, tid: int, memo: Dict[int, tuple]) -> tuple:
    """把一条 Tile（及其依赖子图）归一成可比较的结构签名。

    用于跨 program 比较「同一条表达式」：两个 program 各自 trace 出来的
    ``pid*N + arange(...)`` 指令 id 不同，但结构相同，必须判等才能证明
    生产者的写入地址与消费者的读取地址逐 lane 对齐。
    """
    if tid in memo:
        return memo[tid]
    node = program.instrs[tid]
    if not isinstance(node, Tile):
        raise TypeError(f"#{tid} 不是值节点")
    key = (
        node.op,
        node.shape,
        np.dtype(node.dtype).str,
        tuple(sorted((k, repr(v)) for k, v in node.attrs.items())),
        tuple(_sig(program, arg, memo) for arg in node.args),
    )
    memo[tid] = key
    return key


def _downstream_nodes(program: Program, root: int) -> list:
    """返回 B 里所有（直接或间接）依赖 root 的 **Tile 与 StoreInstr**。

    早期只沿 Tile.args 走，于是 StoreInstr 从来不是「下游」——判据 7 里
    「检查 store 的掩码」那段是死代码，B 多写一个无掩码输出会被静默放行。
    现在把 store 的 offsets/value/mask 也当成依赖边，store 就能被检查到。
    """
    dependents: Dict[int, list] = {}
    for node in program.instrs:
        if isinstance(node, Tile):
            for arg in node.args:
                dependents.setdefault(arg, []).append(node)
        elif isinstance(node, StoreInstr):
            for ref in (node.offsets, node.value, node.mask):
                if ref is not None:
                    dependents.setdefault(ref, []).append(node)
    seen_ids: set = set()
    out: list = []
    stack = [root]
    while stack:
        cur = stack.pop()
        for node in dependents.get(cur, ()):
            key = id(node)
            if key not in seen_ids:
                seen_ids.add(key)
                out.append(node)
                if isinstance(node, Tile):   # store 不是值节点，无下游可走
                    stack.append(node.id)
    return out


def _loads_of(program: Program, buffer: str) -> list:
    return [n for n in program.instrs
            if isinstance(n, Tile) and n.op == "load" and n.attrs.get("buffer") == buffer]


def _stores_to(program: Program, buffer: str) -> list:
    return [n for n in program.instrs
            if isinstance(n, StoreInstr) and n.buffer == buffer]


def can_fuse(a: Program, b: Program, buffer: str) -> tuple[bool, str]:
    """判断 B 能否融合进 A（返回 ``(可否, 拒绝原因)``）。

    把判据单独拿出来是为了让「门能红」：测试可以拿一对**不该融合**的程序
    调它，断言它确实拒绝——如果它永远返回 True，那融合就是装饰。

    这不是一个「通用安全闸」，而是**保守的一对一、逐 lane 对齐**融合：
    只有当 B 对中间缓冲区的每一次读，其地址表达式与 A 的写地址在结构上完全
    相同、shape/dtype 一致、且 B 对该值的全部使用都保持原 shape 并落在与 A
    相同的掩码之下时才放行。任一条不满足就拒绝——宁可少融合一个 kernel，
    也不能产出融合前后数值不同却静默通过的 kernel。
    """
    # 1) 同一套 block 网格
    if tuple(a.grid) != tuple(b.grid):
        return False, f"grid 不一致：{a.grid} vs {b.grid}"

    # 2) A 对 buffer 必须**恰好写一次**（多次写时会话只保留最后一次，前一次被丢弃）
    stores_a = _stores_to(a, buffer)
    if not stores_a:
        return False, f"{buffer} 不是 {a.name} 的输出"
    if len(stores_a) != 1:
        return False, f"{a.name} 对 {buffer} 写了 {len(stores_a)} 次，融合只能保一次"
    store_a = stores_a[0]

    # 3) A 不能也读 buffer（就地 read-modify-write 时融合会 pop 掉输入 → KeyError）
    if _loads_of(a, buffer):
        return False, f"{a.name} 自己也读 {buffer}（就地读写无法融合）"

    # 4) B 对 buffer 必须**恰好读一次**
    loads_b = _loads_of(b, buffer)
    if not loads_b:
        return False, f"{b.name} 没有读取 {buffer}"
    if len(loads_b) != 1:
        return False, f"{b.name} 对 {buffer} 读了 {len(loads_b)} 次，无法一一替换"
    load_b = loads_b[0]

    # 5) 写地址 == 读地址（逐 lane 对齐）；shape 与 dtype 一致
    memo_a: Dict[int, tuple] = {}
    memo_b: Dict[int, tuple] = {}
    if _sig(a, store_a.offsets, memo_a) != _sig(b, load_b.args[0], memo_b):
        return False, (f"{a.name} 写 {buffer} 的地址与 {b.name} 读 {buffer} 的地址不同"
                       "（偏移/halo 不一致，融合后会错位）")
    value_a = a.tile(store_a.value)
    if tuple(value_a.shape) != tuple(load_b.shape):
        return False, (f"shape 不一致：{a.name} 写 {value_a.shape}，{b.name} 读 {load_b.shape}")
    if np.dtype(value_a.dtype) != np.dtype(load_b.dtype):
        return False, (f"dtype 不一致：{a.name} 写 {np.dtype(value_a.dtype).name}，"
                       f"{b.name} 读 {np.dtype(load_b.dtype).name}")

    # 6) 掩码一致：B 读 buffer 的掩码、以及 B 里所有用到该值的 store 掩码，
    #    都必须与 A 写 buffer 的掩码结构相同（否则被掩掉的 lane 语义会变）
    store_mask = None if store_a.mask is None else _sig(a, store_a.mask, memo_a)
    load_mask_tid = load_b.args[1] if len(load_b.args) > 1 else None
    load_mask = None if load_mask_tid is None else _sig(b, load_mask_tid, memo_b)
    if store_mask != load_mask:
        return False, f"{a.name} 写 {buffer} 与 {b.name} 读 {buffer} 的掩码不一致"

    # 7) B 对读到的值必须「逐 lane」使用：下游任何 Tile 的形状都必须与读到的 tile 相同
    #    （sum/max/dot/跨 lane 广播会改变形状 → 被掩掉的 lane 会污染结果）
    for node in _downstream_nodes(b, load_b.id):
        if isinstance(node, Tile) and tuple(node.shape) != tuple(load_b.shape):
            return False, (f"{b.name} 对 {buffer} 做了跨 lane 归约/广播（形状 "
                           f"{load_b.shape} → {node.shape}），融合不安全")
        if isinstance(node, StoreInstr):
            store_m = None if node.mask is None else _sig(b, node.mask, memo_b)
            if store_m != store_mask:
                return False, (f"{b.name} 把 {buffer} 的结果写到 {node.buffer} 时掩码与生产者不一致")

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
