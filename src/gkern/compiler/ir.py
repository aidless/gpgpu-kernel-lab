"""中间表示（IR）：Tile 值节点 + Program（一个 block 级程序）。

为什么需要 IR
-------------
如果直接把 Python 函数当 kernel 跑，就只能在「标量逐线程」和「整张量 numpy」
两个极端里选：前者慢到不可用，后者没有块级语义，算不出分块/融合/自动调优的收益。
Triton 的做法是先**符号执行**一遍 kernel，把 `tl.load/tl.dot/tl.sum` 这些块级
原语记录下来形成 IR，再对 IR 做 pass 和 codegen。

本模块就是那套 IR 的最小可用版本：
- :class:`Tile` 是 IR 里的值（带 shape/dtype/op/依赖），
- :class:`StoreInstr` 是副作用指令（写全局内存），
- :class:`Program` 是一次 block 级程序（含 grid、常量参数、指令序列）。

IR 是**纯数据**，不依赖 numpy 的求值，所以同一份 IR 可以被三种后端消费：
numpy 执行器（正确性参考）、代价模型（估算访存/算力）、SIMT 设备模拟器（实测发散与事务）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Tile:
    """IR 中的一个值。``id`` 是它在 Program 指令表里的位置。"""

    id: int
    op: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    args: Tuple[int, ...] = ()
    attrs: Dict[str, Any] = field(default_factory=dict)

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    @property
    def nbytes(self) -> int:
        return self.numel * int(np.dtype(self.dtype).itemsize)

    @property
    def is_scalar(self) -> bool:
        return len(self.shape) == 0

    # ---- 运算符重载：让 kernel 写起来像 numpy/Triton ---------------------
    def _tracer(self):
        from .dsl import _t
        return _t()

    def __add__(self, other):
        return self._tracer().add(self, other)

    def __radd__(self, other):
        return self._tracer().add(other, self)

    def __sub__(self, other):
        return self._tracer().sub(self, other)

    def __rsub__(self, other):
        return self._tracer().sub(other, self)

    def __mul__(self, other):
        return self._tracer().mul(self, other)

    def __rmul__(self, other):
        return self._tracer().mul(other, self)

    def __truediv__(self, other):
        return self._tracer().div(self, other)

    def __rtruediv__(self, other):
        return self._tracer().div(other, self)

    def __and__(self, other):
        return self._tracer().and_(self, other)

    def __or__(self, other):
        return self._tracer().or_(self, other)

    def __neg__(self):
        return self._tracer().neg(self)

    def __lt__(self, other):
        return self._tracer().lt(self, other)

    def __gt__(self, other):
        return self._tracer().gt(self, other)

    def __le__(self, other):
        return self._tracer().le(self, other)

    def __ge__(self, other):
        return self._tracer().ge(self, other)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"Tile#{self.id}({self.op}, shape={self.shape}, {np.dtype(self.dtype).name})"


@dataclass(frozen=True)
class StoreInstr:
    """写全局内存。value/offsets/mask 都指向 Tile 的 id。"""

    buffer: str
    offsets: int
    value: int
    mask: Optional[int] = None
    order: int = 0


@dataclass(frozen=True)
class SmemStoreInstr:
    """写共享内存。和 :class:`StoreInstr` 分开，是为了让代价模型不会把它算成 HBM 流量。

    这是本仓库最重要的一条区分：**共享内存写不是全局内存写**。
    如果两者共用一个指令类型，`costmodel.analyze()` 会把 smem 分块版算成
    「多了几倍访存」，结论会完全反过来。
    """

    smem: str
    offsets: int
    value: int
    mask: Optional[int] = None
    order: int = 0


@dataclass
class Program:
    """一个 block 级程序：一次 grid 发射里每个 block 执行的指令序列。"""

    name: str
    instrs: list[Any] = field(default_factory=list)
    inputs: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    constexprs: Dict[str, Any] = field(default_factory=dict)
    smem: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = field(default_factory=dict)
    grid: Tuple[int, ...] = (1,)
    num_warps: int = 4

    # ---- 构造 -----------------------------------------------------------
    def emit(self, op: str, shape: Sequence[int], dtype: np.dtype,
             args: Sequence[int] = (), **attrs: Any) -> Tile:
        tile = Tile(
            id=len(self.instrs),
            op=op,
            shape=tuple(int(s) for s in shape),
            dtype=np.dtype(dtype),
            args=tuple(int(a) for a in args),
            attrs=dict(attrs),
        )
        self.instrs.append(tile)
        return tile

    def emit_store(self, buffer: str, offsets: Tile, value: Tile,
                   mask: Optional[Tile] = None) -> StoreInstr:
        ins = StoreInstr(buffer=buffer, offsets=offsets.id, value=value.id,
                         mask=None if mask is None else mask.id,
                         order=len(self.instrs))
        self.instrs.append(ins)
        if buffer not in self.outputs:
            self.outputs.append(buffer)
        return ins

    def emit_smem_store(self, smem: str, offsets: Tile, value: Tile,
                        mask: Optional[Tile] = None) -> SmemStoreInstr:
        """写共享内存：产生副作用，但**不**进 ``outputs``、**不**算 HBM 流量。"""
        if smem not in self.smem:
            raise KeyError(f"共享内存 {smem} 没有分配过")
        ins = SmemStoreInstr(smem=smem, offsets=offsets.id, value=value.id,
                             mask=None if mask is None else mask.id,
                             order=len(self.instrs))
        self.instrs.append(ins)
        return ins

    def alloc_smem(self, name: str, shape: Sequence[int],
                   dtype: Any = np.float32) -> Tuple[Tuple[int, ...], np.dtype]:
        if name in self.smem:
            raise ValueError(f"共享内存 {name} 重复分配")
        spec = (tuple(int(s) for s in shape), np.dtype(dtype))
        self.smem[name] = spec
        return spec

    # ---- 查询 -----------------------------------------------------------
    def tile(self, tid: int) -> Tile:
        node = self.instrs[tid]
        if not isinstance(node, Tile):
            raise TypeError(f"#{tid} 不是值节点：{type(node).__name__}")
        return node
    def stores(self) -> list[StoreInstr]:
        return [n for n in self.instrs if isinstance(n, StoreInstr)]

    def tiles(self) -> list[Tile]:
        return [n for n in self.instrs if isinstance(n, Tile)]

    def clone(self, name: Optional[str] = None) -> "Program":
        """深拷贝（pass 里大量用到，避免就地改动原程序）。"""
        p = Program(
            name=name or self.name,
            inputs=dict(self.inputs),
            outputs=list(self.outputs),
            constexprs=dict(self.constexprs),
            smem=dict(self.smem),
            grid=self.grid,
            num_warps=self.num_warps,
        )
        for node in self.instrs:
            if isinstance(node, Tile):
                p.instrs.append(
                    Tile(id=node.id, op=node.op, shape=node.shape, dtype=node.dtype,
                         args=node.args, attrs=dict(node.attrs))
                )
            elif isinstance(node, SmemStoreInstr):
                p.instrs.append(
                    SmemStoreInstr(smem=node.smem, offsets=node.offsets,
                                   value=node.value, mask=node.mask, order=node.order)
                )
            else:
                p.instrs.append(
                    StoreInstr(buffer=node.buffer, offsets=node.offsets, value=node.value,
                               mask=node.mask, order=node.order)
                )
        return p

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"Program({self.name}, instrs={len(self.instrs)}, grid={self.grid})"
