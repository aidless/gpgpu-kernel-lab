"""Triton 风格的块级 DSL 前端：把 Python kernel 符号执行成 IR。

写法与 Triton 基本一一对应::

    @jit
    def vector_add(X, Y, OUT, n, BLOCK: constexpr):
        pid = program_id(0)
        offs = pid * BLOCK + arange(0, BLOCK)
        mask = offs < n
        x = load(X, offs, mask)
        y = load(Y, offs, mask)
        store(OUT, offs, x + y, mask)

约定：
- 缓冲区参数是 :class:`Ptr`（带名字/形状/dtype），标量参数是 int，
  常量参数用 ``constexpr`` 注解标记（参与分块，不参与求值）；
- 所有算子返回 :class:`~gkern.compiler.ir.Tile`，形状在追踪期就已确定，
  因此 ``dot``、``sum(axis=)`` 这些需要形状信息的原语可以正常推导；
- 追踪期不求值，所以同一份 kernel 可以用不同 BLOCK 反复追踪（自动调优需要）。

与 Triton 的差异（诚实声明）：这里没有 mask 广播的隐式规则，mask 必须显式传；
没有 pointer arithmetic 的隐式类型提升；不支持 atomic / scatter。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .ir import Program, Tile

# ---------------------------------------------------------------------------
# 类型标记
# ---------------------------------------------------------------------------


class constexpr:
    """标注「这是编译期常量」，例如 BLOCK 大小。"""

    def __init__(self, value: Any = None) -> None:
        self.value = value


class Ptr:
    """一个全局缓冲区参数。"""

    def __init__(self, name: str, shape: Sequence[int], dtype: Any = np.float32) -> None:
        self.name = name
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)


class SmemRef:
    """一块共享内存（每个 block 一份，整张 grid 共用同一个分配描述）。"""

    def __init__(self, name: str, shape: Sequence[int], dtype: Any = np.float32) -> None:
        self.name = name
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"SmemRef({self.name}, shape={self.shape}, {self.dtype.name})"


# ---------------------------------------------------------------------------
# 追踪上下文
# ---------------------------------------------------------------------------


class _Tracer:
    def __init__(self, program: Program) -> None:
        self.program = program

    # ---- 索引 -----------------------------------------------------------
    def program_id(self, axis: int = 0) -> Tile:
        return self.program.emit("program_id", (), np.int64, axis=axis)

    def arange(self, start: int, end: int) -> Tile:
        if end < start:
            raise ValueError(f"arange({start},{end}) 非法")
        return self.program.emit("arange", (end - start,), np.int64, start=start, end=end)

    # ---- 访存 -----------------------------------------------------------
    def load(self, ptr: Ptr, offsets: Tile, mask: Optional[Tile] = None,
             other: float = 0.0) -> Tile:
        args = [offsets.id] + ([mask.id] if mask is not None else [])
        return self.program.emit("load", offsets.shape, ptr.dtype, args,
                                 buffer=ptr.name, other=float(other))

    def store(self, ptr: Ptr, offsets: Tile, value: Tile,
              mask: Optional[Tile] = None) -> None:
        self.program.emit_store(ptr.name, offsets, value, mask)

    # ---- 共享内存 -------------------------------------------------------
    def smem_alloc(self, name: str, shape: Sequence[int],
                   dtype: Any = np.float32) -> SmemRef:
        self.program.alloc_smem(name, shape, dtype)
        return SmemRef(name, shape, dtype)

    def smem_store(self, ref: SmemRef, offsets: Tile, value: Tile,
                   mask: Optional[Tile] = None) -> None:
        self.program.emit_smem_store(ref.name, offsets, value, mask)

    def smem_load(self, ref: SmemRef, offsets: Tile,
                  mask: Optional[Tile] = None) -> Tile:
        args = [offsets.id] + ([mask.id] if mask is not None else [])
        return self.program.emit("smem_load", offsets.shape, ref.dtype, args,
                                 smem=ref.name)

    # ---- 逐元素 ---------------------------------------------------------
    def const(self, value: Any, dtype: Any = None) -> Tile:
        dt = np.dtype(dtype) if dtype is not None else np.dtype(
            np.int64 if isinstance(value, int) else np.float32)
        return self.program.emit("const", (), dt, value=value)

    def _as_tile(self, v: Any) -> Tile:
        if isinstance(v, Tile):
            return v
        if isinstance(v, (int, float, np.integer, np.floating)):
            return self.const(v)
        raise TypeError(f"不支持的操作数类型：{type(v).__name__}")

    def _binary(self, op: str, a: Any, b: Any) -> Tile:
        a, b = self._as_tile(a), self._as_tile(b)
        shape = np.broadcast_shapes(a.shape, b.shape)
        dtype = np.result_type(a.dtype, b.dtype)
        return self.program.emit(op, shape, dtype, (a.id, b.id))

    def _compare(self, op: str, a: Any, b: Any) -> Tile:
        a, b = self._as_tile(a), self._as_tile(b)
        shape = np.broadcast_shapes(a.shape, b.shape)
        return self.program.emit(op, shape, np.bool_, (a.id, b.id))

    def add(self, a: Any, b: Any) -> Tile:
        return self._binary("add", a, b)

    def sub(self, a: Any, b: Any) -> Tile:
        return self._binary("sub", a, b)

    def mul(self, a: Any, b: Any) -> Tile:
        return self._binary("mul", a, b)

    def div(self, a: Any, b: Any) -> Tile:
        return self._binary("div", a, b)

    def maximum(self, a: Any, b: Any) -> Tile:
        return self._binary("maximum", a, b)

    def lt(self, a: Any, b: Any) -> Tile:
        return self._compare("lt", a, b)

    def gt(self, a: Any, b: Any) -> Tile:
        return self._compare("gt", a, b)

    def le(self, a: Any, b: Any) -> Tile:
        return self._compare("le", a, b)

    def ge(self, a: Any, b: Any) -> Tile:
        return self._compare("ge", a, b)

    def eq(self, a: Any, b: Any) -> Tile:
        return self._compare("eq", a, b)

    def and_(self, a: Any, b: Any) -> Tile:
        a, b = self._as_tile(a), self._as_tile(b)
        return self.program.emit("and", np.broadcast_shapes(a.shape, b.shape), np.bool_,
                                 (a.id, b.id))

    def or_(self, a: Any, b: Any) -> Tile:
        a, b = self._as_tile(a), self._as_tile(b)
        return self.program.emit("or", np.broadcast_shapes(a.shape, b.shape), np.bool_,
                                 (a.id, b.id))

    def where(self, cond: Any, a: Any, b: Any) -> Tile:
        cond, a, b = self._as_tile(cond), self._as_tile(a), self._as_tile(b)
        shape = np.broadcast_shapes(cond.shape, a.shape, b.shape)
        dtype = np.result_type(a.dtype, b.dtype)
        return self.program.emit("where", shape, dtype, (cond.id, a.id, b.id))

    def _unary(self, op: str, a: Tile) -> Tile:
        return self.program.emit(op, a.shape, a.dtype, (a.id,))

    def exp(self, a: Tile) -> Tile:
        return self._unary("exp", a)

    def log(self, a: Tile) -> Tile:
        return self._unary("log", a)

    def sqrt(self, a: Tile) -> Tile:
        return self._unary("sqrt", a)

    def rsqrt(self, a: Tile) -> Tile:
        return self._unary("rsqrt", a)

    def neg(self, a: Tile) -> Tile:
        return self._unary("neg", a)

    def tanh(self, a: Tile) -> Tile:
        return self._unary("tanh", a)

    # ---- 归约与矩阵乘 ---------------------------------------------------
    def sum(self, a: Tile, axis: int, keep_dims: bool = False) -> Tile:
        shape = list(a.shape)
        shape.pop(axis)
        if keep_dims:
            shape.insert(axis, 1)
        return self.program.emit("sum", shape, a.dtype, (a.id,), axis=axis,
                                 keep_dims=keep_dims)

    def max(self, a: Tile, axis: int, keep_dims: bool = False) -> Tile:
        shape = list(a.shape)
        shape.pop(axis)
        if keep_dims:
            shape.insert(axis, 1)
        return self.program.emit("max", shape, a.dtype, (a.id,), axis=axis,
                                 keep_dims=keep_dims)

    def dot(self, a: Tile, b: Tile) -> Tile:
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise ValueError(f"dot 只支持 2D，收到 {a.shape} x {b.shape}")
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"dot 维度不匹配：{a.shape} x {b.shape}")
        return self.program.emit("dot", (a.shape[0], b.shape[1]),
                                 np.result_type(a.dtype, b.dtype), (a.id, b.id))

    def broadcast(self, a: Tile, shape: Sequence[int]) -> Tile:
        return self.program.emit("broadcast", tuple(shape), a.dtype, (a.id,))

    def expand_dims(self, a: Tile, axis: int) -> Tile:
        shape = list(a.shape)
        shape.insert(axis, 1)
        return self.program.emit("expand_dims", tuple(shape), a.dtype, (a.id,), axis=axis)

    def cast(self, a: Tile, dtype: Any) -> Tile:
        return self.program.emit("cast", a.shape, np.dtype(dtype), (a.id,))


# ---------------------------------------------------------------------------
# 追踪入口
# ---------------------------------------------------------------------------


def trace(fn: Callable, buffers: Dict[str, np.ndarray],
          scalars: Dict[str, int], constexprs: Dict[str, Any],
          grid: Sequence[int], name: Optional[str] = None,
          num_warps: int = 4) -> Program:
    """符号执行一个 kernel 函数，产出 :class:`Program`。

    参数
    ----
    fn        : kernel 函数（第一个参数之后按签名顺序传缓冲区/标量/常量）
    buffers   : 名字 -> numpy 数组（用于取形状与 dtype）
    scalars   : 名字 -> int（运行期标量）
    constexprs : 名字 -> 值（编译期常量，如 BLOCK）
    grid      : grid 维度
    """
    program = Program(name=name or getattr(fn, "__name__", "kernel"),
                      grid=tuple(int(g) for g in grid), num_warps=num_warps)
    tracer = _Tracer(program)

    args: list[Any] = []
    for pname in _param_names(fn):
        if pname in buffers:
            arr = np.asarray(buffers[pname])
            ptr = Ptr(pname, arr.shape, arr.dtype)
            program.inputs[pname] = (ptr.shape, ptr.dtype)
            args.append(ptr)
        elif pname in constexprs:
            args.append(constexprs[pname])
            program.constexprs[pname] = constexprs[pname]
        elif pname in scalars:
            val = scalars[pname]
            args.append(val if isinstance(val, (int, float, bool)) else int(val))
        else:
            raise KeyError(f"kernel {program.name} 的参数 {pname} 既不在 buffers/scalars 也不在 constexprs 中")

    _install(tracer)
    try:
        fn(*args)
    finally:
        _uninstall()
    return program


def _param_names(fn: Callable) -> list[str]:
    import inspect

    return [p.name for p in inspect.signature(fn).parameters.values()]


# ---------------------------------------------------------------------------
# 全局原语（Triton 的 tl.* 对应物）
# ---------------------------------------------------------------------------

_ACTIVE: list[_Tracer] = []


def _install(tracer: _Tracer) -> None:
    _ACTIVE.append(tracer)


def _uninstall() -> None:
    _ACTIVE.pop()


def _t() -> _Tracer:
    if not _ACTIVE:
        raise RuntimeError("当前不在 kernel 追踪上下文中：这些原语只能在 @jit kernel 内调用")
    return _ACTIVE[-1]


def program_id(axis: int = 0) -> Tile:
    return _t().program_id(axis)


def arange(start: int, end: int) -> Tile:
    return _t().arange(start, end)


def load(ptr: Ptr, offsets: Tile, mask: Optional[Tile] = None, other: float = 0.0) -> Tile:
    return _t().load(ptr, offsets, mask, other)


def store(ptr: Ptr, offsets: Tile, value: Tile, mask: Optional[Tile] = None) -> None:
    _t().store(ptr, offsets, value, mask)


def smem_alloc(name: str, shape: Sequence[int], dtype: Any = np.float32) -> SmemRef:
    return _t().smem_alloc(name, shape, dtype)


def smem_store(ref: SmemRef, offsets: Tile, value: Tile,
               mask: Optional[Tile] = None) -> None:
    _t().smem_store(ref, offsets, value, mask)


def smem_load(ref: SmemRef, offsets: Tile, mask: Optional[Tile] = None) -> Tile:
    return _t().smem_load(ref, offsets, mask)


def exp(x: Tile) -> Tile:
    return _t().exp(x)


def log(x: Tile) -> Tile:
    return _t().log(x)


def sqrt(x: Tile) -> Tile:
    return _t().sqrt(x)


def rsqrt(x: Tile) -> Tile:
    return _t().rsqrt(x)


def tanh(x: Tile) -> Tile:
    return _t().tanh(x)


def maximum(a: Any, b: Any) -> Tile:
    return _t().maximum(a, b)


def where(cond: Any, a: Any, b: Any) -> Tile:
    return _t().where(cond, a, b)


def full(shape: Sequence[int], value: float, dtype: Any = np.float32) -> Tile:
    """生成一个常量填充的 tile（等价于 Triton 里先 broadcast 一个标量）。"""
    return _t().broadcast(_t().const(value, dtype), tuple(shape))


def sum(x: Tile, axis: int, keep_dims: bool = False) -> Tile:  # noqa: A001 - 对齐 Triton API
    return _t().sum(x, axis, keep_dims)


def max(x: Tile, axis: int, keep_dims: bool = False) -> Tile:  # noqa: A001
    return _t().max(x, axis, keep_dims)


def dot(a: Tile, b: Tile) -> Tile:
    return _t().dot(a, b)


def broadcast(x: Tile, shape: Sequence[int]) -> Tile:
    return _t().broadcast(x, shape)


def cast(x: Tile, dtype: Any) -> Tile:
    return _t().cast(x, dtype)


def expand_dims(x: Tile, axis: int) -> Tile:
    return _t().expand_dims(x, axis)


def zeros(shape: Sequence[int], dtype: Any = np.float32) -> Tile:
    return full(shape, 0.0, dtype)


def jit(fn: Callable) -> Callable:
    """标记一个函数是 DSL kernel（真正执行走 :func:`gkern.compiler.run`）。"""
    fn.__dsl_kernel__ = True  # type: ignore[attr-defined]
    return fn
