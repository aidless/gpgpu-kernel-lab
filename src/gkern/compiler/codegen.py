"""numpy 执行后端：把 IR 逐 block 执行一遍，得到与 torch 参考实现可比的结果。

它不是「模拟」，而是**真的算**：每个 program id 对应一个 block，块内 tile 运算
用 numpy 一次性完成。这样做的价值是：
- 正确性可以逐元素对拍 torch（这是所有性能数字的前提）；
- 融合前后的 IR 走同一套执行器，唯一的差别就是中间结果是否落回全局内存，
  因此「融合省了多少访存」是 IR 层面可验证的事实，不是文案。

注意：执行器不做线程级展开，warp 级度量由 :mod:`gkern.gpgpu` 那套设备模型负责。
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

from .ir import Program, StoreInstr, Tile

_BINARY = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "maximum": np.maximum,
}

_COMPARE = {
    "lt": np.less,
    "gt": np.greater,
    "le": np.less_equal,
    "ge": np.greater_equal,
    "eq": np.equal,
}

def _relu(x):
    return np.maximum(x, 0.0)


def _sigmoid(x):
    # 分正负两支，避免 exp 溢出
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _gelu_exact(x):
    """PyTorch ``F.gelu`` 默认用的精确 erf 形式（不是 tanh 近似）。

    与 :func:`gkern.ops.elementwise.scale_gelu` 的 tanh 近似是两个不同的东西：
    前者是 torch 的语义，后者是很多 kernel 为了省一次 erf 而做的近似。
    混用会让「和 torch 对拍」变成假通过，所以这里显式区分。
    """
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


_UNARY = {
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
    "neg": np.negative,
    "tanh": np.tanh,
    "relu": _relu,
    "sigmoid": _sigmoid,
    "gelu": _gelu_exact,
}


def execute(program: Program, buffers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """执行一个 Program，返回输出缓冲区（就地写回 ``buffers``）。"""
    for name in program.inputs:
        if name not in buffers:
            raise KeyError(f"缺少输入缓冲区 {name}")
    grid = _grid_offsets(program.grid)
    for pid_coord in grid:
        _execute_block(program, buffers, pid_coord)
    return {name: buffers[name] for name in program.outputs}


def _grid_offsets(grid) -> Iterable[tuple]:
    g = tuple(int(x) for x in grid)
    if len(g) == 1:
        return [(i,) for i in range(g[0])]
    return np.ndindex(*g)


def _execute_block(program: Program, buffers: Dict[str, np.ndarray],
                   pid_coord: tuple) -> None:
    values: Dict[int, np.ndarray] = {}
    for node in program.instrs:
        if isinstance(node, StoreInstr):
            offs = values[node.offsets]
            val = values[node.value]
            mask = values[node.mask] if node.mask is not None else None
            buf = buffers[node.buffer].reshape(-1)   # GPU 的地址空间是一维的
            if mask is None:
                buf[offs] = val
            else:
                shape = np.broadcast_shapes(offs.shape, np.shape(mask))
                offs_b = np.broadcast_to(offs, shape)
                val_b = np.broadcast_to(val, shape)
                m = np.broadcast_to(mask, shape)
                buf[offs_b[m]] = val_b[m]
            continue
        values[node.id] = _eval(node, values, buffers, pid_coord)


def _eval(node: Tile, values: Dict[int, np.ndarray],
          buffers: Dict[str, np.ndarray], pid_coord: tuple) -> np.ndarray:
    op = node.op
    if op == "program_id":
        axis = node.attrs.get("axis", 0)
        return np.int64(pid_coord[axis] if axis < len(pid_coord) else 0)
    if op == "const":
        return np.asarray(node.attrs["value"], dtype=node.dtype)
    if op == "arange":
        return np.arange(node.attrs["start"], node.attrs["end"], dtype=np.int64)
    if op == "load":
        offs = np.asarray(values[node.args[0]], dtype=np.int64)
        buf = buffers[node.attrs["buffer"]].reshape(-1)   # 线性地址空间
        other = node.attrs.get("other", 0.0)
        if len(node.args) > 1:
            valid = np.broadcast_to(values[node.args[1]], offs.shape).astype(bool)
        else:
            valid = np.ones(offs.shape, dtype=bool)
        valid = valid & (offs >= 0) & (offs < buf.shape[0])
        # 越界下标先夹到合法范围再取，最后用 where 把无效位置换回 other：
        # 这样既不会 IndexError，也保住了「越界读不产生数据依赖」的语义。
        safe = np.clip(offs, 0, buf.shape[0] - 1)
        vals = buf[safe]
        return np.where(valid, vals, np.asarray(other, dtype=buf.dtype))
    if op in _COMPARE:
        return _COMPARE[op](values[node.args[0]], values[node.args[1]])
    if op == "and":
        return np.logical_and(values[node.args[0]], values[node.args[1]])
    if op == "or":
        return np.logical_or(values[node.args[0]], values[node.args[1]])
    if op == "where":
        return np.where(values[node.args[0]], values[node.args[1]], values[node.args[2]])
    if op in _BINARY:
        return _BINARY[op](values[node.args[0]], values[node.args[1]])
    if op in _UNARY:
        return _UNARY[op](values[node.args[0]])
    if op == "sum":
        return np.sum(values[node.args[0]], axis=node.attrs["axis"],
                      keepdims=node.attrs.get("keep_dims", False))
    if op == "max":
        return np.max(values[node.args[0]], axis=node.attrs["axis"],
                      keepdims=node.attrs.get("keep_dims", False))
    if op == "dot":
        return np.matmul(values[node.args[0]], values[node.args[1]])
    if op == "broadcast":
        return np.broadcast_to(values[node.args[0]], node.shape)
    if op == "expand_dims":
        return np.expand_dims(values[node.args[0]], axis=node.attrs["axis"])
    if op == "cast":
        return values[node.args[0]].astype(node.dtype)
    raise NotImplementedError(f"执行器不认识的 IR 指令：{op}")


def run(program: Program, buffers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """对外入口：执行并返回输出。"""
    return execute(program, buffers)
