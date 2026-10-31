"""编译器子包：Triton 风格 DSL → IR → pass → 执行/代价分析。

对外最常用的两个入口：

.. code-block:: python

    from gkern import compiler as gc

    prog = gc.compile_kernel(my_kernel, buffers, scalars, constexprs, grid=(4,))
    out = gc.run(my_kernel, buffers, scalars, constexprs, grid=(4,))
    cost = gc.costmodel.analyze(prog)
"""

from . import codegen, costmodel, dsl, passes
from .dsl import (
    Ptr,
    arange,
    broadcast,
    cast,
    expand_dims,
    zeros,
    constexpr,
    dot,
    exp,
    full,
    jit,
    load,
    log,
    max,          # noqa: A004 - 对齐 Triton 的 tl.max
    maximum,
    program_id,
    rsqrt,
    sqrt,
    store,
    sum,          # noqa: A004
    tanh,
    trace,
    where,
)
from .ir import Program, StoreInstr, Tile
from .passes import TuneResult, autotune, fuse_programs

__all__ = [
    "Ptr", "Tile", "Program", "StoreInstr",
    "jit", "constexpr", "trace",
    "program_id", "arange", "load", "store",
    "exp", "log", "sqrt", "rsqrt", "tanh", "maximum", "where", "full",
    "sum", "max", "dot", "broadcast", "cast", "expand_dims", "zeros",
    "codegen", "costmodel", "passes", "dsl",
    "autotune", "fuse_programs", "TuneResult",
    "compile_kernel", "run",
]


def compile_kernel(fn, buffers, scalars=None, constexprs=None,
                   grid=(1,), num_warps: int = 4) -> Program:
    """把一个 DSL kernel 追踪成 IR（不执行）。"""
    return trace(fn, buffers, scalars or {}, constexprs or {}, grid,
                 num_warps=num_warps)


def run(fn, buffers, scalars=None, constexprs=None, grid=(1,),
        num_warps: int = 4) -> dict:
    """追踪并执行，返回输出缓冲区字典。"""
    program = compile_kernel(fn, buffers, scalars, constexprs, grid, num_warps)
    return codegen.run(program, buffers)
