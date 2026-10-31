"""示例内核：用来演示「访存是否合并」「分支是否发散」这两个真实性能开关。

这三个内核的数学结果完全一样（或者只差一次置换），但硬件行为天差地别。
把它们放进包里而不是 examples/，是为了让测试能直接引用同一份代码——
示例和断言共用实现，才不存在「文档说 A、代码跑 B」。

内核签名约定：``fn(ctx, *extra)``，缓冲区通过 ``ctx.buffers`` 按名字访问，
标量参数走 ``extra``（见 :func:`gkern.gpgpu.kernel.launch`）。
"""

from __future__ import annotations

import numpy as np

from .kernel import kernel

BLOCK = 256


@kernel(block=(BLOCK,))
def vec_add_coalesced(ctx, n):
    """相邻 lane 读相邻地址：一个 warp 的 32 个 float32 正好是 128B = 1 个事务。"""
    g = ctx.block.x * BLOCK + ctx.tid
    ctx.mask = g < n
    x = ctx.load("a", g)
    y = ctx.load("b", g)
    ctx.store("out", g, x + y)


@kernel(block=(BLOCK,))
def vec_add_strided(ctx, n):
    """同样的加法，但 lane 之间隔 32 个元素：一个 warp 触碰 32 个不同的 128B 段。

    这是真实世界里「按列访问行主序数组」的典型形态，也是 AoS/SoA 之争的根源。
    """
    g = ctx.block.x * BLOCK + ctx.tid
    ctx.mask = g < n
    src = (g * 32) % n          # 跨步 32 个元素
    x = ctx.load("a", src)
    y = ctx.load("b", src)
    ctx.store("out", g, x + y)


@kernel(block=(BLOCK,))
def relu_branchy(ctx, n):
    """用 if/else 写 ReLU：同一个 warp 里正负混合时，两条路径都要串行跑一遍。"""
    g = ctx.block.x * BLOCK + ctx.tid
    ctx.mask = g < n
    x = ctx.load("a", g)

    def then_body():
        ctx.store("out", g, x)

    def else_body():
        ctx.store("out", g, np.zeros_like(x))

    ctx.branch(x > 0.0, then_body, else_body)


def run_sample(fn, n=1024, block=BLOCK, seed=0):
    """构造输入并发射示例内核，返回 ``(LaunchResult, a, b)``。"""
    from .kernel import launch

    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    grid = ((n + block - 1) // block,)
    if fn is relu_branchy:
        res = launch(fn, {"a": a, "out": out}, grid=grid, block=(block,), extra=(n,))
    else:
        res = launch(fn, {"a": a, "b": b, "out": out}, grid=grid, block=(block,),
                     extra=(n,))
    return res, a, b
