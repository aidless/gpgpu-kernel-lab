"""内核启动入口：把 grid/block 层级、SIMT warp、访存统计拼成一次可度量的发射。

用法（也是本仓库所有算子的写法）::

    @kernel(block=(256,))
    def vec_add(ctx, n):
        i = ctx.block.x * 256 + ctx.tid      # 全局线性下标 = block 偏移 + 块内线程号
        ctx.mask = i < n                     # 尾部不足一个 warp 的 lane 被掩掉
        x = ctx.load("a", i)                 # 缓冲区按名字从 ctx.buffers 取
        y = ctx.load("b", i)
        ctx.store("out", i, x + y)

    res = launch(vec_add, {"a": a, "b": b, "out": out},
                 grid=(cdiv(n, 256),), block=(256,), extra=(n,))
    res.outputs["out"]          # 结果
    res.report.efficiency       # warp 效率
    res.memory.coalescing_efficiency

关键点：``ctx.tid`` 是**长度 32 的数组**，一个 warp 的一次调用就覆盖 32 个 lane，
所以模拟器既保住了 SIMT 语义（发散、掩码、事务数、bank 冲突都从真实索引集合上算），
又不会因为「每线程一次 Python 调用」而慢到不可用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .device import Device, DeviceInfo
from .memory import GlobalMemory, MemoryStats, SharedMemory
from .warp import DivergenceReport, Warp, WarpTrace


@dataclass
class KernelMeta:
    """``@kernel`` 记录下来的静态元信息。"""

    fn: Callable
    grid: Tuple[int, ...] | None = None
    block: Tuple[int, ...] = (256,)
    shared_elements: int = 0
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = getattr(self.fn, "__name__", "kernel")


def kernel(grid: Optional[Sequence[int]] = None, block: Sequence[int] = (256,),
           shared_elements: int = 0) -> Callable[[Callable], Callable]:
    """装饰器：把普通函数标记为 GPGPU 内核，并记录发射配置。"""

    def deco(fn: Callable) -> Callable:
        meta = KernelMeta(fn=fn, grid=tuple(grid) if grid else None,
                          block=tuple(block), shared_elements=shared_elements)
        fn.__gkernel__ = meta   # type: ignore[attr-defined]
        return fn

    return deco


class WarpContext:
    """一个 warp 在执行内核时拿到的上下文。

    它把「我这条指令作用在哪些 lane 上」这件事显式化：
    ``mask`` 决定活跃 lane，``load/store`` 只对活跃 lane 产生访存事务。
    """

    def __init__(self, warp: Warp, block: Any, tid: np.ndarray,
                 gmem: GlobalMemory, smem: Optional[SharedMemory],
                 buffers: Dict[str, np.ndarray]) -> None:
        self.warp = warp
        self.block = block
        self.tid = tid                      # 长度 = warp_size 的 int 数组
        self.lane = np.arange(warp.size, dtype=np.int64)
        self.gmem = gmem
        self.smem = smem
        self.buffers = buffers

    # ---- 掩码 -----------------------------------------------------------
    # 掩码**只有一份真相**：warp.mask。早期这里另外存了一份 self.mask，
    # 结果 branch() 在 then/else 内部改了 warp.mask，ctx._mask 却还是旧值，
    # 于是两个分支都按「全活跃」写入，后写的覆盖先写的（ReLU 全变 0）。
    # 现在 mask 是 warp.mask 的视图，读写都直连。
    @property
    def mask(self) -> np.ndarray:
        return self.warp.mask

    @mask.setter
    def mask(self, pred: Any) -> None:
        self.warp.mask = np.asarray(pred, dtype=bool) & self.warp.mask

    def reset_mask(self) -> None:
        self.warp.mask = np.ones(self.warp.size, dtype=bool)

    # ---- 全局访存 -------------------------------------------------------
    def load(self, name: str, index: np.ndarray) -> np.ndarray:
        """读全局缓冲区 ``name`` 的若干下标（活跃 lane 才计入事务）。"""
        self.warp.op()
        buf = self.buffers[name]
        return self.gmem.load(buf, index, self.mask)

    def store(self, name: str, index: np.ndarray, value: np.ndarray) -> None:
        self.warp.op()
        buf = self.buffers[name]
        self.gmem.store(buf, index, value, self.mask)

    # ---- 共享内存 -------------------------------------------------------
    def smem_load(self, index: np.ndarray) -> np.ndarray:
        if self.smem is None:
            raise RuntimeError("该内核未申请共享内存（shared_elements=0）")
        self.warp.op()
        return self.smem.load(index, self.mask)

    def smem_store(self, index: np.ndarray, value: np.ndarray) -> None:
        if self.smem is None:
            raise RuntimeError("该内核未申请共享内存（shared_elements=0）")
        self.warp.op()
        self.smem.store(index, value, self.mask)

    # ---- 控制流 ---------------------------------------------------------
    def branch(self, predicate: np.ndarray, then_body: Callable[[], None],
               else_body: Optional[Callable[[], None]] = None) -> None:
        """SIMT 分支；发散时指令数按路径数增加。

        then/else 体内部的 ``ctx.load/store`` 读到的 ``ctx.mask`` 就是当前路径的
        活跃掩码——这正是 GPU 上「两条路径各自带掩码串行执行」的语义。
        """
        self.warp.branch(predicate, then_body, else_body)

    def op(self) -> None:
        """记一条纯计算指令（不含访存）。"""
        self.warp.op()


@dataclass
class LaunchResult:
    """一次发射的产物：输出缓冲 + 三份度量报告 + 墙钟时间。"""

    outputs: Dict[str, np.ndarray]
    report: DivergenceReport
    memory: MemoryStats
    elapsed_ms: float
    blocks: int
    warps: int
    grid: Tuple[int, ...]
    block: Tuple[int, ...]
    device: DeviceInfo
    per_warp: list[WarpTrace] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "kernel": None,
            "grid": list(self.grid),
            "block": list(self.block),
            "blocks": self.blocks,
            "warps": self.warps,
            "elapsed_ms": round(self.elapsed_ms, 4),
            "warp_efficiency": round(self.report.efficiency, 4),
            "divergence_rate": round(self.report.divergence_rate, 4),
            "instructions": self.report.instructions,
            **self.memory.as_dict(),
        }


def launch(fn: Callable, buffers: Dict[str, np.ndarray],
           grid: Optional[Sequence[int]] = None,
           block: Optional[Sequence[int]] = None,
           device: Optional[Device] = None,
           shared_elements: Optional[int] = None,
           extra: Tuple[Any, ...] = ()) -> LaunchResult:
    """发射一个内核。

    参数
    ----
    fn         : 被 ``@kernel`` 装饰（或至少接受 ``ctx`` 为第一个参数）的函数
    buffers    : 名字 -> numpy 数组；内核通过名字读写
    grid/block : 覆盖装饰器里的默认配置
    shared_elements : 每个 block 的共享内存元素数
    extra      : 追加传给内核的位置参数
    """
    meta: KernelMeta | None = getattr(fn, "__gkernel__", None)
    grid_t = tuple(grid or (meta.grid if meta else (1,)))
    block_t = tuple(block or (meta.block if meta else (256,)))
    smem_n = shared_elements if shared_elements is not None else (
        meta.shared_elements if meta else 0)

    dev = device or Device()
    problems = dev.validate(grid_t, block_t)
    if problems:
        raise ValueError("；".join(problems))

    stats = MemoryStats()
    gmem = GlobalMemory(stats)
    for name, arr in buffers.items():
        gmem.alloc(name, arr)

    report = DivergenceReport()
    traces: list[WarpTrace] = []
    block_total = int(np.prod(block_t))
    warp_size = dev.info.warp_size
    warps_per_block = (block_total + warp_size - 1) // warp_size

    t0 = time.perf_counter()
    blocks = 0
    for block in dev.iter_blocks(grid_t):
        blocks += 1
        smem = SharedMemory(smem_n, stats, warp_size=warp_size) if smem_n else None
        for w in range(warps_per_block):
            lane_ids = np.arange(w * warp_size, (w + 1) * warp_size, dtype=np.int64)
            # block 内线程号超过 block 维度的 lane 需要被掩掉
            valid = lane_ids < block_total
            warp = Warp(warp_id=f"{block}/w{w}", size=warp_size)
            warp.mask = valid
            ctx = WarpContext(warp, block, lane_ids, gmem, smem, buffers)
            fn(ctx, *extra)
            trace = warp.finish()
            report.add(trace)
            traces.append(trace)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return LaunchResult(
        outputs=buffers,
        report=report,
        memory=stats,
        elapsed_ms=elapsed_ms,
        blocks=blocks,
        warps=len(traces),
        grid=grid_t,
        block=block_t,
        device=dev.info,
        per_warp=traces,
    )


# ---- 常用小工具 ---------------------------------------------------------

def cdiv(a: int, b: int) -> int:
    """向上取整除法，写 grid 时几乎每个 kernel 都要用。"""
    return (a + b - 1) // b


def dtype_of(arr: np.ndarray) -> np.dtype:
    return np.asarray(arr).dtype
