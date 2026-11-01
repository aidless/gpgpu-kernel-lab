"""代价模型：从 IR 直接估出访存字节、浮点运算量和受限于谁。

GPU 优化的第一性问题永远是「这个 kernel 是带宽受限还是算力受限」。
本模型按 roofline 的算法给出估算：

    arithmetic_intensity = FLOPs / Bytes
    predicted_ms = max(bytes / bandwidth, flops / peak_flops)

所有量都从 IR 的 shape/dtype 推出来，不读墙钟，因此可以在**不执行**的情况下
比较不同 BLOCK/分块方案的代价——自动调优器就是靠它做搜索的。

诚实的边界：这个模型不含 cache 命中、不含指令发射开销、不含 bank 冲突，
所以它是「下界估计」。它和 :mod:`gkern.gpgpu` 设备模型在同一 kernel 上的
交叉校验见 ``tests/test_compiler.py``：对 elementwise 这类访存主导的 kernel，
两者预测的字节数必须一致（不一致就是模型写错了）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .ir import Program, SmemStoreInstr, StoreInstr, Tile
from ..gpgpu.device import DeviceInfo


@dataclass
class Cost:
    """一个 Program 在整张 grid 上的代价估计。"""

    bytes_read: int
    bytes_written: int
    flops: int
    programs: int
    device: DeviceInfo
    #: True 表示这是「每个输入只从 HBM 读一次」的理想复用上界，而不是逐块重读的下界
    reuse: bool = False

    @property
    def bytes_total(self) -> int:
        return self.bytes_read + self.bytes_written

    @property
    def arithmetic_intensity(self) -> float:
        if self.bytes_total == 0:
            return float("inf")
        return self.flops / float(self.bytes_total)

    @property
    def memory_ms(self) -> float:
        bandwidth = self.device.theoretical_bandwidth_gbps * 1e9   # B/s
        return self.bytes_total / bandwidth * 1e3 if bandwidth else 0.0

    @property
    def compute_ms(self) -> float:
        peak = self.device.peak_flops_tflops * 1e12
        return self.flops / peak * 1e3 if peak else 0.0

    @property
    def predicted_ms(self) -> float:
        return max(self.memory_ms, self.compute_ms)

    @property
    def bound(self) -> str:
        return "memory" if self.memory_ms >= self.compute_ms else "compute"

    def as_dict(self) -> dict:
        return {
            "reuse": self.reuse,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "bytes_total": self.bytes_total,
            "flops": self.flops,
            "programs": self.programs,
            "arithmetic_intensity": round(self.arithmetic_intensity, 4),
            "memory_ms": round(self.memory_ms, 6),
            "compute_ms": round(self.compute_ms, 6),
            "predicted_ms": round(self.predicted_ms, 6),
            "bound": self.bound,
        }


def analyze(program: Program, device: DeviceInfo | None = None) -> Cost:
    """统计一个 Program 的整张 grid 的访存与算力。"""
    dev = device or DeviceInfo()
    read = written = flops = 0
    for node in program.instrs:
        if isinstance(node, SmemStoreInstr):
            # 共享内存写不是 HBM 流量。这条 if 是 smem 分块版不被算成「多几倍访存」的原因。
            continue
        if isinstance(node, StoreInstr):
            val = program.tile(node.value)
            written += val.nbytes
            continue
        if node.op == "load":
            read += node.nbytes
        elif node.op == "dot":
            a = program.tile(node.args[0])
            b = program.tile(node.args[1])
            flops += 2 * a.shape[0] * a.shape[1] * b.shape[1]
        elif node.op in ("add", "sub", "mul", "div", "maximum"):
            # 只统计浮点运算：地址算术（int64 的乘加）不是 FLOP。
            # 不排除它的话，GEMM 的 AI 会被偏移量计算抬高约 4.8%，roofline 判定跟着漂。
            if np.issubdtype(node.dtype, np.floating):
                flops += node.numel
        elif node.op in ("exp", "log", "sqrt", "rsqrt", "neg", "tanh"):
            # 先按 1 FLOP 计，实际 SFU 吞吐低得多；这里只用于「是否算力受限」的量级判断
            if np.issubdtype(node.dtype, np.floating):
                flops += node.numel
        elif node.op in ("sum", "max"):
            src = program.tile(node.args[0])
            if np.issubdtype(src.dtype, np.floating):
                flops += src.numel
    programs = int(np.prod(program.grid))
    return Cost(
        bytes_read=read * programs,
        bytes_written=written * programs,
        flops=flops * programs,
        programs=programs,
        device=dev,
    )


def crossover_intensity(device: DeviceInfo | None = None) -> float:
    """roofline 的交叉点：AI 超过它才是 compute bound。本机假设机为 256 FLOP/B。"""
    dev = device or DeviceInfo()
    return dev.peak_flops_tflops * 1e12 / (dev.theoretical_bandwidth_gbps * 1e9)


def reuse_bound(program: Program, device: DeviceInfo | None = None) -> Cost:
    """理想 L2 复用下的访存**上界**：每个输入只从 HBM 读一次，每个输出只写一次。

    这个数与 :func:`analyze` 的下界配对使用，回答的是两个不同问题：

    - ``analyze``：如果每个 block 都老实从 HBM 重新读自己的面板（无缓存），
      代价是多少？——下界口径，矩阵乘法下一定是 memory bound。
    - ``reuse_bound``：如果 L2 足够大、面板被跨 block 复用命中，代价是多少？
      ——才是真实 GPU 能摸到的口径。

    方阵 GEMM 在完美复用下 ``AI = 2N^3 / (12N^2) = N/6``，
    所以交叉点 256 FLOP/B 需要 **N >= 1536**——这就是「光调 tile 翻不动 bound」的算术原因。
    """
    dev = device or DeviceInfo()
    written_names = {node.buffer for node in program.instrs
                     if isinstance(node, StoreInstr)}
    read = written = 0
    for name, (shape, dtype) in program.inputs.items():
        nbytes = Tile(0, "buffer", shape, dtype).nbytes
        if name in written_names:
            written += nbytes
        else:
            read += nbytes
    flops = analyze(program, dev).flops
    return Cost(bytes_read=read, bytes_written=written, flops=flops,
                programs=int(np.prod(program.grid)), device=dev, reuse=True)
