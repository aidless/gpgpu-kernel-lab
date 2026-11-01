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

from .ir import Program, StoreInstr, Tile
from ..gpgpu.device import DeviceInfo


@dataclass
class Cost:
    """一个 Program 在整张 grid 上的代价估计。"""

    bytes_read: int
    bytes_written: int
    flops: int
    programs: int
    device: DeviceInfo

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
            flops += node.numel
        elif node.op in ("exp", "log", "sqrt", "rsqrt", "neg", "tanh"):
            # 先按 1 FLOP 计，实际 SFU 吞吐低得多；这里只用于「是否算力受限」的量级判断
            flops += node.numel
        elif node.op in ("sum", "max"):
            src = program.tile(node.args[0])
            flops += src.numel
    programs = int(np.prod(program.grid))
    return Cost(
        bytes_read=read * programs,
        bytes_written=written * programs,
        flops=flops * programs,
        programs=programs,
        device=dev,
    )
