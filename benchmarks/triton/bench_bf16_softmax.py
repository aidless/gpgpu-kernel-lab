"""在真实 GPU 上量「bf16 累加」的代价：误差有多大、带宽有没有变。

CPU 侧的 `gkern.ops.mixed_precision` 能算出「字节完全一样、FLOP 只多几次舍入」，
但算不出两件事：真实 kernel 上的误差量级，以及 bf16 **存储**（而非只舍入累加器）
到底省不省带宽。这个脚本就是去 GPU 上把这两件事量出来。

三种变体，同一个行 softmax：

1. `fp32`：全程 float32，输入输出都是 4 字节。
2. `bf16acc`：存取仍是 float32，只在四个断点把中间值舍到 bf16（输入/行最大值/
   指数结果/归约和）——这正是 `gkern.ops.mixed_precision.softmax_rows_bf16acc`
   在 IR 里显式写出的那四点，到这里变成 `.to(tl.bfloat16).to(tl.float32)`。
3. `bf16io`：输入输出张量本身就是 bfloat16，计算仍升到 float32——**这一档才可能省带宽**。

必须诚实的一点：变体 2 的字节数与变体 1 完全相同，所以如果它看起来「更快」，
那只能是发射/寄存器层面的巧合，不是带宽。脚本会把 GB/s 一起打出来，让这件事
可以自己判断，而不是听结论。

用法（云上）::

    python bench_bf16_softmax.py --out /kaggle/working/bf16_softmax.json

无 CUDA 时直接退出（不伪造数字）。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_fp32(out_ptr, in_ptr, row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * row_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, 0)
    e = tl.exp(x)
    s = tl.sum(e, 0)
    tl.store(out_ptr + row * row_stride + cols, e / s, mask=mask)


@triton.jit
def _softmax_bf16acc(out_ptr, in_ptr, row_stride, n_cols, BLOCK: tl.constexpr):
    """四个断点各舍一次 bf16——与 CPU 侧 IR 里的四处 round_bf16 一一对应。"""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * row_stride + cols, mask=mask, other=-float("inf"))
    x = x.to(tl.bfloat16).to(tl.float32)
    m = tl.max(x, 0).to(tl.bfloat16).to(tl.float32)
    e = tl.exp(x - m)
    e = e.to(tl.bfloat16).to(tl.float32)
    s = tl.sum(e, 0).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + row * row_stride + cols, e / s, mask=mask)


def _launch(kernel, x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty((rows, cols), device=x.device, dtype=out_dtype)
    block = triton.next_power_of_2(cols)
    kernel[(rows,)](out, x, x.stride(0), cols, BLOCK=block, num_warps=8)
    return out


def _errors(got: torch.Tensor, ref_fp64: torch.Tensor) -> tuple[float, float]:
    diff = (got.double() - ref_fp64).abs()
    return float(diff.max()), float(diff.mean())


def _ref_fp64(x: torch.Tensor) -> torch.Tensor:
    a = x.double()
    e = (a - a.max(dim=-1, keepdim=True).values).exp()
    return e / e.sum(dim=-1, keepdim=True)


def _bench(fn, warmup: int = 25, rep: int = 100) -> float:
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bf16_softmax.json")
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--cols", default="512,1024,2048,4096")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("这个脚本必须在真实 GPU 上跑；CUDA 不可用就退出（不要伪造数字）")

    dev = torch.cuda.get_device_properties(0)
    rows = args.rows
    shapes = [int(c) for c in args.cols.split(",")]
    rng = torch.Generator(device="cuda").manual_seed(20261101)

    results = []
    for cols in shapes:
        x = (torch.randn((rows, cols), generator=rng, device="cuda") * 2.0)
        ref = _ref_fp64(x)

        cases = []
        out32 = _launch(_softmax_fp32, x, torch.float32)
        kbytes = rows * cols * 4 * 2
        cases.append({
            "variant": "fp32",
            "bytes_per_elem": 4,
            "ms": _bench(lambda: _launch(_softmax_fp32, x, torch.float32)),
            "bytes_moved": kbytes,
            "max_error": _errors(out32, ref)[0],
            "mean_error": _errors(out32, ref)[1],
        })

        out_acc = _launch(_softmax_bf16acc, x, torch.float32)
        cases.append({
            "variant": "bf16acc",
            "bytes_per_elem": 4,
            "ms": _bench(lambda: _launch(_softmax_bf16acc, x, torch.float32)),
            "bytes_moved": kbytes,
            "max_error": _errors(out_acc, ref)[0],
            "mean_error": _errors(out_acc, ref)[1],
        })

        x16 = x.to(torch.bfloat16)
        out_io = _launch(_softmax_fp32, x16, torch.bfloat16)
        ybytes = rows * cols * 2 * 2
        cases.append({
            "variant": "bf16io",
            "bytes_per_elem": 2,
            "ms": _bench(lambda: _launch(_softmax_fp32, x16, torch.bfloat16)),
            "bytes_moved": ybytes,
            # 输出本身是 bf16，误差下限就是 bf16 的量化步长，不能拿 fp32 的标准要求它
            "max_error": _errors(out_io.float(), ref)[0],
            "mean_error": _errors(out_io.float(), ref)[1],
        })

        # torch eager 基线（fp32），用来判断手写 kernel 到底有没有赢
        cases.append({
            "variant": "torch_softmax_fp32",
            "bytes_per_elem": 4,
            "ms": _bench(lambda: torch.softmax(x, dim=-1)),
            "bytes_moved": kbytes,
            "max_error": _errors(torch.softmax(x, dim=-1), ref)[0],
            "mean_error": _errors(torch.softmax(x, dim=-1), ref)[1],
        })

        for c in cases:
            c["gbs"] = round(c["bytes_moved"] / (c["ms"] * 1e-3) / 1e9, 2)
            c["gb_per_s"] = c["gbs"]
        results.append({"rows": rows, "cols": cols, "cases": cases})
        print(f"[{rows}x{cols}] " + " | ".join(
            f"{c['variant']} {c['ms']*1000:.1f}us {c['gb_per_s']:.1f}GB/s "
            f"max_err {c['max_error']:.2e}" for c in cases))

    payload = {
        "measured_on": "2026-11-01",
        "device": dev.name,
        "sm_count": dev.multi_processor_count,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "script_in_repo": "gpgpu-kernel-lab/benchmarks/triton/bench_bf16_softmax.py",
        "how": "kaggle kernels push（T4，GPU 模式），结果 JSON 由本脚本写出，"
               "原样提交进仓库 benchmarks/results/",
        "results": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
