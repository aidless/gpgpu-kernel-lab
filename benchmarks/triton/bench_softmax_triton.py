"""在**真实 GPU** 上跑 Triton fused softmax，量出带宽与「CPU 教学模型差多远」。

这个文件是整个仓库里唯一允许产生「GFLOPS / GB/s 实测数字」的地方：它必须在
真实 GPU 上跑（Kaggle T4 / 云上 A100），本地 CPU 只做语法检查。

用法（云上）::

    python bench_softmax_triton.py --out triton_softmax.json

它回答三个问题，且每个答案都带出处：

1. **Triton fused softmax 在 T4 上到底跑多少 GB/s**——对比理论的
   `memory_clock × 2 × bus_width/8`。
2. **BLOCK_SIZE 从 1024 扫到 8192，带宽怎么走**——rmem 溢出会掉档，掉在哪一档。
3. **和 torch eager（非 fused，多次读写）比快多少倍**——这是「融合」这件事
   在真实硬件上的定价，而不是本仓库 CPU 模型里的字节数换算。

注意：本脚本**不**声称任何数字能代表 `gkern` 的 CPU 教学模型性能。两者不可比——
CPU 模型的数字是「机制演示」，这里的数字是「硬件实测」。混用就是骗自己。
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """一行一个 program 的 fused softmax：读一次、写一次。"""
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def triton_softmax_rows(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block = triton.next_power_of_2(n_cols)
    softmax_kernel[(n_rows,)](
        out, x, x.stride(0), out.stride(0), n_cols,
        BLOCK_SIZE=block, num_warps=8,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="triton_softmax.json")
    ap.add_argument("--rows", type=int, default=4096)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("这个脚本必须在真实 GPU 上跑；CUDA 不可用就退出（不要伪造数字）")

    dev = torch.cuda.get_device_properties(0)
    meta = {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_gb": round(dev.total_memory / 1024**3, 2),
        "sm_count": dev.multi_processor_count,
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "cuda_version": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    rows = []
    for n_cols in (512, 1024, 2048, 4096, 8192):
        x = torch.randn(args.rows, n_cols, device="cuda", dtype=torch.float32)
        ref = torch.softmax(x, dim=1)

        got = triton_softmax_rows(x)
        max_diff = float((got - ref).abs().max().item())
        block = triton.next_power_of_2(n_cols)

        # 字节数：读一次 + 写一次，各 rows*cols*4
        nbytes = 2 * args.rows * n_cols * 4
        ms_triton = triton.testing.do_bench(lambda: triton_softmax_rows(x),
                                            warmup=25, rep=100)
        ms_eager = triton.testing.do_bench(lambda: torch.softmax(x, dim=1),
                                           warmup=25, rep=100)

        rows.append({
            "rows": args.rows,
            "cols": n_cols,
            "block_size": block,
            "bytes_read_write": nbytes,
            "triton_ms": round(ms_triton, 5),
            "triton_gbps": round(nbytes / (ms_triton * 1e6), 2),
            "eager_ms": round(ms_eager, 5),
            "eager_gbps": round(nbytes / (ms_eager * 1e6), 2),
            "speedup_vs_eager": round(ms_eager / ms_triton, 3),
            "max_abs_diff": max_diff,
        })
        del x, ref, got
        torch.cuda.empty_cache()

    result = {"meta": meta, "rows": rows, "measured_at": args.out}
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    print(f"GPU: {meta['gpu_name']}  SMs={meta['sm_count']}  "
          f"torch={meta['torch_version']}  triton={meta['triton_version']}")
    print("| rows | cols | BLOCK | triton ms | triton GB/s | eager ms | eager GB/s |"
          " x faster | max|diff| |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['rows']} | {r['cols']} | {r['block_size']} | {r['triton_ms']} | "
              f"{r['triton_gbps']} | {r['eager_ms']} | {r['eager_gbps']} | "
              f"{r['speedup_vs_eager']} | {r['max_abs_diff']:.3e} |")


if __name__ == "__main__":
    main()
