"""把云端实测（T4）与本仓库代价模型的预测放回同一把尺上对账。

它不产生任何新测量，只做三件事：

1. 读 `benchmarks/results/triton_softmax_t4.json`（**唯一**的硬件实测证据，
   带 provenance 字段）；
2. 用 `compiler.costmodel` 在 `DeviceInfo.tesla_t4()` 上重算同一 shape 的
   `softmax_rows` 程序，得到「理想下界」预测；
3. 输出 `实测 / 预测` 比值与效率，并据此判断模型是在低估还是高估。

关键纪律：**比值的分子来自落盘证据，分母来自可复算的模型**。任何一方手改都会
让这张表变成故事，所以 `scripts/verify_numbers.py` 会从这两个来源重新算出比值。

用法::

    PYTHONPATH=src python3 benchmarks/analyze_triton.py --write-report
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from gkern.compiler import compile_kernel, costmodel
from gkern.gpgpu.device import DeviceInfo
from gkern.ops.softmax import softmax_rows

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "benchmarks" / "results" / "triton_softmax_t4.json"
REPORT = ROOT / "benchmarks" / "results" / "reconciliation_t4.md"


def model_prediction(rows: int, cols: int, dev: DeviceInfo) -> costmodel.Cost:
    """用代价模型算「同一 shape 的行 softmax」在给定设备上的理想下界。"""
    block = int(2 ** np.ceil(np.log2(cols)))
    prog = compile_kernel(softmax_rows,
                          {"X": np.zeros((rows, cols), np.float32),
                           "Y": np.zeros((rows, cols), np.float32)},
                          {"M": rows, "N": cols}, {"BLOCK_N": block},
                          (rows,))
    return costmodel.analyze(prog, dev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-report", action="store_true")
    args = ap.parse_args()

    dev = DeviceInfo.tesla_t4()
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    rows = evidence["rows"]

    out = []
    for r in rows:
        cost = model_prediction(r["rows"], r["cols"], dev)
        predicted_ms = cost.predicted_ms
        ratio = r["triton_ms"] / predicted_ms
        out.append({
            "cols": r["cols"],
            "bytes": r["bytes_read_write"],
            "model_bytes": cost.bytes_total,
            "model_ms": round(predicted_ms, 5),
            "triton_ms": r["triton_ms"],
            "eager_ms": r["eager_ms"],
            "measured_over_predicted": round(ratio, 3),
            "efficiency_vs_theory": round(1.0 / ratio, 3),
            "bound": cost.bound,
        })

    header = ("| cols | 字节 | 模型预测 (ms) | Triton 实测 (ms) | eager 实测 (ms) | "
              "实测/预测 | 效率 | 受限 |")
    lines = [header, "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in out:
        lines.append(f"| {r['cols']} | {r['bytes']} | {r['model_ms']} | {r['triton_ms']} | "
                     f"{r['eager_ms']} | {r['measured_over_predicted']} | "
                     f"{r['efficiency_vs_theory']} | {r['bound']} |")
    table = "\n".join(lines)
    print(table)
    print()
    print(f"设备预设：{dev.name}  SM={dev.sm_count}  理论带宽={dev.theoretical_bandwidth_gbps:.1f} GB/s  "
          f"FP32 峰值={dev.peak_flops_tflops:.3f} TFLOPS  "
          f"交叉点={costmodel.crossover_intensity(dev):.2f} FLOP/B")

    if args.write_report:
        body = "\n".join([
            "# 云端实测 vs 代价模型（Tesla T4）",
            "",
            f"证据文件：`benchmarks/results/triton_softmax_t4.json`"
            f"（{evidence['provenance']['measured_on']}，"
            f"{evidence['provenance']['kernel_url']}）。",
            "本表由 `benchmarks/analyze_triton.py --write-report` 生成，不要手改。",
            "",
            table,
            "",
            f"设备预设 `DeviceInfo.tesla_t4()`：理论带宽 "
            f"{dev.theoretical_bandwidth_gbps:.1f} GB/s、FP32 峰值 "
            f"{dev.peak_flops_tflops:.3f} TFLOPS、roofline 交叉点 "
            f"{costmodel.crossover_intensity(dev):.2f} FLOP/B。",
            "",
        ])
        REPORT.write_text(body, encoding="utf-8")
        print(f"报告已写入 {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
