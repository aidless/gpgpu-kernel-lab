"""基准脚本：把「模型预测」和「numpy 实际执行墙钟」并排量出来。

    PYTHONPATH=src python3 benchmarks/run_benchmarks.py

**这个脚本不测 GPU**（本机没有 GPU，见 README「边界」）。它量的是两件在
CPU 上也能诚实回答的事：

1. numpy 执行器跑这些形状要多久（这是「真的算了」的实锤）；
2. 代价模型对同一份 IR 给出的理论下界是多少（这是解析估计）。

两者相差几个数量级是**正常的**：numpy 执行器逐 block 走 Python 循环，
而模型算的是"如果这些访存/FLOP 发生在设备上"的时间。放在一起的意义是让
读者一眼看到差距来自哪里，而不是拿其中一个冒充另一个。
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from gkern.compiler import compile_kernel, costmodel
from gkern.ops import (matmul, run_fused_mlp, run_matmul, run_smem_column,
                       run_softmax, run_vector_add)

OUT_DIR = pathlib.Path(__file__).resolve().parent / "results"


def timeit(fn, repeats: int = 3) -> float:
    """取 ``repeats`` 次里的最好成绩（毫秒）。"""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def main() -> None:
    rng = np.random.default_rng(0)
    rows = []

    # 逐元素：规模扫描，看事务数与模型预测怎么随 n 走
    for n in (1 << 12, 1 << 16, 1 << 18):
        x = rng.standard_normal(n).astype(np.float32)
        res = run_vector_add(x, x)
        ms = timeit(lambda: run_vector_add(x, x))
        rows.append({
            "kernel": "vector_add",
            "shape": f"({n},)",
            "wall_ms": round(ms, 3),
            "predicted_us": round(res.cost.predicted_ms * 1000, 3),
            "bytes_total": res.cost.bytes_total,
            "flops": res.cost.flops,
            "arithmetic_intensity": round(res.cost.arithmetic_intensity, 4),
            "bound": res.cost.bound,
        })

    # 归约：固定行数，扫描宽度（BLOCK_N 必须 ≥ N）
    for width in (64, 256, 1024):
        x = rng.standard_normal((64, width)).astype(np.float32)
        res = run_softmax(x)
        ms = timeit(lambda: run_softmax(x))
        rows.append({
            "kernel": "softmax_rows",
            "shape": f"(64, {width})",
            "wall_ms": round(ms, 3),
            "predicted_us": round(res.cost.predicted_ms * 1000, 3),
            "bytes_total": res.cost.bytes_total,
            "flops": res.cost.flops,
            "arithmetic_intensity": round(res.cost.arithmetic_intensity, 4),
            "bound": res.cost.bound,
        })

    # matmul：tile 越大，算术强度越高（复用越多）
    for tile in (16, 32, 64):
        a = rng.standard_normal((256, 256)).astype(np.float32)
        b = rng.standard_normal((256, 256)).astype(np.float32)
        res = run_matmul(a, b, bm=tile, bn=tile, bk=16)
        ms = timeit(lambda: run_matmul(a, b, bm=tile, bn=tile, bk=16))
        rows.append({
            "kernel": "matmul_tiled",
            "shape": f"(256,256)x(256,256) tile={tile}",
            "wall_ms": round(ms, 3),
            "predicted_us": round(res.cost.predicted_ms * 1000, 3),
            "bytes_total": res.cost.bytes_total,
            "flops": res.cost.flops,
            "arithmetic_intensity": round(res.cost.arithmetic_intensity, 4),
            "bound": res.cost.bound,
        })

    # 融合增益：省下的字节 == 2×|C|，**前提是分块整除 M/N**。
    # 当 BN ∤ N 时，代价模型按「整块 tile」计字节（含被掩掉的填充 lane），
    # 于是 counted bytes > 真实张量字节——这一行故意留着，好让人看到差异从哪来。
    cases = [
        (64, 32, 48, 32, 16, "BN 整除 N，等式严格成立"),
        (128, 64, 96, 32, 32, "BN 整除 N，等式严格成立"),
        (256, 128, 192, 32, 32, "BN 整除 N，等式严格成立"),
        (64, 32, 48, 32, 32, "BN 不整除 N(48)，tile 补齐使计数字节偏大"),
    ]
    fusion_rows = []
    for (m, k, n, bm, bn, note) in cases:
        a = rng.standard_normal((m, k)).astype(np.float32)
        w = rng.standard_normal((k, n)).astype(np.float32)
        bias = rng.standard_normal(n).astype(np.float32)
        fused = run_fused_mlp(a, w, bias, bm=bm, bn=bn, bk=16, fuse=True)
        unfused = run_fused_mlp(a, w, bias, bm=bm, bn=bn, bk=16, fuse=False)
        saved = unfused.cost.bytes_total - fused.cost.bytes_total
        fusion_rows.append({
            "shape": f"A({m},{k}) W({k},{n}) bm={bm} bn={bn}",
            "unfused_bytes": unfused.cost.bytes_total,
            "fused_bytes": fused.cost.bytes_total,
            "saved_bytes": saved,
            "expected_2_times_intermediate": 2 * m * n * 4,
            "saved_fraction": round(saved / unfused.cost.bytes_total, 4),
            "match": bool(saved == 2 * m * n * 4),
            "note": note,
            "outputs_close": bool(np.allclose(fused.output, unfused.output, atol=1e-5)),
        })

    # 共享内存分块：全局字节必须与 global-only 版**完全相等**（smem 不改 HBM 流量）
    smem_rows = []
    for tile in (32, 64):
        a = rng.standard_normal((256, 256)).astype(np.float32)
        b = rng.standard_normal((256, 256)).astype(np.float32)
        plain = run_matmul(a, b, bm=tile, bn=tile, bk=16)
        smem = run_matmul(a, b, bm=tile, bn=tile, bk=16, use_smem=True)
        smem_rows.append({
            "tile": tile,
            "bytes_plain": plain.cost.bytes_total,
            "bytes_smem": smem.cost.bytes_total,
            "bytes_equal": bool(plain.cost.bytes_total == smem.cost.bytes_total),
            "close": bool(np.allclose(smem.output, plain.output, atol=1e-5)),
        })

    # bank 冲突：行宽 32 与 33 的对比（padding 的作用）
    col_no_pad = run_smem_column(32)
    col_pad = run_smem_column(33)
    bank_rows = [
        {"pitch": 32, "accesses": col_no_pad.stats.shared_accesses,
         "conflicts": col_no_pad.stats.shared_bank_conflicts,
         "rate": round(col_no_pad.stats.shared_conflict_rate, 4)},
        {"pitch": 33, "accesses": col_pad.stats.shared_accesses,
         "conflicts": col_pad.stats.shared_bank_conflicts,
         "rate": round(col_pad.stats.shared_conflict_rate, 4)},
    ]

    # 复用上界：只 trace 不执行（n=2048 跑一遍要几十秒，而代价模型不需要执行）
    reuse_rows = []
    for n in (512, 1024, 2048):
        z = np.zeros((n, n), dtype=np.float32)
        prog = compile_kernel(matmul.matmul_tiled, {"A": z, "B": z, "C": z},
                              {"M": n, "N": n}, {"BM": 64, "BN": 64, "BK": 16, "K": n},
                              (n // 64, n // 64))
        lower = costmodel.analyze(prog)
        upper = costmodel.reuse_bound(prog)
        reuse_rows.append({
            "n": n,
            "lower_bytes": lower.bytes_total,
            "lower_ai": round(lower.arithmetic_intensity, 4),
            "lower_bound": lower.bound,
            "reuse_bytes": upper.bytes_total,
            "reuse_ai": round(upper.arithmetic_intensity, 4),
            "reuse_bound": upper.bound,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "bench.json").write_text(
        json.dumps({"kernels": rows, "fusion": fusion_rows, "smem": smem_rows,
                    "bank": bank_rows, "reuse": reuse_rows},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")

    header = "| kernel | shape | numpy 墙钟 (ms) | 模型预测 (µs) | 访存 (B) | FLOP | 算术强度 | 受限类型 |"
    sep = "|---|---|---:|---:|---:|---:|---:|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(f"| {r['kernel']} | {r['shape']} | {r['wall_ms']} | {r['predicted_us']} | "
                     f"{r['bytes_total']} | {r['flops']} | {r['arithmetic_intensity']} | {r['bound']} |")
    lines.append("")
    lines.append("| 融合用例 | 未融合 (B) | 融合后 (B) | 省下 (B) | 两倍中间张量 (B) | 省下比例 | 等式成立 | 数值一致 | 说明 |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|---|")
    for f in fusion_rows:
        lines.append(f"| {f['shape']} | {f['unfused_bytes']} | {f['fused_bytes']} | "
                     f"{f['saved_bytes']} | {f['expected_2_times_intermediate']} | "
                     f"{f['saved_fraction']} | {f['match']} | {f['outputs_close']} | {f['note']} |")
    lines.append("")
    lines.append("| tile | 未用 smem 访存 (B) | 用 smem 访存 (B) | 字节相等 | 数值一致 |")
    lines.append("|---|---:|---:|---|---|")
    for r in smem_rows:
        lines.append(f"| {r['tile']} | {r['bytes_plain']} | {r['bytes_smem']} | "
                     f"{r['bytes_equal']} | {r['close']} |")
    lines.append("")
    lines.append("| 共享内存行宽 (float) | 访问次数 | bank 冲突轮次 | 冲突率 |")
    lines.append("|---|---:|---:|---:|")
    for r in bank_rows:
        lines.append(f"| {r['pitch']} | {r['accesses']} | {r['conflicts']} | {r['rate']} |")
    lines.append("")
    lines.append("| 方阵 n | 逐块重读下界 (B) | 下界 AI | 下界受限 | 理想复用上界 (B) | 上界 AI | 上界受限 |")
    lines.append("|---|---:|---:|---|---:|---:|---|")
    for r in reuse_rows:
        lines.append(f"| {r['n']} | {r['lower_bytes']} | {r['lower_ai']} | {r['lower_bound']} | "
                     f"{r['reuse_bytes']} | {r['reuse_ai']} | {r['reuse_bound']} |")
    table = "\n".join(lines)
    (OUT_DIR / "bench.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
