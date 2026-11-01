"""在 TPU（JAX/XLA）上量同一组 bf16 舍入点的误差——T4 额度用尽时的平行证据。

它不是「T4 的替代品」，而是另一块真实加速卡上的**同一问题的另一次提问**：

- 变体 `fp32` / `bf16acc` / `bf16io` 的**语义**与 `benchmarks/triton/bench_bf16_softmax.py`
  完全对齐（`bf16acc` 仍是「存取 float32、只在输入/行最大值/指数结果/归约和四处
  把中间值舍到 bf16」）；
- 差别在**执行栈**：XLA 会把整条行 softmax 融成一片，而 Triton 里每个 program
  是一个 block。所以耗时不与 T4 可比，**只有误差量级**是可以跨平台对照的那一格。

写这个脚本的原因是 Kaggle 的每周 GPU 额度已用尽（2026-11-01 实测 32.75h/30h，
剩余 0.00h），而 TPU 额度 20h 仍在。与其把 bf16 误差这一格留成空白，不如在另一块
真实加速卡上把它量出来，并把「哪一格可以跨平台对照」写清楚。

用法（Kaggle TPU 内核）::

    python bench_bf16_softmax_jax.py --out /kaggle/working/bf16_softmax_tpu.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def _require_jax():
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
    except Exception as exc:  # pragma: no cover - 只在无 JAX 环境触发
        sys.exit(f"这个脚本需要 JAX/XLA；当前环境不可用：{exc}")
    import jax
    import jax.numpy as jnp
    return jax, jnp


def _build(jnp):
    def softmax_fp32(x):
        m = jnp.max(x, axis=-1, keepdims=True)
        e = jnp.exp(x - m)
        return e / jnp.sum(e, axis=-1, keepdims=True)

    def _r16(a):
        return a.astype(jnp.bfloat16).astype(jnp.float32)

    def softmax_bf16acc(x):
        # 四处舍入，与 CPU/IR 侧的 round_bf16 位置一一对应
        x = _r16(x)
        m = _r16(jnp.max(x, axis=-1, keepdims=True))
        e = _r16(jnp.exp(x - m))
        s = _r16(jnp.sum(e, axis=-1, keepdims=True))
        return e / s

    return softmax_fp32, softmax_bf16acc


def _error(got: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    d = np.abs(got.astype(np.float64) - ref)
    return float(d.max()), float(d.mean())


def _time_ms(fn, arg, jax, repeats: int = 50) -> float:
    fn(arg).block_until_ready()  # 先编译/热身
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(arg).block_until_ready()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bf16_softmax_tpu.json")
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--cols", default="512,1024,2048")
    args = ap.parse_args()

    jax, jnp = _require_jax()
    softmax_fp32, softmax_bf16acc = _build(jnp)
    jsoft32 = jax.jit(softmax_fp32)
    jsoftacc = jax.jit(softmax_bf16acc)

    rows = args.rows
    shapes = [int(c) for c in args.cols.split(",")]
    rng = np.random.default_rng(20261101)

    results = []
    for cols in shapes:
        x_np = (rng.standard_normal((rows, cols)) * 2.0).astype(np.float32)
        x = jnp.asarray(x_np)
        # 参考解在 float64 上算（TPU 不保证 fp64，所以用 CPU numpy 算）
        a = x_np.astype(np.float64)
        e = np.exp(a - a.max(axis=-1, keepdims=True))
        ref = e / e.sum(axis=-1, keepdims=True)

        out32 = np.asarray(jsoft32(x))
        outacc = np.asarray(jsoftacc(x))
        x16 = jnp.asarray(x_np.astype(np.float32)).astype(jnp.bfloat16)
        outio = np.asarray(jsoft32(x16))

        kbytes = rows * cols * 4 * 2
        io_bytes = rows * cols * 2 * 2
        cases = []
        for name, fn, arg, mv, bytes_per_elem in (
            ("fp32", jsoft32, x, out32, 4),
            ("bf16acc", jsoftacc, x, outacc, 4),
            ("bf16io", jsoft32, x16, outio, 2),
        ):
            mx, mean = _error(np.asarray(fn(arg)), ref)
            ms = _time_ms(fn, arg, jax)
            cases.append({
                "variant": name,
                "bytes_per_elem": bytes_per_elem,
                "ms": round(ms, 5),
                "bytes_moved": io_bytes if name == "bf16io" else kbytes,
                "gbs": round((io_bytes if name == "bf16io" else kbytes) / (ms * 1e-3) / 1e9, 2),
                "max_error": mx,
                "mean_error": mean,
            })
        results.append({"rows": rows, "cols": cols, "cases": cases})
        print(f"[{rows}x{cols}] " + " | ".join(
            f"{c['variant']} {c['ms']:.3f}ms {c['gbs']:.1f}GB/s max_err {c['max_error']:.2e}"
            for c in cases))

    payload = {
        "measured_on": "2026-11-01",
        "platform": "kaggle-tpu",
        "jax": jax.__version__,
        "python": platform.python_version(),
        "note": "误差量级可与 T4 对照；耗时不与 T4 可比（XLA 融合 vs Triton block）",
        "script_in_repo": "gpgpu-kernel-lab/benchmarks/tpu/bench_bf16_softmax_jax.py",
        "results": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
