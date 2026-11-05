"""在 TPU（JAX/XLA）上量同一组 bf16 舍入点的误差。

本仓库没有这份脚本的输出。没有 TPU 设备时它直接退出，不写 JSON。
CPU 上的 JAX 同样退出：那不是 TPU 结果。

变体 `fp32` / `bf16acc` / `bf16io` 的语义与
`benchmarks/triton/bench_bf16_softmax.py` 对齐。耗时不能拿去和别的平台比。

用法（仅在 jax.devices() 里真有 TPU 时）::

    python bench_bf16_softmax_jax.py --out bf16_softmax_tpu.json
"""

from __future__ import annotations

import argparse
import datetime
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
    platforms = [device.platform for device in jax.devices()]
    if "tpu" not in platforms:
        sys.exit(f"需要 TPU 设备；当前是 {platforms}。不写结果文件。")
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
        "measured_on": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platforms,
        "jax": jax.__version__,
        "python": platform.python_version(),
        "note": "只有在 jax 报告了 TPU 设备时才会写出这个文件。耗时不能拿去和别的平台比。",
        "script_in_repo": "benchmarks/tpu/bench_bf16_softmax_jax.py",
        "results": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
