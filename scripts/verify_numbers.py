"""复算闸：把 README / 文档里写的每个数字从代码重算一遍，不一致就 exit 1。

用法：
    PYTHONPATH=src python3 scripts/verify_numbers.py

为什么要有这个文件：文档里的数字会漂移——改了 kernel，README 里的 74112 没跟着改，
读者就会读到一句「有出处的假话」。铁律是「任何百分数/字节数必须能从落盘证据重算」，
这个脚本就是那条铁律的可执行版本。它同时也是 AGENTS「门必须能失败」的实践：
把任意一个期望值改错（值本身在 EXPECTED 里），脚本必须 exit≠0。
"""

from __future__ import annotations

import sys

import numpy as np

from gkern.gpgpu.device import DeviceInfo
from gkern.gpgpu.memory import count_transactions
from gkern.gpgpu.samples import run_sample, vec_add_coalesced, vec_add_strided
from gkern.ops import run_fused_mlp

dev = DeviceInfo()

# ---- 期望值表（唯一真相；改这里 = 改文档里的数字） ----------------------
EXPECTED = {
    "peak_flops_tflops": 26.2144,
    "bandwidth_gbps": 102.4,
    "roofline_crossover": 256.0,
    "coalesced_load_transactions": 64,
    "strided_load_transactions": 2048,
    "transaction_amplification": 32,
    "divergence_rate_mixed": 1.0,
    "divergence_rate_positive": 0.0,
    "lane_efficiency_mixed_rounded": 0.667,
    "lane_efficiency_positive_rounded": 1.0,
    "fusion_unfused_bytes": 74112,
    "fusion_fused_bytes": 49536,
    "fusion_saved_bytes": 24576,
    # 共享内存与复用上界（P0-A 新增）
    "smem_bytes_tile32": 4456448,
    "bank_conflicts_pitch32": 992,
    "bank_conflicts_pitch33": 0,
    "reuse_crossover_n": 1536,
    "reuse_ai_n512_rounded": 88.0,
    "reuse_ai_n2048_rounded": 352.0,
    # 云端实测对账（证据 = benchmarks/results/triton_softmax_t4.json）
    "t4_bandwidth_gbps": 320.0,
    "t4_crossover_rounded": 25.44,
    "t4_measured_over_predicted_4096": 1.398,
    "t4_speedup_vs_eager_512": 0.741,
    # 混合精度（bf16 累加）—— 舍入必须与 torch.bfloat16 逐位一致
    "bf16_round_tie_down": 1.0,
    "bf16_round_tie_up": 1.015625,
    "bf16_error_ratio_floor": 10000,
}

failures: list[str] = []


def check(name: str, got, want) -> None:
    ok = (abs(got - want) < 1e-6) if isinstance(want, float) else (got == want)
    print(f"{'ok  ' if ok else 'FAIL'} {name}: got={got!r} want={want!r}")
    if not ok:
        failures.append(name)


# 1. 设备模型（解析值）
check("peak_flops_tflops", round(dev.peak_flops_tflops, 4), EXPECTED["peak_flops_tflops"])
check("bandwidth_gbps", round(dev.theoretical_bandwidth_gbps, 2), EXPECTED["bandwidth_gbps"])
check("roofline_crossover",
      round(dev.peak_flops_tflops * 1e12 / (dev.theoretical_bandwidth_gbps * 1e9), 1),
      EXPECTED["roofline_crossover"])

# 2. 合并访存：同一份数据、两种访问形态
n = 1024
coal, _, _ = run_sample(vec_add_coalesced, n=n)
strid, _, _ = run_sample(vec_add_strided, n=n)
cs, ss = coal.summary(), strid.summary()
check("coalesced_load_transactions",
      cs["global_load_transactions"], EXPECTED["coalesced_load_transactions"])
check("strided_load_transactions",
      ss["global_load_transactions"], EXPECTED["strided_load_transactions"])
check("byte_requests_identical",
      cs["global_bytes_requested"] == ss["global_bytes_requested"], True)
check("transaction_amplification",
      ss["global_load_transactions"] // cs["global_load_transactions"],
      EXPECTED["transaction_amplification"])

# 3. 分支发散
from gkern.gpgpu.samples import relu_branchy  # noqa: E402

rng = np.random.default_rng(0)
mixed, _, _ = run_sample(relu_branchy, n=n)
ms = mixed.summary()
check("divergence_rate_mixed", ms["divergence_rate"], EXPECTED["divergence_rate_mixed"])
check("lane_efficiency_mixed_rounded",
      round(ms["warp_efficiency"], 3), EXPECTED["lane_efficiency_mixed_rounded"])

# 3b. 全正输入：没有 lane 走 else 分支 → 发散率必须为 0、lane 效率必须为 1
#     （早期这个 EXPECTED 条目写了却从没断言，是一行死代码；现在补上真检查）
from gkern.gpgpu.kernel import launch  # noqa: E402

a_pos = np.abs(rng.standard_normal(n)).astype(np.float32) + 0.01
out_pos = np.zeros(n, dtype=np.float32)
pos_res = launch(relu_branchy, {"a": a_pos, "out": out_pos},
                 grid=((n + 255) // 256,), block=(256,), extra=(n,))
ps = pos_res.summary()
check("divergence_rate_positive", ps["divergence_rate"], EXPECTED["divergence_rate_positive"])
check("lane_efficiency_positive_rounded",
      round(ps["warp_efficiency"], 3), EXPECTED["lane_efficiency_positive_rounded"])

# 4. 融合省访存（等式，不是估计）
m, k, nn = 64, 32, 48
a = rng.standard_normal((m, k)).astype(np.float32)
w = rng.standard_normal((k, nn)).astype(np.float32)
bias = rng.standard_normal(nn).astype(np.float32)
fused = run_fused_mlp(a, w, bias, bm=32, bn=16, bk=16, fuse=True)
unfused = run_fused_mlp(a, w, bias, bm=32, bn=16, bk=16, fuse=False)
check("fusion_unfused_bytes", unfused.cost.bytes_total, EXPECTED["fusion_unfused_bytes"])
check("fusion_fused_bytes", fused.cost.bytes_total, EXPECTED["fusion_fused_bytes"])
check("fusion_saved_bytes",
      unfused.cost.bytes_total - fused.cost.bytes_total, EXPECTED["fusion_saved_bytes"])
check("fusion_saved_equals_2_times_C",
      unfused.cost.bytes_total - fused.cost.bytes_total == 2 * m * nn * 4, True)
check("fusion_numerically_equal",
      bool(np.allclose(fused.output, unfused.output, atol=1e-5)), True)

# 4b. 文档里的测试数必须与真实收集数一致（漂移过两次：58→68→74）
#     数字类文档断言最容易腐烂，所以把它也做成可复算项，而不是靠人记得改。
import ast  # noqa: E402
import pathlib  # noqa: E402
import re  # noqa: E402

_root = pathlib.Path(__file__).resolve().parent.parent

# 判定用「测试函数个数」而不是「pytest 收集到的用例数」：后者会随可选扩展是否
# 构建而变（C++ 扩展未构建时 22 条用例整模块跳过），把它当文档数就会让同一条
# 闸在不同机器上给出不同结论。函数个数是源码的性质，机器无关。
_func_count = 0
for _f in sorted((_root / "tests").glob("test_*.py")):
    _tree = ast.parse(_f.read_text(encoding="utf-8"))
    _func_count += sum(1 for node in _tree.body
                       if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"))

for doc in ("README.md", "docs/jd-mapping.md"):
    text = (_root / doc).read_text(encoding="utf-8")
    claimed = re.search(r"(\d+) 个测试函数", text)
    if claimed is None:
        failures.append(f"{doc}_test_count_missing")
        print(f"FAIL {doc}_test_count_missing: 文档里找不到「N 个测试函数」")
        continue
    check(f"doc_test_count[{doc}]", int(claimed.group(1)), _func_count)

# 4c. 共享内存：不改全局字节、bank 冲突按 warp 分组、复用上界只在 n>=1536 翻转
from gkern.compiler import compile_kernel, costmodel  # noqa: E402
from gkern.ops import matmul as _mm  # noqa: E402
from gkern.ops import run_matmul, run_smem_column  # noqa: E402

_a256 = rng.standard_normal((256, 256)).astype(np.float32)
_b256 = rng.standard_normal((256, 256)).astype(np.float32)
_plain = run_matmul(_a256, _b256, bm=32, bn=32, bk=16)
_smem = run_matmul(_a256, _b256, bm=32, bn=32, bk=16, use_smem=True)
check("smem_bytes_tile32", _plain.cost.bytes_total, EXPECTED["smem_bytes_tile32"])
check("smem_bytes_equal_global", _smem.cost.bytes_total, _plain.cost.bytes_total)
check("smem_numerically_equal",
      bool(np.allclose(_smem.output, _plain.output, atol=1e-5)), True)

_col32 = run_smem_column(32)
_col33 = run_smem_column(33)
check("bank_conflicts_pitch32", _col32.stats.shared_bank_conflicts,
      EXPECTED["bank_conflicts_pitch32"])
check("bank_conflicts_pitch33", _col33.stats.shared_bank_conflicts,
      EXPECTED["bank_conflicts_pitch33"])

check("reuse_crossover_n", int(6 * costmodel.crossover_intensity(dev)),
      EXPECTED["reuse_crossover_n"])
for _n, _key, _bound in ((512, "reuse_ai_n512_rounded", "memory"),
                         (2048, "reuse_ai_n2048_rounded", "compute")):
    _z = np.zeros((_n, _n), dtype=np.float32)
    _prog = compile_kernel(_mm.matmul_tiled, {"A": _z, "B": _z, "C": _z},
                           {"M": _n, "N": _n},
                           {"BM": 64, "BN": 64, "BK": 16, "K": _n},
                           (_n // 64, _n // 64))
    _up = costmodel.reuse_bound(_prog)
    check(_key, round(_up.arithmetic_intensity, 4), EXPECTED[_key])
    check(f"reuse_bound_n{_n}", _up.bound, _bound)
    check(f"reuse_is_upper_bound_n{_n}",
          bool(_up.bytes_total < costmodel.analyze(_prog).bytes_total), True)

# 4d. 云端实测对账：分子来自落盘证据、分母来自可复算的模型
from gkern.compiler import compile_kernel as _ck  # noqa: E402
from gkern.gpgpu.device import DeviceInfo as _DI  # noqa: E402
from gkern.ops.softmax import softmax_rows as _softmax_rows  # noqa: E402

_t4 = _DI.tesla_t4()
check("t4_bandwidth_gbps", round(_t4.theoretical_bandwidth_gbps, 1),
      EXPECTED["t4_bandwidth_gbps"])
check("t4_crossover_rounded",
      round(costmodel.crossover_intensity(_t4), 2), EXPECTED["t4_crossover_rounded"])

_ev_path = _root / "benchmarks" / "results" / "triton_softmax_t4.json"
_ev = __import__("json").loads(_ev_path.read_text(encoding="utf-8"))
check("cloud_evidence_rows", len(_ev["rows"]), 5)
check("cloud_evidence_has_provenance", "provenance" in _ev, True)
_r4096 = next(r for r in _ev["rows"] if r["cols"] == 4096)
_m = _ck(_softmax_rows,
         {"X": np.zeros((_r4096["rows"], 4096), np.float32),
          "Y": np.zeros((_r4096["rows"], 4096), np.float32)},
         {"M": _r4096["rows"], "N": 4096}, {"BLOCK_N": 4096}, (_r4096["rows"],))
_pred = costmodel.analyze(_m, _t4).predicted_ms
check("t4_measured_over_predicted_4096",
      round(_r4096["triton_ms"] / _pred, 3),
      EXPECTED["t4_measured_over_predicted_4096"])
_r512 = next(r for r in _ev["rows"] if r["cols"] == 512)
check("t4_speedup_vs_eager_512", _r512["speedup_vs_eager"],
      EXPECTED["t4_speedup_vs_eager_512"])

# 4e. 混合精度：舍入语义 + 精度代价
from gkern.compiler.codegen import _bf16_round as _r16  # noqa: E402
from gkern.ops.mixed_precision import run_softmax_precision as _prec  # noqa: E402

_tie_down = np.float32(1.0) + np.float32(2.0 ** -8)
_tie_up = np.float32(1.0) + np.float32(3 * 2.0 ** -8)
_rounded = _r16(np.array([_tie_down, _tie_up], np.float32))
check("bf16_round_tie_down", float(_rounded[0]), EXPECTED["bf16_round_tie_down"])
check("bf16_round_tie_up", float(_rounded[1]), EXPECTED["bf16_round_tie_up"])

_rng16 = np.random.default_rng(20261101)
_x16 = (_rng16.standard_normal((256, 1024)) * 3.0).astype(np.float32)
_p32 = _prec(_x16, accumulate="fp32")
_p16 = _prec(_x16, accumulate="bf16")
check("precision_bytes_equal", _p16.cost.bytes_total == _p32.cost.bytes_total, True)
check("precision_error_ratio_floor",
      bool(_p16.max_error / _p32.max_error > EXPECTED["bf16_error_ratio_floor"]), True)

# 5. 事务计数函数本身的 128B 分段语义（防「计数尺」漂移）
check("count_transactions_128B_split",
      count_transactions(np.array([0, 128, 256], dtype=np.int64)), 3)
check("count_transactions_same_segment",
      count_transactions(np.array([0, 4, 8], dtype=np.int64)), 1)

# ---- 自检：闸必须能失败（AGENTS「门必须能失败」） ----------------------
if "--selftest" in sys.argv:
    # 故意把一项期望值改错，再跑同一套检查，必须报错才算闸有效。
    EXPECTED["fusion_saved_bytes"] += 1
    failures.clear()
    unfused2 = run_fused_mlp(a, w, bias, bm=32, bn=16, bk=16, fuse=False)
    fused2 = run_fused_mlp(a, w, bias, bm=32, bn=16, bk=16, fuse=True)
    check("selftest_injected_wrong_value",
          unfused2.cost.bytes_total - fused2.cost.bytes_total,
          EXPECTED["fusion_saved_bytes"])
    if failures:
        print("\nselftest ok：注入错误期望值后闸确实红了。")
        sys.exit(0)
    print("\nselftest FAILED：注入错误期望值后闸仍然是绿的——这个闸是装饰。")
    sys.exit(2)

if failures:
    print(f"\n{len(failures)} 项不一致：{failures}")
    sys.exit(1)
print("\n全部数字复算通过。")
