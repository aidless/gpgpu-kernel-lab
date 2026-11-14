"""五分钟上手：一条命令看清「算子 → IR → 代价 → 融合收益 → torch.compile」。

    PYTHONPATH=src python3 examples/quickstart.py

这个脚本刻意不引入任何图表库，也不铺满参数，只把本仓库能做什么、
每个数字从哪来，按顺序讲一遍。
"""

from __future__ import annotations

import numpy as np

import torch

from gkern.gpgpu import device, samples
from gkern.ops import run_fused_mlp, run_matmul, run_softmax, run_vector_add


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    dev = device.DeviceInfo()
    banner("1. 设备模型（假设机，不是真卡）")
    print(f"SM 数          : {dev.sm_count}")
    print(f"warp 宽度      : {dev.warp_size}")
    print(f"峰值 FP32 算力 : {dev.peak_flops_tflops:.4f} TFLOPS")
    print(f"理论带宽       : {dev.theoretical_bandwidth_gbps:.1f} GB/s")
    print(f"roofline 交叉点: {dev.peak_flops_tflops * 1000 / dev.theoretical_bandwidth_gbps:.1f} FLOP/B")

    banner("2. 同一个加法，访存合并与否差多少")
    n = 1024
    good, _, _ = samples.run_sample(samples.vec_add_coalesced, n=n)
    bad, _, _ = samples.run_sample(samples.vec_add_strided, n=n)
    print(f"合并访存   : {good.memory.global_load_transactions:>4} 个 load 事务，"
          f"合并效率 {good.memory.coalescing_efficiency:.2f}")
    print(f"跨步访存   : {bad.memory.global_load_transactions:>4} 个 load 事务，"
          f"合并效率 {bad.memory.coalescing_efficiency:.2f}")
    print(f"事务放大   : {bad.memory.global_load_transactions / good.memory.global_load_transactions:.1f}×")

    banner("3. 分支发散")
    from gkern.gpgpu.kernel import launch

    rng0 = np.random.default_rng(11)
    mixed_in = rng0.standard_normal(n).astype(np.float32)      # 正负混合
    positive_in = np.abs(mixed_in) + 0.01                     # 全正，走同一个 then 分支

    def run_relu(values):
        out = np.zeros(n, dtype=np.float32)
        return launch(samples.relu_branchy, {"a": values, "out": out},
                      grid=(n // samples.BLOCK,), block=(samples.BLOCK,), extra=(n,))

    mixed = run_relu(mixed_in)
    uniform = run_relu(positive_in)
    print(f"正负混合输入: 发散率 {mixed.report.divergence_rate:.2f}，"
          f"lane 效率 {mixed.report.efficiency:.3f}")
    print(f"全正输入    : 发散率 {uniform.report.divergence_rate:.2f}，"
          f"lane 效率 {uniform.report.efficiency:.3f}")
    print(f"结果正确    : {np.allclose(mixed.outputs['out'], np.maximum(mixed_in, 0))}")

    banner("4. 算子：IR + 代价模型")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4096).astype(np.float32)
    add = run_vector_add(x, x)
    sm = run_softmax(rng.standard_normal((8, 37)).astype(np.float32))
    mm = run_matmul(rng.standard_normal((64, 64)).astype(np.float32),
                    rng.standard_normal((64, 64)).astype(np.float32))
    for res in (add, sm, mm):
        c = res.cost
        print(f"{res.program.name:<22} 访存 {c.bytes_total:>8} B  "
              f"FLOP {c.flops:>9}  AI {c.arithmetic_intensity:>6.2f}  "
              f"{c.bound:<8} 预测 {c.predicted_ms * 1000:>7.2f} µs")

    banner("5. 算子融合：省掉的访存可以逐字节复算")
    a = rng.standard_normal((64, 32)).astype(np.float32)
    w = rng.standard_normal((32, 48)).astype(np.float32)
    bias = rng.standard_normal(48).astype(np.float32)
    fused = run_fused_mlp(a, w, bias, bm=32, bn=16, bk=16, fuse=True)
    unfused = run_fused_mlp(a, w, bias, bm=32, bn=16, bk=16, fuse=False)
    saved = unfused.cost.bytes_total - fused.cost.bytes_total
    print(f"未融合: {unfused.cost.bytes_total} B   融合: {fused.cost.bytes_total} B")
    print(f"省下  : {saved} B = 2 × |C| = 2 × 64 × 48 × 4 B（中间张量的写回 + 读出）")
    print(f"结果一致: {np.allclose(fused.output, unfused.output, atol=1e-5)}")
    print(f"融合后输入缓冲区: {sorted(fused.program.inputs)}（中间缓冲区 C 已消失）")

    banner("6. torch.compile 自定义后端")
    from gkern.torch_ext.backend import REPORTS, gkern_backend

    torch._dynamo.reset()
    REPORTS.clear()

    def fn(x_, y_):
        import torch.nn.functional as F
        return F.gelu(x_ * 1.7 + y_) * 0.5 + x_

    tx, ty = torch.randn(37), torch.randn(37)
    compiled = torch.compile(fn, backend=gkern_backend, fullgraph=True, dynamic=False)
    out = compiled(tx, ty)
    report = next(iter(REPORTS.values()))
    print(f"融合成功: {report.fused}  节点数 {report.nodes}  原因: {report.reason}")
    print(f"与 eager 一致: {torch.allclose(out, fn(tx, ty), atol=1e-6)}")
    print(f"编译结论: {report.as_dict()}")


if __name__ == "__main__":
    main()
