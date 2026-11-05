# 云端归档 vs 代价模型：T4 上的行 softmax 对账

> 状态：归档，不是 OBSERVED (CPU)。分子来自 `benchmarks/results/triton_softmax_t4.json`，
> 这份文件不由离线路径或 CI 重跑。真实 GPU 行为在复算口径里仍然是 UNKNOWN。
> 下面的比值是「归档毫秒 / 当前模型预测」，不是一次新的测量。

这份文档存在的理由：本仓库其余部分的数字都是 **CPU 上可复算的机制演示**。
这里单独放一份旧的 GPU 日志，避免把它和 CPU 计数写成同一类结论。

## 证据

| 项 | 值 |
|---|---|
| 测量时间 | 2026-10-31（America/Los_Angeles） |
| 平台 | Kaggle Notebook GPU kernel，私有，Tesla T4 |
| 内核地址 | https://www.kaggle.com/code/liuzewen123/gkern-triton-softmax-bandwidth |
| 采集脚本（仓库内） | `benchmarks/triton/bench_softmax_triton.py` |
| 原始落盘 | `benchmarks/results/triton_softmax_t4.json`（含 `provenance` 字段） |
| 对账脚本 | `benchmarks/analyze_triton.py --write-report` |
| 环境 | torch 2.11.0+cu128、triton 3.6.0、CUDA 12.8、Python 3.13.15 |
| 图形 | Tesla T4，40 SM，可用显存 14.56 GB |
| 计量方式 | `triton.testing.do_bench(warmup=25, rep=100)`，取中位数 |

数值正确性同时被钉住：五档 shape 的 `max|diff|` 全部 ≤ 1.5e-08（与 `torch.softmax` 对拍）。

## 对账表（由脚本生成，不要手改）

| cols | 字节 | 模型预测 (ms) | Triton 实测 (ms) | eager 实测 (ms) | 实测/预测 | 效率 | 受限 |
|---|---:|---:|---:|---:|---:|---:|---|
| 512 | 16777216 | 0.05243 | 0.10603 | 0.07856 | 2.022 | 0.494 | memory |
| 1024 | 33554432 | 0.10486 | 0.14212 | 0.15622 | 1.355 | 0.738 | memory |
| 2048 | 67108864 | 0.20972 | 0.2926 | 0.31683 | 1.395 | 0.717 | memory |
| 4096 | 134217728 | 0.41943 | 0.58654 | 0.55779 | 1.398 | 0.715 | memory |
| 8192 | 268435456 | 0.83886 | 1.17603 | 1.13988 | 1.402 | 0.713 | memory |

设备预设 `DeviceInfo.tesla_t4()`：理论带宽 320.0 GB/s、FP32 峰值 8.141 TFLOPS
（与 T4 公开的 8.1 TFLOPS 相符）、roofline 交叉点 25.44 FLOP/B。

## 三条结论，每条都带「模型错在哪」

**① 代价模型是**严格下界**，实测/预测 = 1.36～1.40，即 71%～74% 的理论带宽。**
cols ≥ 1024 时比值稳定在 1.35～1.40，说明这不是噪声，而是系统性缺口：
模型不含 DRAM 刷写、不含 L2 未命中、不含指令发射与 warp 调度开销。
把 `predicted_ms` 当成「能跑到的时间」是这类模型最典型的误用；
它真正的用途是**判断受限类型与相对排序**（五档全部判为 `memory`，与实测一致）。

**② 小 shape 掉到 49%：模型完全看不见 launch 与占用率。**
cols=512 时 4096 行只有 4096 个 program、每 program 1 个 warp，
T4 的 40 个 SM 上驻留的 warp 数远不足以掩盖启动延迟；
模型照样给 0.0524 ms，实测 0.106 ms。这个 2.02 倍偏差不是「模型不准」，
而是**模型的问题定义里就没有这一项**——它在 `docs/ARCHITECTURE.md` §8.2 的缺口清单里。

**③ 不舒服的那条：手写 Triton fused softmax 并没有稳定赢过 `torch.softmax`。**

| cols | Triton 相对 eager 的加速比 |
|---:|---:|
| 512 | 0.741（慢 26%） |
| 1024 | 1.099 |
| 2048 | 1.083 |
| 4096 | 0.951 |
| 8192 | 0.969 |

CPU 侧的字节账（融合省掉中间张量的写回+读出）是真的，但它**推不出「手写 kernel 更快」**：
PyTorch 的 eager 行 softmax 本身已经是单遍、带宽受限的实现，没有可以省掉的中间张量；
真正决定胜负的是映射方式（本脚本每行一个 program、8 个 warp），
它在窄行时占用率不足，在宽行时又被 T4 的带宽天花板卡住。
所以「融合」在这里买到的是**不再有第二遍访存**，而不是**更快的墙钟**——
这两件事在本仓库其他地方被刻意分开写，这里给出硬件侧的实证。

## 这份对账**不能**用来说明什么

- 不能说明 `gkern` 的 CPU 教学模型「准」：它只是下界，且已知缺口五项（见 ARCHITECTURE §8.2）。
- 不能代表国产加速卡：T4 的交叉点是 25.44 FLOP/B，假设国产卡的预设是 256.0，
  相差一个数量级，同一 kernel 在两边的受限类型可能相反。
- 不能当简历上的 GFLOPS 实绩：这里唯一的算力数字是**理论峰值**，实测的是带宽（GB/s），
  且只有 T4 一个平台。

## 还没量的一格：bf16 与 TPU

CPU 侧能算的是：bf16 舍入与 fp32 的访存字节相同，因为存储仍是 float32
（`src/gkern/ops/mixed_precision.py`）。下面两件事是 UNKNOWN：

1. 真实 kernel 上四处舍入的误差是不是仍是 1e-2 量级（CPU 上这一输入是 5.87e-03）；
2. bf16 **存储**（不是只舍入累加器）改不改搬运的字节。

`benchmarks/results/` 里没有 bf16 GPU JSON，也没有 TPU JSON。
下面两个脚本在没有对应设备时退出，并且不写文件：

- `benchmarks/triton/bench_bf16_softmax.py`（需要 CUDA）
- `benchmarks/tpu/bench_bf16_softmax_jax.py`（需要 JAX 报告 TPU；CPU 上的 JAX 也退出）

在有结果文件之前，不要写「bf16 在 GPU 上快多少」。
