# 云端实测 vs 代价模型：T4 上的行 softmax 对账

这份文档存在的理由：本仓库其余部分的数字都是 **CPU 上可复算的机制演示**，
唯独这里出现**真实 GPU 的实测值**。两者混用是这类项目最常见的自欺方式，
所以把它单独拎出来，写清「证据在哪、模型预测多少、差在哪、为什么差」。

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

## 还没量的一格：bf16 累加在真实 GPU 上的误差与带宽

CPU 侧已经能算准两件事：`bf16acc` 与 `fp32` 的**访存字节完全相同**、FLOP 只多四次舍入
（`src/gkern/ops/mixed_precision.py`，数字见 README）。但算不出两件只有硬件能回答的事：

1. 真实 kernel 上四处舍入带来的误差是不是仍是 1e-2 量级（CPU 上是 5.87e-03）；
2. **bf16 存储**（不是只舍入累加器）到底省不省带宽——这才是真正可能提速的那一档。

量这两件事的脚本已经写好并提交：`benchmarks/triton/bench_bf16_softmax.py`。它跑三种变体
（`fp32` / `bf16acc` / `bf16io`）+ torch eager 基线，输出 ms、GB/s、相对 float64 参考的
最大/平均误差，无 CUDA 时直接退出（不伪造数字）。

**为什么现在还没有数字**：Kaggle 的每周 GPU 额度已用尽。实测
（`python3 -m kaggle quota`，2026-11-01）：

| 资源 | 已用 | 剩余 | 总额 | 恢复时间 |
|---|---:|---:|---:|---|
| GPU | 32.75h | **0.00h** | 30.00h | 2026-11-07T00:00:00 |
| TPU | 0.00h | 20.00h | 20.00h | 2026-11-07T00:00:00 |

`kaggle kernels push` 被平台直接拒绝：`Maximum weekly GPU quota of 30.00 hours reached.`

所以这一格是**明确的空白**，不是被省略的结论。额度恢复后一条命令即可补齐：

```bash
# 把 benchmarks/triton/bench_bf16_softmax.py 与一个调用它的 kernel_main.py 一起 push
python3 -m kaggle kernels push      # T4 / GPU 模式，--out /kaggle/working/bf16_softmax.json
python3 -m kaggle kernels output liuzewen123/gkern-bf16-softmax-error -p .
# 然后把 bf16_softmax.json 原样提交进 benchmarks/results/，并在本文件补一节对账
```

在此之前，任何关于「bf16 在 GPU 上快多少」的说法都不该出现在这个仓库里——那正是
`docs/cloud-reconciliation.md` 存在的意义：把「模型说的」和「量到的」分开写。
