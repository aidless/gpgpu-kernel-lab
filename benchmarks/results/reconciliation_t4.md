# 云端实测 vs 代价模型（Tesla T4）

证据文件：`benchmarks/results/triton_softmax_t4.json`（2026-10-31 (America/Los_Angeles)，https://www.kaggle.com/code/liuzewen123/gkern-triton-softmax-bandwidth）。
本表由 `benchmarks/analyze_triton.py --write-report` 生成，不要手改。

| cols | 字节 | 模型预测 (ms) | Triton 实测 (ms) | eager 实测 (ms) | 实测/预测 | 效率 | 受限 |
|---|---:|---:|---:|---:|---:|---:|---|
| 512 | 16777216 | 0.05243 | 0.10603 | 0.07856 | 2.022 | 0.494 | memory |
| 1024 | 33554432 | 0.10486 | 0.14212 | 0.15622 | 1.355 | 0.738 | memory |
| 2048 | 67108864 | 0.20972 | 0.2926 | 0.31683 | 1.395 | 0.717 | memory |
| 4096 | 134217728 | 0.41943 | 0.58654 | 0.55779 | 1.398 | 0.715 | memory |
| 8192 | 268435456 | 0.83886 | 1.17603 | 1.13988 | 1.402 | 0.713 | memory |

设备预设 `DeviceInfo.tesla_t4()`：理论带宽 320.0 GB/s、FP32 峰值 8.141 TFLOPS、roofline 交叉点 25.44 FLOP/B。
