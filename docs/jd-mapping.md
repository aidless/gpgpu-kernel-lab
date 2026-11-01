# 岗位 JD ↔ 本仓库证据对照

左边是招聘要求，右边是「在本仓库的哪一处能当场验证」。**status 一列是诚实标注**：
`覆盖` = 有可运行代码与可复算数字；`部分` = 只有模型/小规模，缺真实规模；`缺口` = 没有。

## 硬性任职资格

| JD 要求 | 本仓库证据（可运行） | status |
|---|---|---|
| 熟悉 Linux 开发环境 | 全部开发与 89 个测试在 Linux/CPU 上跑通；`benchmarks/run_benchmarks.py` 产出机器可读 JSON | 覆盖 |
| 掌握 C++ / Python | Python 全覆盖（`src/gkern/**`）；**C++ 无代码** | 部分 |
| 熟悉 CUDA 编程模型及 GPU 执行与内存体系 | `gpgpu/warp.py` 掩码与分支发散、`gpgpu/memory.py` 128B 事务计数与 32 bank 冲突、`gpgpu/device.py` SM/寄存器/占用率、`compiler/costmodel.py` roofline 受限判断 | 覆盖（**概念与计数层**，非 CUDA 源码） |
| 熟悉至少一种主流 AI 框架 | PyTorch：`torch.library` 自定义算子（实现/fake/autograd/schema）`+ torch.compile` 自定义后端 `+ FX` 图下沉；Triton 到源码级阅读（见 `docs/ecosystem.md` A 节） | 覆盖 |
| LLM/VLM/搜广推 任一领域训练或推理优化经验 | Transformer 推理最核心的几类 kernel：`softmax_rows`、`layernorm_rows`、`fused_mlp`（matmul + bias + gelu 融合）、roofline 判断 memory/compute 受限 | 部分（**算子级**，非端到端模型，无训练） |

## 岗位职责方向

| JD 职责 | 本仓库证据 | status |
|---|---|---|
| 框架组件在国产 GPGPU 上的适配、开发与系统级优化 | `torch_ext/` 三层（自定义算子 / compile 后端 / 预留 PrivateUse1）；所有「接不住」的图都给出回退原因 | 部分（CPU 模拟，**无真卡**） |
| 多领域模型端到端性能分析及优化 | `benchmarks/` 给出 shape 扫描 × 墙钟 × 模型预测 × 受限类型；融合省访存等式 | 部分（算子/小图级） |
| 兼容能力开发及高性能组件（如向量数据库）研发 | 无 | 缺口 |
| 跟踪 AI 框架及场景前沿并落地 | `docs/ecosystem.md`：全文 51 行 / 47 个唯一仓库（正表 30 行 / 28 唯一 + 附表 21 行）逐个用 GitHub API 验真 stars/pushed_at/language/archived，含源码入口级读点与四周任务卡 | 覆盖 |

## 加分项

| 加分项 | 本仓库证据 | status |
|---|---|---|
| a. CPU/GPU 算子优化、并行计算背景 | `ops/matmul.py` 分块 GEMM + `autotune` 枚举；合并访存 32× 事务差、发散 lane 效率 0.667 的实测；`SGEMM_CUDA` 优化阶梯（1→12 级）阅读路径 | 覆盖（模型级） |
| b. TVM/MLIR/XLA/Triton 编译或图算优化 | `compiler/` 自建最小编译栈：DSL → IR → pass（融合/分块）→ codegen + 代价模型；`docs/ecosystem.md` C 节给出 MLIR Linalg / TVM `s_tir` / IREE codegen 的**当前有效源码路径**（TVM 2026 已重构，旧教程路径 404） | 覆盖 |
| c. 国产加速卡经验 | 无（本机无 GPU，Triton 无 CPU driver，远程主机无卡） | 缺口（硬缺口） |
| d. 异构集群经验 | 无 | 缺口 |
| e. 大模型优化经验 | 融合/roofline/attention 的算子级能力 + `docs/ecosystem.md` 的 `vllm/platforms`、`vllm-ascend/attention` 阅读清单 | 部分 |

## 三个必须在面试里承认的缺口，以及怎么补

1. **真实国产卡上的实测**（jd 加分 c）。本仓库所有设备数字都是「假设机」的解析值。
   补法：拿到卡（AMD Radeon Cloud 中国站 / 昇腾云 / 寒武纪云任一）后，把
   `gpgpu/device.py` 的 `DeviceInfo` 换成真卡参数，把 `torch_ext/custom_ops.py` 的
   numpy 实现换成真 kernel 调用，用同一套测试对拍。**代码结构已为此预留，不需要重写。**
2. **C++ / CUDA 源码**。补法：读透 `SGEMM_CUDA` 的 12 级优化阶梯并用 `ncu` 跑出每级的
   GFLOPS 曲线（`docs/ecosystem.md` A 节给了逐文件路径），把「事务放大 32×」「bank 冲突度」
   这些在本仓库里被**建模**的量在真卡上**实测**出来对照。
3. **端到端模型 / 异构集群**。补法：把 `fused_mlp` 扩到一层 Transformer block，
   再用 `vllm/platforms/interface.py` 的抽象去理解「一套代码接多卡」的工程边界。

**一句话定位**：本仓库证明的是「我理解 GPU 为什么慢、也理解编译器怎么把它变快，并且
能把这份理解写成可运行、可复算、可测的代码」；它不证明「我在某张国产卡上调过 kernel」。
后者需要算力，前者不需要——而前者恰恰是多数候选人的薄弱处。
