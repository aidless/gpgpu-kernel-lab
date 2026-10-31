# 生态地图：国产 GPGPU 上的框架适配与算子优化

> 数据来源：GitHub REST API（`/repos/{owner}/{repo}`、`/orgs/{org}/repos`、`/repos/{o}/{r}/contents/{path}`、`/search/repositories`），取样时间 2026-10-31。stars 为当时快照。每个仓库的 `stargazers_count` / `pushed_at` / `language` / `archived` 均逐条取自 API 返回 JSON，未取到的项目已剔除（剔除清单见 F 节）。
> 本仓库面向的岗位：国产 GPGPU 上 AI 框架适配 / 算子与系统级性能优化。

## 0. 一句话结论

先补「能不能把 kernel 写快」这一层（A 类，CUDA/Triton 手感 + ncu 读数能力），再补「能不能把框架接到新卡上」这一层（B 类，PrivateUse1 一套接口 + vLLM 平台抽象），最后把这两层用编译器串起来（C 类，图算融合与 codegen），D 类厂商仓库只作为「真实适配代码长什么样」的参照物、不作为学习主线——因为它们的实际可运行性依赖厂商 SDK（CANN/BANG/MUSA），本机无法验证，只能当源码读物。

具体顺序：**A 的 triton `python/tutorials/06-fused-attention.py` → B 的 `torch/utils/backend_registration.py` → C 的 `vllm/platforms/interface.py` + `iree/compiler/.../Codegen` → D 的 `Ascend/triton-ascend` 与 `Cambricon/mlu-ops`**。前三个是「能迁移到任何卡」的通用能力，D 类是「迁移到国产卡」时才会用到的落地证据。

---

## A. 算子 / 内核

| 仓库 | stars | 最近 push | 语言 | 一句话定位 | 对应岗位要求 |
|---|---|---|---|---|---|
| [triton-lang/triton](https://github.com/triton-lang/triton) | 20498 | 2026-10-31 | MLIR | Triton 语言与编译器官方仓库，Python DSL → MLIR → GPU codegen 的完整参考实现 | 算子优化 / 编译优化 |
| [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | 9732 | 2026-09-09 | C++ | CUDA 官方示例集，覆盖访存、矩阵乘、性能测量到新 Tile 编程模型 | 算子优化 |
| [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA) | 1351 | 2025-09-02 | Cuda | 从朴素版本逐级优化 SGEMM 到接近 cuBLAS，每级 kernel 带 GFLOPS 曲线 | 算子优化 |
| [AlphaGPU/leetgpu-challenges](https://github.com/AlphaGPU/leetgpu-challenges) | 1227 | 2026-10-30 | Python | LeetGPU 官方题库，按难度分级的单算子 CUDA/Triton 练习 | 算子优化 |
| [ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench) | 1306 | 2026-03-24 | Jupyter Notebook | 「LLM 能不能写 GPU kernel」的评测集 + Torch→CUDA 工具链 | 算子优化 / LLM 推理 |
| [gpu-mode/Triton-Puzzles](https://github.com/gpu-mode/Triton-Puzzles) | 2655 | 2026-04-01 | Jupyter Notebook | Triton 渐进式练习 notebook，从 broadcast 一路做到 attention | 算子优化 |
| [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | 10681 | 2026-10-28 | C++ | 高性能 GEMM 模板库，CuTe layout 代数 + CUTLASS mainloop/epilogue 设计 | 算子优化 |
| [feifeibear/LLMRoofline](https://github.com/feifeibear/LLMRoofline) | 126 | 2024-03-13 | Jupyter Notebook | 用 Roofline 模型跨硬件对比 LLM 推理的理论上限 | 算子优化 / LLM 推理 |

**读它学什么**

- **triton**：`python/tutorials/` 下 14 个文件本身就是一条难度梯度——`01-vector-add.py`（最简 launch 模型）→ `02-fused-softmax.py`（reduce + 分块）→ `03-matrix-multiplication.py`（`tl.dot` + block tiling）→ `06-fused-attention.py`（online softmax）→ `09-persistent-matmul.py`（persistent 调度）。想搞清楚「为什么改一个 block size 就慢一半」，去读 `lib/Dialect/TritonGPU/IR` 和 `lib/Dialect/TritonGPU/Transforms` 里的 layout 与 `convert_layout`——swizzling 决定了 shared memory bank conflict。算子语义定义在 `python/triton/language/core.py`（`tl.load` / `tl.store` / `tl.dot` 的约束都写在这里）。
- **cuda-samples**：`cpp/0_Introduction`（vectorAdd、matrixMul，对照 CUDA 编程模型）→ `cpp/6_Performance`（带宽与矩阵乘的性能对照）→ `cpp/9_CUDA_Tile`（新的 Tile 级编程模型，和 Triton 的 block 抽象是同一思路）。公共头文件在 `Common/`，编译后用 `ncu` 跑一遍再谈优化。
- **SGEMM_CUDA**：`src/kernels/` 下 12 个文件本身就是优化阶梯，按编号读：`1_naive.cuh` → `2_kernel_global_mem_coalesce.cuh`（合并访存）→ `3_kernel_shared_mem_blocking.cuh`（shared memory 分块）→ `4_kernel_1D_blocktiling.cuh` → `5_kernel_2D_blocktiling.cuh` → `6_kernel_vectorize.cuh`（向量化）→ `7_kernel_resolve_bank_conflicts.cuh` 与 `8_kernel_bank_extra_col.cuh`（bank conflict 两种解法）→ `9_kernel_autotuned.cuh` → `10_kernel_warptiling.cuh` → `11/12_kernel_double_buffering.cuh`。根目录 `simplest_kernel.cu` 是最小可跑版本，`cuBLAS_sgemm.cu` 是性能上限基准，每级用 `benchmark_results/` 的曲线自证提升幅度。这是「优化要有数据支撑」最便宜的范本。
- **leetgpu-challenges**：`challenges/` 下分 `easy` / `medium` / `hard` / `core` 四档，`scripts/` 是评测脚本。适合每天 1-2 题保持手感，难度可控、反馈即时。
- **KernelBench**：`KernelBench/level1`（单算子）→ `level2`（算子融合）→ `level3`（完整模型结构）→ `level4`（Level 3 + 性能剖析），评测 harness 在 `src/kernelbench`。读 level2 能看清「融合」到底省掉了哪次 HBM 往返。
- **Triton-Puzzles**：整仓只有 `Triton-Puzzles.ipynb` 一个 notebook，12 个 puzzle 各自带测试。适合做完 triton 官方 tutorial 后自测，卡住的 puzzle 就是知识盲区。
- **cutlass**：`include/cute`（layout algebra：`Layout`、`Swizzle`、`copy` 原子）与 `include/cutlass`（`gemm` mainloop、epilogue 的 policy 设计）。国产卡适配算子时，很多 mlu-ops / TileOPs 的实现思路与之同源，读了才有对照。
- **LLMRoofline**：`perf_model.py` 是解析式 roofline 计算（算力/带宽两条上限），`hardware_dict.py` 与 `model_dict.py` 是硬件参数表与模型参数表——你可以直接改这两个 dict 把目标国产卡加进去，`roofline.ipynb` 出图。这是把「理论天花板」量化的最小工具。

---

## B. 框架适配层

| 仓库 | stars | 最近 push | 语言 | 一句话定位 | 对应岗位要求 |
|---|---|---|---|---|---|
| [pytorch/pytorch](https://github.com/pytorch/pytorch) | 105030 | 2026-10-31 | Python | PrivateUse1 / out-of-tree backend 机制的定义方（`torch/utils/backend_registration.py`） | 框架适配 / 编译优化 |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | 95971 | 2026-10-31 | Python | 多硬件推理引擎，`vllm/platforms/` 是「一套代码接多卡」的抽象样板 | 框架适配 / LLM 推理 / 异构集群 |
| [Ascend/pytorch](https://github.com/Ascend/pytorch) | 598 | 2026-10-31 | Python | torch_npu：昇腾 NPU 的 PyTorch 适配插件，国产卡适配最完整的公开样本 | 框架适配 / 算子优化 |
| [MooreThreads/torch_musa](https://github.com/MooreThreads/torch_musa) | 519 | 2026-09-23 | Python | 摩尔线程 MUSA 的 PyTorch 适配，骨架与 torch_npu 同构，适合横向对比 | 框架适配 |
| [pytorch/xla](https://github.com/pytorch/xla) | 2806 | 2026-10-20 | C++ | PyTorch on XLA：整图 lazy 执行路线，与逐 op 下沉路线形成对照 | 框架适配 / 编译优化 |
| [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) | 3038 | 2026-10-31 | Python | 昇腾上的 vLLM 硬件插件，attention 后端实现最值得读 | 框架适配 / LLM 推理 |
| [Cambricon/torch_mlu](https://github.com/Cambricon/torch_mlu) | 61 | 2025-03-15 | Python | 寒武纪 MLU 的 PyTorch 适配；默认分支停在 `r1.22_pt2.1.0`，读时注意版本错位 | 框架适配 |
| [EnflameTechnology/torch-gcu](https://github.com/EnflameTechnology/torch-gcu) | 5 | 2026-06-23 | C++ | 燧原 GCU 的 PyTorch 适配，公开内容以 `torch_gcu/csrc` 为主 | 框架适配 |

**读它学什么**

- **pytorch**：只读一个文件就够入门——`torch/utils/backend_registration.py`（约 24 KB），里面是 `rename_privateuse1_backend`、`generate_methods_for_privateuse1_backend`、`get_privateuse1_backend_name` 这套 API 的全部语义，也就是「接一张新卡要实现哪些接口」的清单来源。之后看 `torch/_inductor`（96 个条目）了解后端如何注册自己的 codegen；`torch/npu` 目录在 2026-10-31 的 main 上不存在（404），不要按旧博客去找。
- **vllm**：`vllm/platforms/` 只有 8 个文件（`interface.py`、`cuda.py`、`rocm.py`、`cpu.py`、`tpu.py`、`xpu.py`、`zen_cpu.py`、`__init__.py`）——先读 `interface.py` 的抽象方法，再看 `cuda.py` 怎么落地，最后看 `__init__.py` 的惰性导入分发。这是全仓库性价比最高的 8 个文件。
- **Ascend/pytorch**：`torch_npu/csrc/InitNpuBindings.cpp` 是绑定入口，往下 `torch_npu/csrc/aten`（算子注册）、`torch_npu/csrc/core`（device/stream 管理）、`torch_npu/csrc/distributed`、`torch_npu/csrc/profiler`；编译后端在 `torch_npu/_inductor` 与 `torch_npu/csrc/inductor`。把这条链与自己手写的 PrivateUse1 对照，能看出「最小可用」与「生产可用」差多少。
- **torch_musa**：`torch_musa/csrc/{aten,core,inductor,distributed,profiler}` 与 torch_npu 一一对应，`torch_musa/_inductor` 同样存在。两家国产卡在同一套 PyTorch 扩展点上的实现差异，就是最好的比较样本。
- **pytorch/xla**：`torch_xla/csrc` 是 lazy tensor 与图构建的核心，`torch_xla/_dynamo`、`torch_xla/_internal` 接 torch.compile。读它是为了判断一个问题：新卡的图优化应该做在框架层（懒执行）还是编译器层（Inductor 后端）。
- **vllm-ascend**：`vllm_ascend/attention/` 有 19 个文件，`mla_v1.py` / `dsa_v1.py` / `fa_v1.py` / `flashmla.py` 各自对应一种 attention 后端，是「同一算子在不同卡上怎么写」的最全集合；`vllm_ascend/_cann_ops_custom` 是自定义 CANN 算子，`vllm_ascend/compilation` 是图模式接入。
- **torch_mlu / torch-gcu**：两者公开内容都比 torch_npu 薄（torch-gcu 只有 `torch_gcu/csrc` 等 7 项，star 5）。读它们不是为了学完整链路，而是验证「同一套私有后端接口，小团队能做到什么程度」。

---

## C. 编译与图优化

| 仓库 | stars | 最近 push | 语言 | 一句话定位 | 对应岗位要求 |
|---|---|---|---|---|---|
| [llvm/llvm-project](https://github.com/llvm/llvm-project) | 41274 | 2026-10-31 | LLVM | MLIR 本体；Linalg dialect 的 tiling/fusion/vectorization 是图算融合的公共底座 | 编译优化 |
| [apache/tvm](https://github.com/apache/tvm) | 13885 | 2026-10-30 | Python | 计算图 → TE/TIR → 目标代码的完整编译栈，含 autoTVM/MetaSchedule 自动调优 | 编译优化 / 算子优化 |
| [tile-ai/tilelang](https://github.com/tile-ai/tilelang) | 8865 | 2026-10-30 | Python | 面向 tile 的 kernel DSL，已在做多后端（含 ascend 目录） | 编译优化 / 算子优化 |
| [iree-org/iree](https://github.com/iree-org/iree) | 4001 | 2026-10-31 | C++ | 基于 MLIR 的可重定向编译+运行时，Codegen 目录按后端切分，最适合学「加一个新 target」 | 编译优化 / 框架适配 |
| [llvm/torch-mlir](https://github.com/llvm/torch-mlir) | 1938 | 2026-10-30 | C++ | PyTorch 图 → MLIR（Torch dialect → Linalg/StableHLO），框架与编译器之间的桥 | 编译优化 / 框架适配 |
| [Cambricon/triton-linalg](https://github.com/Cambricon/triton-linalg) | 230 | 2025-02-07 | C++ | Triton → Linalg 的转换实现，国产厂商在 MLIR 上做 Triton 下沉的公开样本 | 编译优化 |

**读它学什么**

- **llvm/llvm-project**：`mlir/lib/Dialect/Linalg/Transforms`（tiling、fusion、vectorization 的实现）、`mlir/lib/Dialect/Linalg/IR`（`linalg.generic` 的 op 定义）。只读 Linalg 这一个 dialect 就能覆盖「图算融合」80% 的概念，不要从 MLIR 全量开始。
- **apache/tvm**：**2026 主线已重构，旧路径失效**——`python/tvm/tir` 与 `src/target` 在 2026-10-31 的 main 上均返回 404，TIR 已独立为 `s_tir`。正确的读法：`python/tvm/s_tir`（Python 侧）、`src/s_tir/{transform,schedule,meta_schedule}`（调度与自动调优）、`src/relax`（图级 IR）、`src/s_tir/backend`。按网上 2023-2024 的 TVM 教程读会找不到文件。
- **tilelang**：`examples/` 有 58 个目录，`examples/ascend/` 已包含 13 个文件（`example_gemm.py`、`example_rmsnorm.py`、`flash_attention/` 等）——说明 tilelang 已经把昇腾作为一等后端，这是「一个 DSL 同时覆盖 CUDA 与国产 NPU」的现成案例。C++ 侧在 `src/{cuda,ascend,rocm,cpu,op,layout}`。
- **iree**：`compiler/src/iree/compiler/Codegen` 下按后端切分为 `Common` / `LLVMGPU` / `LLVMCPU` / `SPIRV` / `VMVX`。想知道「给一张新卡加 codegen 要动哪些文件」，读 `Common`（公共 pass pipeline）+ 任意一个后端目录（具体 lowering）即可，结构非常干净。
- **torch-mlir**：`lib/Conversion` 下 12 个 conversion，核心是 `TorchToLinalg`（算子下沉到 Linalg）与 `TorchToStablehlo`（走另一条生态）。读它理解「框架算子集」如何映射到「编译器算子集」，这是适配国产卡时最先崩的地方。
- **triton-linalg**：`lib/`（转换实现）、`include/`、`backend/`、`test/`。注意它最后 push 是 2025-02-07，已一年多未更新，只能当方法论参考，不能当上游依赖。

---

## D. 国产 GPGPU 生态

先说结论：**在 GitHub 上公开仓库最多、层次最全的是昇腾（98 个公开仓库，覆盖 PyTorch 适配、Triton 后端、MLIR 编译、profiling 四层）；寒武纪、摩尔线程、沐曦、天数智芯、燧原各能挑出 1-2 个可用仓库；壁仞只有归档的周边工具，没有算子或框架适配代码。** 需要注意的是，多家仓库的 description 自述为 GitCode 镜像（`Ascend/triton-ascend`、`Ascend/AscendNPU-IR` 均写明 mirror of gitcode.com），所以「GitHub 上的公开度」低于厂商的实际投入。

组织级公开仓库数（`/orgs/{org}` 的 `public_repos`，2026-10-31）：Ascend 98、MooreThreads 193、MetaX-MACA 66、EnflameTechnology 33、Deep-Spark 27、Cambricon 24、BirenTechnology 3、Iluvatar-CoreX 3。

| 厂商 | 仓库 | stars | 最近 push | 语言 | 一句话定位 | 对应岗位要求 |
|---|---|---|---|---|---|---|
| 昇腾 | [Ascend/pytorch](https://github.com/Ascend/pytorch) | 598 | 2026-10-31 | Python | torch_npu，昇腾 PyTorch 适配主体（与 B 节同一仓库，此处按厂商视角列） | 框架适配 / 算子优化 |
| 寒武纪 | [Cambricon/mlu-ops](https://github.com/Cambricon/mlu-ops) | 183 | 2026-10-30 | C++ | MLU 上的高效算子实现库，BANG C 写的 54 组算子 | 算子优化 |
| 摩尔线程 | [MooreThreads/torch_musa](https://github.com/MooreThreads/torch_musa) | 519 | 2026-09-23 | Python | MUSA 的 PyTorch 适配（与 B 节同一仓库） | 框架适配 |
| 沐曦 | [MetaX-MACA/TileOPs-Metax](https://github.com/MetaX-MACA/TileOPs-Metax) | 7 | 2026-10-31 | Python | 基于 TileLang 的高性能 LLM 算子库，`src/tileops` 下按 gemm/moe/attention 分模块 | 算子优化 / LLM 推理 |
| 天数智芯 | [Deep-Spark/iluvatar-corex-ixrt](https://github.com/Deep-Spark/iluvatar-corex-ixrt) | 23 | 2026-10-16 | Python | IxRT 推理运行时插件与部署工具，`ixrt/`、`samples/` 可直接对照 | 框架适配 / LLM 推理 |
| 天数智芯 | [Deep-Spark/vllm-iluvatar](https://github.com/Deep-Spark/vllm-iluvatar) | 11 | 2026-10-16 | Python | 天数 GPU 的 vLLM 硬件插件（自述 community-maintained），`csrc/` 是胶水层 | 框架适配 / LLM 推理 |
| 燧原 | [EnflameTechnology/vllm-gcu](https://github.com/EnflameTechnology/vllm-gcu) | 40 | 2026-07-07 | Python | 燧原 GCU 上的 vLLM 适配，`csrc/` + `benchmarks/` | 框架适配 / LLM 推理 |
| 壁仞 | [BirenTechnology/k8s-device-plugin](https://github.com/BirenTechnology/k8s-device-plugin) | 11 | 2024-10-17 | Go | 壁仞 GPU 的 K8s device plugin，**已 archived** | 异构集群 |

**读它学什么 / 落地提示**

- **昇腾**：表内只列 `Ascend/pytorch` 作为入口，读法与 B 节一致（`torch_npu/csrc/InitNpuBindings.cpp` → `torch_npu/csrc/aten`）。昇腾在 GitHub 上还有三个更专门的仓库未入表：`Ascend/triton-ascend`（130 stars，MLIR，Triton 的昇腾后端）、`Ascend/AscendNPU-IR`（61 stars，C++，昇腾的 MLIR 中间表示）、`Ascend/msprof`（3 stars，profiling 工具，对标 ncu），数据见 F.4。注意 `Ascend/msprof` 只有 3 stars 但持续更新到 2026-10-31，star 数在国产项目里不代表质量。
- **寒武纪 BANG**：`Cambricon/mlu-ops/kernels/` 下有 54 个算子目录（`abs`、`adam_w`、`ball_query`、`deform_conv`、`box_iou_rotated`……），头部是 `bangc_kernels.h` 与 `bangc_helper_dtype.h`——这就是 BANG C 算子实现的直接样本。想学「非 CUDA 的 SIMD 式加速器怎么写算子」，从这里读比读任何文档都快。
- **沐曦**：`MetaX-MACA/TileOPs-Metax/src/tileops/` 已按 `gemm.py` / `moe.py` / `attention.py` / `mamba.py` / `kernels/` 分模块，是「用 tile DSL 而非手写汇编做算子库」的最新实践（用 TileLang，和 C 节的 tilelang 同一技术栈）。
- **天数智芯**：`Deep-Spark` 是实际在维护的组织（27 个仓库、持续 push），但 `vllm-iluvatar` 自述 community-maintained、只有 11 stars，属社区/小团队体量；`iluvatar-corex-ixrt` 的 `ixrt/` + `include/` + `samples/` 是三个里最适合当「推理运行时怎么接 PyTorch」来读的。
- **壁仞 = 公开仓库稀少，证据如下**：`users/birentech` 返回 User 且 `public_repos=0`；正确组织是 `BirenTechnology`，但只有 3 个公开仓库，最近 push 是 2024-12-17，其中 2 个已 `archived`。用 `/search/repositories?q=birentech+in:name,description` 只返回 2 条结果，均为第三方 profile 仓库（`api-evangelist/birentech`、`api-evangelist/birentechnology`），无官方项目；用中文 `壁仞` 搜索返回 7 条，全部是无关的个人仓库（书籍列表、比赛页面等）。结论：壁仞在 GitHub 上没有可用的公开算子/框架适配代码。
- **天数智芯的组织名坑**：`IluvatarAI` 组织**不存在（404）**。实际公开的是 `Deep-Spark`（活跃，见上）与 `Iluvatar-CoreX`（3 个仓库，全部最后 push 2024-01-04，0 stars，内容是 SHARK-Turbine / iree / cutlass 的镜像）。
- **CANN 不在 GitHub**：`/orgs/Ascend/repos?per_page=100` 共 98 个仓库，逐个检查后没有任何名为 `cann` 的仓库，只有 `cann_container-image`、`canncamp`、`cann_op_contrib`（0 star，最后 push 2023-09-02）这类周边。CANN 主体需去 GitCode 获取，GitHub 上无法验证其代码。

---

## E. 优先读的源码入口，以及本项目能接住什么

不排周计划。下面每行都是「今天就能打开的那个文件」+「第一小时的具体动作」+「落在 `gkern` 的哪个模块上」——`gkern` 是本仓库（`src/gkern/{ops,gpgpu,compiler,torch_ext}`，纯 CPU 可跑），所以每个入口都有可验收的落点。

| 能力 | 先打开这一个源码入口 | 第一小时做什么 | 接到 gkern 的哪里 |
|---|---|---|---|
| 算子手感 | `triton-lang/triton` → `python/tutorials/02-fused-softmax.py` | 本地跑通，把 `BLOCK_SIZE` 从 1024 遍历到 8192，记耗时曲线并解释拐点 | `src/gkern/ops/softmax.py`：把 Triton 的分块规则对齐到现有 CPU 实现 |
| 优化阶梯 | `siboehm/SGEMM_CUDA` → `src/kernels/1_naive.cuh` 至 `5_kernel_2D_blocktiling.cuh` | 逐级读懂「每级多了什么」，抄下每级的分块/向量化参数 | `src/gkern/ops/matmul.py` + `src/gkern/gpgpu/memory.py`：把 shared-mem 分块与 bank conflict 写成 cost 项 |
| 读数能力 | `NVIDIA/cuda-samples` → `cpp/6_Performance` | 跑 bandwidthTest 与 matrixMul，用 ncu 取 DRAM throughput、achieved occupancy | `src/gkern/compiler/costmodel.py`：把实测带宽代入，量出它的预测误差 |
| 理论上限 | `feifeibear/LLMRoofline` → `perf_model.py`、`hardware_dict.py` | 在 `hardware_dict.py` 加一张目标卡参数，算 7B decode 的 roofline 上限 | `src/gkern/compiler/costmodel.py`：roofline 当 baseline，单测断言预测误差 |
| 框架适配 | `pytorch/pytorch` → `torch/utils/backend_registration.py` | 抄出「PrivateUse1 必须实现的接口」清单并数清项数（不许抄 README） | `src/gkern/torch_ext/backend.py`：注册一个 gkern 假后端跑通最小图 |
| 推理引擎适配 | `vllm-project/vllm` → `vllm/platforms/interface.py`、`cuda.py` | 画出 `platforms/__init__.py` → `cuda.py` 的惰性导入调用链 | 暂不接（依赖重），只作 D 节对照阅读 |
| 编译图谱 | `llvm/llvm-project` → `mlir/lib/Dialect/Linalg/Transforms` | 读 tiling、fusion 两个 pass 的接口，写 200 字机制总结 | `src/gkern/compiler/passes.py`：实现一个 elementwise+matmul 融合最小 pass |
| 国产卡落地 | `Cambricon/mlu-ops` → `kernels/` 挑 `abs` 与 `deform_conv` | 对照同功能 CUDA 写法，列 BANG C 与 CUDA 的语法差异 | 不接代码，作面试对照材料 |

**可衔接的下一任务（三条，都有硬验收）**

- **T1 基准落地**：在空目录 `benchmarks/` 里加 `bench_matmul.py`，用 `pytest tests/` 先确认基线全绿，再输出「naive vs 分块版」在 512/1024/2048 三个尺寸的 GFLOPS 表。验收 = 表 + 命令可复现 + 测试仍全绿。
- **T2 cost model 对齐**：让 `src/gkern/compiler/costmodel.py` 对 matmul、softmax、elementwise 三个算子给出 roofline 预测，与 `benchmarks/` 实测对比，输出误差列。验收 = 误差表 + 误差 >50% 的项必须有归因。
- **T3 假后端闭环**：用 `src/gkern/torch_ext/backend.py` 注册一个 PrivateUse1 后端，`examples/` 加一个 20 行脚本证明一张最小图能跑通。验收 = 脚本 + `pytest tests/test_torch_ext.py` 全绿。

> 拿不到国产卡时，D 类只读不跑：`Ascend/pytorch` 的 `torch_npu/csrc/InitNpuBindings.cpp` 与 `vllm-ascend/attention/mla_v1.py` 对照 B 节读，产出「同一 attention 在 CUDA / NPU 上的差异点」四处（分块、dtype、通信、融合），不需要卡。

---

## F. 调研方法与未验证项

### F.1 用过的 API 调用

```bash
# 单仓库元数据（表中每一行的 stars/push/language/archived 都来自这里）
curl -sS -H "Authorization: Bearer $TOKEN" https://api.github.com/repos/<owner>/<repo>

# 组织信息与组织仓库列表（用于 D 节「公开仓库稀少」的判定）
curl -sS -H "Authorization: Bearer $TOKEN" https://api.github.com/orgs/<org>
curl -sS -H "Authorization: Bearer $TOKEN" "https://api.github.com/orgs/<org>/repos?per_page=100&sort=updated"

# 路径存在性核验（「读它学什么」里每个文件/目录都查过）
curl -sS -H "Authorization: Bearer $TOKEN" https://api.github.com/repos/<owner>/<repo>/contents/<path>

# 关键词检索
curl -sS -H "Authorization: Bearer $TOKEN" "https://api.github.com/search/repositories?q=<query>&per_page=10"
```

### F.2 限流与失败情况（如实记录）

- **未认证限流**：首次 `curl https://api.github.com/rate_limit` 返回 `core.remaining = 0`（60/h 已用尽，`reset = 1793457355` = 2026-10-31T14:35:55Z）。之后改用本机 `gh auth token` 取到的 token，认证后 `core.limit = 5000`，本文件全部数据在认证状态下取得。
- **404 / 不存在的名字**（已修正或剔除，不留残项）：
  - `gpu-mode/lecture` → Not Found；正确仓库是 `gpu-mode/lectures`（6938 stars）。
  - `MooreThreads/musa-samples` → 404，该仓库不存在，已剔除。
  - `IluvatarAI` 组织 → 404；实际为 `Deep-Spark` / `Iluvatar-CoreX`。
  - `Ascend/ModelZoo-PyTorch`、`Ascend/tensorflow`、`Ascend/ascend-for-pytorch` → 404，已剔除。
  - `apache/tvm` 的 `python/tvm/tir`、`src/target` → 404（主线重构为 `s_tir`，见 C 节）。
  - `pytorch/pytorch` 的 `torch/npu` → 404（该目录在 2026-10-31 的 main 上不存在）。

### F.3 证据不足 / 未验证项

1. **「官方 vs 社区」归属未经厂商官网核对**：`Deep-Spark/vllm-iluvatar` 的 description 自述 community-maintained；`Ascend/triton-ascend`、`Ascend/AscendNPU-IR` 自述是 gitcode 镜像。除此之外的归属判断仅基于组织名与仓库 description，未与厂商官网逐一比对。
2. **star 数对国产卡项目参考价值有限**：多为内部镜像或工具仓库，`Ascend/msprof` 仅 3 stars 却每日更新。本文件只把它当快照记录，不当作质量排序依据。另：stars 是活数，热门仓库在采样当日会以分钟级速度 ±1 跳动（复核时 `pytorch/pytorch` 与 `tile-ai/tilelang` 各变动 1，已按复核时刻的读数修正），故表中数字只代表 2026-10-31 的快照，不作为长期引用值。
3. **D 类全部只做到仓库层验证，没有跑通**：本机无国产加速卡，无法验证任何国产仓库的实际可运行性（依赖 CANN/BANG/MUSA SDK，均未安装）。因此 D 节所有「读它学什么」都只保证「路径存在 + 内容形态」，不保证「照着读就能跑」。
4. **壁仞的 kernel 层项目未找到**：只有 K8s device plugin 与 ModelZoo 两类周边，且均已 archived。是否存在未公开的 kernel 库无法判断。
5. **roofline 工具只有小项目**：GitHub 上无厂商官方 roofline 工具；`feifeibear/LLMRoofline`（126 stars，2024-03-13 后未更新）是能找到的最贴近需求的一个。另有 `/search/repositories?q=roofline` 返回的 `hahnyuan/LLM-Viewer`（689 stars）与 `champ-hub/carm-roofline`（35 stars），数字同样取自 API，但只核对了元数据、没有读代码，故未入表。

### F.4 已验真但未列入表格的仓库（按每方向 4-8 个的上限裁剪，数据均为 2026-10-31 快照）

| 仓库 | stars | 最近 push | 语言 | 备注 |
|---|---|---|---|---|
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | 25232 | 2026-10-31 | Python | **强烈建议补进 A 类**：读 attention kernel 的行业标准样本，因每组上限未入表 |
| [gpu-mode/lectures](https://github.com/gpu-mode/lectures) | 6938 | 2026-09-09 | Jupyter Notebook | GPU MODE 课程，`lecture_001`… 共 39 项 |
| [NVIDIA/nvbench](https://github.com/NVIDIA/nvbench) | 942 | 2026-10-23 | Cuda | CUDA kernel 微基准库 |
| [gpu-mode/reference-kernels](https://github.com/gpu-mode/reference-kernels) | 321 | 2026-09-18 | Python | `problems/` 下按 amd / nvidia / helion / linalg 分栏 |
| [thunlp/TritonBench](https://github.com/thunlp/TritonBench) | 143 | 2025-06-14 | Python | Triton 算子生成 benchmark |
| [Ascend/triton-ascend](https://github.com/Ascend/triton-ascend) | 130 | 2026-05-18 | MLIR | 昇腾的 Triton 后端，D 节正文引用；自述为 gitcode 镜像 |
| [Ascend/AscendNPU-IR](https://github.com/Ascend/AscendNPU-IR) | 61 | 2026-10-31 | C++ | 昇腾 MLIR 中间表示与编译，D 节正文引用；gitcode 镜像 |
| [Ascend/msprof](https://github.com/Ascend/msprof) | 3 | 2026-10-31 | Python | 昇腾 profiling 工具（对标 ncu），D 节正文引用 |
| [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops) | 23 | 2026-09-29 | Python | 昇腾 Triton 算子集 |
| [MooreThreads/mccl](https://github.com/MooreThreads/mccl) | 3 | 2026-10-19 | C++ | MUSA 集合通信库（对标 NCCL） |
| [MetaX-MACA/vllm-omni-metax](https://github.com/MetaX-MACA/vllm-omni-metax) | 3 | 2026-09-16 | Python | 沐曦卡上的 vLLM 适配 |
| [Cambricon/triton-linalg](https://github.com/Cambricon/triton-linalg) | 230 | 2025-02-07 | C++ | 同时列在 C 节；寒武纪的 Triton→Linalg 转换 |
| [EnflameTechnology/torch-gcu](https://github.com/EnflameTechnology/torch-gcu) | 5 | 2026-06-23 | C++ | 同时列在 B 节；燧原 PyTorch 适配 |
| [BirenTechnology/ModelZoo](https://github.com/BirenTechnology/ModelZoo) | 9 | 2024-12-17 | Python | 壁仞模型仓库，**已 archived** |
| [Ascend/torchair](https://github.com/Ascend/torchair) | 28 | 2026-06-08 | Python | 昇腾图模式（torch.compile 侧） |
| [Ascend/op-plugin](https://github.com/Ascend/op-plugin) | 14 | 2026-10-30 | Python | 昇腾算子插件 |
| [Cambricon/vllm-mlu](https://github.com/Cambricon/vllm-mlu) | 123 | 2026-05-11 | Python | 寒武纪 vLLM 适配 |
| [MetaX-MACA/TileKernels-Metax](https://github.com/MetaX-MACA/TileKernels-Metax) | 3 | 2026-07-08 | Python | 沐曦 TileLang kernel 库 |
| [Deep-Spark/DeepSparkInference](https://github.com/Deep-Spark/DeepSparkInference) | 37 | 2026-10-23 | Python | 天数智芯 300 模型推理基准集 |
| [EnflameTechnology/DeepGEMM](https://github.com/EnflameTechnology/DeepGEMM) | 1 | 2026-08-26 | Python | 燧原 DeepGEMM 移植 |
| [Ascend/samples](https://github.com/Ascend/samples) | 168 | 2023-11-22 | Python | 昇腾样例，但停更近 3 年，仅作历史参考 |

### F.5 一句话的采集统计

A 类入表 8 个、B 类 8 个、C 类 6 个、D 类 8 个（跨 7 家厂商），A–D 共 30 行、去重后 28 个唯一仓库（`Ascend/pytorch` 与 `MooreThreads/torch_musa` 同时出现在 B 与 D）；F.4 另有 21 行、19 个唯一仓库（`EnflameTechnology/torch-gcu`、`Cambricon/triton-linalg` 各与 B/C 重复一次）。全文合计 51 行表格、47 个唯一仓库（可直接数：`grep -c 'https://github.com/'`）。stars 全部在 2026-10-31 用 `/repos/{owner}/{repo}` 复核一遍；复核时刻 `vllm-project/vllm` 与 `llvm/llvm-project` 各 +1，已按最新读数修正。失败/404 共 9 个名字，全部在 F.2 列出，未做任何记忆填充。
