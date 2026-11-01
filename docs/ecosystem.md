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
| [triton-lang/triton](https://github.com/triton-lang/triton) | 20499 | 2026-10-31 | MLIR | Triton 语言与编译器官方仓库，Python DSL → MLIR → GPU codegen 的完整参考实现 | 算子优化 / 编译优化 |
| [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | 9732 | 2026-09-09 | C++ | CUDA 官方示例集，覆盖访存、矩阵乘、性能测量到新 Tile 编程模型 | 算子优化 |
| [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA) | 1351 | 2025-09-02 | Cuda | 从朴素版本逐级优化 SGEMM 到接近 cuBLAS，每级 kernel 带 GFLOPS 曲线 | 算子优化 |
| [AlphaGPU/leetgpu-challenges](https://github.com/AlphaGPU/leetgpu-challenges) | 1228 | 2026-10-30 | Python | LeetGPU 官方题库，按难度分级的单算子 CUDA/Triton 练习 | 算子优化 |
| [ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench) | 1307 | 2026-03-24 | Jupyter Notebook | 「LLM 能不能写 GPU kernel」的评测集 + Torch→CUDA 工具链 | 算子优化 / LLM 推理 |
| [gpu-mode/Triton-Puzzles](https://github.com/gpu-mode/Triton-Puzzles) | 2655 | 2026-04-01 | Jupyter Notebook | Triton 渐进式练习 notebook，从 broadcast 一路做到 attention | 算子优化 |
| [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | 10682 | 2026-10-28 | C++ | 高性能 GEMM 模板库，CuTe layout 代数 + CUTLASS mainloop/epilogue 设计 | 算子优化 |
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
| [pytorch/pytorch](https://github.com/pytorch/pytorch) | 105047 | 2026-10-31 | Python | PrivateUse1 / out-of-tree backend 机制的定义方（`torch/utils/backend_registration.py`） | 框架适配 / 编译优化 |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | 96006 | 2026-10-31 | Python | 多硬件推理引擎，`vllm/platforms/` 是「一套代码接多卡」的抽象样板 | 框架适配 / LLM 推理 / 异构集群 |
| [Ascend/pytorch](https://github.com/Ascend/pytorch) | 598 | 2026-10-31 | Python | torch_npu：昇腾 NPU 的 PyTorch 适配插件，国产卡适配最完整的公开样本 | 框架适配 / 算子优化 |
| [MooreThreads/torch_musa](https://github.com/MooreThreads/torch_musa) | 519 | 2026-09-23 | Python | 摩尔线程 MUSA 的 PyTorch 适配，骨架与 torch_npu 同构，适合横向对比 | 框架适配 |
| [pytorch/xla](https://github.com/pytorch/xla) | 2806 | 2026-10-20 | C++ | PyTorch on XLA：整图 lazy 执行路线，与逐 op 下沉路线形成对照 | 框架适配 / 编译优化 |
| [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) | 3039 | 2026-10-31 | Python | 昇腾上的 vLLM 硬件插件，attention 后端实现最值得读 | 框架适配 / LLM 推理 |
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
| [llvm/llvm-project](https://github.com/llvm/llvm-project) | 41282 | 2026-10-31 | LLVM | MLIR 本体；Linalg dialect 的 tiling/fusion/vectorization 是图算融合的公共底座 | 编译优化 |
| [apache/tvm](https://github.com/apache/tvm) | 13886 | 2026-10-30 | Python | 计算图 → TE/TIR → 目标代码的完整编译栈，含 autoTVM/MetaSchedule 自动调优 | 编译优化 / 算子优化 |
| [tile-ai/tilelang](https://github.com/tile-ai/tilelang) | 8866 | 2026-10-30 | Python | 面向 tile 的 kernel DSL，已在做多后端（含 ascend 目录） | 编译优化 / 算子优化 |
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

下面这张表是**每周的具体读点**（不是泛读书单）：每行给「今天就能打开的那个文件」+「第一小时的具体动作」+「落在 `gkern` 的哪个模块上」。`gkern` 是本仓库（`src/gkern/{ops,gpgpu,compiler,torch_ext}`，纯 CPU 可跑），所以每个入口都有可验收的落点。表后 E.1 钉两条口径、E.2 是四周任务卡（每周都指名用本表哪几行当读点）。

| 能力 | 先打开这一个源码入口 | 第一小时做什么 | 接到 gkern 的哪里 |
|---|---|---|---|
| 算子手感 | `triton-lang/triton` → `python/tutorials/02-fused-softmax.py` | 云 GPU 上跑通（本机无 CUDA），扫 `BLOCK_SIZE` 1024/2048/4096/8192，每档记 shape、dtype、耗时与有效带宽 | `src/gkern/ops/softmax.py`：把 Triton 的分块规则对齐到现有 CPU 实现 |
| 优化阶梯 | `siboehm/SGEMM_CUDA` → `src/kernels/1_naive.cuh` 至 `5_kernel_2D_blocktiling.cuh` | 逐级读懂「每级多了什么」，抄下每级的分块/向量化参数 | `src/gkern/ops/matmul.py` + `src/gkern/gpgpu/memory.py`：把 shared-mem 分块与 bank conflict 写成 cost 项 |
| 读数能力 | `NVIDIA/cuda-samples` → `cpp/6_Performance` | 云 GPU 上跑 bandwidthTest 与 matrixMul，用 ncu 取 DRAM throughput、achieved occupancy | `src/gkern/compiler/costmodel.py`：把实测带宽代入，量出它的预测误差 |
| 理论上限 | `feifeibear/LLMRoofline` → `perf_model.py`、`hardware_dict.py` | 在 `hardware_dict.py` 加一张目标卡参数（纯计算，本机可做），算 7B decode 的 roofline 上限 | `src/gkern/compiler/costmodel.py`：roofline 当 baseline，单测断言预测误差 |
| 框架适配 | `pytorch/pytorch` → `torch/utils/backend_registration.py` | 只读 `rename_privateuse1_backend` / `generate_methods_for_privateuse1_backend` 两个函数签名，写清「设备后端」要补哪些件（allocator/stream/event/生成器），与「torch.compile backend」分开记 | 见本节「PrivateUse1 vs torch.compile backend」表：`src/gkern/torch_ext/backend.py` 是 compile backend，不是设备后端 |
| 推理引擎适配 | `vllm-project/vllm` → `vllm/platforms/interface.py`、`cuda.py` | 画出 `platforms/__init__.py` → `cuda.py` 的惰性导入调用链 | 暂不接（依赖重），只作 D 节对照阅读 |
| 编译图谱 | `llvm/llvm-project` → `mlir/lib/Dialect/Linalg/Transforms` | 读 tiling、fusion 两个 pass 的接口，写 200 字机制总结 | `src/gkern/compiler/passes.py`：实现一个 elementwise+matmul 融合最小 pass |
| 国产卡落地 | `Cambricon/mlu-ops` → `kernels/` 挑 `abs` 与 `deform_conv` | 对照同功能 CUDA 写法，列 BANG C 与 CUDA 的语法差异 | 不接代码，作面试对照材料 |

> 表的行序（E.2 的「读点」按这个编号引用）：**1** 算子手感、**2** 优化阶梯、**3** 读数能力、**4** 理论上限、**5** 框架适配、**6** 推理引擎适配、**7** 编译图谱、**8** 国产卡落地。

### E.1 两条先立好的口径（写任务卡前必须钉死）

**口径 1：CPU 教学模型不许宣称 GFLOPS 实绩。** `gkern` 是纯 CPU 的解析/模拟模型（`src/gkern/compiler/costmodel.py` 用 `DeviceInfo` 的 SM 数 × FP32 核 × 2 × 频率算 `peak_flops_tflops`，默认 `generic-domestic-gpgpu` 得 26.2144 TFLOPS；带宽 102.4 GB/s、roofline 交叉点 256.0 FLOP/B 全是**假设参数**，不是真卡）。本机实测：无 `nvidia-smi`、无 `nvcc`，`torch 2.14.1+cu130` 但 `torch.cuda.is_available()=False`，`triton 3.8.0` 无 GPU driver（报 `0 active drivers`）。所以：

- 允许说：「模型预测 X 微秒」「算术强度 Y FLOP/B」「受限类型 memory」「访存事务数 / 合并效率 / bank 冲突度」。
- **禁止**说：「优化后达到 N GFLOPS」「比 baseline 快 N 倍」「实测吞吐」——除非该数字来自云 GPU 上的真实 Triton/CUDA 运行。
- 真实 GPU 数字一律放云上跑并贴原始输出：Kaggle GPU 通道可用，`kaggle quota` 实测 2026-10-31 剩 **8.43 h（30 h/周配额，2026-11-07 刷新）**，CPU 侧只做编排与解析。

**口径 2：`torch.compile` backend ≠ PrivateUse1 设备后端。** `src/gkern/torch_ext/backend.py` 里的 `gkern_backend(gm, example_inputs)` 经 `torch._dynamo.register_backend(name="gkern", ...)` 注册，它是 **Dynamo 的图编译器后端**：吃一个 `fx.GraphModule`、吐一个 callable，算子仍然落在 CPU 上执行，靠 `fx_lower.try_lower` 把纯逐元素图换成融合实现、不支持就回退 eager。它**没有**注册任何新设备。注册一个 `PrivateUse1` 设备后端是另一套 API（`torch.utils.backend_registration.rename_privateuse1_backend` + `generate_methods_for_privateuse1_backend`，本机 torch 2.14.1 实测就暴露这两个），需要你自己提供 allocator / stream / event / 生成器，让 `torch.empty(..., device="gkern")` 真能跑。两者互不替代：

| 维度 | `torch.compile` backend（`backend.py` 现状） | PrivateUse1 设备后端（尚未实现） |
|---|---|---|
| 注册 API | `torch._dynamo.register_backend(name=..., compiler_fn=...)` | `torch.utils.backend_registration.rename_privateuse1_backend(name)` + `generate_methods_for_privateuse1_backend()` |
| 输入/输出 | 输入 `fx.GraphModule`+`example_inputs`，输出 callable | 提供设备工厂、allocator、stream/event、RNG 等，让 tensor 能落在新 device 上 |
| 作用面 | 只接管「图 → 代码」这一层 | 接管「设备 → 存储/执行」这一层 |
| 现状 | 已实现、`tests/test_torch_ext.py` 有覆盖 | 本项目未实现；这类实现在树外仓库里（如 `Ascend/pytorch` 的 `torch_npu/csrc/`），`pytorch/pytorch` 主线本身无 `torch/npu`（实测 404） |

### E.2 四周项目任务卡（每天 2–3 小时；每周有可验收产出）

**第 1 周｜把「读数能力」落到 CPU 模型上（0 GPU）**
- 目标：让 `costmodel.py` 的 roofline 输出能对上 `benchmarks/` 的算术强度实测，并写出「模型假设 vs 实测」的边界。
- 读点（本表第 2、3、4 行）：`NVIDIA/cuda-samples` → `cpp/6_Performance`、`siboehm/SGEMM_CUDA` → `src/kernels/1_naive.cuh`…`5_kernel_2D_blocktiling.cuh`、`feifeibear/LLMRoofline` → `hardware_dict.py`。
- 改哪个文件：`src/gkern/compiler/costmodel.py`（对齐 roofline 输出）、`benchmarks/run_benchmarks.py`（补 matmul tile=16/32/64 的算术强度列）、`docs/ARCHITECTURE.md`（加一节「模型假设 vs 实测」）。
- 跑什么命令：`PYTHONPATH=src python3 benchmarks/run_benchmarks.py` 与 `PYTHONPATH=src python3 scripts/verify_numbers.py`。
- 验收：`benchmarks/results/bench.md` 里 matmul 256×256 的三档 tile 均判定为 memory-bound，且算术强度随 tile 单调升（当前实测 4.27→8.03→14.93）；`verify_numbers.py` 全绿（含 `doc_test_count`：文档里的测试数必须与真实收集数一致）。产出里不许出现任何 GFLOPS 词。
- 需不需要真实卡：不需要，纯 CPU。

**第 2 周｜真实 Triton 算子 + 云 GPU 实测（必须上云）**
- 目标：把「只在 CPU 上算 roofline」升级成「同一个 softmax 在真 GPU 上有实测 ms/GB/s」，拿到第一份可以对外说的真卡数字。
- 读点（本表第 1 行）：`triton-lang/triton` → `python/tutorials/02-fused-softmax.py`（照抄它的 `do_bench` 用法，不要自己造计时器）。
- 改哪个文件：新建 `benchmarks/triton/softmax_bench.py`（照 `triton-lang/triton` → `python/tutorials/02-fused-softmax.py` 改，保留 `triton.testing.do_bench`）、`docs/ARCHITECTURE.md` 记一条实测。
- 跑什么命令：本地先 `PYTHONPATH=src python3 -m py_compile benchmarks/triton/softmax_bench.py` 过语法；再在 Kaggle GPU kernel 里 `python benchmarks/triton/softmax_bench.py`（Kaggle 的 notebook 用 `!python ...`，working dir 需 `%cd` 到仓库根）。
- 验收：贴出 GPU 型号、`nvidia-smi` 输出、`BLOCK_SIZE ∈ {1024,2048,4096,8192}` 的 `ms` 与 `GB/s` 实测四项（**必须同时记录 shape 与 dtype，否则数字无法复算**）；与 PyTorch eager softmax 同 shape 对比，给出「Triton vs eager」倍数并注明测量环境。**这批数字是唯一可以对外说的真卡带宽/吞吐数据；没跑过就不许写。**
- 需不需要真实卡：**需要**，云 GPU（Kaggle T4/P100，实测 2026-10-31 配额剩 8.43 h）；本机无 CUDA，跑不了。

**第 3 周｜PyTorch 适配双层走通（0 GPU，纯 CPU）**
- 目标：把「compile backend」和「PrivateUse1 设备后端」两层分开落地——前者已有实现、后者只做诚实的缺口清单，不许混称。
- 读点（本表第 5 行）：`pytorch/pytorch` → `torch/utils/backend_registration.py`（只看 `rename_privateuse1_backend` 与 `generate_methods_for_privateuse1_backend` 两个签名）。
- 改哪个文件：改 `src/gkern/torch_ext/backend.py`（明确它只是 compile backend，docstring 写清）、新建 `src/gkern/torch_ext/device_backend.py`（用 `rename_privateuse1_backend("gkern")` 尝试注册，失败要给出缺哪几个件）、`tests/test_torch_ext.py` 增一条「backend 注册名可查」断言、`docs/ARCHITECTURE.md` 落 E.1 那张对比表。
- 跑什么命令：`PYTHONPATH=src python3 -m pytest tests/test_torch_ext.py -q`。
- 验收：跑下面这条，输出必须是 `registered: True` / `backends: ['gkern']`（2026-10-31 已实测通过，可直接当回归断言）：

  ```bash
  PYTHONPATH=src python3 -c "import torch, gkern.torch_ext.backend; print('registered:', 'gkern' in torch._dynamo.list_backends()); print('backends:', [x for x in torch._dynamo.list_backends() if 'gkern' in x])"
  ```

  另外 `device_backend.py` 必须给出「注册 PrivateUse1 还缺 allocator/stream/event」的显式报错清单，不许假装成功。
- 需不需要真实卡：不需要，纯 CPU。

**第 4 周｜成本模型 vs 云上实测的对账报告（1 天云 GPU + CPU 收尾）**
- 目标：出一份「预测 vs 实测」误差表，让 CPU 模型的假设错在哪儿变成可指认的条目，而不是一句「大致接近」。
- 读点（本表第 3、4 行）：`NVIDIA/cuda-samples` → `cpp/6_Performance` 的 bandwidthTest 结果口径、`feifeibear/LLMRoofline` → `perf_model.py` 的 roofline 公式。
- 改哪个文件：新建 `benchmarks/compare_model_vs_gpu.md`（表：算子 | 模型预测 µs | 云上实测 µs | 误差 % | 归因）、`src/gkern/compiler/costmodel.py` 若同一类误差系统性 >50% 就改假设并写 changelog。
- 跑什么命令：云上跑第 2 周的 softmax_bench 取实测；本地 `PYTHONPATH=src python3 benchmarks/run_benchmarks.py` 取预测；`PYTHONPATH=src python3 scripts/verify_numbers.py` 校数字。
- 验收：对账表三项算子齐全；误差 >50% 的行每行一句归因（写清是假设参数错还是模型缺项）；报告首行标注「预测 = CPU 解析模型，实测 = 云端 GPU，二者不可混称」。
- 需不需要真实卡：**需要**（第 2 周同一云 GPU 通道，1 天即可）；若配额耗尽则本卡顺延并如实标注「未实测」。

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
  - `gpu-mode/lecture` → Not Found；正确仓库是 `gpu-mode/lectures`（6939 stars，2026-10-31 复核值）。
  - `MooreThreads/musa-samples` → 404，该仓库不存在，已剔除。
  - `IluvatarAI` 组织 → 404；实际为 `Deep-Spark` / `Iluvatar-CoreX`。
  - `Ascend/ModelZoo-PyTorch`、`Ascend/tensorflow`、`Ascend/ascend-for-pytorch` → 404，已剔除。
  - `apache/tvm` 的 `python/tvm/tir`、`src/target` → 404（主线重构为 `s_tir`，见 C 节）。
  - `pytorch/pytorch` 的 `torch/npu` → 404（该目录在 2026-10-31 的 main 上不存在）。

### F.3 证据不足 / 未验证项

1. **「官方 vs 社区」归属未经厂商官网核对**：`Deep-Spark/vllm-iluvatar` 的 description 自述 community-maintained；`Ascend/triton-ascend`、`Ascend/AscendNPU-IR` 自述是 gitcode 镜像。除此之外的归属判断仅基于组织名与仓库 description，未与厂商官网逐一比对。
2. **star 数对国产卡项目参考价值有限**：多为内部镜像或工具仓库，`Ascend/msprof` 仅 3 stars 却每日更新。本文件只把它当快照记录，不当作质量排序依据。另：stars 是活数，热门仓库在采样当日会以分钟级速度跳动。本文件做了两轮 API 复核：第二轮（2026-10-31 收尾）47 个唯一仓库里有 13 个已发生漂移，其中 `pytorch/pytorch` +17（105030→105047）、`vllm-project/vllm` +35（95971→96006）、`llvm/llvm-project` +8（41274→41282），其余 10 个各 +1（triton、leetgpu-challenges、KernelBench、cutlass、vllm-ascend、apache/tvm、tilelang、flash-attention、lectures、nvbench），表中已按第二轮读数改写。故这些数字只是 2026-10-31 的快照，不作为长期引用值。
3. **D 类全部只做到仓库层验证，没有跑通**：本机无国产加速卡，无法验证任何国产仓库的实际可运行性（依赖 CANN/BANG/MUSA SDK，均未安装）。因此 D 节所有「读它学什么」都只保证「路径存在 + 内容形态」，不保证「照着读就能跑」。
4. **壁仞的 kernel 层项目未找到**：只有 K8s device plugin 与 ModelZoo 两类周边，且均已 archived。是否存在未公开的 kernel 库无法判断。
5. **roofline 工具只有小项目**：GitHub 上无厂商官方 roofline 工具；`feifeibear/LLMRoofline`（126 stars，2024-03-13 后未更新）是能找到的最贴近需求的一个。另有 `/search/repositories?q=roofline` 的搜索命中 `hahnyuan/LLM-Viewer`（689 stars）与 `champ-hub/carm-roofline`（35 stars），这两个数字来自**搜索结果里的 `stargazers_count` 字段**（不是逐仓库 `/repos/` 直取），且只核对元数据、没有读代码，故未入表。

### F.4 已验真但未列入表格的仓库（按每方向 4-8 个的上限裁剪，数据均为 2026-10-31 快照）

| 仓库 | stars | 最近 push | 语言 | 备注 |
|---|---|---|---|---|
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | 25233 | 2026-10-31 | Python | **强烈建议补进 A 类**：读 attention kernel 的行业标准样本，因每组上限未入表 |
| [gpu-mode/lectures](https://github.com/gpu-mode/lectures) | 6939 | 2026-09-09 | Jupyter Notebook | GPU MODE 课程，`lecture_001`… 共 39 项 |
| [NVIDIA/nvbench](https://github.com/NVIDIA/nvbench) | 943 | 2026-10-23 | Cuda | CUDA kernel 微基准库 |
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

正表 A–D：A 类 8 行、B 类 8 行、C 类 6 行、D 类 8 行（跨 7 家厂商），共 **30 行**；其中 `Ascend/pytorch`、`MooreThreads/torch_musa` 两个仓库在 B 与 D 各出现一次，故 A–D 去重后 **28 个唯一仓库**。F.4 附表 **21 行 = 21 个不同仓库**，其中 `EnflameTechnology/torch-gcu`、`Cambricon/triton-linalg` 已分别在 B、C 出现，故相对 A–D 只 **新增 19 个**（不重复的仓库）。全文合计 **51 行表格、47 个唯一仓库**（51 − 4 个跨节重复 = 47，脚本按表格行 URL 去重实测）。

> 计数口径与复核方式：`grep` 抓全文以 `|` 开头的表格行，用正则 `github\.com/([^)\s]+)` 提 URL，按仓库全名去重。4 个跨节重复分别是 `Ascend/pytorch`（第 46 行 B、第 95 行 D）、`MooreThreads/torch_musa`（47 B、97 D）、`EnflameTechnology/torch-gcu`（51 B、248 F.4）、`Cambricon/triton-linalg`（74 C、247 F.4）——行号按本文件当前版本实测。此前口头汇报里出现过的「49」是错的（把 F.4 的 19 个新增当成了 21），以本段实测的 **47** 为准。

stars 全部在 2026-10-31 用 `/repos/{owner}/{repo}` 直取，共两轮：第一轮 51 行全查、第二轮 47 个唯一仓库全查，第二轮发现 13 个漂移（明细见 F.3 第 2 条），表中按第二轮读数定稿。失败/404 共 9 个名字，全部在 F.2 列出，未做任何记忆填充。
