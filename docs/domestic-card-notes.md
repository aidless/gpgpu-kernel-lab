# 国产加速卡上的 attention：源码对照笔记

取样时间 2026-10-31；本笔记为源码阅读，不含任何实测性能数字

（实际抓取发生在宿主机 2026-11-01 00:00–00:15 UTC-7，取样日期按任务约定记为 2026-10-31。所有行号来自本次 raw 直取的内容，未从记忆填写。）

取样来源与分支：

| 仓库 | 分支 | 说明 |
| --- | --- | --- |
| `vllm-project/vllm-ascend` | `main` | NPU 侧 attention backend、MLA、mask、KV 传输 fence |
| `Ascend/pytorch` | `master` | `torch_npu` 的 stream 池与 op 提交通道 |
| `Dao-AILab/flash-attention` | `main` | CUDA 侧 FA2 前向 kernel 的分块与模板 |
| `vllm-project/vllm` | `main` | CUDA 侧 attention backend 的 dtype 白名单与 KV 写入 |

下文引用一律写成 `owner/repo/路径:行号`。

## 1. 分块（tiling）差在哪

CUDA 侧的 tile 是**编译期模板参数**，写在可读源码里，能被搜索和改写：

- `Dao-AILab/flash-attention/csrc/flash_attn/src/kernel_traits.h:66-72`：`kBlockM` / `kBlockN` 是模板实参；69 行 `static_assert(kHeadDim % 32 == 0)` 是唯一的 head_dim 对齐约束；70-72 行的 `kBlockKSmem`（64 或 32）、`kBlockKGmem`（128/64/32）、`kSwizzle`（3 或 2）全部由 head_dim 是否被 64/128 整除决定。
- `Dao-AILab/flash-attention/csrc/flash_attn/src/flash_fwd_launch_template.h:173-184`：split-KV 路径固定 `kBlockM = 64`，`kBlockN = Headdim <= 64 ? 256 : (Headdim <= 128 ? 128 : 64)`；183 行注释写明 headdim 96 配 64×256、headdim 192 配 64×128 会让 nvcc segfault，所以分块不只是性能旋钮，还是编译器约束。
- `Dao-AILab/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h:67-68` 从 `Kernel_traits` 取 `kBlockM/kBlockN`；90-94 行按 `window_size_left/right` 与 `actual_seqlen_k` 动态算 `n_block_min/n_block_max`；149-150 行 `local_tile(..., Shape<Int<kBlockM>, Int<kHeadDim>>{}, ...)` 把 m、n 两维 tile 显式展开。这三处合起来说明：**m/n 二维分块在 CUDA kernel 内是可见、可改、可自动搜索的量。**

上层也把它当参数：`vllm-project/vllm/vllm/v1/attention/backends/flash_attn.py:93-95` 定义 `FA4_DENSE_Q_TILE = 128`、`FA4_DENSE_NUM_BLOCKS = 256`、`FA4_DENSE_MAX_SEQLEN_K = 8192`。

NPU 侧同一件事只剩三个**整数枚举型**旋钮，没有 tile 形状：

1. page 粒度：`vllm-project/vllm-ascend/vllm_ascend/attention/attention_v1.py:153-155` 的 `get_supported_kernel_block_sizes()` 直接 `return [128]`，是唯一取值；KV cache 形状由它推（同文件 116-124），每请求块数 `cdiv(max_model_len, 128)`（270-272）。MLA 的 TND 路径反过来不用 paged：`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:1745` 传 `"block_size": 0`。
2. mask 模式：`vllm-project/vllm-ascend/vllm_ascend/attention/attention_v1.py:671-673` 写 `sparse_mode = 4 if self.sliding_window else 3 if attn_metadata.causal else 0`，滑窗上下界走 `pre_tokens/next_tokens`，非滑窗时填 `SWA_INT_MAX = 2147483647`（同文件 70 行）。
3. workspace：尺寸不固定，必须先向 op 问最大占用再下发：`vllm-project/vllm-ascend/vllm_ascend/attention/attention_v1.py:702-724` 调 `torch_npu._npu_fused_infer_attention_score_get_max_workspace(...)`；`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:2970-2973` 调 `_npu_fused_infer_attention_score_v2_get_max_workspace`。

真正约束分块形状的硬限制，源码里写得很直白：

- 头数必须 pad 到 2 的幂：`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:1056-1060`，注释给了反例（GLM-4.7-Flash 的 20 头），`num_heads_padded = 1 << (self.num_heads - 1).bit_length()`，`head_padding = num_heads_padded - num_heads`。CUDA FA 侧没有这条。
- 单请求 token 上限 16：`vllm-project/vllm-ascend/vllm_ascend/attention/attention_v1.py:329-332` 对 `decode_threshold` 断言 `<= 16`，断言文案写明这是 FIA TND layout 的限制；`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:529` 的 `FIA_SEQ_LEN_LIMIT = 16` 把 `actual_seq_lengths_q` 均匀插值到不超 16（同文件 508-530 的 docstring 给的是 `[1,2,...,16,32,36]`）。
- chunk 归约上限 16 对：`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:1652-1657`，`out_list` 一到 `ATTENTION_UPDATE_MAX_INPUTS` 就先 `merge_prefill_chunks` 压成一对，1653 行注释写「CANN npu_attention_update accepts at most 16 output/LSE pairs」。

结论：CUDA 的 tile 是源码里能改的数；NPU 的 tile 形状封在 CANN 二进制里，仓库只能控制 page 粒度、mask 模式和 workspace 尺寸，对不齐时靠 pad（头数补 2 的幂、序列补 16）绕过，而不是调 tile。

## 2. dtype：计算 / 累加 / 强制转换

- CUDA：累加类型是模板参数 `ElementAccum`（`Dao-AILab/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h:33`）。白名单在 `vllm-project/vllm/vllm/v1/attention/backends/flash_attn.py:289-296`：计算 `[float16, bfloat16]`，KV cache 支持 `auto/float16/bfloat16/fp8/fp8_e4m3`。
- NPU：dtype 不是白名单而是**入口处强制转换**。`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:1689-1697`，注释直接写 `FIA with TND layout only supports bfloat16`，非 bf16 输入全部 `.to(torch.bfloat16)` 再进 op；`1738-1740` 再把输出转回原 dtype。也就是说 fp16 模型走 TND 路径时必然先上一趟 bf16，精度语义由调用点决定，不由 op 决定。
- 累加精度在同一个仓库里两处不一致：MLA 的 LSE 固定 fp32（`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:1700`，`attn_lse = torch.empty(..., dtype=torch.float32)`），而 GQA 路径的 `softmax_lse` 跟着 query dtype 走（`vllm-project/vllm-ascend/vllm_ascend/attention/attention_v1.py:668` 与 `768` 的 `dtype=query.dtype`）。前者真 fp32 累加，后者 bf16 存 LSE。
- chunk 级归约前显式升 fp32：`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:1559-1560`，`chunk_out` 与 `chunk_lse` 都 `.to(torch.float32)` 再进 `merge_prefill_chunks`。
- op 内部精度只有一个开关：`inner_precise=1`（`vllm-project/vllm-ascend/vllm_ascend/attention/attention_v1.py:683`、`1624`、`1672`）。传进去的是一个整数，具体用什么累加精度在源码里看不到（见第 5 节）。
- fp8 KV 的 descale 是手写的两步：`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:1619-1621`，`torch.mul(kv_c_normed.to(self.fak_descale_float.dtype), self.fak_descale_float).to(torch.bfloat16)`，先升到 descale 的 dtype 相乘再落回 bf16。
- mask 的**数值编码随 dtype 变**：`vllm-project/vllm-ascend/vllm_ascend/attention/attention_mask.py:21-29`，`mask_value = float("-inf") if dtype == torch.float16 else 1`。fp16 时 mask 是可加偏置，其它 dtype 时 mask 是 0/1 标记。CUDA 侧靠 `cu_seqlens` 比较算因果，没有这个按 dtype 换语义的分支。
- 权重布局在加载期一次性转：`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:1280` 用 `npu_format_cast(..., ACL_FORMAT_FRACTAL_ND)`，`1452-1454` 把三个 MLA 权重转 `ACL_FORMAT_FRACTAL_NZ`。运行期不再转格式，代价前置到权重加载。

## 3. 通信与同步：AI Core vs SM/warp 概念对不上的地方

先摆事实：

- NPU 的 stream 是 `aclrtStream` 池，不是 `cudaStream_t`：`Ascend/pytorch/torch_npu/csrc/core/npu/NPUStream.cpp:57-60` 定义 `kStreamsPerPoolBits = 5`（每池 32 条）、`kMaxStreamPriorities = 2`、`kSyncLaunchStreamsPerPool = 4`；当前 stream 是 `thread_local`（同文件 87 行）。
- op 提交有两条路径，源码里是分支：`Ascend/pytorch/torch_npu/csrc/framework/OpCommand.cpp:145-171`。非 sync-launch stream 且开了 task queue 时，把 op 编成 `COMPILE_AND_EXECUTE` 塞进队列（`c10_npu::enCurrentNPUStream`）；否则直接 `aclCmd->Run(sync, sync_index, outputTensor)`，并在 `CheckBlockingEnable()` 为真时立刻 `Sync()`。同步原语只有整条 stream 的 `AclrtSynchronizeStreamWithTimeout`（同文件 222-224、265-267）。
- 显式 event 只出现在跨组件边界：`vllm-project/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/attention_fence.py:45-72`，`torch.npu.Event().record(stream)` 与 `event.synchronize()`。用途写在同文件 32-36 行的 docstring：attention worker 在提交 attention op 前打 event，MemCache 线程等这个 event 才发 H2D/L2G，让传输对齐到 compute stream 真正走到 attention 边界的时刻，而不是 Python 调用点。
- 调用点：`vllm-project/vllm-ascend/vllm_ascend/attention/attention_v1.py:1295` 的 `with attention_transfer_window():`；`vllm-project/vllm-ascend/vllm_ascend/attention/attention_v1.py:1224` 与 `1810` 的 `attn_metadata.reshape_cache_event.record()`；`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:1717` 的 `record_attention_compute_start()`。

概念对不上的四处：

1. **warp/lane 这一层不存在。** NPU 的执行单元是 AI Core（Cube 与 Vector 两组流水），同步与调度的最小可见单位是 stream 和 op。`Ascend/pytorch/torch_npu/csrc/framework/OpCommand.cpp:222-224` 能做到的最细同步是整条 stream，源码里没有任何「32 个 lane 执行同一条指令」的结构。
2. **block 级 barrier 没有对应原语。** CUDA 靠 CTA 内的 barrier 对齐；NPU 侧在仓库可见的代码里找不到等价物，融合 op 内部的核间/流水同步全封在 CANN 里。这也是为什么 workspace 必须预先向 op 申请（`vllm-project/vllm-ascend/vllm_ascend/attention/attention_v1.py:702-724`）。
3. **同步点是围绕融合 op，而不是 op 内部。** `vllm-project/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/attention_fence.py:107-120` 的 `attention_transfer_window()` 注释写明 event 记在 cache 写与前置 collective 之后，即仓库只能对齐 op 与 op 的边界，op 内各流水线的 overlap 无从干预。
4. **硬件分支会改 mask 语义。** `vllm-project/vllm-ascend/vllm_ascend/attention/attention_mask.py:57-70`：非 causal 时必须传 `None`，因为 FIA 会把任何传入的 mask 当 defaultMask（sparse_mode=0），会把双向注意力的上三角错误掩掉；但 62-67 行注释同时写明 310P 的 mask builder 必须覆盖成「显式的非掩码 mask」。同一条 Python 路径在不同卡上语义不同，CUDA 侧没有这种情况。

## 4. 融合边界：哪些被融成一个 fused op

以下每一处都能回指到具体源码行：

- KV 写入 + RMSNorm + RoPE（+ 可选 fp8 descale）融成一个 op：`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:1844-1857` 的 `npu_kv_rmsnorm_rope_cache(kv_no_split, self.kv_a_layernorm.weight, cos, sin, slots.to(torch.int64), ...)`，`cache_mode = "PA_NZ" if self.enable_kv_nz else "PA"`（1842 行）。一个调用同时吃未归一化的 KV、gamma、cos/sin、slot 和 cache。
- MLA 整条预处理链融成一个 op：`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:2353-2363` 的 `npu_mla_prolog_v3(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, ...)`。q_a/kv_a 投影、两处 RMSNorm、RoPE、写 cache 全在一个 op 内，权重以参数形式传入，中间张量不出 op 边界。
- attention 本体：`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:1728-1730` 与 `vllm-project/vllm-ascend/vllm_ascend/attention/attention_v1.py:736`、`959` 的 `npu_fused_infer_attention_score`；decoder 的 page 版是 `_npu_paged_attention`（`vllm-project/vllm-ascend/vllm_ascend/attention/attention_v1.py:831`）。
- KV cache 的 NZ 写：`vllm-project/vllm-ascend/vllm_ascend/attention/attention_v1.py:1816-1827` 的 `npu_scatter_pa_kv_cache(key=..., value=..., key_cache=k_cache_layer, ...)`，1816 行注释写「NZ write path: 5D view + npu_scatter_pa_kv_cache」。
- RoPE 交织：`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:2736` 的 `npu_interleave_rope(x, cos, sin)`。
- 量化：`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:2315` 的 `npu_dynamic_quant(token_x)`、`2323` 的 `npu_dynamic_mx_quant(token_x, dst_type=torch.float8_e4m3fn)`，decode 侧另有 `3082`、`3220`。
- 分块结果的归约也走专用 op：`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:1652-1657` 的注释点名 `npu_attention_update`，上限 16 对。

CUDA 侧对照：attention 本体同样是一个 op，KV 写入是**独立 kernel**（`vllm-project/vllm/vllm/v1/attention/backends/flash_attn.py:53` 从 `fa_utils` 导入 `reshape_and_cache_flash`），RoPE / RMSNorm / 量化在 Python 层各自调一次。所以「一个 fused op 覆盖多少语义」这件事，NPU 侧明显更宽：`npu_mla_prolog_v3` 一次覆盖了 CUDA 侧需要四五个 kernel 串起来的工作。

## 5. 未验证项

1. FIA / PA 内部真正的 tile 形状（Cube 的 fractal 尺寸是 16×16 还是 16×32）——**文件不存在**：这些在 CANN 闭源二进制里，仓库只暴露 `ACL_FORMAT_FRACTAL_ND` / `ACL_FORMAT_FRACTAL_NZ` 这个枚举名（`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:1280`、`1452-1454`）。抓不到，只能上卡测。
2. `inner_precise=1` 时 op 内部用什么累加精度——源码只传整数 `1`（`vllm-project/vllm-ascend/vllm_ascend/attention/attention_v1.py:683`），语义在 CANN 文档里，本地无该文档，未验证。
3. `vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:2987` 的 `npu_fused_infer_attention_score_v2(q_nope, k_nope, k_nope, ...)`：第二个与第三个实参都是 `k_nope`。同文件 2516-2517 行注释提到 PROLOG_V3 返回的是 MLAPO-absorbed query，MLA absorb 形式下 V 就是 latent，所以这可能是刻意写法；但仓库里没找到直接说明「decode 路径 V 位传 k_nope」的注释，**未验证是否刻意**。
4. `Ascend/pytorch` 的 `main` 分支对 `torch_npu/csrc/core/npu/NPUStream.cpp` 等文件返回 **404**，只有 `master` 与 `v2.1.0` 可达。本笔记行号基于 `master`，与 `v2.1.0` 的差异未验证。
5. workspace 的实际字节数（`get_max_workspace` 返回值）需要真卡；本机无 GPU/NPU，未取到，因此**不写任何数值**。
6. `kStreamsPerPoolBits = 5`（`Ascend/pytorch/torch_npu/csrc/core/npu/NPUStream.cpp:57`）与同文件 126-146 行 `GetStreamsPerDevice() == 8` 分支的关系：8 是另一种每设备 stream 配置，改 `StreamsPerPoolBits` 的完整分支我没读全，未验证。

## 6. 与本仓库建模假设的对应关系

| 仓库假设 | 位置 | 在 NPU 上是否成立 |
| --- | --- | --- |
| warp 是 32 lane 同步执行、带活跃掩码 | `src/gkern/gpgpu/warp.py:31`、`105`、`129-158` | 不成立。NPU 无 lane/warp 概念，`Ascend/pytorch/torch_npu/csrc/framework/OpCommand.cpp:145-171` 能看到的最细粒度是 op/stream |
| warp 效率 = 有效 lane 槽 / 发射槽 | `src/gkern/gpgpu/warp.py:38-50` | 不成立。NPU 上对应的量是 AI Core 利用率与流水占用，不是 lane 槽 |
| 共享内存 32 bank × 4B，bank = (byte//4)%32 | `src/gkern/gpgpu/memory.py:24-25`、`77-85` | 不成立（参数层面）。片上存储 bank 与 fractal 布局绑定，源码只暴露 `ACL_FORMAT_FRACTAL_NZ`（`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:1452-1454`），32×4B 是 CUDA 的数 |
| 按 32 lane 分组统计 bank 冲突 | `src/gkern/gpgpu/memory.py:136-149` | 不成立。分组单位本身消失，整块 tile 冲突统计的前提不在了 |
| roofline 单峰：max(bytes/bw, flops/peak) | `src/gkern/compiler/costmodel.py:52-63`、`126-129` | 部分不成立。NPU 有 Cube（矩阵）与 Vector（逐元素/softmax）两组算力，单一 `peak_flops_tflops` 表示不了这种分工 |
| 访存字节只算显式 load/store | `src/gkern/compiler/costmodel.py:85-120` | 不成立。fused op 的 workspace 是不可见中间缓冲（`vllm-project/vllm-ascend/vllm_ascend/attention/attention_v1.py:702-724`），不在 `bytes_total` 里，roofline 会系统性低估 |
| 设备规格：SM 数 / 每 SM warp 上限 / 每 SM FP32 单元 | `src/gkern/gpgpu/device.py:27-37`、`66-68` | 不成立。AI Core 数不等于 SM 数，也没有「每 SM 驻留 warp 上限」这个量 |
| 未建模 block 内 barrier | `src/gkern/gpgpu/kernel.py`（全文无 barrier/`__syncthreads`） | 在 NPU 上反而更贴近现实：NPU 的同步粒度就是 op/stream（`Ascend/pytorch/torch_npu/csrc/framework/OpCommand.cpp:222-224`） |
| 行级 softmax 单遍 + 行内归约，受 BLOCK_N 约束 | `src/gkern/ops/softmax.py:22-38` | 结构性成立，但上限来源不同：NPU 同样分块归约（`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:1641-1650` 分 chunk 算、`1652-1657` 合并），上限来自 op 的 16 对限制，不是 smem 容量 |
| head_dim % 32 == 0 是唯一对齐约束 | `Dao-AILab/flash-attention/csrc/flash_attn/src/kernel_traits.h:69` | 不够用。NPU 还要求头数补到 2 的幂（`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:1056-1060`） |

对应关系归纳成三句：

- `src/gkern/gpgpu/warp.py` 与 `src/gkern/gpgpu/memory.py` 的整套 SIMT 模型（lane 掩码、分支序列化、32 bank 冲突）在 NPU 上**整体失效**，不是参数需要调，而是被建模的对象不存在。要把这套模型用到 NPU，得换成「Cube/Vector 双流水 + op 级同步」的执行模型，`src/gkern/gpgpu/kernel.py` 反而更接近（它本来就没建 barrier）。
- `src/gkern/compiler/costmodel.py` 的 roofline 形式可以保留，但 `peak_flops_tflops` 与 `bytes_total` 两个输入都要改：前者要拆成 Cube/Vector 两档（不同算子走不同档，见 `vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:2353` 的 prolog 与 `2315` 的 quant 同为 fused op 但吃不同单元），后者要把 workspace 这类不可见缓冲算进去。
- `src/gkern/ops/softmax.py` 的行内归约假设在 NPU 上对应的是「分 chunk 算 + 上限 16 对合并」（`vllm-project/vllm-ascend/vllm_ascend/attention/mla_v1.py:1641-1657`），结构一致但上限的来源从 smem 容量换成了 op 参数限制——这条是六节里唯一可以直接迁移的建模假设。
