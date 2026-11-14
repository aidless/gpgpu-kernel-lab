# gpgpu-kernel-lab

[![CI](https://github.com/aidless/gpgpu-kernel-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/aidless/gpgpu-kernel-lab/actions/workflows/ci.yml)

> **English TL;DR**: A GPGPU operator/compiler/PyTorch-adaptation lab that runs entirely on CPU: an executable SIMT model (masks, divergence, coalescing transactions, bank conflicts), a Triton-style DSL→IR→pass→numpy-executor stack with a cost model, six operators checked element-wise against numpy, plus `torch.library` custom ops and a `torch.compile` backend. Run `PYTHONPATH=src python3 -m pytest` and `benchmarks/run_benchmarks.py`; `scripts/verify_numbers.py` recomputes every documented figure. It is explicitly not a cycle-level GPU simulator or production backend; the single real-GPU (T4) measurement is reconciled separately.

一个**在纯 CPU 上就能跑通**的 GPGPU 算子 / 编译器 / PyTorch 适配实验台。

它不假装自己有 GPU。它做的是把真正决定 GPU 性能的机制——warp 掩码、分支发散、
访存合并、共享内存 bank 冲突、roofline 受限判断、算子融合、分块调优、
`torch.compile` 自定义后端、`torch.library` 自定义算子——做成**可执行、可测量、
每个数字都能从本仓库代码重算**的最小实现。

> 定位一句话：**这里的所有结论都来自代码，不来自话术。**
> 你看到的每一个百分数、字节数、事务数，都能用 `pytest` 或
> `benchmarks/run_benchmarks.py` 当场复算出来。

---

## 它是什么 / 不是什么

| 是 | 不是 |
|---|---|
| 一套能在 CPU 上真实执行的 SIMT 执行模型（掩码 / 发散 / 事务 / bank 冲突） | 一个 GPU 模拟器（不做周期级时序仿真） |
| 一个 Triton 风格的 DSL → IR → pass → 执行器 / 代价模型 | 一个能跑在真实国产卡上的算子库 |
| 用 `torch.library` + `torch.compile` 后端做的框架适配样例 | 一个生产可用的 PyTorch 后端 |
| 六个算子（vector_add / scale_gelu / softmax / layernorm / matmul / fused_mlp）与 numpy 逐元素对拍 | 一份性能 benchmark 报告 |

**边界（必须说清楚）**：本机没有 GPU（`nvidia-smi` / `nvcc` 均不存在，
`torch.cuda.is_available() == False`），Triton 3.8 装了但没有 CPU driver
（最小 kernel 直接报 `RuntimeError: 0 active drivers`）。所以：

- 设备模型里的 TFLOPS / 带宽是**假设机**（`DeviceInfo` 默认参数）算出的解析值，
  **不是真卡实测**；
- 代价模型给的是**理论下界**，不含 cache 命中、发射开销、bank 冲突的真实代价；
- numpy 执行器的墙钟是「真的算了」，但它逐 block 走 Python 循环，
  **比模型预测慢 2–3 个数量级是正常的**，不能当性能读数；
- `explain()` 有一个已知局限：同一个原函数分别编译成 static 与 dynamic 两个包装时，
  Dynamo 的 cache entry 是共享的，两个包装会被归到同一份结论（实测两者都返回
  `只支持静态形状…s77`）。不同函数、以及同一包装的重编译都能正确区分。

---

## 快速开始

```bash
cd gpgpu-kernel-lab
# 方式 A（推荐，避免 PEP 668 externally-managed 报错）：源码直跑，不安装
PYTHONPATH=src python3 -m pytest            # 154 个用例通过（含 C++ 扩展；未构建扩展时 132 passed + 1 模块级 skip）
PYTHONPATH=src python3 examples/quickstart.py
PYTHONPATH=src python3 benchmarks/run_benchmarks.py

# 方式 B（正常 venv 内）：editable 安装后直接 pytest
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                       # numpy + torch + pytest（可再加 pybind11）
pytest                                        # 同上 154 个用例
```

2026-11-01 实测：宿主机是 PEP 668 externally-managed（`pip install -e .` 被拒），
所以**方式 A 是在这台机器上的可靠路径**——`PYTHONPATH=src` 让 `pytest`、
`examples/quickstart.py`、`benchmarks/run_benchmarks.py` 三个入口都无需安装即可跑。

`examples/quickstart.py` 会依次打印：设备模型、访存合并对比、分支发散、
六个算子的 IR 与代价、融合省下的字节、`torch.compile` 融合结论。

---

## 三层架构

```
src/gkern/
├── gpgpu/          SIMT 执行模型：这一层回答「硬件为什么慢」
│   ├── device.py     DeviceInfo / Device / BlockCoord / ThreadCoord（峰值算力、带宽、占用率）
│   ├── warp.py       Warp / WarpTrace / DivergenceReport（掩码、分支发散的 lane 效率）
│   ├── memory.py     GlobalMemory / SharedMemory（128B 事务计数、32 bank 冲突计数）
│   ├── kernel.py     @kernel / launch / WarpContext（tid 是长 32 的数组 = 一次一个 warp）
│   └── samples.py    coalesced vs strided 加法、branchy ReLU（示例与测试共用实现）
├── compiler/       Triton 风格编译栈：这一层回答「为什么能更快」
│   ├── ir.py         Tile / StoreInstr / Program（线性指令表，Tile.id == 指令下标）
│   ├── dsl.py        追踪器：arange/load/store/exp/sum/dot/... → IR
│   ├── codegen.py    numpy 执行器（一维线性地址空间，越界读用 clip+where 兜住）
│   ├── costmodel.py  Cost / analyze：由 IR 的 shape/dtype 推访存与 FLOP，给 roofline 受限判断
│   └── passes.py     can_fuse / fuse_programs（生产者-消费者融合）、autotune（分块枚举）
├── ops/            真实算子：这一层回答「能算对吗」
│   └── elementwise/softmax/matmul + run_* runner（返回 输出 + IR + 代价）
└── torch_ext/      PyTorch 适配：这一层回答「能接进框架吗」
    ├── custom_ops.py  torch.library 自定义算子（实现 + fake + autograd + schema 四件套）
    ├── fx_lower.py    FX 图 → gkern IR 的下沉器（每种失败都给原因）
    └── backend.py     torch.compile(backend="gkern") 的自定义后端
```

---

## 可复算的数字（全部来自 `benchmarks/run_benchmarks.py` 与 `pytest`）

**设备模型（解析值，假设机）**

| 量 | 值 | 公式 |
|---|---:|---|
| 峰值 FP32 算力 | 26.2144 TFLOPS | `sm_count(64) × 128 cores × 2 × clock(1.6GHz) / 1000` |
| 理论带宽 | 102.4 GB/s | `1.6GHz × 2 × 256bit/8` |
| roofline 交叉点 | 256.0 FLOP/B | 峰值算力 / 带宽 |

**访存合并（`n=1024`，一个 warp 的 32 lane）**

| 形态 | load 事务数 | 合并效率 |
|---|---:|---:|
| 连续访存（相邻 lane 读相邻地址） | 64 | 1.00 |
| 跨步 32 元素 | 2048 | 0.05 |

→ 同样一次加法，**事务数放大 32×**，字节请求数却完全一样。这就是「访存合并」
比「算法复杂度」更影响性能的直接证据。

**分支发散（branchy ReLU）**

| 输入 | 发散率 | lane 效率 |
|---|---:|---:|
| 正负混合 | 1.00 | 0.667 |
| 全正 | 0.00 | 1.000 |

**融合省下的访存（等式，不是估算）**

`A(64,32) @ W(32,48) + bias → gelu`，`bm=32, bn=16, bk=16`：

```
未融合 74112 B  →  融合后 49536 B
差值   24576 B  == 2 × |C| == 2 × 64 × 48 × 4 B
```

`2 × |C|` 的来历：融合前中间张量 C 要**写回一次全局内存、再读出来一次**，
融合后这两笔都没了。测试 `tests/test_compiler.py::test_fusion_saves_exactly_two_intermediate_writes`
把这条等式钉成断言；`benchmarks` 里还有一个 `BN ∤ N` 的对照行，展示 tile 补齐
会让计数字节大于真实张量字节（代价模型的已知口径，见「边界」）。

---

**共享内存分块：省的不是 HBM 流量**

`A(256,256) @ B(256,256)`，tile=32 / 64，两个版本各自跑一遍：

| tile | 未用 smem 访存 (B) | 用 smem 访存 (B) | 字节相等 | 数值一致 |
|---|---:|---:|---|---|
| 32 | 4456448 | 4456448 | True | True |
| 64 | 2359296 | 2359296 | True | True |

**字节数完全相等**——这是整个仓库里最容易被讲错的一条。共享内存省的是
global → L1 的往返与 bank 冲突，不是 HBM 流量：每个 block 仍然要把自己的 A/B
面板从全局读一遍。所以「加了 smem 就该变成 compute bound」是错的，
`tests/test_smem.py::test_smem_tiling_does_not_change_global_bytes` 就是钉这件事的。

**真正决定 memory/compute 的是复用与规模**

| 方阵 n | 逐块重读下界 (B) | 下界 AI | 下界受限 | 理想复用上界 (B) | 上界 AI | 上界受限 |
|---|---:|---:|---|---:|---:|---|
| 512 | 17825792 | 15.53 | memory | 3145728 | 88.0 | memory |
| 1024 | 138412032 | 16.00 | memory | 12582912 | 176.0 | memory |
| 2048 | 1090519040 | 16.25 | memory | 50331648 | 352.0 | compute |

理想复用（每个输入只从 HBM 读一次）下方阵 GEMM 的 `AI = N/6`，
交叉点 256 FLOP/B 需要 **N ≥ 1536**——而把 tile 从 16 加到 128，
AI 只从 4.27 涨到 29.59，离交叉点差一个数量级。**调 tile 翻不动 bound，
是这里的算术决定的，不是调参不够努力。**

**bank 冲突：同一个算子，行宽差 1 个 float**

| 共享内存行宽 (float) | 访问次数 | bank 冲突轮次 | 冲突率 |
|---|---:|---:|---:|
| 32 | 1056 | 992 | 0.9394 |
| 33 | 1056 | 0 | 0.0 |

按列读行主序面板时，行宽 32 让 32 个 lane 全落 bank 0；行宽改成 33 全部散开。

---

**PrivateUse1 设备后端：`torch.device("gkern")` 能建出来，算子还跑不了**

```python
from gkern.torch_ext import register_device_backend
register_device_backend("gkern")        # 改名 + 注册设备模块 + 生成 .gkern() 方法
torch.device("gkern", 0)                # gkern:0  ← 合法
torch.gkern.device_count()              # 1
torch.empty(4, device="gkern")          # NotImplementedError ← 刻意的
```

最后一行不是 bug，是这一层的**边界**：PrivateUse1 的内存分配只能在 C++ 注册
（`c10::PrivateUse1Allocator` + `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)`），
纯 Python 接管不了 `aten::empty` 的 dispatch。缺口清单在
`device_backend.GAPS` 里（分配器 / 算子内核 / stream-event / 设备侧 autograd / 显存统计），
`describe()` 输出「接了哪几层」。测试把这条失败钉成断言——宁可它红，
也不要一个「数值对、设备掺假」的假成功。

顺带纠一个常见混用：`torch.compile(backend="gkern")` 与 PrivateUse1 设备后端
**不是一件事**——前者只接管图的分段与执行，张量仍在 CPU 上；后者要求张量真的
具有第三种设备类型、dispatch key 切到 PrivateUse1。见 `docs/ecosystem.md` E.1 的对照表。

---

**C++ 侧：同一份语义的第二实现，以及它与 Python 的语义差**

`csrc/gkern_cpp.cpp` 把两个最热的访存判据（事务计数、bank 冲突）用原生代码重写一遍，
通过 pybind11 暴露成可选扩展 `gkern_cpp`：

```bash
pip install pybind11 && pip install -e ".[cpp]" --no-build-isolation
PYTHONPATH=src python3 scripts/verify_cpp.py
```

`scripts/verify_cpp.py` 是硬门禁：模块缺失直接 `exit 2`（不允许「没构建就算通过」），
存在时跑 1200 组随机/边界输入的逐值对拍，再量一次同工作量的耗时
（1M 偏移：`count_transactions` 约 9×、`warp_bank_conflicts` 约 33×，本机 Ryzen 7 5800H）。

写这份代码暴露的真问题不是性能，而是**跨语言语义差**：Python 的 `//` 向下取整，
C++ 的 `/` 向零截断，`[-1, -128, -129, 0]` 在 Python 里是 3 个事务段、在朴素的 C++
里是 2 个。`csrc/gkern_cpp.cpp` 里因此手写了 `floor_div`——这是坑 #11。

**混合精度：精度是免费的，但必须显式选**

`gkern.ops.mixed_precision` 把「累加精度」写成 IR 里的显式舍入点（`tl.round_bf16`），
softmax 在四个位置降精度（输入、行最大值、指数结果、归约和）。实测同一个 256×1024 输入：

| 累加精度 | 最大绝对误差 | 访存 (B) | FLOP |
|---|---:|---:|---:|
| fp32 | 5.96e-08 | 4194304 | 2621440 |
| bf16 | 5.87e-03 | 4194304 | 3671040 |

访存字节**完全相同**、FLOP 只多四次舍入——所以「用不用 bf16 累加」是一个纯粹的
数值决策，而不像很多人以为的那样是性能决策。误差大了约 10 万倍这件事，被写成了
`tests/test_mixed_precision.py` 的断言，防止哪天实现偷偷没降精度却还挂着 bf16 的名字。

**这一格在真实 GPU 上还没量**（Kaggle 每周 GPU 额度 2026-11-01 已用尽：32.75h/30h、
剩余 0.00h，`kaggle kernels push` 被平台拒绝）。脚本 `benchmarks/triton/bench_bf16_softmax.py`
已提交、可直接跑，缺口与补齐命令写在 `docs/cloud-reconciliation.md` 最后一节——
不写「应该差不多」这种话。

---

**云端实测：T4 上量一次，再回头问模型准不准**

本仓库其他数字都是 CPU 上可复算的机制演示；唯独这一节是**真实 GPU 实测**，
两者混用就是自欺，所以证据、脚本、对账三者分开写：

| 项 | 值 |
|---|---|
| 平台 | Kaggle 私有 kernel，Tesla T4（40 SM） |
| 环境 | torch 2.11.0+cu128、triton 3.6.0、Python 3.13.15 |
| 采集 | `benchmarks/triton/bench_softmax_triton.py`（`do_bench` warmup=25/rep=100） |
| 证据 | `benchmarks/results/triton_softmax_t4.json`（带 provenance） |
| 对账 | `benchmarks/analyze_triton.py --write-report` → `docs/cloud-reconciliation.md` |

三条结论（每条都在对账文档里带表）：

1. **模型是严格下界**：实测/预测 = 1.36～1.40，即 71%～74% 的理论带宽；
   五档 shape 的受限类型判得全对（都是 `memory`），但别把 `predicted_ms` 当成能跑到的时间。
2. **小 shape 掉到 49%**：cols=512 时占用率不足，模型的问题定义里就没有这一项。
3. **手写 Triton fused softmax 没稳定赢过 `torch.softmax`**（cols=512 慢 26%，
   1024/2048 快约 1.08～1.10×，4096/8192 反而慢 3%～5%）。CPU 侧的字节账
   「融合省掉中间张量」是真的，但它推不出「手写 kernel 更快」——
   eager 行 softmax 本身已是单遍带宽受限实现。

---

**工作分解：tail 值钱，但只有把工作切细才治得了它**

矩阵乘在硬件上不是「一次算完」，而是把输出切成 tile 铺到 SM 上排队。100 个 tile
铺 40 个 SM：40 + 40 + 20 —— 最后一波一半 SM 空转，整体利用率 0.83。这个账在
`gkern/schedule/` 里是可复算的整数，不是直觉（数字来自 `benchmarks/results/bench.md`）：

| 形状 | 策略 | 单元数 | 波次 | 利用率 | 边界部分和 | 额外流量 (B) | 预测 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1280×1280×528 | naive | 100 | 3 | 0.8333 | 0 | 0 | 0.255034 |
| 1280×1280×528 | split-K(2) | 200 | 5 | 1.0000 | 200 | 26214400 | 0.212528 |
| 1280×1280×528 | stream-K | 3300 | 83 | 0.9940 | 39 | 5111808 | 0.213816 |
| 1280×1280×528 | persistent | 100 | 3 | 0.8333 | 0 | 0 | 0.255034 |
| 256×256×8192 | naive | 4 | 1 | 0.1000 | 0 | 0 | 1.318964 |
| 256×256×8192 | split-K(16) | 64 | 2 | 0.8000 | 64 | 8388608 | 0.164870 |
| 256×256×8192 | stream-K | 2048 | 52 | 0.9846 | 39 | 5111808 | 0.133957 |
| 256×256×8192 | persistent | 4 | 1 | 0.1000 | 0 | 0 | 1.318964 |

四条结论，每条都能从上表读出：

1. **persistent 治不了 tail**：它和 naive 的波指标**完全一样**（0.8333 / 0.1000）。
   常驻块省的是块调度开销，而 tail 是并行度用不满——换谁派发都一样。这一栏留在表里
   就是为了把这个常见误解钉死。
2. **split-K 在宽扁大矩阵上几乎免费**：额外流量 26 MB 看着吓人，但同一形状的计算
   时间是 0.21 ms，26 MB 在 320 GB/s 下只要 0.08 ms——被计算盖住，而利用率从
   0.83 提到 1.00。所以它在这个形状上赢。
3. **stream-K 在「输出极小、K 极深」时翻盘**：256×256 的输出只有 4 个 tile，
   naive 利用率 0.1（40 个槽里 36 个空着）；stream-K 以 k-iter 为单元把粒度细 512 倍，
   利用率 0.98，且只在 39 个跨 worker 边界落部分和（5.1 MB），比 split-K×16 的
   8.4 MB 还少。
4. **同一套模型给出不同推荐**：`recommend()` 在 1280³ 上选 split-K、在 256×256×8192
   上选 stream-K。这是它有用处的证据——如果两个形状都给同一个答案，这套模型就该删。

「边界部分和」那一列**不是闭式公式算的**：`streamk_units()` 把 3300 / 2048 次 k-迭代
真的切成连续区间（分别 139 / 43 段），再从划分里数出「哪些 tile 被多个 worker 累加」。
闭式 `capacity−1 = 39` 只是上界，本机两个形状恰好都等于 39——凑巧相等不算数，
所以测试断言的是「两套账必须一致」，一旦划分改了而表里的数没改，闸就红。

诚实边界：`predicted_ms` 只含「计算 ÷ 利用率」与「字节 ÷ 带宽」两项，
不含发射延迟、占用率爬坡、L2 命中（见 `docs/ARCHITECTURE.md` §8.2 的缺口清单）。
它能在策略之间排序，不能预测墙钟。

---

## 开发中实际踩到并修掉的十一个坑

每一条都是**先有失败的症状、再定位根因**，不是事后编的故事：

1. **掩码状态存了两份**（`gpgpu/kernel.py`）。`WarpContext` 另存了一份 `_mask`，
   而 `branch()` 在 then/else 内部改的是 `warp.mask`。结果两个分支都按「全活跃」
   写入，后写的覆盖先写的——ReLU 输出全 0（实测 1024 个元素里 497 个错）。
   修法：掩码只留一份真相，`mask` 直接是 `warp.mask` 的视图。
2. **标量被 `int()` 截断**（`compiler/dsl.py`）。`scale=1.7` 被绑成 `1`，
   梯度看着对、结果全错。修法：保留 `int/float/bool` 原类型。
3. **LayerNorm 尾部 lane 污染方差**（`compiler/codegen.py` 的归约语义）。
   被掩掉的 lane 参与 `(x-mean)²` 求和 → 方差偏大。修法：`tl.where(mask, x*x, 0)/N`。
   这是所有「带 reduce 的 kernel」的经典坑。
4. **二维缓冲区被当二维索引**（`compiler/codegen.py`）。GPU 的地址空间是一维的，
   2D numpy 数组直接拿算出来的线性下标去索引会 `TypeError`。修法：一律 `reshape(-1)`。
5. **融合后残留中间缓冲区**（`compiler/passes.py`）。融合把 C 的写读都消掉了，
   却还在 `inputs` 里留了 `C`，执行时报 `KeyError: 缺少输入缓冲区 C`。
   修法：`pop` 掉——但**只在 A 自己不读 C 时**成立。就地读写的生产者被 `pop` 之后
   仍会 KeyError，所以这条修法配了判据：`can_fuse` 拒绝「A 也读 buffer」的组合。
6. **融合判据不足会产出静默错结果**（`compiler/passes.py`，最隐蔽的一个）。
   早期 `can_fuse` 只查 grid/输出/被读三条，于是 A 写 `C[i]`、B 读 `C[i+1]` 这种
   halo 访问也被判成可融合，融合后是一整条旋转，`max|Δ| = 1.894` 却**不报错**。
   同批被误判的还有 shape 不匹配、dtype 不匹配、就地读写、多次写、跨 lane 归约。
   修法：判据扩到七条（见 `docs/ARCHITECTURE.md` §4），七类都写成拒绝用例
   （`test_fusion_gate_rejects_*` 共七条），并验证过「拿旧版 `can_fuse` 跑这组测试必红」。
7. **`register_autograd` 签名变了**（`torch_ext/custom_ops.py`）。2.14 上是
   `register_autograd(backward, *, setup_context=...)`，按早期文档写成位置参数
   `(setup, backward)` 直接 `TypeError`。
   另有一个更要命的：FX 图输出是 `((t,),)`，后端回裸 tensor 会被 Dynamo 按序列取
   `[0]` → 结果变成 0 维标量（实测 `torch.Size([]) != torch.Size([37])`）。
8. **编译期与运行期语义不一致**（`torch_ext/`，第二隐蔽的一个）。真实实现经 numpy
   只能算 float32，但 `register_fake` 照单全收任意 dtype：`float64` 输入时 fake 报
   `float64`、real 悄悄降成 `float32`——`torch.compile` 最典型的「图里说一套、跑出来
   另一套」。同批还有三处：`fused_mlp` 的 `K % BK == 0` 只在 real 校验（编译期过、
   运行期炸）、CUDA/meta 张量在 `try_lower` 里没被拒（运行期抛的是与「下沉失败」无关的
   `TypeError`，Dynamo 侧接不住）、`explain(obj)` 无视参数直接回 `REPORTS` 第一条
   （问 A 答 B）。修法：real/fake 共用同一份校验、`try_lower` 先拒非 CPU 与符号形状、
   `REPORTS` 改按 Dynamo 的 `CompileId` 归档。
9. **bank 冲突按整块 tile 算，得出了一个不存在的结论**（`gpgpu/memory.py`）。给
   `SharedMemory` 加共享内存分块后，一次**无冲突**的整块写被算成 31 倍冲突：
   因为 `bank_conflict_degree` 把 1024 个元素当成「一次访问」一起求最大 bank 命中数，
   得出「1024 次访问撞在 32 个 bank 上」。硬件真正一起发射的单位是 **warp**，
   所以必须按 32 个 lane 分组。修法：新增 `warp_bank_conflicts()`，按 warp 切块求和
   （实测行宽 32 从「32736 轮冲突」纠正为 992，行宽 33 从 31744 纠正为 0）。
10. **整数地址算术被当成 FLOP，把算术强度抬高了 4.8%**（`compiler/costmodel.py`）。
   `analyze()` 原来对 `add/mul` 一律按 numel 计浮点，于是 GEMM 里的偏移量计算
   （`rm2 * K + rk_r`，int64）也被算成浮点运算：n=1024 的算术强度从 176.0 虚高到
   184.5，roofline 判定跟着漂。修法：只对浮点 dtype 的节点计 FLOP。

11. **`from ..compiler import run` 拿到的是「编译并跑」，不是「跑一个 Program」**
   （`ops/online_softmax.py`）。把已经编译好的 `Program` 传进去，报的是
   `TypeError: Program(online_softmax_rows, instrs=118, grid=(4,)) is not a callable
   object` —— 报错指向 `inspect`，离真正的原因隔了两层。正确的入口是
   `compiler.codegen.run(program, buffers)`，两者同名不同义。
   这类「同名不同义」的坑比逻辑错更难查：它不产生错误数值，只在调用方式上设一道坎。

---

## 测试与自检闸

```bash
PYTHONPATH=src python3 -m pytest              # 154 passed（含 C++ 扩展；未构建扩展时 132 passed + 1 个模块级 skip）
PYTHONPATH=src python3 scripts/verify_numbers.py            # 重算文档里每个数字
PYTHONPATH=src python3 scripts/verify_numbers.py --selftest # 证明这个闸会红
```

仓库现有 **135 个测试函数**（含 C++ 扩展时 pytest 收集 154 条用例；未构建扩展时
132 passed + 1 个模块级 skip）。这个「函数个数」被 `verify_numbers.py` 当断言盯着——
用例数会随可选扩展变，函数个数不会，所以拿它做机器无关的文档闸。

`scripts/verify_numbers.py` 是「文档数字会漂移」这个问题的可执行对策：它把
README/文档里出现的每个数值（26.2144 TFLOPS、102.4 GB/s、64/2048 事务、
0.667 效率、74112/49536/24576 字节）从代码重算一遍，不一致就 `exit 1`；
`--selftest` 会故意把一项期望值改错、再跑同样的检查，必须报错才算闸有效。

- `tests/test_gpgpu_model.py`：峰值算力/带宽/交叉点、事务计数、合并效率、
  发散率、bank 冲突（含「门能红」的反例）。
- `tests/test_compiler.py`：算子对拍、标量不截断、融合省访存等式、`can_fuse` 双向门、
  roofline 受限判断、缺输入缓冲区报错。
- `tests/test_ops.py`：六个算子的空输入 / shape 不匹配 / `BLOCK_N < N` / `K % BK`
  等**边界拒绝**。
- `tests/test_torch_ext.py`：自定义算子前向/反向对拍、`torch.compile` 端到端、
  每一种不支持语义的回退都必须给出原因；另有**编译期与运行期语义一致性**的一组：
  非 float32 在 real/fake 两端都拒绝、`fused_mlp` 的 `K % 16` 与形状约束两边都校验、
  非 CPU 张量与符号维度在 `try_lower` 就干净回退、`explain(obj)` 只回答**该对象**的结论。
- `tests/test_smem.py`：共享内存分块的**反直觉不变量**（字节数必须与 global-only 完全相等）、
  bank 冲突按 warp 分组的 992/0 两档、复用上界只在 n≥1536 才翻成 compute。
- `tests/test_cpp_kernels.py`：C++ 扩展（`gkern_cpp`）与 numpy 参考实现**逐值对拍**——
  含负偏移这条跨语言语义差（Python 的 `//` 向下取整、C++ 的 `/` 向零截断，
  `[-1,-128,-129,0]` 是 3 段而不是 2 段）、尾块不足 32 lane 时的 warp 冲突计数。
- `tests/test_mixed_precision.py`：`round_bf16` 与 `torch.bfloat16` 的**位模式级**一致
  （含 tie-to-even 与 NaN/Inf）、bf16 累加的误差比 fp32 大 100 倍以上、
  两条路径的访存字节完全相同而 FLOP 只多四次舍入。
- `tests/test_online_softmax.py`：分块在线 softmax 与两遍扫描**数值一致**、
  尾块（1000 列 / 块 256）行和为 1、大偏移不发散，以及字节账
  **差恰好等于一整遍输入**（`4×512×4` 字节）——省的是确定的流量，不是玄学。
- `tests/test_schedule.py`：波量化（100 单元铺 40 槽 → 3 波、最后一波空 20 槽）、
  四种分解的单元数与额外流量，以及**结论随形状翻转**（宽扁大矩阵推荐 split-K，
  输出极小 K 极深推荐 stream-K）。
- `tests/test_package.py`：公开 API 契约（`from gkern.torch_ext import register_backend`
  必须真的可用——这个包曾一度缺 `__init__.py`，`import gkern` 正常但公开名字导入失败）。

---

## 路线图

- [x] SIMT 执行模型（掩码 / 发散 / 事务 / bank 冲突）
- [x] Triton 风格 DSL → IR → pass → 执行器 / 代价模型
- [x] 六个算子与 numpy 逐元素对拍
- [x] 算子融合（省访存等式可复算）与分块 autotune
- [x] `torch.library` 自定义算子（前向 / fake / autograd / schema）
- [x] `torch.compile(backend="gkern")` 自定义后端（含回退原因）
- [x] 共享内存分块 GEMM + 复用上界模型（`reuse_bound`：理想 L2 复用下 `AI = N/6 + N/192`，
      n=2048 首次判为 compute；同时证明 smem 本身不改 HBM 字节）
- [x] `PrivateUse1` 设备后端骨架（设备对象与查询函数接上，分配器 / kernel 缺口清单化）
- [x] 云端 GPU 实测 + 代价模型对账（T4：`docs/cloud-reconciliation.md`）
- [x] 分块在线 softmax（输入只读一遍，字节账差恰好一整遍输入）
- [x] 调度层：波量化 + naive / split-K / stream-K / persistent 四种分解的账与推荐
- [x] C++ 第二实现（pybind11，跨语言语义差用 `floor_div` 钉住）+ bf16 混合精度
- [ ] 真实国产卡上的算子对拍（等算力到位后，把「模拟」换成「实测」）

## 延伸阅读

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)：三档保真度、IR 为什么是线性指令表、融合不变量的边界、代价模型的两个口径、调度层的波量化与工作分解。
- [docs/jd-mapping.md](docs/jd-mapping.md)：招聘要求 ↔ 本仓库可运行证据逐条对照，**含三个必须承认的缺口**。
- [docs/ecosystem.md](docs/ecosystem.md)：正表 30 行 / 28 个唯一仓库，附表 21 行 / 相对正表新增 19 个，全文 51 行 / 47 个唯一仓库（全部 2026-10-31 用 GitHub API 复核）逐个用 GitHub API 验真，含源码入口级阅读路径与失效路径说明。
- [docs/cloud-reconciliation.md](docs/cloud-reconciliation.md)：唯一的硬件实测证据（T4）与代价模型的对账，含「手写 softmax 没赢过 eager」这条不舒服的结论。
- [docs/domestic-card-notes.md](docs/domestic-card-notes.md)：国产卡（昇腾/vllm-ascend）与 CUDA 侧的 attention 源码对照，逐条判定本仓库哪几条建模假设在 NPU 上直接失效。
- [docs/papers.md](docs/papers.md)：8 篇论文的文献地图（在线 softmax / GEMM 分解 / 融合 / 代价模型），每篇都标了「能落成本仓库哪个模块的什么改动」与「我们不做的那部分」。
- [benchmarks/results/bench.md](benchmarks/results/bench.md)：最近一次基准输出（数值墙钟 × 模型预测 × 受限类型 × 融合等式）。

## 许可证

Apache-2.0，见 [LICENSE](LICENSE)。

---

## 验收标准与复算

本仓判为合格的条件是 `python3 eval/run_cases.py` 通过率必须为 100%（20/20），且 `make eval` 退出码必须为 0；任何一条失败即判为不合格。

**这是一条硬线，没有分档**：19/20 和 0/20 在判据上等价（都是不合格）。想让阈值松动，
必须同时改这一句和 `eval/cases.jsonl`，不能只改一边——否则「文档说 100%、题库只跑到 19」
就是一句有出处的假话。

### 题库：`eval/cases.jsonl`（20 条，七键 `id/axis/question/expected/mode/verify/source`）

| axis | 条数 | 这个轴在问什么 | 代表题 |
|---|---:|---|---|
| `device/peak` | 1 | 设备解析峰值算力（公式而非实测） | `DeviceInfo().peak_flops_tflops` = 26.2144 TFLOPS |
| `device/bandwidth` | 1 | 理论显存带宽 | `theoretical_bandwidth_gbps` = 102.4 GB/s |
| `device/t4-roofline` | 1 | 换一张卡（T4）交叉点是否跟着变 | `crossover_intensity(DeviceInfo.tesla_t4())` = 25.44 FLOP/B |
| `memory/coalescing` | 2 | 128B 事务计数：连续 vs 跨步 | 相邻 lane 1 个事务；跨步 32 元素 32 个 |
| `costmodel/roofline` | 1 | 假设机的 roofline 交叉点 | 256.0 FLOP/B |
| `warp/divergence` | 1 | 分支发散的实测发散率 | `relu_branchy` 正负混合 → 1.0 |
| `schedule/wave` | 2 | 波量化的末波单元数与非法入参 | `wave_stats(100, 40)` 末波 20；`units=0` 抛 `ValueError` |
| `schedule/gemm-tiling` | 1 | 输出 tile 数（工作单元粒度） | 1280×1280×528 / 128² → 100 |
| `schedule/recommend` | 1 | 结论随形状翻转 | T4 上 1280×1280×528 → `split-k` |
| `schedule/streamk` | 1 | 部分和份数必须从真实划分数出来 | `streamk_partials(...)` = 39（不是闭式 `capacity-1` 猜的） |
| `ops/softmax` | 2 | 行和归一到 1 + BLOCK_N 护栏 | 8×32 行和偏差 < 1e-5；`block_n=8 < N=16` 抛 `ValueError` |
| `ops/matmul` | 2 | 分块 GEMM 对拍 numpy + K%BK 护栏 | 64×32 @ 32×48 误差 < 1e-4；`bk=12` 抛 `ValueError` |
| `smem/bank-conflict` | 1 | 行宽 32 的按列读冲突轮次 | `run_smem_column(32)` → 992 |
| `smem/padding` | 1 | 行宽 33（+1 padding）冲突归零 | `run_smem_column(33)` → 0 |
| `doc/readme` | 2 | 文档数字与代码是否仍然一致 | README 的 26.2144 TFLOPS 行、256×256×8192 stream-K 行 |

模式分布：`expr` 15 条（真调本仓 API 后取 `result`）、`raises` 3 条（断言异常类型名）、
`doc` 2 条（`文件::正则`，只解析本仓文档）。**18 条 expr/raises 全部真跑本仓代码**，
期望值不是抄 README 来的——改一行实现就会变红，这正是这套题存在的意义。

### 三条命令

```bash
python3 eval/run_cases.py            # 期望 20/20，rc=0
python3 eval/run_cases.py --selftest # 期望 19/20，rc=0（把 GP-01 的期望砸成不可能值，必须变红）
make eval; echo rc=$?                # 期望 rc=0（先跑题库，再跑 154 条 pytest）
```

`--selftest` 的语义是「证明这套题不是装饰」：它把第一条 case 的 `expected` 篡改成
`__selftest_impossible_value__`，若仍然 20/20 说明比较逻辑失效（`run_cases.py` 会返回 rc=3）。
这是 AGENTS.md 第 8/21 条的红→绿两段断言在本题库上的落地。

### 数字怎么复算

1. **期望值全部来自代码**：`expr` 题的 `result` 由 `verify` 里那段代码当场算出，
   `python3 eval/run_cases.py --save` 会把逐条结果写到 `eval/run-report.json`
   （`total/passed/failed/failed_by_axis/results`），可以逐条对上。
2. **与既有闸交叉**：`scripts/verify_numbers.py` 的 `EXPECTED` 表覆盖同一批数字
   （26.2144 / 102.4 / 256.0 / 992 / 0 ……）。两处不一致说明文档或代码其中一处漂移了，
   不是「以谁为准」的选择题——先改代码再重算两边。
3. **复算入口不用安装**：题库和 pytest 都走 `PYTHONPATH=src`（src-layout，裸克隆没装也能跑）。
   `eval/cases.jsonl` 每条 `verify` 第一行自己把 `src/` 塞进 `sys.path`，所以
   `python3 eval/run_cases.py` 在仓库根直接可用，不依赖环境变量。
4. **不许把测量结果当判据**：本题库里没有「耗时 < X ms」这类题目。`bench.json` 里的
   `wall_ms` 是逐 block 走 Python 循环跑出来的，随机器抖动，把它写进 `expected` 等于
   把测量当标准。判据只钉**解析值、整数计数、等式与异常类型**。

### 失败归因（按 axis）

`run_cases.py` 不只报一个 `pass_rate`，失败时按 axis 汇总（`failed_by_axis`），
因为「哪一轴红了」比「红了 1 条」信息量大得多。`--selftest` 的真实输出是：

```
19/20 cases passed
失败归因（按 axis）：
  device/peak: 1 条 —— GP-01
SELFTEST: 期望变红；若上面仍全绿则这套题无效
```

三条判读规则（照这个表定位，别一条条猜）：

| 现象 | 归因 | 处置 |
|---|---|---|
| 同一 axis 的 expr **和** raises 一起红 | 该模块接口/签名被改（如 `DeviceInfo` 默认参数、`wave_stats` 改名） | 读报错栈定位到 `source` 指的那行，先修实现再决定是否改题 |
| 只有 expr 红、raises 绿 | 数值漂移，不是签名破坏（典型：`peak_flops` 公式或 `shared_bank_conflicts` 计数口径变了） | 用 `verify_numbers.py` 复算确认新值，改代码或改题**两处同改** |
| `doc/readme` 红而 expr 全绿 | 文档漂移：代码是对的，README 的数字没跟着改 | 只改 README（本仓历史上漂移过两次：58→68→74→135 个测试函数） |

`make eval` 的两段也是有意的：题库红说明「判据本身站不住」，pytest 红说明「实现坏了」。
两者同时红时先看 pytest——实现坏了，题库红只是它的回声。
