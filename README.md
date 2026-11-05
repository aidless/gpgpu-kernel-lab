# gpgpu-kernel-lab

没有 GPU 时，把 warp 掩码、访存合并、共享内存 bank 冲突、roofline 和算子融合做成可执行、可对拍的 CPU 模型。
本仓库不测量真实 GPU。硬件耗时、达成带宽、bf16 存储流量、国产卡或 TPU 上的行为都是未知，不用模型输出去填。
假设设备上的 predicted_ms 是公式，不是墙钟。CPU 上的计数用 `scripts/offline_cpu.sh` 重算。

## 观测状态

| 主张 | 状态 | 怎么核对 |
|---|---|---|
| 事务数、bank 冲突、融合字节、复用上界、调度整数、tile 算术强度 | OBSERVED (CPU) | `pytest` 与 `scripts/verify_numbers.py` |
| numpy 墙钟（`bench.md` 那一列） | OBSERVED (CPU)，随机器变，不设门槛 | `benchmarks/run_benchmarks.py`；不是设备时间 |
| roofline 的 `predicted_ms` | OBSERVED (CPU) 的公式输出；设备参数是假设的 | 同一套脚本；不是硬件时间 |
| 真实 GPU 耗时、达成带宽、bf16 存储流量、国产卡、TPU | UNKNOWN | 离线路径不跑；`benchmarks/results/` 里没有这些结果 |
| `benchmarks/results/triton_softmax_t4.json` | 归档日志，不是这次观测 | `analyze_triton.py` 只重算模型侧分母；CI 不重跑 GPU |

## 离线 CPU 路径

依赖装好之后不需要网络，也不需要 GPU。CPU 版 PyTorch 就够。仓库现有 **148 个测试函数**。

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
bash scripts/offline_cpu.sh
```

`scripts/offline_cpu.sh` 依次跑 `pytest`、`scripts/verify_numbers.py`（含 `--selftest`）、
`examples/quickstart.py`、`benchmarks/run_benchmarks.py`。它不调用 `benchmarks/triton/`
或 `benchmarks/tpu/`。那些脚本在没有对应设备时退出，并且不写结果文件。

没装 C++ 扩展时 pytest 是 145 passed + 1 个模块级 skip。装了扩展
（`pip install -e ".[cpp]" --no-build-isolation`）是 167 passed。
不想安装包时，用 `PYTHONPATH=src` 跑同样的四个入口。

`examples/quickstart.py` 会打印设备参数卡、访存合并、分支发散、算子 IR 与代价、融合字节。
打印出来的 TFLOPS 是假设参数，不是吞吐。

## 清单

这些个数由 `scripts/inventory.py` 对源码和 `benchmarks/results/bench.json` 重算。
文档改了而树没改，`verify_numbers.py` 会失败。

| 项 | 个数 |
|---|---:|
| test_functions | 148 |
| pytest_items_with_cpp | 167 |
| pytest_items_without_cpp | 145 |
| pytest_module_skips_without_cpp | 1 |
| cpp_pytest_items | 22 |
| bench_kernel_rows | 9 |
| bench_fusion_rows | 4 |
| bench_smem_rows | 2 |
| bench_bank_rows | 2 |
| bench_reuse_rows | 3 |
| bench_precision_rows | 4 |
| bench_schedule_rows | 8 |
| bench_schedule_verdicts | 2 |
| ci_python_versions | 2 |
| failure_modes | 11 |

## 它是什么 / 不是什么

| 是 | 不是 |
|---|---|
| 一套能在 CPU 上执行的 SIMT 模型（掩码 / 发散 / 事务 / bank 冲突） | 周期级 GPU 模拟器，或一张真实加速卡 |
| 一个 Triton 风格的 DSL → IR → pass → 执行器 / 代价模型 | 一个能在国产卡上跑的算子库 |
| `torch.library` + `torch.compile` 后端的 CPU 样例 | 生产可用的 PyTorch 设备后端 |
| 六个算子与 numpy 逐元素对拍 | 一份 GPU 性能报告 |

静默对不上硬件的地方写在 [docs/FAILURE_MODES.md](docs/FAILURE_MODES.md)。短的边界：

- 默认 `DeviceInfo` 的 TFLOPS / 带宽是假设参数，不是某张卡的规格书；
- 代价模型是理论下界，不含 cache、发射开销、bank 冲突的时间；
- numpy 墙钟是 Python 循环，不能当性能读数；
- `explain()` 在同一原函数的 static / dynamic 两个包装之间会共享 Dynamo cache entry。

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

## 可复算的数字（OBSERVED (CPU)，来自 `benchmarks/run_benchmarks.py` 与 `pytest`）

下面的表是 CPU 上的计数或公式。`predicted_ms` / `µs` 用的是假设参数卡，不是硬件时间。
numpy 墙钟只出现在 `benchmarks/results/bench.md`，不在这里当结果引用。

**设备模型（解析值，虚构参数卡 `generic-domestic-gpgpu`）**

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
交叉点 256 FLOP/B 需要 **N ≥ 1536**——而 n=512 时把 tile 从 16 加到 128，
AI 只从 4.0615 涨到 29.3333，离交叉点差一个数量级。**调 tile 翻不动 bound，
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
存在时跑 1200 组随机/边界输入的逐值对拍。脚本会再打印一次耗时比，这个比不设门槛，也不进入清单。

写这份代码暴露的真问题不是性能，而是**跨语言语义差**：Python 的 `//` 向下取整，
C++ 的 `/` 向零截断，`[-1, -128, -129, 0]` 在 Python 里是 3 个事务段、在朴素的 C++
里是 2 个。`csrc/gkern_cpp.cpp` 里因此手写了 `floor_div`——这是坑 #11。

**混合精度：精度是免费的，但必须显式选**

`gkern.ops.mixed_precision` 把「累加精度」写成 IR 里的显式舍入点（`tl.round_bf16`），
softmax 在四个位置降精度（输入、行最大值、指数结果、归约和）。CPU 上同一个 256×1024 输入：

| 累加精度 | 最大绝对误差 | 访存 (B) | FLOP |
|---|---:|---:|---:|
| fp32 | 5.96e-08 | 4194304 | 2621440 |
| bf16 | 5.87e-03 | 4194304 | 3671040 |

访存字节**完全相同**、FLOP 只多四次舍入——所以「用不用 bf16 累加」是一个纯粹的
数值决策，而不像很多人以为的那样是性能决策。误差大了约 10 万倍这件事，被写成了
`tests/test_mixed_precision.py` 的断言，防止哪天实现偷偷没降精度却还挂着 bf16 的名字。

真实 GPU 上 bf16 存储省不省带宽、误差是不是仍是这个量级，都是 UNKNOWN。
`benchmarks/results/` 里没有 bf16 GPU 结果，也没有 TPU 结果。
`benchmarks/triton/bench_bf16_softmax.py` 与 `benchmarks/tpu/bench_bf16_softmax_jax.py`
在没有对应设备时直接退出，不写文件。

---

**归档的 T4 softmax 日志不是 OBSERVED (CPU)**

`benchmarks/results/triton_softmax_t4.json` 是一份带 provenance 的旧日志。
`benchmarks/analyze_triton.py` 可以拿它重算「日志里的毫秒 / 模型预测」。
离线路径和 CI 都不重跑 GPU，所以真实 GPU 行为在本仓库的复算口径里仍然是 UNKNOWN。
这份日志不能说明什么，写在 `docs/cloud-reconciliation.md`。

---

**工作分解：tail 值钱，但只有把工作切细才治得了它**

矩阵乘在硬件上不是「一次算完」，而是把输出切成 tile 铺到 SM 上排队。100 个 tile
铺 40 个 SM：40 + 40 + 20 —— 最后一波一半 SM 空转，整体利用率 0.83。这个账在
`gkern/schedule/` 里是可复算的整数，不是直觉（数字来自 `benchmarks/results/bench.md`）。
表里的「预测 (ms)」用的是 `DeviceInfo.tesla_t4()` 参数卡，不是 T4 墙钟；真实 GPU 时间是 UNKNOWN：

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
   0.83 提到 1.00。所以模型在这个形状上把它排在前面。这是参数卡上的排序，不是测到的墙钟。
3. **stream-K 在「输出极小、K 极深」时翻盘**：256×256 的输出只有 4 个 tile，
   naive 利用率 0.1（40 个槽里 36 个空着）；stream-K 以 k-iter 为单元把粒度细 512 倍，
   利用率 0.98，且只在 39 个跨 worker 边界落部分和（5.1 MB），比 split-K×16 的
   8.4 MB 还少。
4. **同一套模型给出不同推荐**：`recommend()` 在 1280³ 上选 split-K、在 256×256×8192
   上选 stream-K。这是它有用处的证据——如果两个形状都给同一个答案，这套模型就该删。

「边界部分和」那一列**不是闭式公式算的**：`streamk_units()` 把 3300 / 2048 次 k-迭代
真的切成连续区间（分别 139 / 43 段），再从划分里数出「哪些 tile 被多个 worker 累加」。
闭式 `capacity−1 = 39` 只是上界，这两个形状恰好都等于 39——凑巧相等不算数，
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
PYTHONPATH=src python3 -m pytest              # 167 passed（含 C++ 扩展；未构建扩展时 145 passed + 1 个模块级 skip）
PYTHONPATH=src python3 scripts/verify_numbers.py            # 重算文档里每个数字
PYTHONPATH=src python3 scripts/verify_numbers.py --selftest # 证明这个闸会红
```

仓库现有 **148 个测试函数**（含 C++ 扩展时 pytest 收集 167 条用例；未构建扩展时
145 passed + 1 个模块级 skip）。个数在上面的清单里，由 `scripts/inventory.py` 对源码重算——
用例数会随可选扩展变，函数个数不会，所以清单同时记下这两种收集结果。

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
- [x] 归档一份 T4 softmax 日志（离线路径不重跑；见 `docs/cloud-reconciliation.md`）
- [x] 分块在线 softmax（输入只读一遍，字节账差恰好一整遍输入）
- [x] 调度层：波量化 + naive / split-K / stream-K / persistent 四种分解的账与推荐
- [x] C++ 第二实现（pybind11，跨语言语义差用 `floor_div` 钉住）+ bf16 混合精度
- [ ] 真实国产卡上的算子对拍（等算力到位后，把「模拟」换成「实测」）

## 延伸阅读

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)：三档保真度、IR 为什么是线性指令表、融合不变量的边界、代价模型的两个口径、调度层的波量化与工作分解。
- [docs/jd-mapping.md](docs/jd-mapping.md)：招聘要求 ↔ 本仓库可运行证据逐条对照，**含三个必须承认的缺口**。
- [docs/ecosystem.md](docs/ecosystem.md)：正表 30 行 / 28 个唯一仓库，附表 21 行 / 相对正表新增 19 个，全文 51 行 / 47 个唯一仓库（全部 2026-10-31 用 GitHub API 复核）逐个用 GitHub API 验真，含源码入口级阅读路径与失效路径说明。
- [docs/FAILURE_MODES.md](docs/FAILURE_MODES.md)：CPU 模拟器在哪些地方会不报错地给出一个不像真实 GPU 的数。
- [docs/cloud-reconciliation.md](docs/cloud-reconciliation.md)：一份不由 CI 重跑的 T4 softmax 归档，以及它不能说明什么。UNKNOWN 的格子（bf16 GPU、TPU）保持空白。
- [docs/domestic-card-notes.md](docs/domestic-card-notes.md)：国产卡（昇腾/vllm-ascend）与 CUDA 侧的 attention 源码对照，逐条判定本仓库哪几条建模假设在 NPU 上直接失效。
- [docs/papers.md](docs/papers.md)：8 篇论文的文献地图（在线 softmax / GEMM 分解 / 融合 / 代价模型），每篇都标了「能落成本仓库哪个模块的什么改动」与「我们不做的那部分」。
- [benchmarks/results/bench.md](benchmarks/results/bench.md)：最近一次基准输出（数值墙钟 × 模型预测 × 受限类型 × 融合等式）。

## 许可证

Apache-2.0，见 [LICENSE](LICENSE)。
