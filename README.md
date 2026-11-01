# gpgpu-kernel-lab

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
pip install -e ".[dev]"          # numpy + torch + pytest
pytest                            # 92 个测试，约 11 秒
PYTHONPATH=src python3 examples/quickstart.py
PYTHONPATH=src python3 benchmarks/run_benchmarks.py
```

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

## 开发中实际踩到并修掉的八个坑

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

---

## 测试与自检闸

```bash
pytest                                    # 92 passed
PYTHONPATH=src python3 scripts/verify_numbers.py            # 重算文档里每个数字
PYTHONPATH=src python3 scripts/verify_numbers.py --selftest # 证明这个闸会红
```

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
- [ ] 共享内存分块 GEMM（把 matmul 的算术强度真正推到计算受限区）
- [ ] `PrivateUse1` 设备后端骨架（`torch.device("gkern")`）
- [ ] 真实国产卡上的算子对拍（等算力到位后，把「模拟」换成「实测」）

## 延伸阅读

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)：三档保真度、IR 为什么是线性指令表、融合不变量的边界、三个扩展点。
- [docs/jd-mapping.md](docs/jd-mapping.md)：招聘要求 ↔ 本仓库可运行证据逐条对照，**含三个必须承认的缺口**。
- [docs/ecosystem.md](docs/ecosystem.md)：正表 30 行 / 28 个唯一仓库，附表 21 行 / 相对正表新增 19 个，全文 51 行 / 47 个唯一仓库（全部 2026-10-31 用 GitHub API 复核）逐个用 GitHub API 验真，含源码入口级阅读路径与失效路径说明。
- [benchmarks/results/bench.md](benchmarks/results/bench.md)：最近一次基准输出（数值墙钟 × 模型预测 × 受限类型 × 融合等式）。

## 许可证

Apache-2.0，见 [LICENSE](LICENSE)。
