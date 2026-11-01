# 架构与设计取舍

这份文档解释「为什么这么写」——比 README 更深一层，专给要改代码的人看。

## 1. 三档保真度，混用会骗自己

本仓库刻意把「模型」和「实测」分开放在不同层，因为它们量的是**不同的东西**：

| 层 | 文件 | 量的是什么 | 可信边界 |
|---|---|---|---|
| 解析代价模型 | `compiler/costmodel.py` | 由 IR 的 shape/dtype 推访存字节与 FLOP，套 roofline 判 memory/compute 受限 | 是不含 cache 命中、发射开销、真实 bank 冲突代价的**理论下界** |
| SIMT 事务模型 | `gpgpu/memory.py`、`gpgpu/warp.py` | 128B 段事务数、合并效率、lane 效率、bank 冲突度 | 是**计数模型**，不是时序仿真；不预测毫秒 |
| numpy 执行器 | `compiler/codegen.py` | 真的把结果算出来 | 逐 block 走 Python 循环，墙钟**比设备慢 2–3 个数量级**，不能当性能读数 |

**唯一的硬规矩**：任何写进文档/测试的数字，必须标明它来自哪一层。把 numpy 墙钟说成
「这个算子要 40ms」，或者把 roofline 估计说成「实测 TFLOPS」，都是造假。

## 2. IR 为什么是「线性指令表」而不是带循环的图

`Program.instrs` 是一维列表，`Tile.id == instrs 里的下标`。这个决定带来两个后果：

- **好处**：pass 极简（融合 = 删两条 store/load + 改 id 引用），执行器极简（按序 for 循环），
  代价模型极简（扫一遍列表累加）。非常适合「教学 + 可复算」的定位。
- **代价**：**没有循环结构**，所以 `matmul` 的 K 循环必须被 `trace` 展开成 `K/BK` 组指令，
  因此 `K` 必须是 `constexpr`（`ops/matmul.py` 在 `K % BK != 0` 时直接 `raise ValueError`）。
  这也意味着大 K 会让 IR 长度线性增长——这是刻意接受的取舍，不是疏漏。

真实 Triton 用有环的 IR（`scf.for`）来避免展开，代价是 pass 与 codegen 复杂度高一个量级。
本仓库选择「能读懂、能复算」优先。

因此 `Tile.id == 下标` 是一条**不变量**：任何往 `instrs` 中间插入/删除节点的 pass 都必须
重编号。目前唯一做这件事的是 `passes.fuse_programs`，它采用「重建新 Program」而不是就地改，
从结构上避免了重编号 bug。

## 3. 一个 warp 一次调用，tid 是长度 32 的数组

`WarpContext.tid` 不是标量而是 `np.arange(32)`。这是本仓库最关键的实现技巧：

- 如果把 kernel 写成「逐 lane 标量 Python 函数」，1024 元素 × 每 lane 一次调用 = 慢到跑不动；
- 如果写成「整张量 numpy」，就丢掉 SIMT 语义（掩码、发散、事务都对不上）；
- 折中：**内核函数一次处理 32 个 lane**，用 numpy 向量化算这 32 个，但语义上仍是
  「一个 warp 的 32 个 lane 一起执行」。掩码是长度 32 的 bool 数组，`branch()` 分两次调用
  then/else 各自带掩码——这正好是 GPU 上分支发散的真实执行方式。

代价是：`ctx.mask` 必须是**唯一真相**（`warp.mask` 的视图）。开发时踩过的最贵的坑就是
存了两份（`_mask` 与 `warp.mask`），分支里改了后者、`load/store` 读了前者，
导致 ReLU 输出全 0。见 README「八个坑」第 1 条。

## 4. 融合的不变量：省下的字节 == 2×|C|

`passes.fuse_programs(a, b, buffer)` 的前提由 `can_fuse` 七条判据守住。这套判据是
**保守的一对一、逐 lane 对齐**融合，不是一个通用安全闸：

1. `a.grid == b.grid`（同一批 block 一一对应，否则映射不成立）；
2. `a` 对 `buffer` **恰好写一次**（多次写时融合只保留最后一次，前一次会被静默丢弃）；
3. `a` **不读** `buffer`（就地 read-modify-write 时融合会 `pop` 掉输入 → `KeyError`）；
4. `b` 对 `buffer` **恰好读一次**（多次读时无法一一替换）；
5. `a` 写 `buffer` 的地址表达式与 `b` 读它的地址表达式**结构相同**（逐 lane 对齐；
   否则 halo/stencil 会被融合成一条旋转，产生**静默的错误数值**）；
6. 写入值的 `shape` 与 `dtype` 与读到的 tile 相同；
7. `b` 对读到的值只做**逐 lane** 使用——掩码与生产者一致，下游任何 Tile 的 shape
   不得变化（`sum`/`max`/`dot`/跨 lane 广播会改变 shape，被掩掉的 lane 会污染结果）。

判据 5 的实现是 `_sig()`：把 Tile 及其依赖子图归一成结构签名再比。两个 program
各自 trace 出来的指令 id 不同，但 `pid*N + arange(...)` 的结构相同，必须判等。

这七条每条都有对应的拒绝用例（`tests/test_compiler.py` 的七个
`test_fusion_gate_rejects_*`），并且验证过「拿旧版 `can_fuse` 跑这组测试必红」。

融合后省掉的访存恰好是中间缓冲区的一次写 + 一次读 = `2 × |C|`。
**但这只在分块整除 M/N 时严格成立**——代价模型按「整块 tile 面积」计费，不扣尾部被 mask
的 lane，所以 `BN ∤ N` 时它偏大。这条边界在 `tests/test_ops.py::test_fusion_byte_is_static_tile_estimate`
和 `benchmarks/results/bench.md` 的尾块对照行里都钉死了。**不要**把 `2×|C|` 当恒定公式四处引用。

## 5. PyTorch 适配：三层，逐层收缩支持面

```
torch.library 自定义算子 (custom_ops.py)   ← 最稳：单算子，autograd 齐备，任意图都能用
        ↑
torch.compile 自定义后端 (backend.py + fx_lower.py)  ← 最激进：整图融合，支持面最窄
        ↑
（未来）PrivateUse1 设备后端              ← 让 torch.device("gkern") 成为一等公民
```

`fx_lower.try_lower` 的设计原则：**接不住就抛 `UnsupportedGraph` 并写清原因**，
`backend.py` 捕获后包一层 `fallback` 调 `gm.forward`，并在 `CompileReport` 里记 `fused=False`
与原因。当前明确拒绝的情形：`requires_grad`（numpy 会断 autograd）、非 float32、
非 **CPU** 张量（runner 经 numpy 计算，meta/CUDA 张量必须先拒，否则运行期抛的是
与「下沉失败」无关的 `TypeError`）、**符号/动态形状**（`numel`/`grid` 需编译期常量，
否则抛 data-dependent guard 错误、Dynamo 侧无法按回退处理）、输入形状不一致、
非连续、非 `call_function` 节点（如 `x.sum()` 是 `call_method`）、未知算子、多输出、
以及 `alpha=` / `approximate=` 这类**非默认关键字参数**（硬接等于悄悄算错）。

`torch.library` 自定义算子那一层同样守住「编译期 == 运行期」：真实实现走 numpy
只能算 float32，所以 `register_fake` 也**必须**拒绝非 float32——否则 fake 放行
float64、运行期悄悄降精度，正是 `torch.compile` 最典型的 dtype 事故。
`fused_mlp` 的 `K % BK == 0`（IR 无循环，K 是编译期常量）与形状约束在 real/fake
两端共用同一份校验，不允许「编译期过、运行期炸」。

「静默给错结果」比「明确回退」危险得多，所以这里宁可回退得多。

## 6. 怎么扩展

- **加一个算子**：在 `compiler/dsl.py` 的 `_Tracer` 加方法 + `codegen.py` 的 op 分派加分支 +
  `ops/` 里写三行 kernel + reference 实现对拍 + `tests/test_ops.py` 补正常与非法输入两组用例。
- **加一个 pass**：函数签名 `pass(program) -> Program`，**返回新 Program**（克隆后改），
  不要就地改 `instrs`。补一条「能红」的测试（构造该融合不成立的输入，断言报错或原样返回）。
- **加一个设备**：`gpgpu/device.py` 里加 `DeviceInfo` 预设（改 SM 数/寄存器/带宽参数），
  costmodel 与 SIMT 模型自动跟随，无需改其它代码。
- **接真实卡**：`torch_ext/` 下照 `custom_ops.py` 的「实现 + fake + autograd + schema」四件套
  把 numpy 实现换成真实 kernel 调用即可，上层接口不变。

## 7. 明确的非目标

- 不做周期级 GPU 时序仿真（那不是「实验台」，那是另一个 gem5/gpgpusim 项目）。
- 不追求与 Triton 的 API 兼容（只借用它的心智模型：DSL → IR → pass → codegen）。
- 不在本机伪造 GPU 实测数字。
