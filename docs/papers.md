# 文献地图：从 online softmax 到 GEMM 分解调度

只收 8 篇，按「能落成 gkern 的哪个模块」组织，不按时间线。每篇的 arXiv 号都抓过
`https://arxiv.org/abs/<id>` 页面、比对页面标题与下表标题 normalized 后一致；
`会议/年份` 一栏只写 abs 页可核实的内容——abs 页没有会议标注时写 arXiv 发布年，
不补写会议名（`NeurIPS 2018` 这种才写，因为 Comments 字段里真有）。

## 1. 在线 softmax / FlashAttention 一脉

### FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- Maxim Milakov et al. · arXiv 2022（abs 页未标注会议）
- arXiv: `2205.14135` —— 已验，页面标题逐字一致
- 解决什么：把注意力重写成「按 K/V 分块流式扫描 + 块内重缩放」，使 HBM 读写次数跟着
  分块策略走而不是跟着 N×N 注意力矩阵走；并给出 IO 复杂度的下界论证。
- 落点：`src/gkern/ops/online_softmax.py` 里 `online_softmax_rows` 的块内重缩放已经是
  同一套数学，缺的是把它从「逐行 softmax」推成「两输入（Q,K）扫描 + 第三份输出累加器」；
  具体是在该文件同目录新增 `attention_online.py`，复用 `online_softmax_rows` 的
  `m/s` 修正因子结构，把行内块循环换成 K 维块循环。
- 不做：不实现 backward、不做 block-sparse、不引入真实 SRAM 容量约束（本仓库 IR 无
  片上缓存层级，只有字节账）。

### Online normalizer calculation for softmax
- Maxim Milakov & Natalia Gimelshein · arXiv 2018（abs 页未标注会议）
- arXiv: `1805.02867` —— 已验，页面标题逐字一致
- 解决什么：给出 softmax 单遍计算的递推式（运行最大值 + 按新基准重缩放累加和），
  把「先求 max 再求 sum 再算 exp」的两遍读，压成一遍读；同时说明与 Top-K 融合的收益。
- 落点：这是 `src/gkern/ops/online_softmax.py` 的**直接出处**——`online_softmax_rows`
  里的 `s = s * exp(m_old - m_new) + ...` 就是本文的递推式。可落的具体改动是补一条
  断言：`compare_online_vs_recompute` 除比数值一致外，再比 `costmodel.analyze` 给出的
  `bytes_total` 差值恰好等于一次输入读（当前只比结果数组）。
- 不做：不追 Top-K 融合那条线（本仓库没有 topk 算子），不讨论数值精度上界。

## 2. GEMM 的工作分解与负载均衡

### Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU
- Muhammad Osama et al. · arXiv 2023（abs 页注明内容曾收于作者博士论文）
- arXiv: `2301.03598` —— 已验，页面标题逐字一致
- 解决什么：不再按输出 tile 切工作，而是把**聚合内层迭代次数**按执行单元均分，
  于是「输出 tile 数除不尽 SM 数」造成的尾巴消失；代价是跨 tile 边界要落部分和。
- 落点：`src/gkern/schedule/gemm_decomp.py` 中 `compare_strategies` 的 `"stream-k"`
  分支——现在只算了单元数与额外字节，缺「每个单元覆盖哪些 (tile, k-range)」的划分函数；
  应在同文件补 `streamk_units(shape, capacity)`，返回 `[(tile_id, k_start, k_end)]`，
  由 `wave.py:wave_stats` 直接吃它算波量化。
- 不做：不做真实硬件测量（本仓库只做代价模型预测），不做多卡切分。

### Stream-K++: Adaptive GPU GEMM Kernel Scheduling and Selection using Bloom Filters
- Harisankar Sadasivan et al. · arXiv 2024（v2 2025；abs 页未标注会议）
- arXiv: `2408.11417` —— 已验，页面标题逐字一致
- 解决什么：把 Stream-K 的调度策略从三种扩到七种，并用 Bloom filter 快速剔除明显
  不合适的配置，使「选哪个调度」本身变成可摊薄的在线过程而不是离线穷举。
- 落点：`src/gkern/schedule/gemm_decomp.py:recommend()` 现在是全枚举后取最优；
  可加一层「先用 `costmodel.analyze` 的 `bound` 做廉价相斥剪枝」的前置过滤，
  把 Bloom filter 的角色换成「代价模型下界筛」，语义等价但可解释。`StrategyRow`
  需要多带一个 `pruned_by` 字段记录被哪条下界筛掉。
- 不做：不实现 Bloom filter 数据结构本身（本仓库没有候选集合大到需要它的场景）。

### Stream-K Optimization and Exploration
- Nick Rackley et al. · arXiv 2024（abs 页未标注会议）
- arXiv: `2407.00044` —— 已验，页面标题逐字一致
- 解决什么：把 Stream-K 的理论划分与实际实现对账，指出 padding（把 M/N/K 补齐到
  整 tile）会改变实际的工作单元数与负载分布，并讨论用分块时间预测改进行为。
- 落点：`src/gkern/schedule/gemm_decomp.py:GemmShape` 目前按 `ceil` 算 tile 数，
  隐含「补零的 tile 与满 tile 同价」；应补 `padded_tiles` 与 `useful_fraction` 两个
  字段，并让 `wave_stats` 在算效率时按有用工作量而非单元数计。这是**对现有结论的
  修正**，可能改变 `tests/test_schedule.py` 里 naive 利用率那条断言的口径。
- 不做：不复现文中那个未解决的 block mapping bug，不做实验性参数扫描。

## 3. 算子融合 / 图级优化

### DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion
- Wei Niu et al. · arXiv 2021（abs 页只给了 ACM DOI，未写会议名）
- arXiv: `2108.13342` —— 已验，页面标题逐字一致
- 解决什么：把融合从「手写 pattern 匹配」升级成「算子分类学 + 组合分类学」，
  再用数学性质重写图以扩大可融合面，最后靠轻量 profiling 在候选融合方案里选。
- 落点：`src/gkern/compiler/passes.py:can_fuse` 的 7 条判据现在是**纯结构**判据
  （地址同构、写入次数、掩码一致……），拒绝理由里没有「融了也不划算」这一档；
  可按本文的分类学给 `Tile` 打 `elementwise / reduction / broadcast` 标签，
  在第 7 条之后补一条**收益判据**，让 `can_fuse` 返回 `(False, "可融但无收益")`。
- 不做：不做移动端部署与模型级 benchmark，不做多算子（>2）链式融合。

### TVM: An Automated End-to-End Optimizing Compiler for Deep Learning
- Tianqi Chen et al. · arXiv 2018（abs 页未标注会议）
- arXiv: `1802.04799` —— 已验，页面标题逐字一致
- 解决什么：把「图级算子融合」与「算子级 schedule 搜索」放进同一套 IR 与同一套
  自动调优回路里，使融合判据和分块选择不再分属两个不能互相询问的系统。
- 落点：`src/gkern/compiler/ir.py:Program` 已是线性 IR，缺的是「图」这一层；
  最小改动是在 `passes.fuse_programs` 之外补一个 `fuse_chain(programs, buffers)`
  按拓扑序贪心串融合，并把 `autotune` 的配置空间从「单 kernel 分块」扩到
  「融合 or 不融合」这一维（`TuneResult.config` 加 `fused: bool`）。
- 不做：不移植 TVM 的 schedule 原语与 Ansor 式搜索空间，不做多后端 codegen。

## 4. 代价模型与 autotuning

### Learning to Optimize Tensor Programs
- Tianqi Chen et al. · NeurIPS 2018（abs 页 Comments 字段标注）
- arXiv: `1805.08166` —— 已验，页面标题逐字一致
- 解决什么：用学出来的代价模型（而不是实测）给候选程序变体排序，把「枚举 + 上机测」
  换成「枚举 + 预测」，并支持跨 workload 的模型迁移以减少冷启动成本。
- 落点：`src/gkern/compiler/passes.py:autotune` 现在是**全枚举 + 解析代价模型**
  （`costmodel.analyze`），候选多时是纯线性开销；可加 `--prune` 式的两段式：
  先用 `costmodel.reuse_bound` 粗排、再对前 k 个做完整 `analyze`，`TuneResult.trials`
  保留被粗排淘汰的行并标 `stage` 字段，保证被淘汰的候选也可审计。
- 不做：不训练任何模型（本仓库没有实测数据，学出来的模型无处校准），不做迁移学习。

## 优先级

| 论文 | 落点 | 估计工作量 | 增量 |
|---|---|---|---|
| Stream-K | `schedule/gemm_decomp.py:streamk_units` | 中（新函数 + 对接 `wave_stats`） | 高：当前正在写的模块缺的就是划分函数 |
| Online normalizer | `ops/online_softmax.py` + `costmodel` 字节账断言 | 小（补断言与文档） | 中：把已实现的东西钉死，防止回归 |
| FlashAttention | 新建 `ops/attention_online.py` | 大（新算子 + 两输入扫描） | 高：把 online softmax 从玩具变成可用结构 |
| Stream-K Optimization | `GemmShape.padded_tiles` / `useful_fraction` | 中（改口径，会动既有断言） | 中：修正当前隐含的「补齐 tile 免费」假设 |
| Stream-K++ | `recommend()` 前置下界剪枝 | 中（改 `StrategyRow` schema） | 低：8 篇里候选规模最小，收益有限 |
| DNNFusion | `compiler/passes.py:can_fuse` 第 8 条收益判据 | 中（要给 Tile 打标签） | 中：让融合从「能不能」变成「值不值」 |
| TVM | `passes.fuse_chain` + `autotune` 加 `fused` 维 | 大（引入图这一层） | 中：是架构性改动，不急 |
| Learning to Optimize | `passes.autotune` 两段式剪枝 | 小（加 stage 字段与粗排） | 低：当前候选规模还不构成瓶颈 |

**先做这两篇**：

1. **Stream-K（2301.03598）**——`schedule/` 正在写，缺的恰好是「单元→(tile, k-range)」
   这一步映射；`wave_stats` 与 `compare_strategies` 都已就位，补一个纯函数即可闭环，
   且能直接让 `tests/test_schedule.py` 的四种分解断言有更细的口径。
2. **Online normalizer（1805.02867）**——`online_softmax_rows` 已经是它的实现，
   现在只差把字节账断言钉进测试；成本最小、又能立刻保护一块刚补完的代码。
