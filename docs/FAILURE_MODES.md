# FAILURE_MODES

CPU 模拟器在下面这些点上会**正常返回一个数字，并且不抛异常**。那个数字回答的是
模型自己的假设，不是真实 GPU。把它写成「卡上测到的」就是静默失真。

每一节的名字对应 `tests/test_failure_modes.py` 里的 `MODES` 和一条 `test_*`。
测试钉住的是「它给出这个数、不报错」，不是「这个数等于某张卡」。

离线路径 `scripts/offline_cpu.sh` 不补这些空白。`benchmarks/results/` 里没有
bf16 GPU JSON，也没有 TPU JSON。`triton_softmax_t4.json` 是一份归档日志，
CI 不重跑它，不能把它算进 OBSERVED (CPU)。

## sector-128

`count_transactions` 把地址折进固定的 128 字节段，函数没有段长参数。
32 个连续的 float32 因此永远是 1 个事务。段长 32 字节的硬件上，同一组地址是 4 段。
调用方拿不到这个差别：返回值就是 1。

## banks-32x4

默认共享内存是 32 个 bank、每个 4 字节。同一组下标在 `bank_count=64` 时冲突轮次不同，
但 `warp_bank_conflicts(indices)` 不看设备，直接用 32×4。列读演示
（行宽 32 → 992）也是这个默认几何，不是从某张卡读出来的。

## compiler-wave-32

编译器执行器里的 `SharedMemory` 不接收 `DeviceInfo.warp_size`，冲突按 32 个 lane 切。
64 个连续 float 的一次共享内存读写因此报 0 冲突。wave64 上同一次访问每个 bank 会被碰到两次
（额外 64 个槽）。SIMT 启动器 `launch()` 会把设备的 `warp_size` 传进去；编译器这条路径不会。

## roofline-parameter-card

`predicted_ms = max(字节 / 带宽, FLOP / 峰值)`。带宽和峰值来自调用方手里的 `DeviceInfo`。
默认卡是虚构的 `generic-domestic-gpgpu`（26.2144 TFLOPS、102.4 GB/s 都是假设参数）。
同一份 IR 换上 `tesla_t4()` 参数，字节不变、预测时间变，两边都不报错。
这个时间不是墙钟，也不是那张卡的测量。

## smem-time-unchanged

共享内存分块不改变模型里的全局字节，于是 `predicted_ms` 也不变。
真实 GPU 上，把面板放进片上存储通常会改变耗时。模型给不出这个差，它只重复同一个下界。

## persistent-equals-naive

persistent 和 naive 的单元数、波次、`predicted_ms` 相同。模型里没有发射开销，
所以「常驻块省掉重复 launch」这件事不会出现在数字里。用这张表否定 persistent 的墙钟收益，是失真。

## bf16-storage-invisible

`round_bf16` 把数值舍入后再存成 float32。两条 softmax 的访存字节因此相同。
真实 GPU 上把输入输出改成 bf16 存储会改变搬运的字节。这个模型算不出那一档，
树里也没有对应的 GPU 结果文件。误差变大是 CPU 上的数值事实，不是带宽事实。

## occupancy-ignores-registers

`Device.occupancy` 只做 `block 内 warp 数 / max_warps_per_sm`，上限 1。
把 `registers_per_sm` 从 64K 改成 32，占用率不变。寄存器、共享内存容量导致的
驻留下降，模型看不见，但仍然返回一个 0 到 1 之间的数。

## padded-tile-bytes

`BN` 不整除 `N` 时，代价模型按整块 tile 计字节，不把被掩掉的 lane 扣掉。
函数照样返回字节数，而且省下的字节不再等于 `2×|C|`。拿这个数去对硬件 profiler 的
DRAM 计数，会对不上，并且没有任何异常。

## python-elapsed-is-not-device-time

`LaunchResult.elapsed_ms` 是 Python 里逐 warp 循环的墙钟。它没有 `gpu_ms` 字段。
同一个向量加，这份墙钟比默认参数卡的 roofline 大两个数量级以上。
`benchmarks/results/bench.md` 的「numpy 墙钟」列是同一类数：OBSERVED (CPU)，随机器变，不是设备时间。

## aggregate-warp-size

这是已经修上的一个静默错误，测试防止它回来。`DivergenceReport.efficiency` 曾经用
`instructions * 32` 当总槽数。warp 宽 64、一条指令、64 个 lane 都活跃时，效率会算成 2，
而不是报错。现在总槽数来自每条轨迹自己的 `warp_size`。这只让模拟器内部一致；
它仍然不是硬件周期。
