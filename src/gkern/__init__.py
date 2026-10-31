"""gkern：一个可以在纯 CPU 上跑通的 GPGPU 算子/编译/框架适配实验台。

三个子包：
- :mod:`gkern.gpgpu`     SIMT 执行模型（warp 发散、访存事务、bank 冲突）
- :mod:`gkern.compiler`  Triton 风格 DSL → IR → 融合/调优 pass → 执行与代价分析
- :mod:`gkern.ops`       真实算子（elementwise / softmax / layernorm / matmul / 融合 MLP）

为什么是「模拟」
----------------
本仓库的开发机没有 GPU。与其写一堆跑不起来的 CUDA 代码假装在做 GPU 工程，
不如把 GPU 上**真正决定性能的那几个机制**（warp 掩码、分支发散、访存合并、
共享内存 bank、roofline、算子融合、分块调优）做成可执行、可测量、可测试的模型，
再用同一套 IR 对接 PyTorch 的扩展机制（见 :mod:`gkern.torch_ext`）。

所有数字都能从本仓库的代码重算，没有任何「引用自某处」的估值。
"""

from . import compiler, gpgpu, ops

__version__ = "0.1.0"

__all__ = ["compiler", "gpgpu", "ops", "__version__"]
