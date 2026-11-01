"""扩展构建：C++ 参考实现 `gkern_cpp`。

它是可选件——`pip install -e .` 不构建，`pip install -e ".[cpp]"` 才构建。
缺 pybind11 时这里只打印一条警告，不阻断纯 Python 安装（教学模型本身不需要它）；
需要「C++ 必须存在」的场合由 `scripts/verify_cpp.py` 当硬门禁。
"""

from __future__ import annotations

import sys

from setuptools import setup

ext_modules = []
cmdclass = {}

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    ext_modules = [
        Pybind11Extension(
            "gkern_cpp",
            ["csrc/gkern_cpp.cpp"],
            cxx_std=17,
        ),
    ]
    cmdclass = {"build_ext": build_ext}
except ImportError:  # pragma: no cover - 只影响构建环境
    print(
        "warning: 未安装 pybind11，跳过 gkern_cpp 扩展构建（纯 Python 安装继续）。"
        ' 需要它请用 pip install -e ".[cpp]"',
        file=sys.stderr,
    )

setup(ext_modules=ext_modules, cmdclass=cmdclass)
