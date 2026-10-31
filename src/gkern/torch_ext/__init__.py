"""PyTorch 适配层：把 gkern 接到框架上。

三个层次的入口：

- :mod:`gkern.torch_ext.custom_ops` —— ``torch.library`` 自定义算子（实现 + fake + autograd + schema）
- :mod:`gkern.torch_ext.fx_lower` —— FX 图 → gkern IR 的下沉器（接不住必须给原因）
- :mod:`gkern.torch_ext.backend` —— ``torch.compile(backend="gkern")`` 的自定义后端

注意：``torch`` 是本包的运行时依赖（在 ``pyproject.toml`` 里声明），但为了让
``import gkern`` 在没装 torch 的环境里也能用（只想跑 SIMT 模型/编译器部分的人），
这里**不**在模块顶部 import torch —— 各子模块自己按需 import。
"""

from __future__ import annotations

__all__ = ["custom_ops", "fx_lower", "backend", "CompileReport", "register_backend", "compile"]


def __getattr__(name: str):
    """惰性再导出：访问 ``gkern.torch_ext.register_backend`` 时才真正 import torch。"""
    if name in ("CompileReport", "try_lower", "UnsupportedGraph"):
        from . import fx_lower

        return getattr(fx_lower, name)
    if name in ("register_backend", "compile", "explain", "gkern_backend", "REPORTS"):
        from . import backend

        return getattr(backend, name)
    if name in ("custom_ops", "fx_lower", "backend"):
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
