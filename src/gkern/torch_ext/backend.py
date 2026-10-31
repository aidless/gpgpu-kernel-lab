"""``torch.compile(backend="gkern")``：把 FX 图下沉到 gkern IR 的自定义后端。

这是 JD 里「PyTorch 框架组件适配」那条职责的最小可运行样本：
Dynamo 负责抓图，我们把图**接过来自己编译**，而不是交给 inductor。
不支持的图（含 matmul / reduction / 动态形状）明确回退 eager，并在
``CompileReport`` 里留下原因——宁可慢，不可错。
"""

from __future__ import annotations

from typing import Any, Callable, List

import torch
import torch.fx as fx
from torch._dynamo import register_backend

from . import fx_lower

__all__ = ["gkern_backend", "compile", "explain", "REPORTS"]

#: 每次编译的结论，key 是图的 id，value 是 :class:`fx_lower.CompileReport`
REPORTS: dict = {}


def gkern_backend(gm: fx.GraphModule, example_inputs) -> Callable:
    """torch.compile 的后端入口（签名由 Dynamo 规定）。"""
    try:
        runner = fx_lower.try_lower(gm, example_inputs)
    except fx_lower.UnsupportedGraph as exc:
        # 回退 eager，但把「为什么没融合」记下来，别让用户以为编译生效了
        report = fx_lower.CompileReport(fused=False, reason=str(exc),
                                       nodes=len(list(gm.graph.nodes)))
        REPORTS[id(gm)] = report

        def fallback(*args, **kwargs):
            return gm.forward(*args, **kwargs)

        fallback.__gkern_fused__ = False  # type: ignore[attr-defined]
        return fallback

    report = fx_lower.CompileReport(
        fused=True,
        reason="纯逐元素图，压成单个 kernel",
        nodes=len(list(gm.graph.nodes)),
        matched=[str(n.target) for n in gm.graph.nodes if n.op == "call_function"],
    )
    REPORTS[id(gm)] = report
    runner.__gkern_fused__ = True  # type: ignore[attr-defined]
    runner.__gkern_report__ = report  # type: ignore[attr-defined]
    return runner


def compile(fn: Callable, **kwargs: Any) -> Callable:
    """``torch.compile(fn, backend="gkern")`` 的糖。"""
    return torch.compile(fn, backend="gkern", **kwargs)


def explain(compiled: Any) -> dict:
    """尽力取出某个被编译对象的编译结论（取不到就说明还没编译过）。"""
    for report in REPORTS.values():
        return report.as_dict()
    return {"fused": None, "reason": "尚未编译任何图"}


register_backend(name="gkern", compiler_fn=gkern_backend, tags=())
