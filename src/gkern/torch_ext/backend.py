"""``torch.compile(backend="gkern")``：把 FX 图下沉到 gkern IR 的自定义后端。

这是 JD 里「PyTorch 框架组件适配」那条职责的最小可运行样本：
Dynamo 负责抓图，我们把图**接过来自己编译**，而不是交给 inductor。
不支持的图（含 matmul / reduction / 动态形状）明确回退 eager，并在
``CompileReport`` 里留下原因——宁可慢，不可错。
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.fx as fx
from torch._dynamo import register_backend

from . import fx_lower

__all__ = ["gkern_backend", "compile", "explain", "REPORTS"]

#: 每次编译的结论，按 ``CompileId`` 归档（退化为 ``id(gm)``）。
#: CompileId 由 Dynamo 分配、在每个被编译帧里唯一，因此能把结论精确归属到
#: 「哪个被 compile 的对象」，而不是笼统地取 REPORTS 里的第一条。
REPORTS: dict = {}

#: 写入顺序（key → 递增序号）。不能靠 dict 的插入顺序判「最新」：
#: ``torch._dynamo.reset()`` 之后 CompileId 会从 0/0 重新分配，重编译同一个
#: 帧是**原地覆盖**旧 key，它的插入位置仍停在最前面。
_ORDER: dict = {}
_SEQ = 0


def _report_key(gm: fx.GraphModule):
    """一次编译的稳定 key：优先用 Dynamo 的 CompileId，退化到 id(gm)。"""
    cid = gm.meta.get("dynamo_compile_id")
    return str(cid) if cid is not None else id(gm)


def _record(gm: fx.GraphModule, report: "fx_lower.CompileReport") -> None:
    """归档一次编译结论，并记下它的写入序号（用于「最新一次」的判定）。"""
    global _SEQ
    key = _report_key(gm)
    _SEQ += 1
    _ORDER[key] = _SEQ
    REPORTS[key] = report


def gkern_backend(gm: fx.GraphModule, example_inputs) -> Callable:
    """torch.compile 的后端入口（签名由 Dynamo 规定）。"""
    try:
        runner = fx_lower.try_lower(gm, example_inputs)
    except fx_lower.UnsupportedGraph as exc:
        # 回退 eager，但把「为什么没融合」记下来，别让用户以为编译生效了
        report = fx_lower.CompileReport(fused=False, reason=str(exc),
                                       nodes=len(list(gm.graph.nodes)))
        _record(gm, report)

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
    _record(gm, report)
    runner.__gkern_fused__ = True  # type: ignore[attr-defined]
    runner.__gkern_report__ = report  # type: ignore[attr-defined]
    return runner


def compile(fn: Callable, **kwargs: Any) -> Callable:
    """``torch.compile(fn, backend="gkern")`` 的糖。"""
    return torch.compile(fn, backend="gkern", **kwargs)


def _lookup_report(compiled: Any):
    """从被编译对象反查它自己那次编译的 :class:`CompileReport`。

    走 Dynamo 的 cache entry：对 ``compiled`` 的原始函数的 code object 取
    cache entries，用它们的 ``compile_id`` 去 ``REPORTS`` 里取对应结论。
    同一个 code object 可能有多条 cache entry（不同 guard 特化），所以按写入
    序号取**最近一次**，而不是取列表里的第一条。
    """
    orig = getattr(compiled, "_torchdynamo_orig_callable", None)
    if orig is None:
        orig = getattr(compiled, "_orig_mod", None)
    code = getattr(orig, "__code__", None)
    if code is None:
        return None
    try:
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list
        entries = _debug_get_cache_entry_list(code)
    except Exception:
        return None
    best_key = None
    best_seq = -1
    for entry in entries:
        key = str(getattr(entry, "compile_id", ""))
        if key not in REPORTS:
            continue
        seq = _ORDER.get(key, -1)
        if seq > best_seq:
            best_key, best_seq = key, seq
    return REPORTS[best_key] if best_key is not None else None


def explain(compiled: Any = None) -> dict:
    """取出编译结论：给对象就返回「这个对象」的结论，不给就返回最近一次编译。

    早先的实现无视参数、直接回 ``REPORTS`` 的第一条，导致「问的是 A、答的是
    更早的 B」——结论错配比没有结论更糟。
    """
    if compiled is None:
        if REPORTS:
            latest = max(REPORTS, key=lambda k: _ORDER.get(k, -1))
            return REPORTS[latest].as_dict()
        return {"fused": None, "reason": "尚未编译任何图"}
    report = _lookup_report(compiled)
    if report is not None:
        return report.as_dict()
    return {"fused": None, "reason": "尚未编译任何图"}


register_backend(name="gkern", compiler_fn=gkern_backend, tags=())
