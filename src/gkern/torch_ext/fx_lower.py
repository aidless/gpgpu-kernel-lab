"""把 FX 图下沉成 gkern DSL 的 IR（目前只支持「纯逐元素图」）。

这是 ``torch.compile(backend="gkern")`` 的核心：Dynamo 抓到一张 FX 图之后，
我们判断它是不是一串逐元素算子（add/mul/gelu/sigmoid/...）作用在同样形状的
输入上；如果是，就把整条链压成**一个** gkern kernel，中间结果不落全局内存；
如果有任何节点不认识，就明确回退到 eager（并在 ``CompileReport`` 里记下原因），
绝不静默给错结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.fx as fx

from ..compiler import codegen, costmodel
from ..compiler.ir import Program

_UNARY = {
    "relu": "relu",
    "gelu": "gelu",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "exp": "exp",
    "neg": "neg",
    "sqrt": "sqrt",
    "rsqrt": "rsqrt",
}

_BINARY = {"add": "add", "sub": "sub", "mul": "mul", "div": "div"}


class UnsupportedGraph(RuntimeError):
    """这张图 gkern 接不住。

    下沉失败**必须给出原因**（而不是悄悄回 None）：回退到 eager 是允许的，
    但用户有权知道为什么没融合——这条对应 JD 里「框架适配要能说清行为差异」。
    """


@dataclass
class CompileReport:
    """一次编译的结论（用于 benchmark 与测试断言）。"""

    fused: bool
    reason: str = ""
    nodes: int = 0
    matched: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"fused": self.fused, "reason": self.reason,
                "nodes": self.nodes, "matched": self.matched}


def _op_name(node: fx.Node) -> str:
    """把 FX 节点的 target 归一成裸算子名。

    三种形态都要处理，否则「明明认识 gelu 却说 unsupported」：
    - ``symbolic_trace`` 抓到的是 builtin 函数：``<built-in function gelu>`` → ``__name__``
    - Dynamo 抓到的是 aten OpOverload：``aten::gelu`` → ``_schema.name`` 的后半段
    - 少数情况是 ``aten.gelu.default`` 这样的字符串
    """
    target = node.target
    schema = getattr(target, "_schema", None)
    if schema is not None and hasattr(schema, "name"):
        return str(schema.name).split("::")[-1]
    name = getattr(target, "__name__", None)
    if isinstance(name, str) and name:
        return name.split(".")[0]
    text = str(target)
    if text.startswith("aten."):
        return text.split(".")[1]
    return text


def _is_symbolic(dim: Any) -> bool:
    """维度是不是符号值（``torch.SymInt``）。"""
    return isinstance(dim, torch.SymInt)


def try_lower(gm: fx.GraphModule, example_inputs) -> Callable:
    """成功返回可调用的融合实现；接不住就抛 :class:`UnsupportedGraph` 说明原因。"""
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    if not placeholders:
        raise UnsupportedGraph("图里没有输入占位符")
    tensors = [a for a in example_inputs if isinstance(a, torch.Tensor)]

    # Dynamo 在 dynamic=True 时会把形状符号作为额外占位符塞进图里（SymInt），
    # 此时「张量数 != 占位符数」。如果只说数量对不上，用户根本看不出是动态
    # 形状导致的 —— 所以先点名 SymInt，再说数量。
    sym_placeholders = sorted(
        str(a) for a in example_inputs
        if not isinstance(a, torch.Tensor) and isinstance(a, torch.SymInt))
    if sym_placeholders:
        raise UnsupportedGraph(
            f"只支持静态形状，图里含符号维度占位符 {sym_placeholders}")

    if len(tensors) != len(placeholders):
        raise UnsupportedGraph(
            f"张量输入数 {len(tensors)} 与占位符数 {len(placeholders)} 不一致")

    # 需要 autograd 的图不能下沉：gkern runner 用 numpy 算，返回的张量没有
    # grad_fn，硬接会把反向传播悄悄断掉（比回退 eager 危险得多）。
    if any(t.requires_grad for t in tensors):
        raise UnsupportedGraph("输入 requires_grad，gkern 只有前向，无法提供 autograd")
    if any(t.dtype != torch.float32 for t in tensors):
        raise UnsupportedGraph(
            f"只支持 float32，收到 {sorted({str(t.dtype) for t in tensors})}")

    # 非 CPU 设备必须先拒：runner 走 numpy，meta 张量根本没有数据，
    # 硬接会在 runner 里抛出一个和「下沉失败」无关的 TypeError。
    off_device = sorted({str(t.device.type) for t in tensors if t.device.type != "cpu"})
    if off_device:
        raise UnsupportedGraph(
            f"只支持 CPU 张量（runner 经 numpy 计算），收到设备 {off_device}")

    # 动态/符号形状回退：numel 与 grid 都要在编译期算成常量，符号维度算不出来；
    # 不做显式判断就会在 _build_runner 里抛 GuardOnDataDependentSymNode，
    # 那是个「下沉失败」之外的错误类型，Dynamo 侧无法按回退处理。
    symbolic = sorted({str(d) for t in tensors for d in t.shape if _is_symbolic(d)})
    if symbolic:
        raise UnsupportedGraph(
            f"只支持静态形状，收到符号维度 {symbolic}（numel/grid 需编译期常量）")

    shape = tuple(tensors[0].shape)
    if any(tuple(t.shape) != shape for t in tensors):
        raise UnsupportedGraph("输入形状不一致，逐元素融合要求同形状")
    if any(not t.is_contiguous() for t in tensors):
        raise UnsupportedGraph("输入非连续，先 .contiguous() 再下沉")

    # 只接受「整张图都是逐元素」的情况：出现 matmul/reduction 等就回退
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        if node.op != "call_function":
            raise UnsupportedGraph(f"不支持的节点类型 {node.op}: {node.target}")
        name = _op_name(node)
        if name not in _UNARY and name not in _BINARY:
            raise UnsupportedGraph(f"不支持的算子 {name}")
        # 只实现「默认参数」版本：alpha=、approximate=、rounding_mode= 这类
        # 关键字参数会改变语义，硬接就等于悄悄算错，所以一律回退。
        if node.kwargs or len(node.args) != (1 if name in _UNARY else 2):
            raise UnsupportedGraph(f"不支持 {name} 的非默认参数")

    return _build_runner(gm, tensors, shape)


def _unwrap_single_output(value: Any) -> tuple[fx.Node, Callable[[torch.Tensor], Any]]:
    """把图输出拆成「单个张量 + 恢复原结构的包装函数」。"""
    if isinstance(value, fx.Node):
        return value, lambda t: t
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise UnsupportedGraph(f"输出是 {len(value)} 个张量，gkern 只支持单输出融合")
        item = value[0]
        if not isinstance(item, fx.Node):
            raise UnsupportedGraph("输出结构里含非张量元素")
        maker = tuple if isinstance(value, tuple) else list
        return item, lambda t: maker([t])
    raise UnsupportedGraph("输出结构不是单个张量")


def _build_runner(gm: fx.GraphModule, tensors, shape) -> Callable:
    """用 Program 手工搭 IR（等价于 DSL 追踪的结果，只是来源换成 FX 图）。"""
    numel = int(np.prod(shape))
    block = 256
    grid = ((numel + block - 1) // block,)

    program = Program(name="fx_pointwise_fused", grid=grid, num_warps=4)
    for i, t in enumerate(tensors):
        program.inputs[f"in{i}"] = (tuple(t.shape), np.dtype(np.float32))

    pid = program.emit("program_id", (), np.int64, axis=0)
    block_c = program.emit("const", (), np.int64, value=block)
    start = program.emit("mul", (), np.int64, (pid.id, block_c.id))
    arange = program.emit("arange", (block,), np.int64, start=0, end=block)
    offs = program.emit("add", (block,), np.int64, (start.id, arange.id))
    n_c = program.emit("const", (), np.int64, value=numel)
    mask = program.emit("lt", (block,), np.bool_, (offs.id, n_c.id))

    values: Dict[fx.Node, int] = {}
    for i, node in enumerate(n for n in gm.graph.nodes if n.op == "placeholder"):
        ld = program.emit("load", (block,), np.float32, (offs.id, mask.id),
                          buffer=f"in{i}", other=0.0)
        values[node] = ld.id

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        name = _op_name(node)
        args = [values[a] for a in node.args if isinstance(a, fx.Node)]
        scalars = [a for a in node.args if isinstance(a, (int, float))]
        if name in _UNARY:
            values[node] = program.emit(name, (block,), np.float32, (args[0],)).id
        elif name in _BINARY:
            if len(args) == 2:
                values[node] = program.emit(name, (block,), np.float32,
                                            (args[0], args[1])).id
            else:
                c = program.emit("const", (), np.float32, value=float(scalars[0]))
                a, b = (args[0], c.id) if isinstance(node.args[0], fx.Node) else (c.id, args[0])
                values[node] = program.emit(name, (block,), np.float32, (a, b)).id
        else:
            raise NotImplementedError(name)

    out_node = next(n for n in gm.graph.nodes if n.op == "output")
    out_value, wrap = _unwrap_single_output(out_node.args[0])
    if out_value not in values:
        raise UnsupportedGraph("输出不是已匹配的逐元素结果")
    program.emit_store("out", offs, program.tile(values[out_value]), mask)

    cost = costmodel.analyze(program)

    def runner(*inputs):
        bufs = {f"in{i}": np.ascontiguousarray(t.detach().numpy(), dtype=np.float32)
                for i, t in enumerate(inputs)}
        out = np.zeros(numel, dtype=np.float32)
        bufs["out"] = out
        codegen.run(program, bufs)
        result = torch.from_numpy(out.reshape(shape))
        # Dynamo 的图输出是 ((t,),)，backend 必须回同样结构；早期直接回裸
        # tensor，Dynamo 按序列取 [0] → 结果变成 0 维标量。
        return wrap(result)

    runner.__gkern_cost__ = cost  # type: ignore[attr-defined]
    runner.__gkern_program__ = program  # type: ignore[attr-defined]
    return runner
