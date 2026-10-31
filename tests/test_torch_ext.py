"""验证真实 PyTorch 调用、梯度与编译结果；不 mock 编译器。"""
import pytest
import torch
import torch.nn.functional as F


def test_custom_scale_gelu_forward_and_backward():
    from gkern.torch_ext.custom_ops import scale_gelu
    x = torch.linspace(-3, 3, 37, requires_grad=True)
    out = scale_gelu(x, 1.7)
    ref = F.gelu(x * 1.7, approximate="tanh")
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    actual_grad, = torch.autograd.grad(out.sum(), x)
    expected_grad, = torch.autograd.grad(ref.sum(), x)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=1e-5)


def test_custom_fused_mlp_gradients():
    from gkern.torch_ext.custom_ops import fused_mlp
    torch.manual_seed(3)
    a = torch.randn(4, 16, requires_grad=True)
    w = torch.randn(16, 8, requires_grad=True)
    b = torch.randn(8, requires_grad=True)
    out = fused_mlp(a, w, b)
    ref = F.gelu(a @ w + b, approximate="tanh")
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    got = torch.autograd.grad(out.sum(), (a, w, b))
    want = torch.autograd.grad(ref.sum(), (a, w, b))
    for g, r in zip(got, want):
        torch.testing.assert_close(g, r, atol=1e-4, rtol=1e-4)


def test_compile_pointwise_preserves_output_tuple():
    from gkern.torch_ext.backend import gkern_backend, REPORTS
    torch._dynamo.reset()
    REPORTS.clear()
    def fn(x, y):
        return F.gelu(x * 1.7 + y) * 0.5 + x
    x, y = torch.randn(37), torch.randn(37)
    compiled = torch.compile(fn, backend=gkern_backend, fullgraph=True, dynamic=False)
    torch.testing.assert_close(compiled(x, y), fn(x, y))
    assert any(r.fused for r in REPORTS.values())


@pytest.mark.parametrize("case", ["grad", "double", "alpha", "multi", "tanh_gelu"])
def test_unsupported_semantics_fall_back_without_change(case):
    from gkern.torch_ext.backend import gkern_backend
    torch._dynamo.reset()
    x = torch.randn(37, dtype=torch.float64 if case == "double" else torch.float32,
                    requires_grad=(case == "grad"))
    y = torch.randn_like(x)
    if case == "alpha":
        fn = lambda a, b: torch.add(a, b, alpha=3)
    elif case == "multi":
        fn = lambda a, b: (a + b, a * b)
    elif case == "tanh_gelu":
        fn = lambda a, b: F.gelu(a + b, approximate="tanh")
    else:
        fn = lambda a, b: a * 1.7 + b
    compiled = torch.compile(fn, backend=gkern_backend, fullgraph=True, dynamic=False)
    actual, expected = compiled(x, y), fn(x, y)
    torch.testing.assert_close(actual, expected)
    if case == "grad":
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0],
                                   torch.autograd.grad(expected.sum(), x)[0])


# ---------------------------------------------------------------------------
# 下沉失败的每一种原因都必须说清楚（不能只回一个 None）
# ---------------------------------------------------------------------------

def _trace(fn, *args):
    import torch.fx as fx
    return fx.symbolic_trace(fn), args


def test_lowering_reports_reason_for_requires_grad():
    from gkern.torch_ext.fx_lower import UnsupportedGraph, try_lower
    gm, args = _trace(lambda x: x * 1.7 + 1.0, torch.randn(37, requires_grad=True))
    with pytest.raises(UnsupportedGraph, match="autograd"):
        try_lower(gm, args)


def test_lowering_reports_reason_for_non_float32():
    from gkern.torch_ext.fx_lower import UnsupportedGraph, try_lower
    gm, args = _trace(lambda x: x * 2.0, torch.randn(37, dtype=torch.float64))
    with pytest.raises(UnsupportedGraph, match="float32"):
        try_lower(gm, args)


def test_lowering_reports_reason_for_multiple_outputs():
    from gkern.torch_ext.fx_lower import UnsupportedGraph, try_lower
    gm, args = _trace(lambda x: (x + 1.0, x * 2.0), torch.randn(37))
    with pytest.raises(UnsupportedGraph, match="输出"):
        try_lower(gm, args)


def test_lowering_reports_reason_for_unsupported_op():
    from gkern.torch_ext.fx_lower import UnsupportedGraph, try_lower
    gm, args = _trace(lambda x: torch.sin(x), torch.randn(37))
    with pytest.raises(UnsupportedGraph, match="sin"):
        try_lower(gm, args)


def test_lowering_reports_reason_for_method_call_and_kwargs():
    """reduce 走 call_method、alpha 走 kwargs —— 两种都必须说明原因。"""
    from gkern.torch_ext.fx_lower import UnsupportedGraph, try_lower
    gm, args = _trace(lambda x: x.sum(), torch.randn(37))
    with pytest.raises(UnsupportedGraph, match="call_method"):
        try_lower(gm, args)

    gm2, args2 = _trace(lambda x: torch.add(x, x, alpha=3), torch.randn(37))
    with pytest.raises(UnsupportedGraph, match="非默认参数"):
        try_lower(gm2, args2)


def test_fused_runner_matches_graph_output_structure():
    """回归：Dynamo 的图输出是 ``((t,),)``，backend 必须回一个单元素 tuple。

    早期返回裸 tensor，Dynamo 按序列取 [0] → 结果变成 0 维标量
    （实测 ``torch.Size([]) != torch.Size([37])``）。
    """
    from gkern.torch_ext.backend import REPORTS, gkern_backend
    torch._dynamo.reset()
    REPORTS.clear()
    fn = lambda x, y: F.gelu(x * 1.7 + y) * 0.5 + x
    x, y = torch.randn(37), torch.randn(37)
    compiled = torch.compile(fn, backend=gkern_backend, fullgraph=True, dynamic=False)
    out = compiled(x, y)
    assert out.shape == torch.Size([37])
    assert any(r.fused for r in REPORTS.values())
