"""验证真实 PyTorch 调用、梯度与编译结果；不 mock 编译器。"""
import pytest
import torch
import torch.fx as fx
import torch.nn.functional as F

from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv


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


# ---------------------------------------------------------------------------
# F5：真实实现会把输入压成 float32，fake 实现却照单全收 —— 编译期与运行期
# 的 dtype 语义必须一致，所以非 float32 一律在两边都显式拒绝。
# ---------------------------------------------------------------------------

def test_scale_gelu_rejects_non_float32_real_and_fake():
    from gkern.torch_ext.custom_ops import scale_gelu
    x64 = torch.randn(8, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="float32"):
        scale_gelu(x64, 1.0)
    with FakeTensorMode():
        fake64 = torch.randn(8, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="float32"):
            scale_gelu(fake64, 1.0)


def test_scale_gelu_fake_float32_preserves_metadata():
    from gkern.torch_ext.custom_ops import scale_gelu
    with FakeTensorMode():
        out = scale_gelu(torch.randn(8), 2.0)
        assert out.dtype == torch.float32
        assert out.shape == torch.Size([8])


def test_fused_mlp_rejects_non_float32_real_and_fake():
    from gkern.torch_ext.custom_ops import fused_mlp
    a, w, b = (torch.randn(4, 16, dtype=torch.float64),
               torch.randn(16, 3, dtype=torch.float64),
               torch.randn(3, dtype=torch.float64))
    with pytest.raises(RuntimeError, match="float32"):
        fused_mlp(a, w, b)
    with FakeTensorMode():
        fa = torch.randn(4, 16, dtype=torch.float64)
        fw = torch.randn(16, 3, dtype=torch.float64)
        fb = torch.randn(3, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="float32"):
            fused_mlp(fa, fw, fb)


# ---------------------------------------------------------------------------
# F6：fused_mlp 的 K % 16 约束（以及形状）必须在 real / fake 两边一致校验，
# 否则编译期放行、运行期报错，或者更糟：编译期接受一个真实实现算不了的形状。
# ---------------------------------------------------------------------------

def test_fused_mlp_rejects_k_not_multiple_of_block_real_and_fake():
    from gkern.torch_ext.custom_ops import fused_mlp
    a, w, b = torch.randn(4, 8), torch.randn(8, 3), torch.randn(3)
    with pytest.raises(RuntimeError, match="16"):
        fused_mlp(a, w, b)
    with FakeTensorMode():
        fa, fw, fb = torch.randn(4, 8), torch.randn(8, 3), torch.randn(3)
        with pytest.raises(RuntimeError, match="16"):
            fused_mlp(fa, fw, fb)


@pytest.mark.parametrize("shapes, needle", [
    (((4, 16), (9, 3), (3,)), "内维"),      # A/W 内维不匹配
    (((4, 16), (16, 3), (5,)), "bias"),     # bias 形状错
    (((16,), (16, 3), (3,)), "二维"),        # A 不是二维
])
def test_fused_mlp_rejects_bad_shapes_real_and_fake(shapes, needle):
    from gkern.torch_ext.custom_ops import fused_mlp
    a = torch.randn(*shapes[0])
    w = torch.randn(*shapes[1])
    b = torch.randn(*shapes[2])
    with pytest.raises(RuntimeError, match=needle):
        fused_mlp(a, w, b)
    with FakeTensorMode():
        fa = torch.randn(*shapes[0])
        fw = torch.randn(*shapes[1])
        fb = torch.randn(*shapes[2])
        with pytest.raises(RuntimeError, match=needle):
            fused_mlp(fa, fw, fb)


def test_fused_mlp_rejects_symbolic_k_not_multiple_of_bk():
    """符号形状下 K 不整除 BK 必须在**编译期**就被拒，而不是放过 export、运行期才炸。

    复现路径：``torch.export`` 用 ``Dim("K")`` 把内维标成符号维，喂 K=10
    （不是 16 的整数倍）。修复前 export 成功、真跑才算错——正是「编译期过、
    运行期炸」；修复后 fake 实现用 ``torch._check`` 把约束写进图，export 当场失败。
    """
    from gkern.torch_ext.custom_ops import fused_mlp

    class M(torch.nn.Module):
        def forward(self, a, w, b):
            return fused_mlp(a, w, b)

    a = torch.randn(4, 10)
    w = torch.randn(10, 8)
    b = torch.randn(8)
    dim_k = torch.export.Dim("K")
    with pytest.raises(Exception, match="整数倍|BK"):
        torch.export.export(
            M(), (a, w, b),
            dynamic_shapes=({1: dim_k}, {0: dim_k}, None),
        )


def test_fused_mlp_accepts_symbolic_k_multiple_of_bk():
    """把符号维约束成 16 的整数倍（``16*_K``）后必须能 export 成功。

    证明上面那条拒绝不是无差别乱拒：约束可被求解器证明满足时放行。
    """
    from gkern.torch_ext.custom_ops import fused_mlp

    class M(torch.nn.Module):
        def forward(self, a, w, b):
            return fused_mlp(a, w, b)

    a = torch.randn(4, 32)
    w = torch.randn(32, 8)
    b = torch.randn(8)
    _K = torch.export.Dim("_K")
    ep = torch.export.export(
        M(), (a, w, b),
        dynamic_shapes=({1: 16 * _K}, {0: 16 * _K}, None),
    )
    assert ep is not None


def test_fused_mlp_fake_valid_shape_reports_correct_output_shape():
    from gkern.torch_ext.custom_ops import fused_mlp
    with FakeTensorMode():
        out = fused_mlp(torch.randn(4, 16), torch.randn(16, 3), torch.randn(3))
        assert out.shape == torch.Size([4, 3])
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# F7：try_lower 必须拒绝非 CPU 张量（meta 可在无 GPU 环境下构造），
# 并让动态/符号形状干净地回退，而不是抛出一个未被捕获的 guard 错误。
# ---------------------------------------------------------------------------

def test_lowering_rejects_non_cpu_tensor_meta():
    from gkern.torch_ext.fx_lower import UnsupportedGraph, try_lower
    gm = fx.symbolic_trace(lambda x: x * 1.7 + 1.0)
    meta = torch.empty(37, device="meta")
    with pytest.raises(UnsupportedGraph, match="设备|device"):
        try_lower(gm, [meta])


def test_lowering_rejects_symbolic_shapes_cleanly():
    from gkern.torch_ext.fx_lower import UnsupportedGraph, try_lower
    gm = fx.symbolic_trace(lambda x: x * 1.7 + 1.0)
    se = ShapeEnv()
    with FakeTensorMode(shape_env=se) as mode:
        u = se.create_unbacked_symint()
        sym = torch.empty(u, device="cpu")
        assert isinstance(sym.shape[0], torch.SymInt)
        with pytest.raises(UnsupportedGraph, match="符号|动态|SymInt"):
            try_lower(gm, [sym])


def test_dynamic_shapes_fall_back_cleanly_and_stay_correct():
    from gkern.torch_ext.backend import REPORTS, gkern_backend
    torch._dynamo.reset()
    REPORTS.clear()
    fn = lambda x, y: F.gelu(x * 1.7 + y) * 0.5 + x
    compiled = torch.compile(fn, backend=gkern_backend, dynamic=True)
    for n in (37, 53, 64):
        x, y = torch.randn(n), torch.randn(n)
        torch.testing.assert_close(compiled(x, y), fn(x, y))
    assert REPORTS, "应当留下回退记录"
    assert all(not r.fused for r in REPORTS.values())
    assert any(("符号" in r.reason or "动态" in r.reason or "SymInt" in r.reason)
               for r in REPORTS.values())


# ---------------------------------------------------------------------------
# F8：explain 必须返回「被问的那个对象」的结论，而不是随便 REPORTS 里的第一条。
# ---------------------------------------------------------------------------

def test_explain_attributes_report_to_the_compiled_object():
    from gkern.torch_ext.backend import REPORTS, explain, gkern_backend
    torch._dynamo.reset()
    REPORTS.clear()
    fused_fn = lambda x: torch.sigmoid(x) * 2.0          # 纯逐元素 → 融合
    fallback_fn = lambda x, y: torch.sin(x) + y          # sin 不认识 → 回退
    cf = torch.compile(fused_fn, backend=gkern_backend, fullgraph=True, dynamic=False)
    cb = torch.compile(fallback_fn, backend=gkern_backend, fullgraph=True, dynamic=False)
    cf(torch.randn(31))
    cb(torch.randn(31), torch.randn(31))

    rep_f = explain(cf)
    rep_b = explain(cb)
    assert rep_f["fused"] is True
    assert rep_b["fused"] is False
    assert "sin" in rep_b["reason"]


def test_explain_without_argument_reports_latest_compile():
    from gkern.torch_ext.backend import REPORTS, explain, gkern_backend
    torch._dynamo.reset()
    REPORTS.clear()
    assert explain()["fused"] is None
    fn = lambda x: torch.sigmoid(x) * 2.0
    torch.compile(fn, backend=gkern_backend, fullgraph=True, dynamic=False)(torch.randn(9))
    latest = explain()
    assert latest["fused"] is True


def test_explain_latest_is_recent_compile_not_first_inserted_key():
    """同一函数先融合再因 dtype 变化回退，查询必须回答最近一次编译。

    Dynamo 的 cache entry 顺序不能当作编译时间顺序；reset 后 CompileId
    还会复用，原地覆盖的 dict key 也必须算作最近一次记录。
    """
    from gkern.torch_ext.backend import REPORTS, explain, gkern_backend
    torch._dynamo.reset()
    REPORTS.clear()
    fn = lambda x: torch.sigmoid(x) * 2.0
    cf = torch.compile(fn, backend=gkern_backend, fullgraph=True, dynamic=False)
    cf(torch.randn(31))                            # float32 → 融合
    assert explain()["fused"] is True
    cf(torch.randn(31, dtype=torch.float64))       # dtype 变 → 重编译 → 回退
    assert explain()["fused"] is False
    assert "float32" in explain()["reason"]
    assert explain(cf)["fused"] is False
    assert "float32" in explain(cf)["reason"]


def test_explain_latest_is_recency_not_dict_insertion_order():
    """reset 后重编译旧帧会覆盖 key，插入顺序不再等于编译顺序。"""
    from gkern.torch_ext.backend import REPORTS, explain, gkern_backend
    torch._dynamo.reset()
    REPORTS.clear()
    fused_fn = lambda x: torch.sigmoid(x) * 2.0
    fallback_fn = lambda x, y: torch.sin(x) + y
    cf = torch.compile(fused_fn, backend=gkern_backend, fullgraph=True, dynamic=False)
    cb = torch.compile(fallback_fn, backend=gkern_backend, fullgraph=True, dynamic=False)
    cf(torch.randn(31))
    cb(torch.randn(31), torch.randn(31))
    assert explain()["fused"] is False
    assert "sin" in explain()["reason"]

    torch._dynamo.reset()
    cf(torch.randn(41))                            # 重编译旧帧 → 原地覆盖旧 key
    assert explain()["fused"] is True
    assert explain(cf)["fused"] is True
