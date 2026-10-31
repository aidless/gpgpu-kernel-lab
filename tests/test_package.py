"""包级 API 契约：公开入口必须真的能 import，不能只是「看起来有」。

这条测试存在的理由：`src/gkern/torch_ext/` 曾一度缺 `__init__.py`，
`import gkern` 正常、`from gkern.torch_ext import register_backend` 却报
`ImportError: cannot import name 'register_backend' (unknown location)`——
包能导入不等于公开 API 可用，所以这里逐个钉死。
"""

from __future__ import annotations

import numpy as np
import torch


def test_top_level_version_and_subpackages():
    import gkern

    assert gkern.__version__ == "0.1.0"
    # 三个子包都应可导入（torch_ext 走惰性再导出，也必须有 __init__.py）
    from gkern import compiler, gpgpu, ops, torch_ext  # noqa: F401


def test_ops_public_runners():
    from gkern.ops import (  # noqa: F401
        OpResult,
        run_fused_mlp,
        run_layernorm,
        run_matmul,
        run_scale_gelu,
        run_softmax,
        run_vector_add,
    )


def test_torch_ext_public_api_is_importable():
    from gkern.torch_ext import (  # noqa: F401
        CompileReport,
        backend,
        custom_ops,
        fx_lower,
        register_backend,
    )


def test_torch_ext_register_backend_really_registers():
    """注册后 ``torch.compile(backend="gkern")`` 必须能跑，而不是只在字典里挂个名。"""
    from gkern.torch_ext import register_backend

    register_backend()
    fn = torch.compile(lambda x: torch.relu(x * 1.5), backend="gkern", fullgraph=True)
    x = torch.randn(37)
    got = fn(x)
    expected = torch.relu(x * 1.5)
    assert torch.allclose(got, expected, atol=1e-5)
    assert got.shape == x.shape


def test_torch_ext_compile_report_shape():
    from gkern.torch_ext import CompileReport

    r = CompileReport(fused=True, nodes=3, matched=["mul", "relu"])
    d = r.as_dict()
    assert set(d) == {"fused", "reason", "nodes", "matched"}
    assert d["fused"] is True


def test_ops_result_is_namedtuple_like():
    from gkern.ops import run_vector_add

    res = run_vector_add(np.zeros(4, np.float32), np.ones(4, np.float32))
    assert res.output.shape == (4,)
    assert res.program.name
    assert res.cost.bytes_total > 0
