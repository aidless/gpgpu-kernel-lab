"""用 ``torch.library`` 把 gkern 的 kernel 注册成 PyTorch 自定义算子。

一个「能被 torch 生态正常使用」的自定义算子需要四件东西，缺一件就会在某个
环节静默出错：

1. **CPU/CUDA 实现**——真的算；
2. **meta 实现**（fake tensor）——否则 ``torch.compile`` 抓图时直接失败；
3. **autograd**——否则反向传播断链（或者更糟：悄悄不求导）；
4. **schema**——参数名/可变性声明，决定它能不能被 functionalize。

本模块用 ``torch.library.custom_op`` 把这四件一次补齐，并顺手给了一个
``gkern::fused_mlp`` 的复合算子（matmul + bias + gelu 融合）。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from ..ops import run_fused_mlp, run_scale_gelu

__all__ = ["scale_gelu", "fused_mlp", "gelu_tanh_grad"]

_C = 0.7978845608028654   # sqrt(2/pi)
_K = 0.044715


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(_C * (x + _K * x * x * x)))


def gelu_tanh_grad(x: torch.Tensor) -> torch.Tensor:
    """tanh 近似 GELU 的导数（手推，用于 autograd 注册与梯度校验）。

    g(x) = 0.5x(1+tanh(u)),  u = c(x + kx^3)
    g'(x) = 0.5(1+tanh(u)) + 0.5x(1-tanh^2(u))·c(1+3kx^2)
    """
    u = _C * (x + _K * x * x * x)
    t = torch.tanh(u)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * _C * (1.0 + 3.0 * _K * x * x)


@torch.library.custom_op("gkern::scale_gelu", mutates_args=())
def scale_gelu(x: torch.Tensor, scale: float) -> torch.Tensor:
    """``gelu_tanh(scale * x)``，走 gkern 的 DSL kernel（不是 torch 的 kernel）。"""
    out = run_scale_gelu(x.detach().cpu().numpy(), scale=float(scale)).output
    return torch.from_numpy(out).to(x.device)


@scale_gelu.register_fake
def _(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.empty_like(x)


def _scale_gelu_setup(ctx, inputs, output):
    x, scale = inputs
    ctx.save_for_backward(x)
    ctx.scale = scale


def _scale_gelu_backward(ctx, grad_out):
    (x,) = ctx.saved_tensors
    s = ctx.scale
    return gelu_tanh_grad(s * x) * s * grad_out, None


# 注意签名：register_autograd(backward, *, setup_context=None)。
# 早期按 PyTorch 2.x 早期文档写成位置参数 (setup, backward)，
# 在 2.14 上直接 TypeError。
scale_gelu.register_autograd(_scale_gelu_backward,
                             setup_context=_scale_gelu_setup)


@torch.library.custom_op("gkern::fused_mlp", mutates_args=())
def fused_mlp(a: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """``gelu_tanh(a @ w + bias)``：matmul 与 epilogue 在 IR 层融合成一个 kernel。"""
    res = run_fused_mlp(a.detach().cpu().numpy(), w.detach().cpu().numpy(),
                        bias.detach().cpu().numpy(), fuse=True)
    return torch.from_numpy(res.output).to(a.device)


@fused_mlp.register_fake
def _(a: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return a.new_empty((a.shape[0], w.shape[1]))


def _fused_mlp_setup(ctx, inputs, output):
    a, w, bias = inputs
    ctx.save_for_backward(a, w, bias)
    ctx.out = output


def _fused_mlp_backward(ctx, grad_out):
    a, w, bias = ctx.saved_tensors
    z = a @ w + bias
    g = gelu_tanh_grad(z)
    dz = grad_out * g
    return dz @ w.t(), a.t() @ dz, dz.sum(dim=0)


fused_mlp.register_autograd(_fused_mlp_backward,
                            setup_context=_fused_mlp_setup)
