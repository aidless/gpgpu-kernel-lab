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

#: matmul kernel 的 BK 步长。gkern 的 IR 是线性指令表、没有循环，K 必须能被
#: BK 整除（见 ``gkern.ops.matmul`` 的模块文档），所以这是一条硬约束。
_BK = 16


def _check_float32(op: str, *tensors: torch.Tensor) -> None:
    """真实实现会经 numpy 走 float32，所以非 float32 必须显式拒绝。

    fake 实现也必须调用它：否则编译期（fake）放行 float64、运行期（real）
    悄悄降精度，两边语义不一致 —— 这正是 ``torch.compile`` 最典型的
    「图里说 float64、跑出来是 float32」事故。
    """
    bad = sorted({str(t.dtype) for t in tensors if t.dtype != torch.float32})
    if bad:
        raise RuntimeError(f"{op} 只支持 float32，收到 {bad}")


def _known_int(dim) -> "int | None":
    """把维度转成 int；符号维度（SymInt）返回 None。

    动态形状下维度可能是 SymInt，对它做 ``%``/``==`` 会生成 SymBool，
    在 Python ``if`` 里会抛 data-dependent guard 错误 —— 所以具体维度用
    普通比较，符号维度改用 :func:`torch._check`（见 ``_check_fused_mlp``）。
    """
    return dim if isinstance(dim, int) else None


def _require(op: str, cond, message: str) -> None:
    """对具体 bool 直接抛，对符号 SymBool 用 ``torch._check`` 落成图内断言。

    ``cond`` 是 Python ``bool`` 时走普通分支（真实实现、具体形状走的都是这条）；
    是 ``SymBool`` 时 ``torch._check`` 会把断言写进图里，于是 ``torch.export`` /
    ``torch.compile`` 在**编译期**就会因为不满足的符号约束而失败，而不是放过
    一个真实 kernel 算不了的形状、等到运行期才炸。
    """
    if isinstance(cond, bool):
        if not cond:
            raise RuntimeError(message)
    else:
        torch._check(cond, lambda: message)


def _check_fused_mlp(op: str, a: torch.Tensor, w: torch.Tensor,
                     bias: torch.Tensor) -> None:
    """``fused_mlp`` 的形状约束，real / fake 共用同一份校验。

    fake 实现如果不校验，编译期就会接受一个真实 kernel 算不了的形状
    （例如 K 不是 BK 的整数倍），把错误推迟到运行期甚至静默算错。
    """
    if a.dim() != 2:
        raise RuntimeError(f"{op} 的 A 要求二维输入，收到 shape={tuple(a.shape)}")
    if w.dim() != 2:
        raise RuntimeError(f"{op} 的 W 要求二维输入，收到 shape={tuple(w.shape)}")

    k = a.shape[1]
    k2 = w.shape[0]
    n = w.shape[1]
    _require(op,
             k == k2,
             f"{op} 权重形状不匹配：A{tuple(a.shape)} @ W{tuple(w.shape)}"
             f"（内维 {k} != {k2}）")

    if bias.dim() != 1:
        raise RuntimeError(
            f"{op} 偏置 bias 形状必须为 ({n},)，收到 {tuple(bias.shape)}")
    b_len = bias.shape[0]
    _require(op,
             b_len == n,
             f"{op} 偏置 bias 形状必须为 ({n},)，收到 {tuple(bias.shape)}")

    _require(op,
             k % _BK == 0,
             f"{op} 的 K={k} 必须是 BK={_BK} 的整数倍（IR 无循环，K 为编译期常量）")


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
    _check_float32("scale_gelu", x)
    out = run_scale_gelu(x.detach().cpu().numpy(), scale=float(scale)).output
    return torch.from_numpy(out).to(x.device)


@scale_gelu.register_fake
def _(x: torch.Tensor, scale: float) -> torch.Tensor:
    _check_float32("scale_gelu", x)
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
    _check_float32("fused_mlp", a, w, bias)
    _check_fused_mlp("fused_mlp", a, w, bias)
    res = run_fused_mlp(a.detach().cpu().numpy(), w.detach().cpu().numpy(),
                        bias.detach().cpu().numpy(), fuse=True)
    return torch.from_numpy(res.output).to(a.device)


@fused_mlp.register_fake
def _(a: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    _check_float32("fused_mlp", a, w, bias)
    _check_fused_mlp("fused_mlp", a, w, bias)
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
