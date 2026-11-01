"""PrivateUse1 设备后端骨架：让 ``torch.device("gkern")`` 真的存在。

## 这一层在 PyTorch 里对应什么

国产加速卡的 PyTorch 适配，公开形态几乎都是「一个 PrivateUse1 后端」：

- `torch.utils.rename_privateuse1_backend("xxx")` —— 把预留的私有后端改名成厂商名；
- `torch._register_device_module("xxx", module)` —— 提供 `device_count` /
  `current_device` / `synchronize` 等设备查询函数；
- `torch.utils.generate_methods_for_privateuse1_backend()` —— 给 `Tensor` /
  `Storage` / `Module` 生成 `.xxx()` 这类迁移方法；
- 真正的重活（**分配器**与**算子内核**）在 C++ 侧：`c10::PrivateUse1Allocator`、
  `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)`。

本模块只做前三条，第四条明确**不做**——这正是它作为「骨架」的价值：
把「能让设备对象存在」与「能真的在上面跑算子」之间的界线画清楚，
而不是假装两者是一回事。

## 为什么不能只用一个 ``torch.compile`` 后端充数

`torch.compile(backend="gkern")` 与 PrivateUse1 设备后端是两件不同的事：
前者只接管 **图的分段与执行**，张量仍然在 CPU/CUDA 上，设备语义不变；
后者要求张量真的具有第三种设备类型，`aten` 的 dispatch key 会切到 PrivateUse1。
详见 ``docs/ecosystem.md`` E.1 的对照表。

## 为什么分配器必须是 C++

`aten::empty` 在 PrivateUse1 上会走 dispatch 到该后端注册的实现；没有注册就是
`NotImplementedError`。Python 层无法注入这条 dispatch 路径（没有公开的
`torch.library.impl(..., "PrivateUse1")` 能接管内存分配本身），所以
``torch.empty(4, device="gkern")`` 在本仓库里**必须失败**——
`tests/test_device_backend.py::test_allocation_on_the_fake_device_fails_loudly_not_silently`
把这条失败钉成断言：宁可红，也不要一个「数值对、设备掺假」的假成功。
"""

from __future__ import annotations

import torch

__all__ = ["GAPS", "register_device_backend", "device_module", "describe"]

#: 明确未实现的缺口清单。写成机器可读的列表，而不是散落在注释里。
GAPS = [
    "分配器（allocator）：PrivateUse1 的内存分配只能在 C++ 注册"
    "（c10::PrivateUse1Allocator / TORCH_LIBRARY_IMPL），纯 Python 接管不了 "
    "aten::empty，因此 torch.empty(4, device='gkern') 会抛 NotImplementedError。",
    "算子内核（kernel）：aten::* 一个都还没实现，需要逐个注册或走 fallback，"
    "否则每个算子都在 dispatch 处失败。",
    "stream / event：设备流（Stream）、事件（Event）只是占位对象，"
    "没有真正的异步与跨流同步语义。",
    "autograd：反向传播依赖真实存储与设备流；当前只有 CPU 上的 numpy 路径，"
    "设备侧 autograd 未接线。",
    "显存统计：memory_allocated / max_memory_reserved 恒为 0，"
    "因为它们本应来自分配器。",
]


class _FakeStream:
    """占位流对象：只为让 ``torch.gkern.current_stream()`` 之类调用不崩。"""

    def __init__(self, device: int = 0) -> None:
        self.device = torch.device("privateuseone", device)

    def wait_stream(self, other) -> None:  # noqa: D401 - 空实现是刻意的
        return None

    def record_event(self, event=None):
        return _FakeEvent(self.device)

    def synchronize(self) -> None:
        return None

    def __repr__(self) -> str:  # pragma: no cover - 只影响打印
        return f"<gkern stream device={self.device}>"


class _FakeEvent:
    """占位事件对象：没有真实时序，只表达「这里本该有一次同步」。"""

    def __init__(self, device=None) -> None:
        self.device = device

    def record(self, stream=None) -> None:
        return None

    def wait(self, stream=None) -> None:
        return None

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None


class _DeviceModule:
    """``torch._register_device_module`` 期望的那组设备查询函数。

    这些函数被 ``torch.gkern.*`` 直接暴露；缺了任何一个，上层（例如
    ``torch.cuda`` 风格的设备管理、``torch.utils`` 里的设备枚举）就会 AttributeError。
    """

    def __init__(self) -> None:
        self._current_device = 0

    # ---- 设备枚举与选择 ---------------------------------------------------
    def device_count(self) -> int:
        return 1

    def current_device(self) -> int:
        return self._current_device

    def set_device(self, device) -> None:
        idx = int(device) if not isinstance(device, torch.device) else device.index or 0
        if idx < 0:
            raise RuntimeError("设备索引不能为负")
        self._current_device = idx

    def is_available(self) -> bool:
        # 注意：这里返回 True 说的是「设备对象可用」，不是「能跑算子」。
        # 能跑算子的判据是分配器与 kernel，见 GAPS。
        return True

    # ---- 同步 -------------------------------------------------------------
    def synchronize(self, device=None) -> None:
        return None

    def device(self, device=None) -> torch.device:
        idx = self._current_device if device is None else int(device)
        return torch.device("privateuseone", idx)

    # ---- 属性 -------------------------------------------------------------
    def get_device_name(self, device=None) -> str:
        return "gkern-simulated-device"

    def get_device_properties(self, device=None) -> dict:
        from ..gpgpu.device import DeviceInfo

        info = DeviceInfo()
        return {
            "name": "gkern-simulated-device",
            "sm_count": info.sm_count,
            "warp_size": info.warp_size,
            "shared_memory_per_block": info.shared_memory_per_block,
            "total_memory_bytes": info.l2_bytes * 1024,  # 占位：真实值来自分配器
        }

    # ---- 显存（恒 0，因为分配器不在 Python 侧） ---------------------------
    def memory_allocated(self, device=None) -> int:
        return 0

    def max_memory_allocated(self, device=None) -> int:
        return 0

    def memory_reserved(self, device=None) -> int:
        return 0

    def max_memory_reserved(self, device=None) -> int:
        return 0

    def empty_cache(self) -> None:
        return None

    # ---- 流与事件 ---------------------------------------------------------
    def current_stream(self, device=None):
        return _FakeStream(self._current_device)

    def default_stream(self, device=None):
        return _FakeStream(self._current_device)

    def set_stream(self, stream) -> None:
        return None

    def Stream(self, device=None, **kwargs):
        return _FakeStream(self._current_device)

    def Event(self, **kwargs):
        return _FakeEvent()


_MODULE = _DeviceModule()
_REGISTERED_NAME = None


def device_module() -> _DeviceModule:
    """返回模块单例（测试里直接用它查设备状态）。"""
    return _MODULE


def register_device_backend(name: str = "gkern") -> dict:
    """把 PrivateUse1 后端改名为 ``name`` 并注册设备模块；返回注册摘要。

    幂等：重复调用不会因为「后端已被改名」而炸——第二次调用直接把上次的
    ``previous_name`` 返回。
    """
    global _REGISTERED_NAME

    previous = torch._C._get_privateuse1_backend_name()
    if previous != name:
        torch.utils.rename_privateuse1_backend(name)
    _REGISTERED_NAME = name
    torch._register_device_module(name, _MODULE)
    # 给 Tensor / Storage / Module 生成 .gkern() 这类迁移方法；对纯 Python
    # 后端它们最终仍会撞到「没有 kernel」这堵墙，但至少方法存在。
    torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)
    return {
        "backend": name,
        "previous_name": previous,
        "registered": True,
        "gaps": list(GAPS),
    }


def describe() -> dict:
    """当前接线状态的机器可读摘要（README / 文档引用它，避免手写漂移）。"""
    name = torch._C._get_privateuse1_backend_name()
    constructible = True
    try:
        torch.device(name, 0)  # noqa: F841 - 只验证构造不抛异常
    except Exception:  # pragma: no cover - 只在改名失败时触发
        constructible = False
    return {
        "backend": name,
        "device_constructible": constructible,
        "allocator": False,
        "kernels": False,
        "streams": False,
        "gaps": list(GAPS),
    }
