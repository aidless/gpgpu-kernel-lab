"""PrivateUse1 设备后端骨架的测试。

为什么这些用例必须跑在**子进程**里：``torch.utils.rename_privateuse1_backend``
改的是进程全局状态，而且不可逆——一旦在本进程把后端名从 ``privateuseone`` 改成
``gkern``，后续所有测试看到的都是被改过的世界（``torch._C._get_privateuse1_backend_name()``
再也回不去）。所以这里用 ``subprocess`` 起一个干净的 Python 进程，让污染只发生在
那个进程里。这也顺便测了「用户拿这段代码能否在裸环境里跑通」。
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 子进程脚本：注册后端，然后把「观察到的事实」以 JSON 打出来。
PROBE = r"""
import json
import torch
import gkern.torch_ext.device_backend as db

info = db.register_device_backend("gkern")
mod = db.device_module()


def _try_alloc():
    try:
        torch.empty(4, device="gkern:0")
        return "ok"
    except Exception as exc:          # 记录异常类型，不吞掉结论
        return type(exc).__name__


out = {
    "backend_name": torch._C._get_privateuse1_backend_name(),
    "previous_name": info["previous_name"],
    "device_count": mod.device_count(),
    "current_device": mod.current_device(),
    "is_available": mod.is_available(),
    "device_repr": str(torch.device("gkern", 0)),
    "alloc_result": _try_alloc(),
    "torch_module_present": hasattr(torch, "gkern"),
    "gaps": list(db.GAPS),
    "describe": db.describe(),
}
print(json.dumps(out, ensure_ascii=False))
"""


_CACHE: dict = {}


def run_probe() -> dict:
    """跑一次子进程并把结果缓存——每条用例各起一次会白等 3×7 秒的 torch import。"""
    if _CACHE:
        return _CACHE
    env = {**os.environ, "PYTHONPATH": "src"}
    proc = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT,
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, f"子进程失败：\n{proc.stdout}\n{proc.stderr}"
    _CACHE.update(json.loads(proc.stdout.strip().splitlines()[-1]))
    return _CACHE


def test_register_device_backend_renames_and_exposes_a_device():
    info = run_probe()
    assert info["backend_name"] == "gkern"
    assert info["previous_name"] == "privateuseone"
    assert info["device_count"] == 1
    assert info["current_device"] == 0
    assert info["is_available"] is True
    assert info["device_repr"] == "gkern:0"
    assert info["torch_module_present"] is True


def test_allocation_on_the_fake_device_fails_loudly_not_silently():
    """没有 C++ 分配器时，``torch.empty(..., device="gkern")`` 必须**明确报错**。

    这是反向断言：如果哪天有人用 Python 假装分配成功（偷偷把张量放 CPU 再改
    device 字段），这里会先红——那正是最坏的一种假成功：数值对了，设备语义全假。
    """
    info = run_probe()
    assert info["alloc_result"] in ("NotImplementedError", "RuntimeError"), \
        f"设备分配既没成功也没明确报错，而是 {info['alloc_result']}"


def test_gap_list_is_explicit_and_nonempty():
    """缺口必须写成清单交给用户看，而不是藏在 docstring 里。"""
    info = run_probe()
    gaps = info["gaps"]
    assert len(gaps) >= 3
    joined = " ".join(gaps)
    assert "分配器" in joined or "allocator" in joined.lower()
    assert "kernel" in joined.lower() or "算子" in joined


def test_describe_reports_what_is_and_is_not_wired():
    info = run_probe()
    desc = info["describe"]
    assert desc["backend"] == "gkern"
    assert desc["device_constructible"] is True
    assert desc["allocator"] is False        # 纯 Python 注册不了分配器
    assert isinstance(desc["gaps"], list) and desc["gaps"]
