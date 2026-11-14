"""C++ 扩展的硬门禁：**模块必须存在**、**语义必须与 numpy 逐值相同**、**性能要能量出来**。

与 `scripts/verify_numbers.py` 的分工：

- `verify_numbers.py` 管「README/文档里的数字与代码复算一致」；
- 本脚本管「C++ 那条实现路径真的被构建了，而且真的和 Python 语义一样，不是摆设」。

所以它在本模块缺失时**直接 exit 2**——不允许「没构建就算通过」的静默绿。

用法::

    python3 scripts/build_cpp.py          # 或者 pip install -e ".[cpp]" --no-build-isolation
    python3 scripts/verify_cpp.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

import gkern.gpgpu.memory as mem

BUILD_HINT = (
    '先构建扩展：pip install pybind11 && pip install -e ".[cpp]" --no-build-isolation'
)


def _fail(msg: str, code: int = 1) -> None:
    print(f"FAIL {msg}")
    sys.exit(code)


def main() -> None:
    try:
        import gkern_cpp
    except ImportError:
        _fail(f"gkern_cpp 未构建（{BUILD_HINT}）", code=2)

    rng = np.random.default_rng(20261101)
    checked = 0

    # 1) 事务计数：随机 + 边界
    for _ in range(200):
        n = int(rng.integers(0, 400))
        offsets = rng.integers(-5000, 200_000, size=n).astype(np.int64)
        got = gkern_cpp.count_transactions(offsets)
        want = mem.count_transactions(offsets)
        if got != want:
            _fail(f"count_transactions 不一致：got={got} want={want} n={n}")
        checked += 1

    # 2) bank 冲突度：随机，含负偏移与自定义 bank 数
    for _ in range(200):
        n = int(rng.integers(0, 400))
        offsets = (rng.integers(-500, 4096, size=n) * 4).astype(np.int64)
        for bank_count, bank_width in ((32, 4), (16, 4), (64, 8)):
            got = gkern_cpp.bank_conflict_degree(offsets, bank_count, bank_width)
            want = mem.bank_conflict_degree(offsets, bank_count, bank_width)
            if got != want:
                _fail(f"bank_conflict_degree 不一致：got={got} want={want} "
                      f"bank_count={bank_count} bank_width={bank_width}")
            checked += 1

    # 3) warp 冲突：随机，含非 32 倍数长度与 item_bytes=8
    for _ in range(200):
        n = int(rng.integers(0, 500))
        indices = rng.integers(-64, 8192, size=n).astype(np.int64)
        for item_bytes in (4, 8):
            got = gkern_cpp.warp_bank_conflicts(indices, item_bytes)
            want = mem.warp_bank_conflicts(indices, item_bytes)
            if got != want:
                _fail(f"warp_bank_conflicts 不一致：got={got} want={want} item_bytes={item_bytes}")
            checked += 1

    # 4) 参数校验必须真的抛错（C++ 侧不能默默接受非法参数）
    for call, desc in (
        (lambda: gkern_cpp.count_transactions(np.arange(4, dtype=np.int64), 0), "transaction_bytes=0"),
        (lambda: gkern_cpp.bank_conflict_degree(np.arange(4, dtype=np.int64), 0, 4), "bank_count=0"),
        (lambda: gkern_cpp.warp_bank_conflicts(np.arange(4, dtype=np.int64), 0), "item_bytes=0"),
    ):
        try:
            call()
        except Exception:
            pass
        else:
            _fail(f"C++ 侧未对非法参数报错：{desc}")

    # 5) 性能：同一份工作量的 numpy vs C++（只报数，不设阈值——它服务的是
    #    「这份计算值不值得写 C++」这个判断，而不是一个必须达到的加速比）
    n = 1 << 20
    offsets = (rng.integers(0, 1 << 28, size=n) * 4).astype(np.int64)
    t0 = time.perf_counter()
    for _ in range(5):
        mem.count_transactions(offsets)
    py_tx = (time.perf_counter() - t0) / 5
    t0 = time.perf_counter()
    for _ in range(5):
        gkern_cpp.count_transactions(offsets)
    cpp_tx = (time.perf_counter() - t0) / 5

    t0 = time.perf_counter()
    for _ in range(5):
        mem.warp_bank_conflicts(offsets)
    py_bank = (time.perf_counter() - t0) / 5
    t0 = time.perf_counter()
    for _ in range(5):
        gkern_cpp.warp_bank_conflicts(offsets)
    cpp_bank = (time.perf_counter() - t0) / 5

    print(f"ok   语义对拍：{checked} 组随机/边界输入全部一致")
    print("ok   非法参数校验：3/3 抛错")
    print(f"     count_transactions  1M 个偏移：numpy {py_tx * 1e3:.2f} ms / "
          f"C++ {cpp_tx * 1e3:.2f} ms → {py_tx / cpp_tx:.2f}×")
    print(f"     warp_bank_conflicts 1M 个索引：numpy {py_bank * 1e3:.2f} ms / "
          f"C++ {cpp_bank * 1e3:.2f} ms → {py_bank / cpp_bank:.2f}×")
    print("C++ 扩展门禁通过。")


if __name__ == "__main__":
    main()
