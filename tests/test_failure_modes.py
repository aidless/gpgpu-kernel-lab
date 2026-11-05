"""Lock the places where the CPU model returns a number and does not raise.

Those numbers are not real-GPU measurements. ``docs/FAILURE_MODES.md`` names
each one; this file is the executable copy of that list.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gkern.compiler import codegen, compile_kernel, costmodel
from gkern.compiler import dsl as tl
from gkern.compiler.dsl import constexpr
from gkern.gpgpu.device import Device, DeviceInfo
from gkern.gpgpu.kernel import kernel, launch
from gkern.gpgpu.memory import count_transactions, warp_bank_conflicts
from gkern.gpgpu.samples import run_sample, vec_add_coalesced
from gkern.gpgpu.warp import DivergenceReport, Warp
from gkern.ops import run_fused_mlp, run_matmul
from gkern.ops.mixed_precision import run_softmax_precision
from gkern.schedule import GemmShape, compare_strategies

ROOT = Path(__file__).resolve().parents[1]

# Headings in docs/FAILURE_MODES.md must stay in this order.
MODES = (
    "sector-128",
    "banks-32x4",
    "compiler-wave-32",
    "roofline-parameter-card",
    "smem-time-unchanged",
    "persistent-equals-naive",
    "bf16-storage-invisible",
    "occupancy-ignores-registers",
    "padded-tile-bytes",
    "python-elapsed-is-not-device-time",
    "aggregate-warp-size",
)


def test_sector_128() -> None:
    """32 contiguous float32 addresses are one 128B sector. A 32B sector would be four."""
    offsets = np.arange(0, 128, 4, dtype=np.int64)
    assert count_transactions(offsets) == 1
    assert np.unique(offsets // 32).size == 4
    assert "byte_offsets" in count_transactions.__code__.co_varnames
    assert "transaction_bytes" not in count_transactions.__code__.co_varnames


def test_banks_32x4() -> None:
    """Default bank geometry is 32 banks × 4 bytes. Another geometry disagrees, and nothing raises."""
    indices = np.arange(32, dtype=np.int64) * 32
    cuda_like = warp_bank_conflicts(indices)
    wider = warp_bank_conflicts(indices, bank_count=64)
    assert cuda_like == 32 * 31
    assert wider != cuda_like
    assert wider > 0


def test_compiler_wave_32() -> None:
    """The compiler executor chunks shared memory by 32 lanes and has no device argument."""
    n = 64

    def kernel_fn(OUT, N: constexpr):
        tile = tl.smem_alloc("tile", (N,))
        offs = tl.arange(0, N)
        tl.smem_store(tile, offs, tl.cast(offs, np.float32))
        tl.store(OUT, offs, tl.smem_load(tile, offs))

    out = np.zeros((n,), dtype=np.float32)
    program = compile_kernel(kernel_fn, {"OUT": out}, None, {"N": n}, (1,))
    outputs, stats = codegen.run_with_stats(program, {"OUT": out})
    indices = np.arange(n, dtype=np.int64)
    assert warp_bank_conflicts(indices) == 0
    assert warp_bank_conflicts(indices, warp_size=64) == 64
    assert stats.shared_bank_conflicts == 0
    assert np.allclose(outputs["OUT"], indices.astype(np.float32))


def test_roofline_parameter_card() -> None:
    """Same IR, two parameter cards, two predicted times, no 'this is not a measurement' error."""
    from tests.test_smem import _matmul_program

    program = _matmul_program(64, bm=32, bn=32)
    fictional = costmodel.analyze(program, DeviceInfo())
    t4 = costmodel.analyze(program, DeviceInfo.tesla_t4())
    assert fictional.device.name == "generic-domestic-gpgpu"
    assert fictional.bytes_total == t4.bytes_total
    assert fictional.predicted_ms != t4.predicted_ms
    assert fictional.predicted_ms > 0 and t4.predicted_ms > 0


def test_smem_time_unchanged() -> None:
    """Equal HBM bytes make predicted_ms equal. The model cannot express a shared-memory speedup."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((64, 64)).astype(np.float32)
    b = rng.standard_normal((64, 64)).astype(np.float32)
    plain = run_matmul(a, b, bm=32, bn=32, bk=16)
    smem = run_matmul(a, b, bm=32, bn=32, bk=16, use_smem=True)
    assert plain.cost.bytes_total == smem.cost.bytes_total
    assert plain.cost.predicted_ms == smem.cost.predicted_ms
    assert plain.cost.bound == smem.cost.bound == "memory"


def test_persistent_equals_naive() -> None:
    """Persistent kernels do not change the model's time. Launch overhead is invisible."""
    shape = GemmShape(1280, 1280, 528, 128, 128, 16)
    rows = compare_strategies(shape, DeviceInfo.tesla_t4(), splits=2)
    assert rows["persistent"]["predicted_ms"] == rows["naive"]["predicted_ms"]
    assert rows["persistent"]["efficiency"] == rows["naive"]["efficiency"]
    assert rows["persistent"]["extra_bytes"] == 0


def test_bf16_storage_invisible() -> None:
    """Rounding to bf16 keeps float32 storage, so the byte count does not move."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 64)).astype(np.float32)
    fp32 = run_softmax_precision(x, accumulate="fp32")
    bf16 = run_softmax_precision(x, accumulate="bf16")
    assert fp32.cost.bytes_total == bf16.cost.bytes_total
    assert fp32.output.dtype == bf16.output.dtype == np.float32
    assert bf16.max_error > fp32.max_error


def test_occupancy_ignores_registers() -> None:
    """Shrinking the register file does not change occupancy. The model never looks at it."""
    full = Device(DeviceInfo(registers_per_sm=64 * 1024))
    starved = Device(DeviceInfo(registers_per_sm=32))
    assert full.occupancy((256,)) == starved.occupancy((256,))
    assert full.occupancy((256,)) == pytest.approx(8 / 32)


def test_padded_tile_bytes() -> None:
    """When BN does not divide N the model still returns a byte count, larger than 2×|C|."""
    rng = np.random.default_rng(0)
    m, k, n = 64, 32, 48
    a = rng.standard_normal((m, k)).astype(np.float32)
    w = rng.standard_normal((k, n)).astype(np.float32)
    bias = rng.standard_normal((n,)).astype(np.float32)
    unfused = run_fused_mlp(a, w, bias, bm=32, bn=32, bk=16, fuse=False)
    fused = run_fused_mlp(a, w, bias, bm=32, bn=32, bk=16, fuse=True)
    saved = unfused.cost.bytes_total - fused.cost.bytes_total
    assert saved != 2 * m * n * 4
    assert saved > 0
    assert np.allclose(fused.output, unfused.output, atol=1e-5)


def test_python_elapsed_is_not_device_time() -> None:
    """LaunchResult.elapsed_ms is the Python loop, orders of magnitude above the roofline."""
    res, _, _ = run_sample(vec_add_coalesced, n=4096)
    assert not hasattr(res, "gpu_ms")
    requested = 4096 * 3 * 4
    predicted_ms = requested / (DeviceInfo().theoretical_bandwidth_gbps * 1e9) * 1e3
    assert res.elapsed_ms > predicted_ms * 100
    assert predicted_ms < 0.01


def test_aggregate_warp_size() -> None:
    """A 64-lane warp must not report efficiency above 1, and its bank groups follow that width."""
    report = DivergenceReport()
    warp = Warp("w", size=64)
    warp.op()
    report.add(warp.finish())
    assert report.efficiency == 1.0
    assert report.issued_lane_slots == 64

    @kernel(block=(64,))
    def store_all(ctx):
        ctx.smem_store(ctx.tid, ctx.tid.astype(np.float32))

    wide = launch(store_all, {}, grid=(1,), block=(64,), shared_elements=64,
                  device=Device(DeviceInfo(warp_size=64)))
    narrow = launch(store_all, {}, grid=(1,), block=(64,), shared_elements=64,
                    device=Device(DeviceInfo(warp_size=32)))
    assert wide.report.efficiency == 1.0
    assert wide.memory.shared_bank_conflicts == 64
    assert narrow.memory.shared_bank_conflicts == 0


def test_accelerator_scripts_exit_without_writing_json(tmp_path: Path) -> None:
    """GPU and TPU scripts must not invent a result file on the offline CPU path."""
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA is available; this guard is for the CPU path")
    scripts = (
        ROOT / "benchmarks" / "triton" / "bench_softmax_triton.py",
        ROOT / "benchmarks" / "triton" / "bench_bf16_softmax.py",
        ROOT / "benchmarks" / "tpu" / "bench_bf16_softmax_jax.py",
    )
    for script in scripts:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode != 0, script.name
        assert list(tmp_path.glob("*.json")) == [], script.name
