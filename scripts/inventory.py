"""Drift guard: documented counts must match the tree.

Also rejects badge / PyPI widgets and absolute author paths. Run via
``scripts/verify_numbers.py`` (and ``tests/test_inventory.py``).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INVENTORY_KEYS = (
    "test_functions",
    "pytest_items_with_cpp",
    "pytest_items_without_cpp",
    "pytest_module_skips_without_cpp",
    "cpp_pytest_items",
    "bench_kernel_rows",
    "bench_fusion_rows",
    "bench_smem_rows",
    "bench_bank_rows",
    "bench_reuse_rows",
    "bench_precision_rows",
    "bench_schedule_rows",
    "bench_schedule_verdicts",
    "ci_python_versions",
    "failure_modes",
)

_TEXT_SUFFIXES = {".md", ".py", ".yml", ".yaml", ".toml", ".json", ".txt", ".sh"}
_SKIP_DIRS = {".git", ".venv", "__pycache__", "build", "dist", ".pytest_cache"}


def _abs_path_re() -> re.Pattern[str]:
    # Pieced together so this source file does not contain a path the scan matches.
    posix = "/" + "(?:tmp|home|Users)/"
    win = "[A-Za-z]" + ":" + "\\x5c"
    return re.compile(rf"(?<![A-Za-z0-9_.])(?:{posix}|{win})")


def _badge_needles() -> tuple[str, ...]:
    return (
        "img." + "shields.io",
        "static." + "pepy.tech",
        "badge." + "fury.io",
        "pypi." + "org/project",
        "pypistats." + "org",
    )


def _iter_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS or part.endswith(".egg-info") for part in path.parts):
            continue
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        yield path


def _parametrize_width(dec: ast.AST) -> int | None:
    if not isinstance(dec, ast.Call):
        return None
    func = dec.func
    if not (isinstance(func, ast.Attribute) and func.attr == "parametrize"):
        return None
    if len(dec.args) < 2:
        raise ValueError("parametrize is missing its case list")
    cases = dec.args[1]
    if isinstance(cases, (ast.List, ast.Tuple)):
        return len(cases.elts)
    raise ValueError("parametrize cases must be a list or tuple literal so the count can be audited")


def _module_skips_on_import(tree: ast.AST) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "importorskip":
            return True
    return False


def _pytest_items(path: Path) -> tuple[int, int, bool]:
    """Return (functions, collected items, module-level importorskip)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = 0
    items = 0
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name.startswith("test_"):
                    raise ValueError(f"{path.name} has class-based tests; the counter does not expand those")
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue
        functions += 1
        width = 1
        for dec in node.decorator_list:
            factor = _parametrize_width(dec)
            if factor is not None:
                width *= factor
        items += width
    return functions, items, _module_skips_on_import(tree)


def collect_test_counts(root: Path) -> dict[str, int]:
    functions = items = cpp_items = module_skips = 0
    for path in sorted((root / "tests").glob("test_*.py")):
        fn, collected, skips = _pytest_items(path)
        functions += fn
        items += collected
        if skips:
            module_skips += 1
            cpp_items += collected
    if module_skips != 1:
        raise ValueError(f"expected exactly one module-level importorskip, found {module_skips}")
    return {
        "test_functions": functions,
        "pytest_items_with_cpp": items,
        "pytest_items_without_cpp": items - cpp_items,
        "pytest_module_skips_without_cpp": module_skips,
        "cpp_pytest_items": cpp_items,
    }


def _load_modes(root: Path) -> tuple[list[str], set[str]]:
    path = root / "tests" / "test_failure_modes.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modes: list[str] | None = None
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MODES" and isinstance(node.value, ast.Tuple):
                    modes = []
                    for elt in node.value.elts:
                        if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str):
                            raise ValueError("MODES entries must be string literals")
                        modes.append(elt.value)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            names.add(node.name)
    if not modes:
        raise ValueError("tests/test_failure_modes.py is missing a MODES tuple")
    return modes, names


def _inventory_table(text: str) -> dict[str, int]:
    found: dict[str, int] = {}
    missing = []
    for key in INVENTORY_KEYS:
        match = re.search(rf"\|\s*{re.escape(key)}\s*\|\s*(\d+)\s*\|", text)
        if match is None:
            missing.append(key)
        else:
            found[key] = int(match.group(1))
    if missing:
        raise ValueError("README inventory table is missing: " + ", ".join(missing))
    return found


def _bench_lengths(report: dict) -> dict[str, int]:
    return {
        "bench_kernel_rows": len(report["kernels"]),
        "bench_fusion_rows": len(report["fusion"]),
        "bench_smem_rows": len(report["smem"]),
        "bench_bank_rows": len(report["bank"]),
        "bench_reuse_rows": len(report["reuse"]),
        "bench_precision_rows": len(report["precision"]),
        "bench_schedule_rows": len(report["schedule"]),
        "bench_schedule_verdicts": len(report["schedule_verdict"]),
    }


def _load_benchmarks():
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location(
        "gkern_run_benchmarks", ROOT / "benchmarks" / "run_benchmarks.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _project(report: dict) -> dict:
    """Drop machine-specific wall clocks and keep the accounting."""
    kernels = [{k: v for k, v in row.items() if k != "wall_ms"} for row in report["kernels"]]
    keep = ("shape", "accumulate", "bytes", "flops", "bound")
    precision = [{k: row[k] for k in keep} for row in report["precision"]]
    projected = {key: value for key, value in report.items() if key not in ("kernels", "precision")}
    projected["kernels"] = kernels
    projected["precision"] = precision
    return projected


def _norm(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_norm(v) for v in value]
    return value


def _tile_ai_pair() -> tuple[float, float]:
    import numpy as np

    from gkern.compiler import compile_kernel, costmodel
    from gkern.ops import matmul as mm

    def one(tile: int) -> float:
        n = 512
        zeros = np.zeros((n, n), dtype=np.float32)
        program = compile_kernel(
            mm.matmul_tiled,
            {"A": zeros, "B": zeros, "C": zeros},
            {"M": n, "N": n},
            {"BM": tile, "BN": tile, "BK": 16, "K": n},
            (n // tile, n // tile),
        )
        return round(costmodel.analyze(program).arithmetic_intensity, 4)

    return one(16), one(128)


def check(root: Path | None = None) -> list[str]:
    root = root or ROOT
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    failures: list[str] = []

    try:
        counts = collect_test_counts(root)
    except ValueError as exc:
        return [str(exc)]

    readme = (root / "README.md").read_text(encoding="utf-8")
    lines = readme.splitlines()
    if len(lines) < 6 or lines[0] != "# gpgpu-kernel-lab" or lines[1] != "" or lines[5] != "":
        failures.append("README must open with a title, one blank line, exactly three problem lines, then a blank line")
    elif not all(line.strip() for line in lines[2:5]):
        failures.append("the three README problem lines must be non-empty")
    opening = "\n".join(lines[:50])
    if "OBSERVED" not in opening or "UNKNOWN" not in opening:
        failures.append("README opening must name OBSERVED and UNKNOWN")
    if "scripts/offline_cpu.sh" not in readme:
        failures.append("README is missing the offline CPU path scripts/offline_cpu.sh")

    try:
        table = _inventory_table(readme)
    except ValueError as exc:
        failures.append(str(exc))
        table = {}

    bench_path = root / "benchmarks" / "results" / "bench.json"
    committed = json.loads(bench_path.read_text(encoding="utf-8"))
    lengths = _bench_lengths(committed)
    ci_text = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    version_match = re.search(r"python-version:\s*\[([^\]]+)\]", ci_text)
    if version_match is None:
        failures.append("ci.yml is missing python-version")
        versions = []
    else:
        versions = re.findall(r"\d+\.\d+", version_match.group(1))

    modes, test_names = _load_modes(root)
    doc = (root / "docs" / "FAILURE_MODES.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## ([a-z0-9-]+)\s*$", doc, re.M)
    if headings != modes:
        failures.append(f"FAILURE_MODES headings {headings} != MODES {modes}")
    for mode in modes:
        fn = "test_" + mode.replace("-", "_")
        if fn not in test_names:
            failures.append(f"missing {fn} for failure mode {mode}")

    computed = {
        **counts,
        **lengths,
        "ci_python_versions": len(versions),
        "failure_modes": len(modes),
    }
    for key, want in computed.items():
        if key in table and table[key] != want:
            failures.append(f"README inventory {key}={table[key]} but tree has {want}")

    cpp_token = re.search(r"cpp_pytest_items=(\d+)", ci_text)
    if cpp_token is None:
        failures.append("ci.yml is missing cpp_pytest_items=<n>")
    elif int(cpp_token.group(1)) != counts["cpp_pytest_items"]:
        failures.append(
            f"ci.yml cpp_pytest_items={cpp_token.group(1)} but tree has {counts['cpp_pytest_items']}"
        )

    phrase = f"{counts['test_functions']} 个测试函数"
    for rel in ("README.md", "docs/jd-mapping.md"):
        text = (root / rel).read_text(encoding="utf-8")
        found = re.findall(r"(\d+) 个测试函数", text)
        if not found:
            failures.append(f"{rel} is missing 「N 个测试函数」")
        elif any(int(n) != counts["test_functions"] for n in found):
            failures.append(f"{rel} says {found} 个测试函数, tree has {counts['test_functions']}")
        elif phrase not in text:
            failures.append(f"{rel} is missing {phrase}")

    try:
        ai16, ai128 = _tile_ai_pair()
    except Exception as exc:  # pragma: no cover - import/runtime failure is a real gate failure
        failures.append(f"tile AI recomputation failed: {exc}")
        ai16 = ai128 = None
    if ai16 is not None:
        for rel in ("README.md", "docs/ARCHITECTURE.md"):
            text = (root / rel).read_text(encoding="utf-8")
            for value in (ai16, ai128):
                if str(value) not in text:
                    failures.append(f"{rel} does not quote live tile AI {value}")

    stale = ("29.59", "8.03→14.93", "Ryzen", "坑 #12", "4.27→")
    for path in _iter_text_files(root):
        if path.suffix.lower() not in {".md", ".yml", ".yaml"}:
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root).as_posix()
        for token in stale:
            if token in text:
                failures.append(f"{rel} still contains stale claim {token!r}")

    benches = _load_benchmarks()
    rendered = benches.render_markdown(committed)
    markdown = (root / "benchmarks" / "results" / "bench.md").read_text(encoding="utf-8")
    if rendered != markdown:
        failures.append("benchmarks/results/bench.md does not match render(bench.json)")
    fresh = benches.build_report(time_kernels=False)
    if _norm(_project(fresh)) != _norm(_project(committed)):
        failures.append("bench.json accounting drifted from benchmarks/run_benchmarks.py (wall_ms ignored)")

    abs_re = _abs_path_re()
    needles = _badge_needles()
    for path in _iter_text_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(root).as_posix()
        if abs_re.search(text):
            failures.append(f"absolute author path in {rel}")
        for needle in needles:
            if needle in text:
                failures.append(f"badge or PyPI claim {needle!r} in {rel}")

    return failures


def main() -> None:
    failures = check()
    if failures:
        print(f"{len(failures)} inventory failure(s):")
        for item in failures:
            print(f"FAIL {item}")
        sys.exit(1)
    print("inventory ok")


if __name__ == "__main__":
    main()
