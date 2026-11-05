#!/usr/bin/env bash
# Offline CPU path. No Kaggle, no Triton, no JAX, no GPU result files.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"

python3 - <<'PY'
import torch
print(f"torch {torch.__version__} cuda_available={torch.cuda.is_available()}")
print("real GPU behavior: UNKNOWN (this script does not measure a device)")
PY

python3 -m pytest
python3 scripts/verify_numbers.py
python3 scripts/verify_numbers.py --selftest
python3 examples/quickstart.py
python3 benchmarks/run_benchmarks.py
echo "offline CPU path ok"
