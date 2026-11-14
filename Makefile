# --- acceptance_backfill (EVAL_ACCEPTANCE_STANDARD.md P4) ---
# 证据来源: ci.yml 的 `pytest -q` 步骤
# 原命令 `python3 -m pytest -q` 本机 rc=2：8 个测试模块 ModuleNotFoundError:
# gkern（src-layout，需 PYTHONPATH=src）。addopts 已含 -q，重复传 -q 合成 -qq，
# pytest 9 会吞掉「154 passed」那行。
.PHONY: eval

eval:
	PYTHONPATH=src python3 -m pytest
