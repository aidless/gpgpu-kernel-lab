"""The documented inventory has to match the tree."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import inventory  # noqa: E402


def test_documented_inventory_matches_the_tree() -> None:
    failures = inventory.check(ROOT)
    assert failures == [], failures
