"""Pipeline tests (O1.2): determinism, stratification, no leakage.

Run:  pytest -q   (from orion/)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.data import (  # noqa: E402
    assert_no_leakage,
    deterministic_split,
    discover_samples,
    dataset_version,
)
from orion.synth_data import generate  # noqa: E402

COUNTS = {"train_good": 40, "test_good": 15, "test_spot": 8, "test_scratch": 7}


def _make(tmp: str):
    generate(tmp, "cat", size=32, seed=1, counts=COUNTS)


def test_discover_counts():
    with tempfile.TemporaryDirectory() as tmp:
        _make(tmp)
        train_pool, test_pool = discover_samples(tmp, "cat")
        assert len(train_pool) == 40          # all good
        assert len(test_pool) == 15 + 8 + 7   # good + defects
        assert all(s.label == 0 for s in train_pool)
        assert sum(s.label == 1 for s in test_pool) == 15  # 8 spot + 7 scratch


def test_split_is_deterministic_and_leakfree():
    with tempfile.TemporaryDirectory() as tmp:
        _make(tmp)
        train_pool, test_pool = discover_samples(tmp, "cat")
        a_tr, a_val = deterministic_split(train_pool, 0.2, seed=42)
        b_tr, b_val = deterministic_split(train_pool, 0.2, seed=42)
        # determinism: identical file sets across calls
        assert [s.path for s in a_tr] == [s.path for s in b_tr]
        assert [s.path for s in a_val] == [s.path for s in b_val]
        # no leakage across train/val/test
        assert_no_leakage(a_tr, a_val, test_pool)
        # split proportion ~20% val
        assert abs(len(a_val) / len(train_pool) - 0.2) < 0.05


def test_dataset_version_stable():
    with tempfile.TemporaryDirectory() as tmp:
        _make(tmp)
        v1 = dataset_version(tmp, "cat")
        v2 = dataset_version(tmp, "cat")
        assert v1 == v2 and len(v1) == 12
