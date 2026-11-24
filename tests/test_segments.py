import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.ame import AmeError  # noqa: E402
from utils.segments import run_segments, wilson  # noqa: E402


def planted(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 3, n)
    a = np.where(g == 0, rng.choice(["x", "y"], n, p=[.95, .05]), rng.choice(["x", "y"], n, p=[.1, .9]))
    b = np.where(g == 1, rng.choice(["p", "q"], n, p=[.05, .95]), rng.choice(["p", "q"], n, p=[.9, .1]))
    c = rng.choice(list("uvw"), n)
    y = (rng.random(n) < np.where(g == 2, .7, .3)).astype(int)
    return pd.DataFrame({"a": a, "b": b, "c": c, "y": y}), g


def test_wilson_basic():
    lo, hi = wilson(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.60
    assert wilson(0, 10)[0] == 0 or wilson(0, 10)[0] < 1e-9
    assert wilson(10, 10)[1] <= 1.0 + 1e-9


def test_finds_planted_structure():
    df, g = planted()
    r = run_segments(df, ["a", "b", "c"], "y", k=3)
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(g, r.labels) > 0.5
    assert r.profiles.N.sum() == len(df)
    assert r.stability > 0.6


def test_pure_noise_warns():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({c: rng.choice(list("abcd"), 3000) for c in "abcd"})
    df["y"] = rng.integers(0, 2, 3000)
    r = run_segments(df, list("abcd"), "y", k=4)
    assert r.warnings


def test_rejects_tiny_or_single_variable():
    df, _ = planted(100)
    with pytest.raises(AmeError):
        run_segments(df, ["a", "b"], "y")
    df, _ = planted()
    with pytest.raises(AmeError):
        run_segments(df, ["a"], "y")


def test_missing_values_become_a_category():
    df, _ = planted()
    df.loc[:200, "a"] = None
    r = run_segments(df, ["a", "b", "c"], "y", k=3)
    assert len(r.labels) == len(df)
