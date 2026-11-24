"""
tests/test_ame.py
=================
Automated checks for utils/ame.py. Run from the project root with:

    pip install pytest
    python -m pytest tests -v

If any test fails after someone edits utils/ame.py, the calculation has changed
and the results shown to managers can no longer be trusted until it is fixed.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.ame import (NOT_RECORDED, SMALL_GROUP, AmeError, categorise,          # noqa: E402
                       describe_filters, looks_undergraduate, run_ame)


# ---------------------------------------------------------------------------
# Synthetic data with known truth
# ---------------------------------------------------------------------------
def make_data(n=30000, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.choice(["Low", "Mid", "High"], n, p=[0.5, 0.3, 0.2])
    b = rng.choice(["No", "Yes"], n)
    c = rng.choice(["No", "Yes"], n)        # same category names as B on purpose
    eta = (-0.5 + 0.4 * (a == "Mid") - 0.6 * (a == "High")
           + 0.5 * (b == "Yes") - 0.3 * (c == "Yes"))
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    return pd.DataFrame({"out": y, "a": a, "b": b, "c": c})


CHARS = {"A": "a", "B": "b", "C": "c"}
REFS = {"A": "Low", "B": "No", "C": "No"}


def make_clustered(n_apps=4000, seed=7):
    """Applications with 1-3 course rows each, an application-level random effect,
    messy missing values, a rare category and a control: a miniature of the real data."""
    rng = np.random.default_rng(seed)
    apps = pd.DataFrame({
        "application_number": [f"APP{i}" for i in range(n_apps)],
        "eth": rng.choice(["White", "Black", "Asian", "Mixed", "Tiny"], n_apps,
                          p=[0.6, 0.15, 0.15, 0.0985, 0.0015]),
        "gen": rng.choice(["Women", "Men", "Unknown"], n_apps, p=[0.55, 0.40, 0.05]),
        "imd": rng.choice(["1", "2", "3", "4", "5", None], n_apps),
        "u": rng.normal(0, 0.8, n_apps),
    })
    reps = rng.integers(1, 4, n_apps)
    df = apps.loc[apps.index.repeat(reps)].reset_index(drop=True)
    df["year"] = rng.choice([2023, 2024, 2025], len(df))
    eta = (-0.3 + df["u"] - 0.5 * (df["eth"] == "Black") + 0.3 * (df["eth"] == "Asian")
           + 0.4 * (df["gen"] == "Men") + 0.1 * (df["year"] == 2025)
           + 0.2 * (df["imd"] == "5"))
    df["offer"] = (rng.random(len(df)) < 1 / (1 + np.exp(-eta))).astype(int)
    return df.drop(columns="u")


# ---------------------------------------------------------------------------
# Reference implementation: a faithful copy of the maths in the L2 page
# (Newton-Raphson logit, application-clustered covariance, counterfactual AME).
# The shared module must reproduce its numbers.
# ---------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def l2_reference(base, num, chars, controls, refs, drop_missing, min_cat):
    excl = {"unknown", "not applicable", "unanswered", "not known", "nonclassified",
            "don't know", "not assigned", "nan", "<na>", "none", ""}
    all_vars = {**chars, **controls}
    work = base.copy()
    for label, col in all_vars.items():
        s = work[col].astype("string").str.strip()
        work[f"__{label}"] = s.mask(s.isna() | s.str.lower().isin(excl), NOT_RECORDED)
    if drop_missing:
        for label in all_vars:
            work = work[work[f"__{label}"] != NOT_RECORDED]
    for label in all_vars:
        c = f"__{label}"
        counts = work[c].value_counts()
        small = counts[counts < min_cat].index.tolist()
        if small and len(counts) - len(small) >= 1:
            work[c] = work[c].where(~work[c].isin(small), SMALL_GROUP)

    blocks, meta = [], []
    for label in all_vars:
        c = f"__{label}"
        for cat in [v for v in work[c].value_counts().index if v != refs[label]]:
            blocks.append((work[c] == cat).astype(float).rename(f"{label}={cat}"))
            meta.append((label, cat))
    X_df = pd.concat(blocks, axis=1)
    seen, drop = {}, []
    for col in X_df.columns:
        key = X_df[col].values.tobytes()
        if key in seen:
            drop.append(col)
        else:
            seen[key] = col
    keep = [i for i, col in enumerate(X_df.columns) if col not in drop]
    meta = [meta[i] for i in keep]
    X_df = X_df.drop(columns=drop)

    y = work[num].astype(float).to_numpy()
    groups = pd.factorize(work["application_number"].astype(str))[0]
    X = np.column_stack([np.ones(len(X_df)), X_df.to_numpy(dtype=np.float64)])
    n, p = X.shape
    beta, ridge = np.zeros(p), 1e-8
    for _ in range(100):
        mu = _sigmoid(X @ beta)
        w = mu * (1 - mu)
        H = X.T @ (X * w[:, None]) + ridge * np.eye(p)
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    mu = _sigmoid(X @ beta)
    w = mu * (1 - mu)
    H_inv = np.linalg.inv(X.T @ (X * w[:, None]) + ridge * np.eye(p))
    S = pd.DataFrame(X * (y - mu)[:, None]).groupby(groups).sum().values
    G = S.shape[0]
    V = H_inv @ (S.T @ S) @ H_inv * ((G / max(G - 1, 1)) * ((n - 1) / max(n - p, 1)))

    var_cols = {}
    for j, (label, cat) in enumerate(meta, start=1):
        var_cols.setdefault(label, []).append(j)
    rows = []
    for j, (label, cat) in enumerate(meta, start=1):
        X0 = X.copy()
        X0[:, var_cols[label]] = 0.0
        X1 = X0.copy()
        X1[:, j] = 1.0
        p0, p1 = _sigmoid(X0 @ beta), _sigmoid(X1 @ beta)
        ame = float(np.mean(p1 - p0))
        g = (X1 * (p1 * (1 - p1))[:, None]).mean(0) - (X0 * (p0 * (1 - p0))[:, None]).mean(0)
        se = float(np.sqrt(max(g @ V @ g, 0.0)))
        rows.append({"Characteristic": label, "Category": cat, "ame": ame * 100,
                     "lo": (ame - 1.96 * se) * 100, "hi": (ame + 1.96 * se) * 100,
                     "p": 2 * (1 - 0.5 * (1 + math.erf(abs(ame / se) / math.sqrt(2))))})
    return pd.DataFrame(rows), len(work), G


# ---------------------------------------------------------------------------
# 1. Agreement with the L2 page's own calculation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("drop_missing,min_cat", [(False, 30), (True, 0), (False, 0)])
def test_matches_l2_page_maths(drop_missing, min_cat):
    df = make_clustered()
    chars = {"Ethnicity": "eth", "Gender": "gen", "IMD": "imd"}
    controls = {"Cycle year": "year"}
    refs = {"Ethnicity": "White", "Gender": "Women", "IMD": "3", "Cycle year": "2024"}

    ref, n_ref, g_ref = l2_reference(df, "offer", chars, controls, refs, drop_missing, min_cat)
    run = run_ame(df, "offer", chars, refs, controls=controls,
                  group_col="application_number",
                  missing="drop" if drop_missing else "keep", min_cat=min_cat)

    assert run.n_used == n_ref
    assert run.n_clusters == g_ref
    got = run.results.merge(ref.rename(columns={"p": "p_ref"}),
                            on=["Characteristic", "Category"], how="outer", indicator=True)
    assert (got["_merge"] == "both").all(), "different set of categories than the L2 page"
    assert (got["AME (pp)"] - got["ame"]).abs().max() < 0.005
    assert (got["CI Lower (pp)"] - got["lo"]).abs().max() < 0.01
    assert (got["CI Upper (pp)"] - got["hi"]).abs().max() < 0.01
    assert (got["p"] - got["p_ref"]).abs().max() < 0.001


# ---------------------------------------------------------------------------
# 2. Agreement with statsmodels' own marginal effects / brute force
# ---------------------------------------------------------------------------
def _design(df):
    prep = categorise(df, "out", CHARS)
    from utils.ame import _build_design
    X, meta, _ = _build_design(prep.work, CHARS, REFS)
    return prep, X, meta


def test_binary_variables_match_statsmodels():
    df = make_data()
    run = run_ame(df, "out", CHARS, REFS)
    _, X, meta = _design(df)
    y = df["out"].astype(float).loc[X.index]
    ref = sm.Logit(y, sm.add_constant(X)).fit(disp=False)
    me = ref.get_margeff(at="overall", dummy=True).summary_frame() * 100
    for j, (lab, cat) in enumerate(meta):
        if lab in ("B", "C"):
            col = X.columns[j]
            mine = run.results.query("Characteristic == @lab and Category == @cat").iloc[0]
            assert mine["AME (pp)"] == pytest.approx(me.loc[col, "dy/dx"], abs=0.01)


def test_multicategory_matches_brute_force():
    df = make_data()
    run = run_ame(df, "out", CHARS, REFS)
    _, X, meta = _design(df)
    y = df["out"].astype(float).loc[X.index]
    fit = sm.Logit(y, sm.add_constant(X)).fit(disp=False)
    a_cols = [X.columns[j] for j, (lab, _) in enumerate(meta) if lab == "A"]
    for j, (lab, cat) in enumerate(meta):
        if lab != "A":
            continue
        X0 = X.copy(); X0[a_cols] = 0
        X1 = X0.copy(); X1[X.columns[j]] = 1
        p0 = fit.predict(sm.add_constant(X0, has_constant="add")).mean()
        p1 = fit.predict(sm.add_constant(X1, has_constant="add")).mean()
        mine = run.results.query("Characteristic == 'A' and Category == @cat")["AME (pp)"].iloc[0]
        assert mine == pytest.approx((p1 - p0) * 100, abs=0.01)


def test_recovers_known_direction_and_size():
    r = run_ame(make_data(), "out", CHARS, REFS).results.set_index(
        ["Characteristic", "Category"])["AME (pp)"]
    assert r[("A", "Mid")] > 5 and r[("A", "High")] < -8
    assert r[("B", "Yes")] > 8 and r[("C", "Yes")] < -4


def test_same_answer_every_time():
    df = make_data(5000)
    pd.testing.assert_frame_equal(run_ame(df, "out", CHARS, REFS).results,
                                  run_ame(df, "out", CHARS, REFS).results)


# ---------------------------------------------------------------------------
# 3. Clustering
# ---------------------------------------------------------------------------
def test_clustering_widens_intervals_for_application_level_variables():
    df = make_clustered()
    chars = {"Ethnicity": "eth", "Gender": "gen"}
    refs = {"Ethnicity": "White", "Gender": "Women"}
    plain = run_ame(df, "offer", chars, refs)
    clus = run_ame(df, "offer", chars, refs, group_col="application_number")
    width = lambda r: r.results["CI Upper (pp)"] - r.results["CI Lower (pp)"]
    big = plain.results["N in group"] >= 500          # tiny groups have huge, noisy intervals
    ratio = (width(clus) / width(plain))[big]
    assert big.sum() >= 4 and (ratio > 1.03).all(), ratio.tolist()
    # point estimates are identical; only the uncertainty changes
    pd.testing.assert_series_equal(plain.results["AME (pp)"], clus.results["AME (pp)"])


# ---------------------------------------------------------------------------
# 4. Data handling
# ---------------------------------------------------------------------------
def test_group_sizes_correct_when_category_names_repeat():
    df = make_data(5000)
    got = run_ame(df, "out", CHARS, REFS).results.set_index(["Characteristic", "Category"])
    assert got.loc[("B", "Yes"), "N in group"] == (df["b"] == "Yes").sum()
    assert got.loc[("C", "Yes"), "N in group"] == (df["c"] == "Yes").sum()


def test_excluded_values_are_dropped_and_counted():
    df = make_data(5000)
    df.loc[:99, "a"] = "Unknown"
    df.loc[100:149, "b"] = "Not known"
    run = run_ame(df, "out", CHARS, REFS)
    assert run.n_dropped == 150 and run.n_used == 4850
    assert NOT_RECORDED not in run.results["Category"].values


def test_keep_mode_turns_missing_into_a_category():
    df = make_data(5000)
    df.loc[:299, "a"] = "Unknown"
    run = run_ame(df, "out", CHARS, REFS, missing="keep")
    assert run.n_dropped == 0
    assert NOT_RECORDED in run.results.query("Characteristic == 'A'")["Category"].values


def test_small_categories_are_merged_and_reported():
    df = make_clustered()
    run = run_ame(df, "offer", {"Ethnicity": "eth"}, {"Ethnicity": "White"},
                  group_col="application_number", min_cat=30)
    assert "Tiny" in run.merged_notes["Ethnicity"]
    assert "Tiny" not in run.results["Category"].values
    assert SMALL_GROUP in run.results["Category"].values


def test_controls_are_flagged():
    df = make_clustered()
    run = run_ame(df, "offer", {"Gender": "gen"}, {"Gender": "Women", "Cycle year": "2024"},
                  controls={"Cycle year": "year"}, group_col="application_number")
    roles = run.results.groupby("Characteristic")["Role"].first().to_dict()
    assert roles == {"Gender": "Characteristic", "Cycle year": "Control"}


def test_single_category_characteristic_is_skipped_not_fatal():
    df = make_data(5000)
    df["dom"] = "UK"
    run = run_ame(df, "out", {**CHARS, "Domicile": "dom"}, {**REFS, "Domicile": "UK"})
    assert run.skipped == ["Domicile"]
    assert "Domicile" not in run.results["Characteristic"].values


def test_identical_columns_are_dropped_and_reported():
    """Two characteristics carrying the same information cannot both be estimated."""
    df = make_data(5000)
    df["a_copy"] = df["a"]
    run = run_ame(df, "out", {**CHARS, "A again": "a_copy"}, {**REFS, "A again": "Low"})
    assert run.dup_dropped, "duplicates should be reported"
    assert "A again" not in run.results["Characteristic"].values


def test_clear_errors_for_bad_inputs():
    df = make_data(5000)
    with pytest.raises(AmeError, match="Reference category"):
        run_ame(df, "out", CHARS, {**REFS, "A": "Nonexistent"})
    with pytest.raises(AmeError, match="only one value"):
        run_ame(df.assign(out=1), "out", CHARS, REFS)
    with pytest.raises(AmeError, match="coded 0/1"):
        run_ame(df.assign(out=df["out"] * 3), "out", CHARS, REFS)
    with pytest.raises(AmeError, match="usable rows"):
        run_ame(df.head(50), "out", CHARS, REFS)
    with pytest.raises(AmeError, match="not in the data"):
        run_ame(df, "out", {**CHARS, "Bad": "nope"}, {**REFS, "Bad": "x"})


# ---------------------------------------------------------------------------
# 5. Helpers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label,expected", [
    ("Undergraduate", True), ("UG", True), ("BSc (Hons)", True), ("BA Hons", True),
    ("Bachelor of Arts", True), ("Postgraduate Taught", False),
    ("Postgraduate Research", False), ("PGT", False), ("MBA", False), ("MSc", False),
    ("Foundation degree", False),
])
def test_undergraduate_label_detection(label, expected):
    assert looks_undergraduate(label) is expected


def test_describe_filters_is_readable_and_stable():
    assert (describe_filters({"Cycle year": [2025, 2023], "UK only": "Yes"})
            == "Cycle year: 2023, 2025 · UK only: Yes")
