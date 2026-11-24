"""
utils/segments.py
=================
Applicant segments (no Streamlit). Groups applicants who look alike on the chosen
characteristics, then reports each group's size, outcome rate (with a 95% interval) and
what makes it distinctive. Segments describe; they do not explain. Use the Drivers page
for adjusted effects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from utils.ame import AmeError

WEAK_SILHOUETTE = 0.15      # below this the groups overlap heavily
UNSTABLE_ARI = 0.6          # below this, re-running on other rows gives different groups


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson 95% interval for a proportion (well behaved for small n and rates near 0/1)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def encode(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """One-hot every column; each variable gets equal total weight regardless of how many
    categories it has, so a 10-level variable cannot swamp a yes/no one."""
    parts = []
    for c in cols:
        s = df[c].astype("string").fillna("Not recorded")
        d = pd.get_dummies(s, dtype=float)
        parts.append(d / np.sqrt(max(d.shape[1], 1)) * np.sqrt(2))
        parts[-1].columns = [f"{c}={v}" for v in d.columns]
    return pd.concat(parts, axis=1)


@dataclass
class SegmentRun:
    labels: np.ndarray            # one per row of the input
    k: int
    silhouette: float
    stability: float              # mean adjusted Rand index across resamples
    k_scores: pd.DataFrame        # k, silhouette
    profiles: pd.DataFrame        # one row per segment
    traits: dict                  # segment id -> DataFrame of over-represented categories
    warnings: list


def _fit(X, k, seed):
    return KMeans(n_clusters=k, n_init=5, random_state=seed).fit(X)


def run_segments(df: pd.DataFrame, cols: list, target: str, k: int | None = None,
                 k_range=range(2, 8), max_rows: int = 20000, seed: int = 0,
                 n_boot: int = 5) -> SegmentRun:
    if len(cols) < 2:
        raise AmeError("Choose at least two characteristics to build segments from.")
    d = df.reset_index(drop=True)
    if len(d) < 200:
        raise AmeError("Too few rows to build segments (need at least 200).")
    X = encode(d, cols).to_numpy()
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(d), size=min(len(d), max_rows), replace=False)
    Xs = X[sub]

    scores = []
    for kk in k_range:
        if kk >= len(sub):
            continue
        lab = _fit(Xs, kk, seed).labels_
        ss = silhouette_score(Xs, lab, sample_size=min(5000, len(sub)), random_state=seed)
        scores.append({"k": kk, "silhouette": float(ss)})
    k_scores = pd.DataFrame(scores)
    if k is None:
        k = int(k_scores.loc[k_scores.silhouette.idxmax(), "k"])
    model = _fit(Xs, k, seed)
    labels = model.predict(X)
    sil = float(k_scores.loc[k_scores.k == k, "silhouette"].iloc[0]) if (k_scores.k == k).any() \
        else float(silhouette_score(Xs, model.labels_))

    # stability: refit on bootstrap samples, compare on the same rows
    aris = []
    for b in range(n_boot):
        idx = rng.choice(len(sub), size=len(sub), replace=True)
        mb = _fit(Xs[idx], k, seed + b + 1)
        aris.append(adjusted_rand_score(model.labels_, mb.predict(Xs)))
    stability = float(np.mean(aris))

    y = d[target].astype(float).to_numpy()
    overall = y.mean()
    rows, traits = [], {}
    for s in range(k):
        m = labels == s
        n, pos = int(m.sum()), int(y[m].sum())
        lo, hi = wilson(pos, n)
        lifts = []
        for c in cols:
            col = d[c].astype("string").fillna("Not recorded")
            share_all = col.value_counts(normalize=True)
            share_in = col[m].value_counts(normalize=True)
            for v, sh in share_in.items():
                lifts.append({"Characteristic": c, "Category": v,
                              "Share in segment %": sh * 100,
                              "Share overall %": share_all[v] * 100,
                              "Lift": sh / share_all[v]})
        t = pd.DataFrame(lifts)
        t = t[(t["Share in segment %"] >= 20)].sort_values("Lift", ascending=False)
        t = t[t.Lift >= 1.2].head(6).reset_index(drop=True)
        traits[s] = t
        rows.append({"Segment": s, "N": n, "Share of applicants %": n / len(d) * 100,
                     "Outcome rate %": pos / n * 100 if n else float("nan"),
                     "CI Lower %": lo * 100, "CI Upper %": hi * 100,
                     "Gap to overall (pp)": (pos / n - overall) * 100 if n else float("nan"),
                     "Distinguished by": ", ".join(f"{r.Category}" for r in t.head(2).itertuples())
                     or "no clear trait"})
    prof = pd.DataFrame(rows).sort_values("N", ascending=False).reset_index(drop=True)

    warns = []
    if sil < WEAK_SILHOUETTE:
        warns.append("These applicants do not fall into clearly separate groups (weak "
                     "structure). Treat the segments as a convenient way to slice the data, "
                     "not as real types of applicant.")
    if stability < UNSTABLE_ARI:
        warns.append("The groups change noticeably when the rows change slightly (low "
                     "stability). Do not report them as fixed categories.")
    if (prof.N < 100).any():
        warns.append("Some segments have fewer than 100 applicants; their rates are very uncertain.")
    return SegmentRun(labels=labels, k=k, silhouette=sil, stability=stability,
                      k_scores=k_scores, profiles=prof, traits=traits, warnings=warns)
