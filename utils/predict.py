"""
utils/predict.py
================
Probability engine (no Streamlit). Uses the SAME fitted model as the Drivers page, so
"what drives it" and "what is the chance" can never disagree.

* predict_profile  - chance for one chosen applicant profile, with a 95% interval
* holdout_metrics  - honest accuracy: fit on some applications, score on unseen ones
* calibration_table - predicted vs actual by decile
No class weighting: probabilities stay true probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

from utils.ame import AmeError, FittedModel, Prepared, fit_model


def _design(model: FittedModel, rows: pd.DataFrame) -> np.ndarray:
    X = np.zeros((len(rows), len(model.meta) + 1))
    X[:, 0] = 1.0
    for j, (label, cat) in enumerate(model.meta, start=1):
        X[:, j] = (rows[f"__{label}"].to_numpy() == cat).astype(float)
    return X


def predict_profile(model: FittedModel, profile: dict) -> dict:
    """Chance of the outcome for one profile {characteristic label: category}.

    Anything not given sits at its reference category. The interval is built on the
    log-odds scale and transformed back, so it always stays between 0 and 100%.
    """
    x = np.zeros(len(model.beta))
    x[0] = 1.0
    for j, (label, cat) in enumerate(model.meta, start=1):
        chosen = profile.get(label, model.refs_used.get(label))
        if str(chosen) == str(cat):
            x[j] = 1.0
    eta = float(x @ model.beta)
    se = float(np.sqrt(max(x @ model.cov @ x, 0.0)))
    z = stats.norm.ppf(0.975)
    return {"p": float(expit(eta)), "lo": float(expit(eta - z * se)),
            "hi": float(expit(eta + z * se)), "se_logit": se}


@dataclass
class Holdout:
    auc: float
    brier: float
    brier_baseline: float      # Brier score of always predicting the training average
    n_train: int
    n_test: int
    calibration: pd.DataFrame
    scored: pd.DataFrame       # y, p for each test row


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), min(bins, max(2, len(d) // 20)),
                       labels=False)
    out = d.groupby("bin").agg(n=("y", "size"), predicted=("p", "mean"),
                               actual=("y", "mean")).reset_index(drop=True)
    return out


def holdout_metrics(prep: Prepared, target: str, refs: dict, controls=(),
                    test_frac: float = 0.25, seed: int = 0) -> Holdout:
    """Split by application (never within one), fit on the larger part, score the rest."""
    work = prep.work
    keys = (work[prep.group_col].astype(str) if prep.group_col
            else pd.Series(np.arange(len(work)), index=work.index).astype(str))
    uniq = keys.unique()
    rng = np.random.default_rng(seed)
    test_keys = set(rng.choice(uniq, size=max(1, int(len(uniq) * test_frac)), replace=False))
    is_test = keys.isin(test_keys).to_numpy()
    train, test = work[~is_test], work[is_test]
    if len(test) < 50 or train[target].nunique() < 2 or test[target].nunique() < 2:
        raise AmeError("Not enough rows to hold some back for testing.")
    try:
        model = fit_model(replace(prep, work=train), target, refs, controls)
    except AmeError as e:
        raise AmeError(f"The holdout model could not be fitted: {e}") from e
    # unseen categories (absent from the training part) simply get no coefficient
    p = expit(_design(model, test) @ model.beta)
    y = test[target].astype(float).to_numpy()
    base = float(train[target].astype(float).mean())
    return Holdout(auc=_auc(y, p), brier=float(np.mean((p - y) ** 2)),
                   brier_baseline=float(np.mean((base - y) ** 2)),
                   n_train=len(train), n_test=len(test),
                   calibration=calibration_table(y, p),
                   scored=pd.DataFrame({"y": y, "p": p}))
