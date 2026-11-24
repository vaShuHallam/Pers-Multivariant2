import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ame import CHARS, REFS, make_clustered, make_data  # noqa: E402
from utils.ame import AmeError, categorise, fit_model, fit_prepared  # noqa: E402
from utils.predict import holdout_metrics, predict_profile  # noqa: E402


@pytest.fixture(scope="module")
def model():
    df = make_data()
    return fit_model(categorise(df, "out", CHARS), "out", REFS)


def test_reference_profile_is_close_to_true_probability(model):
    # truth: eta = -0.5 + 0 -> p = 0.3775 for A=Low,B=No,C=No
    r = predict_profile(model, {})
    assert abs(r["p"] - 1 / (1 + np.exp(0.5))) < 0.03
    assert r["lo"] < r["p"] < r["hi"]


def test_profile_matches_known_effect(model):
    r = predict_profile(model, {"A": "High", "B": "Yes", "C": "No"})
    truth = 1 / (1 + np.exp(-(-0.5 - 0.6 + 0.5)))
    assert abs(r["p"] - truth) < 0.04
    assert truth > r["lo"] - 0.02 and truth < r["hi"] + 0.02


def test_interval_stays_inside_zero_one():
    df = make_data(n=400, seed=3)
    m = fit_model(categorise(df, "out", CHARS), "out", REFS)
    r = predict_profile(m, {"A": "High"})
    assert 0 <= r["lo"] <= r["p"] <= r["hi"] <= 1


def test_effects_and_prediction_share_one_fit(model):
    run = fit_prepared(model.prep, "out", REFS)
    assert run.n_used == model.n


def test_holdout_beats_baseline_and_is_calibrated():
    df = make_data(n=20000)
    h = holdout_metrics(categorise(df, "out", CHARS), "out", REFS)
    assert 0.55 < h.auc < 0.8
    assert h.brier <= h.brier_baseline + 1e-4
    cal = h.calibration
    assert (cal.predicted - cal.actual).abs().max() < 0.06


def test_holdout_never_splits_an_application():
    df = make_clustered()
    df = df[df.eth != "Tiny"]      # a 7-row group separates in a 75% fit; see Drivers 'Unstable'
    prep = categorise(df, "offer", {"Eth": "eth", "Gen": "gen"}, group_col="application_number",
                      missing="keep", min_cat=30)
    h = holdout_metrics(prep, "offer", {"Eth": "White", "Gen": "Women"})
    assert h.n_train + h.n_test == len(prep.work)


def test_holdout_too_small_raises():
    df = make_data(n=120)
    with pytest.raises(AmeError):
        holdout_metrics(categorise(df, "out", CHARS), "out", REFS)
