"""Every page runs end to end on synthetic data with no error box."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.ame import looks_undergraduate  # noqa: E402


def make_data(seed=3, n_apps=3000):
    rng = np.random.default_rng(seed)
    apps = pd.DataFrame({
        "application_number": [f"A{i}" for i in range(n_apps)],
        "ethnic_group_summary_lvl": rng.choice(["White", "Black", "Asian", "Mixed", "Unknown"], n_apps,
                                               p=[.6, .15, .15, .08, .02]),
        "gender": rng.choice(["Women", "Men", None], n_apps, p=[.5, .45, .05]),
        "age_group_2band": rng.choice(["Young (under 21)", "Mature (21 and over)"], n_apps, p=[.85, .15]),
        "imd_quint_2019": rng.choice(["1", "2", "3", "4", "5", None], n_apps),
        "parental_he_indicator": rng.choice(["Yes", "No", "Not known"], n_apps),
        "disability_indicator": rng.choice(["Yes", "No"], n_apps, p=[.1, .9]),
        "app_domicile_high_lvl_3_lvls": rng.choice(["UK", "EU (excluding UK)", "Not EU", None], n_apps,
                                                   p=[.8, .08, .07, .05]),
        "college": rng.choice(["Arts", "Science", "Health"], n_apps),
        "fsm_indicator": rng.choice(["Yes", "No"], n_apps, p=[.2, .8]),
        "qual_alevel_only": rng.choice(["Y", "N"], n_apps, p=[.6, .4]),
        "qual_btec_only": rng.choice(["Y", "N"], n_apps, p=[.1, .9]),
        "degree_type": rng.choice(["BSc", "BA Hons", "Postgraduate Taught", "MBA"], n_apps,
                                  p=[.5, .3, .15, .05]),
        "u": rng.normal(0, .7, n_apps),
    })
    df = apps.loc[apps.index.repeat(rng.integers(1, 4, n_apps))].reset_index(drop=True)
    df["cycle_year"] = rng.choice([2022, 2023, 2024, 2025, np.nan], len(df), p=[.1, .25, .25, .35, .05])
    df["subject_group_summary_lvl"] = rng.choice(["Law", "Nursing", "Business", "Maths"], len(df))
    df["application"] = 1
    eta = -0.2 + df.u - 0.4 * (df.ethnic_group_summary_lvl == "Black") + 0.3 * (df.gender == "Men")
    df["offer_as_at_30_june"] = (rng.random(len(df)) < 1 / (1 + np.exp(-eta))).astype(int)
    df["acceptance"] = ((rng.random(len(df)) < .6) & (df.offer_as_at_30_june == 1)).astype(int)
    return df.drop(columns="u")



def load(page, df):
    at = AppTest.from_file(str(ROOT / page), default_timeout=300)
    if df is not None:
        at.session_state["df"] = df
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    return at


@pytest.fixture(scope="module")
def df():
    return make_data()


def test_home_without_data():
    load("app.py", None)


def test_home_with_data_shows_checks(df):
    at = load("app.py", df)
    assert any("applications" in i.value for i in at.info)


@pytest.mark.parametrize("page", ["pages/1_Drivers.py", "pages/2_Segments.py", "pages/3_Probability.py"])
def test_pages_run(page, df):
    load(page, df)


@pytest.mark.parametrize("page", ["pages/1_Drivers.py", "pages/2_Segments.py", "pages/3_Probability.py"])
def test_pages_ask_for_data_first(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Home page" in w.value for w in at.warning)


def test_drivers_filters_match_pandas(df):
    at = load("pages/1_Drivers.py", df)
    exp = df[df.cycle_year.isin([2023, 2024, 2025]) & (df.app_domicile_high_lvl_3_lvls == "UK")
             & df.degree_type.map(looks_undergraduate)]
    assert any(f"{len(exp):,}" in m.value for m in at.sidebar.markdown if "Showing" in m.value)


def test_drivers_includes_a_level_and_btec(df):
    at = load("pages/1_Drivers.py", df)
    ticked = [m for m in at.multiselect if m.label == "Characteristics"][0].value
    assert "A-Level only" in ticked and "BTEC only" in ticked


def test_pages_reach_their_results(df):
    assert len(load("pages/1_Drivers.py", df).dataframe) >= 1
    assert len(load("pages/2_Segments.py", df).dataframe) >= 1
    at = load("pages/3_Probability.py", df)
    assert any("AUC" == m.label for m in at.metric)
