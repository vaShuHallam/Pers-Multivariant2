"""
tests/test_central_filters.py
=============================
Checks for the central filter engine. Run from the project root:

    python -m pytest tests -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.central_filters import (MISSING_LABEL, Rule, apply_rules, as_text,       # noqa: E402
                                   options_for, rule_from_selection,
                                   uk_default_values, undergraduate_default_values)


@pytest.fixture
def data():
    return pd.DataFrame({
        "cycle_year": [2022, 2023, 2023, 2024, 2025, 2025, 2025, np.nan, 2025, 2025],
        "dom": ["UK", "UK", "EU (excluding UK)", "Not EU", "UK", "UK", "England", "UK", None, "UK"],
        "degree_type": ["BSc", "BA Hons", "Postgraduate Taught", "MBA", "BSc", "BSc", "UG",
                        "BSc", "BSc", "PGT"],
        "college": list("AABBAABBAA"),
    })


def test_audit_reconciles_at_every_step(data):
    rules = [Rule("Year", "cycle_year", keep=("2023", "2024", "2025")),
             Rule("UK", "dom", keep=("UK", "England"))]
    res = apply_rules(data, rules)
    a = res.audit
    assert a.loc[0, "Rows after"] == len(data)
    assert ((a["Rows before"] - a["Removed"]) == a["Rows after"]).all()
    assert (a["Rows after"].iloc[:-1].values == a["Rows before"].iloc[1:].values).all()
    assert a["Removed"].sum() == len(data) - len(res.df)
    assert res.n_start == len(data)


def test_missing_values_are_never_dropped_silently(data):
    """A year filter that keeps every real year still drops the row with no year,
    and the audit says so explicitly. Choosing '(missing)' keeps it."""
    drops = apply_rules(data, [Rule("Year", "cycle_year", keep=("2022", "2023", "2024", "2025"))])
    row = drops.audit.iloc[1]
    assert row["Removed"] == 1 and row["Removed because value missing"] == 1
    keeps = apply_rules(data, [Rule("Year", "cycle_year",
                                    keep=("2022", "2023", "2024", "2025", MISSING_LABEL))])
    assert keeps.audit.iloc[1]["Removed"] == 0 and len(keeps.df) == len(data)


def test_selecting_everything_means_no_restriction(data):
    notes = []
    options = options_for(data, "cycle_year")
    assert options[-1] == MISSING_LABEL                    # gaps are offered as a choice
    rule = rule_from_selection("Year", "cycle_year", options, options, notes)
    assert apply_rules(data, [rule]).audit.iloc[1]["Removed"] == 0 and notes == []


def test_nothing_selected_is_reported_and_applies_no_restriction(data):
    notes = []
    rule = rule_from_selection("Year", "cycle_year", [], options_for(data, "cycle_year"), notes)
    res = apply_rules(data, [rule], notes)
    assert len(res.df) == len(data)
    assert any("NOT applied" in n for n in res.notes)


def test_float_years_match_integer_choices():
    df = pd.DataFrame({"cycle_year": [2023.0, 2024.0, 2025.0]})
    assert options_for(df, "cycle_year") == ["2023", "2024", "2025"]
    res = apply_rules(df, [Rule("Year", "cycle_year", keep=("2024",))])
    assert res.df["cycle_year"].tolist() == [2024.0]


def test_uk_only_matches_exact_labels_not_substrings(data):
    """The old default matched any label containing 'uk', so 'EU (excluding UK)' was kept."""
    values = [v for v in options_for(data, "dom") if v != MISSING_LABEL]
    uk = uk_default_values(values)
    assert "EU (excluding UK)" not in uk and {"UK", "England"} <= set(uk)
    res = apply_rules(data, [Rule("UK only", "dom", keep=tuple(uk))])
    assert "EU (excluding UK)" not in res.df["dom"].values
    # rows with no domicile are removed AND counted as missing
    assert res.audit.iloc[1]["Removed because value missing"] == 1


def test_undergraduate_default_uses_whole_words(data):
    types = options_for(data, "degree_type")
    ug = undergraduate_default_values(types)
    assert set(ug) == {"BSc", "BA Hons", "UG"}
    for pg in ("Postgraduate Taught", "MBA", "PGT"):
        assert pg not in ug


def test_removed_values_are_reported(data):
    res = apply_rules(data, [Rule("UK only", "dom", keep=("UK", "England"))])
    detail = res.removed_detail["UK only"]
    assert detail["EU (excluding UK)"] == 1 and detail["Not EU"] == 1 and detail[MISSING_LABEL] == 1


def test_missing_column_is_reported_not_fatal(data):
    res = apply_rules(data, [Rule("Gender", "gender", keep=("Women",))])
    assert len(res.df) == len(data)
    assert any("not in the data" in n for n in res.notes)


def test_order_is_stated_and_cascades(data):
    """Rows are attributed to the first filter that removes them."""
    res = apply_rules(data, [Rule("UK", "dom", keep=("UK",)),
                             Rule("Year", "cycle_year", keep=("2025",))])
    assert res.audit["Filter"].tolist() == ["All data loaded", "UK", "Year"]
    assert len(res.df) == (data["dom"].eq("UK") & data["cycle_year"].eq(2025)).sum()


def test_line_summarises_active_filters_only(data):
    res = apply_rules(data, [Rule("Year", "cycle_year", keep=("2025",)), Rule("College", "college")])
    assert "Year: 2025" in res.line and "College" not in res.line
    assert res.line.endswith(f"→ {len(res.df):,} of {len(data):,} rows")


def test_as_text_keeps_gaps_as_gaps():
    s = as_text(pd.Series([1.0, np.nan, 2.5, 3.0]))
    assert s.tolist()[0] == "1" and s.isna().tolist() == [False, True, False, False]
