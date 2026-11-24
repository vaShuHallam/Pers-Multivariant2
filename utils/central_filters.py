"""
utils/central_filters.py
========================
ONE place for every filter, with an audit trail of exactly what each one removed.

Why this exists
---------------
Filters used to be spread over utils/filters.py and over individual pages (UK only,
undergraduate only, college/school/course drill-down). Two pages could therefore
analyse different people without anyone noticing. Here:

  * every filter is a Rule, applied in a stated order;
  * the audit table shows rows before / removed / after for every rule;
  * rows with a missing value are never dropped silently: "(missing)" is an
    explicit choice in every list;
  * 'nothing selected' is reported, not silently treated as 'everything';
  * 'UK only' matches exact UK labels, not any text containing 'uk'
    (which also matched 'EU (excluding UK)');
  * 'undergraduate only' matches whole words, not substrings
    (which also matched 'Postgraduate Taught').

The engine (Rule, apply_rules) has no Streamlit inside and is covered by
tests/test_central_filters.py. The sidebar (central_sidebar_filters) only turns
widget choices into Rules.

utils/filters.py is left untouched, so existing pages behave exactly as before.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from utils.ame import looks_undergraduate

MISSING_LABEL = "(missing)"

# Domicile labels counted as UK. EXACT match on the cleaned, lower-cased label.
UK_LABELS = {"uk", "united kingdom", "england", "wales", "scotland", "northern ireland"}

DEFAULT_YEARS = ("2023", "2024", "2025")


# ---------------------------------------------------------------------------
# Engine (no Streamlit)
# ---------------------------------------------------------------------------
def as_text(s: pd.Series) -> pd.Series:
    """Comparable text version of a column: integers lose '.0', spaces stripped,
    true missing values stay missing (NA)."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        def fmt(v):
            if pd.isna(v):
                return pd.NA
            f = float(v)
            return str(int(f)) if f.is_integer() else str(f)
        return s.map(fmt).astype("string")
    return s.astype("string").str.strip()


@dataclass
class Rule:
    """One filter. With neither `keep` nor `predicate` it means 'no restriction'."""
    label: str
    column: str
    keep: Optional[tuple] = None                      # values to keep (text); may include MISSING_LABEL
    predicate: Optional[Callable] = None              # text Series -> boolean Series
    text: str = ""                                    # human description for the audit

    def describe(self) -> str:
        if self.text:
            return self.text
        if self.keep is None and self.predicate is None:
            return f"{self.label}: all values"
        vals = list(self.keep or [])
        shown = ", ".join(map(str, vals[:6])) + (f" (+{len(vals) - 6} more)" if len(vals) > 6 else "")
        return f"{self.label}: {shown}"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        if self.keep is None and self.predicate is None:
            return pd.Series(True, index=df.index)
        s = as_text(df[self.column])
        if self.keep is not None:
            keep = {str(v) for v in self.keep}
            m = s.isin(keep - {MISSING_LABEL})
            if MISSING_LABEL in keep:
                m = m | s.isna()
        else:
            m = self.predicate(s)
        return pd.Series(m, index=df.index).fillna(False).astype(bool)


@dataclass
class FilterResult:
    df: pd.DataFrame
    audit: pd.DataFrame
    removed_detail: dict = field(default_factory=dict)   # rule label -> Series(value counts removed)
    line: str = ""                                       # one readable line for captions / downloads
    notes: list = field(default_factory=list)            # warnings the user should see
    n_start: int = 0


def apply_rules(df: pd.DataFrame, rules: list, notes: Optional[list] = None) -> FilterResult:
    """Apply rules in order; return the filtered data and a step-by-step audit."""
    notes = list(notes or [])
    start = len(df)
    cur = df
    rows = [{"Step": 0, "Filter": "All data loaded", "Rule": "—", "Rows before": start,
             "Removed": 0, "Rows after": start, "% of start removed": 0.0,
             "Removed because value missing": 0}]
    detail, active_text = {}, []

    for i, r in enumerate(rules, start=1):
        before = len(cur)
        if r.column not in cur.columns:
            notes.append(f"{r.label}: column '{r.column}' is not in the data, so this filter "
                         "could not be applied.")
            rows.append({"Step": i, "Filter": r.label, "Rule": f"column '{r.column}' not in data",
                         "Rows before": before, "Removed": 0, "Rows after": before,
                         "% of start removed": 0.0, "Removed because value missing": 0})
            continue
        m = r.mask(cur)
        removed = cur[~m]
        if len(removed):
            vals = as_text(removed[r.column]).fillna(MISSING_LABEL).value_counts()
            detail[r.label] = vals
        n_missing = int(as_text(removed[r.column]).isna().sum()) if len(removed) else 0
        cur = cur[m]
        rows.append({"Step": i, "Filter": r.label, "Rule": r.describe(),
                     "Rows before": before, "Removed": len(removed), "Rows after": len(cur),
                     "% of start removed": round(100 * len(removed) / start, 2) if start else 0.0,
                     "Removed because value missing": n_missing})
        if r.keep is not None or r.predicate is not None:
            active_text.append(r.describe())

    audit = pd.DataFrame(rows)
    line = " · ".join(active_text) if active_text else "no filters restrict the data"
    return FilterResult(df=cur, audit=audit, removed_detail=detail,
                        line=f"{line} → {len(cur):,} of {start:,} rows",
                        notes=notes, n_start=start)


# ---------------------------------------------------------------------------
# Rule builders (pure, testable)
# ---------------------------------------------------------------------------
def options_for(df: pd.DataFrame, column: str) -> list:
    """Sorted distinct values as text, plus '(missing)' when the column has gaps."""
    s = as_text(df[column])
    vals = sorted(s.dropna().unique().tolist(),
                  key=lambda v: (0, float(v)) if _isnum(v) else (1, v.lower()))
    return vals + ([MISSING_LABEL] if s.isna().any() else [])


def _isnum(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def rule_from_selection(label, column, selected, options, notes) -> "Rule":
    """Selection -> Rule. Everything selected = no restriction; nothing selected is
    reported (and, as on the old pages, applies no restriction)."""
    if not selected:
        notes.append(f"{label}: nothing selected, so this filter was NOT applied.")
        return Rule(label, column)
    if set(selected) == set(options):
        return Rule(label, column)
    return Rule(label, column, keep=tuple(selected))


def uk_default_values(values) -> list:
    """Domicile values counted as UK: exact label match only."""
    return [v for v in values if str(v).strip().lower() in UK_LABELS]


def undergraduate_default_values(values) -> list:
    return [v for v in values if looks_undergraduate(v)]


# ---------------------------------------------------------------------------
# Streamlit sidebar
# ---------------------------------------------------------------------------
CORE = [  # (label, column) in the order the old shared sidebar used
    ("Cycle year", "cycle_year"), ("College", "college"), ("School", "school"),
    ("Gender", "gender"), 
    ("Course name", "course_name"), ("Domicile", "app_domicile_high_lvl_3_lvls"),
    ("Placed status", "placed_status"),
]
REGRESSION_ORDER = ["cycle_year", "college", "school", "gender", 
                    "course_name", "placed_status"]
DRILL = [("College", "college"), ("School", "school"), ("Course name", "course_name")]
DRILL_KEYWORDS = {"college": ["college"], "school": ["school"], "course_name": ["course_name", "coursename", "course"]}
DOMICILE_COLS = ["app_domicile_high_lvl_3_lvls"]     # values: UK / EU / Non EU
DOMICILE_COL = DOMICILE_COLS[0]


def find_column(df: pd.DataFrame, preferred: str, keywords=()) -> Optional[str]:
    """The column to use for a filter: the expected name, else (case-insensitively) the
    first column whose name contains a keyword. None if nothing fits."""
    if preferred in df.columns:
        return preferred
    lower = {str(c).lower(): c for c in df.columns}
    if preferred.lower() in lower:
        return lower[preferred.lower()]
    for kw in keywords:
        hits = [c for c in df.columns if kw in str(c).lower()
                and not str(c).lower().startswith("__")]
        if hits:
            return sorted(hits, key=lambda c: len(str(c)))[0]
    return None


def domicile_column(df: pd.DataFrame) -> Optional[str]:
    return next((c for c in DOMICILE_COLS if c in df.columns), None)


def central_sidebar_filters(df: pd.DataFrame, regression: bool = False, key: str = "cf") -> FilterResult:
    """Draw the sidebar and return a FilterResult.

    regression=False : all core filters as multiselects (same set as utils/filters.py)
    regression=True  : adds the analysis scope (UK only, undergraduate only) and the
                       College -> School -> Course drill-down used by the regression pages
    """
    import streamlit as st

    sb = st.sidebar
    sb.markdown("## 🔽 Filters")
    rules, notes = [], []

    def multiselect(label, column, default=None, expanded_help=None):
        if column not in df.columns:
            return
        options = options_for(df, column)
        dflt = options if default is None else [o for o in options if o in default]
        sel = sb.multiselect(label, options, default=dflt, key=f"{key}_{column}", help=expanded_help)
        rules.append(rule_from_selection(label, column, sel, options, notes))

    if not regression:
        for label, column in CORE:
            multiselect(label, column, default=DEFAULT_YEARS if column == "cycle_year" else None)
    else:
        label_of = dict((c, l) for l, c in CORE)
        drill = {w: l for l, w in DRILL}

        def cascade(label, wanted):
            """College / School / Course name: empty = everything; options follow the rules above."""
            column = find_column(df, wanted, DRILL_KEYWORDS.get(wanted, ()))
            if column is None:
                sb.caption(f"No {label.lower()} column found in this data (looked for '{wanted}').")
                return
            cur = apply_rules(df, rules).df
            options = [v for v in options_for(cur, column) if v != MISSING_LABEL]
            picks = sb.multiselect(label, options, key=f"{key}_drill_{column}",
                                   placeholder="All", help=f"Column: {column}")
            if picks:
                rules.append(Rule(label, column, keep=tuple(picks),
                                  text=f"{label}: " + ", ".join(picks)))

        # Same order as the old shared sidebar (utils/filters.py)
        for column in REGRESSION_ORDER:
            if column in drill:
                cascade(drill[column], column)
            else:
                multiselect(label_of[column], column,
                            default=DEFAULT_YEARS if column == "cycle_year" else None)

        sb.markdown("---")
        sb.markdown("### 📐 Analysis scope")
        dom_col = domicile_column(df)
        if dom_col:
            dom_values = [v for v in options_for(df, dom_col) if v != MISSING_LABEL]
            uk_only = sb.checkbox("UK applicants only", value=True, key=f"{key}_uk_only",
                                  help=f"Uses the column '{dom_col}'. Matching is exact.")
            if uk_only:
                uk_vals = uk_default_values(dom_values)
                if uk_vals:
                    rules.append(Rule("UK only", dom_col, keep=tuple(uk_vals),
                                      text="UK only: " + ", ".join(uk_vals)))
                else:
                    # None of the values look like UK: let the user say which one(s) mean UK.
                    uk_vals = sb.multiselect(
                        "Which value(s) mean UK?", dom_values, key=f"{key}_uk_vals",
                        help=f"None of the values in '{dom_col}' were recognised as UK.")
                    if uk_vals:
                        rules.append(Rule("UK only", dom_col, keep=tuple(uk_vals),
                                          text="UK only: " + ", ".join(uk_vals)))
                    else:
                        notes.append("UK only is ticked but no value in "
                                     f"'{dom_col}' was recognised as UK, so this filter was NOT applied.")
            else:
                multiselect("Domicile", dom_col)
        if "degree_type" in df.columns:
            ug_only = sb.checkbox("Undergraduate only", value=True, key=f"{key}_ug_only")
            if ug_only:
                types = [v for v in options_for(df, "degree_type") if v != MISSING_LABEL]
                ug_vals = sb.multiselect("Degree types counted as undergraduate", types,
                                         default=undergraduate_default_values(types), key=f"{key}_ug_vals",
                                         help="Chosen by whole-word match; edit the list if a label is wrong.")
                if ug_vals:
                    rules.append(Rule("Undergraduate only", "degree_type", keep=tuple(ug_vals),
                                      text="Undergraduate only: " + ", ".join(ug_vals)))
                else:
                    notes.append("Undergraduate only is ticked but no degree type is counted as "
                                 "undergraduate, so this filter was NOT applied.")
        else:
            notes.append("No 'degree_type' column in the data, so there is no undergraduate filter.")

    result = apply_rules(df, rules, notes)
    sb.markdown("---")
    pct = 100 * len(result.df) / result.n_start if result.n_start else 0
    sb.markdown(f"**Showing:** {len(result.df):,} of {result.n_start:,} rows ({pct:.0f}%)")
    for n in result.notes:
        sb.warning(n)
    return result


def show_filter_audit(result: FilterResult, expanded: bool = False) -> None:
    """Main-page panel: what each filter removed, and which values went."""
    import streamlit as st

    removed = int(result.n_start - len(result.df))
    with st.expander(f"🔎 What the filters removed — {removed:,} of {result.n_start:,} rows "
                     f"({100 * removed / result.n_start if result.n_start else 0:.0f}%)",
                     expanded=expanded):
        st.caption("Filters are applied top to bottom, so a row is counted against the first "
                   "filter that removes it.")
        st.dataframe(result.audit, hide_index=True, use_container_width=True)
        for n in result.notes:
            st.warning(n)
        if result.removed_detail:
            pick = st.selectbox("Which values did a filter remove?", list(result.removed_detail),
                                key="cf_audit_pick")
            vc = result.removed_detail[pick].rename("Rows removed").reset_index()
            vc.columns = ["Value", "Rows removed"]
            st.dataframe(vc.head(25), hide_index=True, use_container_width=True)
        st.download_button("⬇️ Download filter audit", data=result.audit.to_csv(index=False).encode("utf-8"),
                           file_name="filter_audit.csv", mime="text/csv", key="cf_audit_dl")