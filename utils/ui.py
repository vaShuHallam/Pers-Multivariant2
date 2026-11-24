"""
utils/ui.py - the shared model-setup block used by the Drivers and Probability pages, so both
configure the model identically (same filters, outcome, characteristics, references).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.ame import (EXCLUDE, NOT_RECORDED, SMALL_GROUP, AmeError, categorise,
                       describe_filters, fit_robust)
from utils.central_filters import central_sidebar_filters, show_filter_audit
from utils.model_config import (DEFAULT_EXCLUDE, DEFAULT_ORDER, DENOM_LABEL, TARGETS,
                                resolve_characteristics, resolve_controls)


def require_data() -> pd.DataFrame:
    if "df" not in st.session_state:
        st.warning("Please load your data on the Home page first.")
        st.stop()
    return st.session_state["df"]


def available_characteristics(df: pd.DataFrame) -> dict:
    return resolve_characteristics(df)


def missing_share(s: pd.Series) -> float:
    return float((s.isna() | s.astype(str).str.strip().str.lower().isin(EXCLUDE)).mean())


def model_setup(with_controls: bool = True):
    """Render filters + model settings; return a dict, or stop the page with a message."""
    raw = require_data()
    filters = central_sidebar_filters(raw, regression=True)
    df = filters.df
    show_filter_audit(filters)

    with st.sidebar:
        st.markdown("---")
        st.markdown("### Model settings")
        missing_mode = st.radio("Missing values", ["Keep as 'Not recorded'", "Drop rows"])
        min_cat = st.slider("Merge categories smaller than", 0, 200, 30, step=10)
        auto = st.checkbox(
            "Simplify automatically if the model can't run", value=True,
            help="On a small selection some groups are too small for the model. When this is ticked "
                 "the page merges small groups and, if needed, leaves out the variable that breaks "
                 "the model, and tells you exactly what it changed. Untick it to see the raw error.")

    outcome = st.radio("Outcome", list(TARGETS), horizontal=True)
    num, den = TARGETS[outcome]
    if num not in df.columns or (den and den not in df.columns):
        st.error(f"`{num}` or `{den}` is not in the data.")
        st.stop()
    base = df if den is None else df[df[den] == 1]
    base = base[base[num].notna()]
    st.caption(f"Modelling {outcome.lower()} over {DENOM_LABEL[den]}: {len(base):,} rows after filters.")
    if len(base) < 200:
        st.warning("Fewer than 200 rows after filters; widen the filters.")
        st.stop()

    char_map = available_characteristics(df)
    if not char_map:
        st.error("None of the expected characteristic columns are in this data.")
        st.stop()
    miss = {k: missing_share(base[c]) for k, c in char_map.items()}
    ordered = [v for v in DEFAULT_ORDER if v in char_map]
    ordered += [v for v in sorted(char_map) if v not in ordered and v not in DEFAULT_EXCLUDE]
    defaults = [v for v in ordered if miss[v] < 0.4] or sorted(char_map, key=miss.get)[:4]
    selected = st.multiselect("Characteristics", sorted(char_map), default=defaults,
                              format_func=lambda k: f"{k} ({100 * miss[k]:.0f}% missing)")
    controls_map = resolve_controls(df)
    controls = []
    if with_controls and controls_map:
        controls = st.multiselect(
            "Controls (adjusted for, not reported)", list(controls_map),
            default=[k for k in ["Cycle year", "Subject group"] if k in controls_map])
        if "College" in controls and "Department" in controls:
            controls.remove("College")
    if not selected:
        st.warning("Select at least one characteristic.")
        st.stop()

    variables = {k: char_map[k] for k in selected}
    variables.update({k: controls_map[k] for k in controls})
    try:
        prep = categorise(base, num, variables,
                          group_col="application_number" if "application_number" in base.columns else None,
                          missing="drop" if missing_mode == "Drop rows" else "keep", min_cat=min_cat)
    except AmeError as e:
        st.warning(str(e))
        st.stop()

    st.markdown("#### Reference groups")
    st.caption("Every effect is a difference from the reference group. Default: largest group.")
    refs, cols = {}, st.columns(3)
    for i, label in enumerate(variables):
        counts = prep.work[f"__{label}"].value_counts()
        opts = [v for v in counts.index if v not in (NOT_RECORDED, SMALL_GROUP)] or list(counts.index)
        with cols[i % 3]:
            refs[label] = st.selectbox(label, opts, key=f"ref_{label}")

    group_col = "application_number" if "application_number" in base.columns else None
    missing = "drop" if missing_mode == "Drop rows" else "keep"
    try:
        fit = fit_robust(base, num, variables, refs, controls, group_col=group_col,
                         missing=missing, min_cat=min_cat, auto=auto)
    except AmeError as e:
        st.error(str(e))
        st.stop()
    if fit.steps:
        st.warning("**The model could not run exactly as selected, so it was simplified:**\n\n"
                   + "\n".join(f"- {x}" for x in fit.steps)
                   + "\n\nResults below are for the simplified model. Untick 'Simplify automatically' "
                     "in the sidebar to see the original error, or widen the filters.")
    kept = [k for k in selected if k in fit.variables]
    kept_controls = [k for k in controls if k in fit.variables]
    line = filters.line + " · " + describe_filters({
        "Outcome": outcome, "Missing values": missing_mode, "Merge under": fit.min_cat,
        "Characteristics": kept, "Controls": kept_controls or "none",
        "Auto-simplified": "yes: " + " ".join(fit.steps) if fit.steps else "no"})
    return dict(filters=filters, df=df, base=base, outcome=outcome, num=num, prep=fit.prep,
                refs=fit.refs, model=fit.model, steps=fit.steps, selected=kept,
                controls=kept_controls, variables=fit.variables, line=line, char_map=char_map)