"""Segments: groups of similar applicants, with outcome rates and honest warnings."""
import os
import sys

import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ame import AmeError
from utils.central_filters import central_sidebar_filters, show_filter_audit
from utils.model_config import TARGETS
from utils.segments import run_segments
from utils.ui import available_characteristics, require_data

st.set_page_config(page_title="Segments", page_icon="🧩", layout="wide")
st.title("🧩 Segments")
st.caption("Groups of applicants who look alike. These describe the data; they do not explain "
           "outcomes. Use Drivers for adjusted effects.")

raw = require_data()
f = central_sidebar_filters(raw, regression=True)
df = f.df
show_filter_audit(f)

outcome = st.radio("Outcome to compare segments on", list(TARGETS), horizontal=True)
num, den = TARGETS[outcome]
if num not in df.columns or (den and den not in df.columns):
    st.error("Outcome columns missing from the data.")
    st.stop()
base = df if den is None else df[df[den] == 1]
base = base[base[num].notna()]

chars = available_characteristics(df)
chosen = st.multiselect("Build segments from", sorted(chars),
                        default=[k for k in ["Age band", "Gender", "Ethnic group (high level)",
                                             "IMD quintile", "Parental HE Indicator"] if k in chars])
auto = st.checkbox("Choose the number of segments automatically", value=True)
k = None if auto else st.slider("Number of segments", 2, 8, 4)
if len(chosen) < 2:
    st.info("Pick at least two characteristics.")
    st.stop()

cols = [chars[c] for c in chosen]
try:
    with st.spinner("Building segments…"):
        r = run_segments(base, cols, num, k=k)
except AmeError as e:
    st.warning(str(e))
    st.stop()

st.caption("**Filters in force:** " + f.line)
for w in r.warnings:
    st.warning(w)
m = st.columns(3)
m[0].metric("Segments", r.k)
m[1].metric("Separation (silhouette)", f"{r.silhouette:.2f}", help="Above ~0.25 is reasonable; below 0.15 is weak.")
m[2].metric("Stability", f"{r.stability:.2f}", help="1 = identical groups when rows are resampled.")

p = r.profiles
fig = go.Figure(go.Bar(
    x=p["Segment"].astype(str), y=p["Outcome rate %"],
    error_y=dict(type="data", symmetric=False, array=p["CI Upper %"] - p["Outcome rate %"],
                 arrayminus=p["Outcome rate %"] - p["CI Lower %"]),
    marker_color="#67072d", customdata=p[["N", "Distinguished by"]],
    hovertemplate="Segment %{x}<br>%{y:.1f}% · N=%{customdata[0]:,}<br>%{customdata[1]}<extra></extra>"))
overall = base[num].astype(float).mean() * 100
fig.add_hline(y=overall, line_dash="dash", annotation_text=f"Overall {overall:.1f}%")
fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", font=dict(color="black"),
                  yaxis_title=f"{outcome} (%) with 95% CI", xaxis_title="Segment", height=380)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(p.round(1), use_container_width=True, hide_index=True)
st.markdown("### What defines each segment")
for s_id in p["Segment"]:
    with st.expander(f"Segment {s_id}: {p.loc[p.Segment == s_id, 'Distinguished by'].iloc[0]}"):
        t = r.traits[s_id]
        st.dataframe(t.round(1), hide_index=True, use_container_width=True) if len(t) else st.write("No clearly over-represented category.")
st.download_button("⬇️ Download segment profiles", p.assign(Filters=f.line).to_csv(index=False).encode(),
                   file_name="segments.csv", mime="text/csv")
