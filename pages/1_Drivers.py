"""Drivers: adjusted marginal effects (unpenalised logit, clustered SEs, delta-method CIs)."""
import os
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ame import effects_table, methods_text
from utils.model_config import short_label
from utils.narrative import headline_findings
from utils.ui import model_setup

st.set_page_config(page_title="Drivers", page_icon="📐", layout="wide")
st.title("📐 Drivers")
st.caption("Which characteristics shift the outcome once the others are held constant?")

s = model_setup()
model = s["model"]
run = effects_table(model)
res, outcome = run.results, s["outcome"]

c = st.columns(4)
c[0].metric("Rows in model", f"{run.n_used:,}")
c[1].metric(outcome, f"{100 * run.outcome_mean:.1f}%")
c[2].metric("Applications (clusters)", f"{run.n_clusters:,}")
c[3].metric("Pseudo-R²", f"{run.pseudo_r2:.3f}")
st.caption("**Filters in force:** " + s["line"])
if run.extreme:
    st.warning("Some coefficients are extreme (near-perfect separation). Do not report flagged groups.")
if run.skipped:
    st.caption("Left out (only one category remains): " + ", ".join(run.skipped))

st.markdown("### Headline findings")
found = headline_findings(res, outcome)
if found:
    for f in found:
        st.markdown("- " + f)
else:
    st.info("No characteristic shows a clear difference after adjustment.")

show = res if st.checkbox("Show controls in chart", value=False) else res[res["Role"] == "Characteristic"]
if not show.empty:
    p = show.iloc[::-1].copy()
    p["label"] = p["Characteristic"].map(short_label) + " — " + p["Category"].astype(str)
    col = np.where(~p["Significant"], "#9aa0a6", np.where(p["AME (pp)"] > 0, "#0d6e7a", "#c94a2d"))
    fig = go.Figure(go.Scatter(
        x=p["AME (pp)"], y=p["label"], mode="markers", marker=dict(color=col, size=10),
        error_x=dict(type="data", symmetric=False, array=p["CI Upper (pp)"] - p["AME (pp)"],
                     arrayminus=p["AME (pp)"] - p["CI Lower (pp)"], color="#555", thickness=1.2, width=4),
        customdata=np.stack([p["N in group"], p["Unadjusted gap (pp)"], p["p"], p["Reference"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>Adjusted %{x:+.1f}pp · Unadjusted %{customdata[1]:+.1f}pp"
                      "<br>vs %{customdata[3]} · p=%{customdata[2]:.3f} · N=%{customdata[0]:,}<extra></extra>"))
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", font=dict(color="black"),
                      xaxis=dict(title="Percentage points vs reference group", gridcolor="#eee", zeroline=False),
                      height=max(380, 26 * len(p)), margin=dict(l=10, r=30, t=10, b=40))
    st.plotly_chart(fig, use_container_width=True)
    refs_line = "; ".join(f"{short_label(k)}: {v}" for k, v in s["refs"].items() if k not in model.controls)
    st.caption("Grey = interval crosses zero. Whiskers: 95% CI, standard errors clustered by application. "
               "Reference groups: " + refs_line + ".")

st.markdown("### Full results")
tbl = res.copy()
for k in ["Rate %", "Unadjusted gap (pp)", "AME (pp)", "CI Lower (pp)", "CI Upper (pp)"]:
    tbl[k] = tbl[k].round(1)
tbl["p"] = tbl["p"].round(4)
st.dataframe(tbl.drop(columns=["Significant"]), use_container_width=True, hide_index=True, height=400)
st.download_button("⬇️ Download results", tbl.assign(Filters=s["line"]).to_csv(index=False).encode(),
                   file_name=f"drivers_{s['num']}.csv", mime="text/csv")
with st.expander("Method note for reports"):
    st.markdown(methods_text(s["refs"], run.n_used, run.n_dropped, run.clustered))