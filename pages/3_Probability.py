"""Probability: estimated chance for a chosen profile, plus an honest accuracy check."""
import os
import sys

import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ame import AmeError
from utils.narrative import prediction_sentence
from utils.predict import holdout_metrics, predict_profile
from utils.ui import model_setup

st.set_page_config(page_title="Probability", page_icon="🎯", layout="wide")
st.title("🎯 Probability")
st.caption("The same model as Drivers, used to estimate the chance for a chosen profile.")

s = model_setup()
model = s["model"]
st.caption("**Filters in force:** " + s["line"])

st.markdown("### Choose a profile")
profile, cols = {}, st.columns(3)
for i, label in enumerate(s["variables"]):
    opts = [c for (l, c) in model.meta if l == label]
    ref = model.refs_used.get(label)
    if ref is None:
        continue
    opts = [ref] + opts
    with cols[i % 3]:
        profile[label] = st.selectbox(label, opts, key=f"prof_{label}")
pr = predict_profile(model, profile)
st.metric(f"Estimated {s['outcome'].lower()}", f"{pr['p'] * 100:.0f}%",
          f"95% interval {pr['lo'] * 100:.0f}% to {pr['hi'] * 100:.0f}%", delta_color="off")
st.write(prediction_sentence(pr, s["outcome"]))

st.markdown("### How accurate is the model?")
st.caption("Some applications are held back, the model is fitted on the rest, and its estimates "
           "are compared with what actually happened on the held-back applications.")
try:
    h = holdout_metrics(s["prep"], s["num"], s["refs"], s["controls"])
except AmeError as e:
    st.warning(str(e))
    st.stop()
a = st.columns(4)
a[0].metric("AUC", f"{h.auc:.2f}", help="0.5 = no better than chance; 1 = perfect.")
a[1].metric("Brier score", f"{h.brier:.3f}", help="Lower is better.")
a[2].metric("Brier if we ignored all characteristics", f"{h.brier_baseline:.3f}")
a[3].metric("Held-back rows", f"{h.n_test:,}")
if h.auc < 0.6:
    st.warning("The characteristics separate outcomes only weakly. Use the figures for "
               "comparing groups, not for predicting individuals.")
cal = h.calibration
fig = go.Figure()
fig.add_trace(go.Scatter(x=cal.predicted * 100, y=cal.actual * 100, mode="markers+lines",
                         name="Deciles", marker_color="#67072d"))
top = max(cal.predicted.max(), cal.actual.max()) * 100
fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Perfect",
                         line=dict(dash="dash", color="#999")))
fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", font=dict(color="black"),
                  xaxis_title="Predicted (%)", yaxis_title="Actual (%)", height=360)
st.plotly_chart(fig, use_container_width=True)
st.caption("Points on the dashed line mean predicted chances match reality.")