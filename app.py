"""Home: load data, run checks. Every other page reads st.session_state['df']."""
import io
import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.data_checks import check_data

st.set_page_config(page_title="Student Insights", page_icon="🎓", layout="wide")
st.title("🎓 Student Insights")
st.markdown("""
Three pages, one set of filters, one model:

* **Drivers** – which characteristics are linked to the outcome, once the others are held constant.
* **Segments** – groups of similar applicants and how each group fares.
* **Probability** – the estimated chance for a chosen profile, with an honest accuracy check.

Results are associations in observational data, not proof of cause.
""")

up = st.file_uploader("Load the applications file (parquet or CSV)", type=["parquet", "csv"])
if up is not None:
    try:
        df = pd.read_parquet(io.BytesIO(up.getvalue())) if up.name.endswith(".parquet") \
            else pd.read_csv(io.BytesIO(up.getvalue()), low_memory=False)
    except Exception as e:
        st.error(f"Could not read that file: {e}")
        st.stop()
    st.session_state["df"] = df
    st.session_state["source_name"] = up.name

if "df" in st.session_state:
    df = st.session_state["df"]
    st.success(f"Loaded {st.session_state.get('source_name', 'data')}: {len(df):,} rows, {df.shape[1]} columns.")
    for level, msg in check_data(df):
        {"error": st.error, "warning": st.warning, "ok": st.info}[level](msg)
else:
    st.info("Load a file to begin.")
