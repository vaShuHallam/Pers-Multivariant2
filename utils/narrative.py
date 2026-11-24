"""utils/narrative.py - plain-English sentences generated from results (no Streamlit)."""
from __future__ import annotations

import pandas as pd


def effect_sentence(row, outcome: str) -> str:
    ame, lo, hi = row["AME (pp)"], row["CI Lower (pp)"], row["CI Upper (pp)"]
    who = f"{row['Characteristic']}: {row['Category']} applicants"
    ref = f"{row['Reference']} applicants"
    if lo <= 0 <= hi:
        return (f"{who} are not distinguishable from {ref} on {outcome.lower()} "
                f"({ame:+.1f} pp, 95% CI {lo:+.1f} to {hi:+.1f}).")
    word = "higher" if ame > 0 else "lower"
    return (f"{who} have a {outcome.lower()} {abs(ame):.1f} percentage points {word} than {ref}, "
            f"after allowing for the other characteristics (95% CI {lo:+.1f} to {hi:+.1f}).")


def headline_findings(results: pd.DataFrame, outcome: str, top: int = 5) -> list:
    r = results[(results["Role"] == "Characteristic") & results["Significant"]
                & ~results["Unstable"]].copy()
    r["_abs"] = r["AME (pp)"].abs()
    return [effect_sentence(x, outcome) for _, x in r.sort_values("_abs", ascending=False).head(top).iterrows()]


def prediction_sentence(p: dict, outcome: str) -> str:
    return (f"For this profile the estimated {outcome.lower()} is {p['p']*100:.0f}% "
            f"(95% interval {p['lo']*100:.0f}% to {p['hi']*100:.0f}%). The interval covers the "
            f"uncertainty in the average chance for people like this, not how any one "
            f"individual will turn out.")
