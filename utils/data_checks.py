"""utils/data_checks.py - plain-language checks run once when data is loaded."""
from __future__ import annotations

import pandas as pd

from utils.model_config import FALLBACK_CHARACTERISTICS, TARGETS


def check_data(df: pd.DataFrame) -> list:
    """Return [(level, message)] with level in {'error','warning','ok'}."""
    out = []
    if len(df) == 0:
        return [("error", "The file has no rows.")]
    outcome_cols = {n for n, _ in TARGETS.values()}
    present = [c for c in outcome_cols if c in df.columns]
    if not present:
        out.append(("error", "None of the outcome columns were found (" +
                    ", ".join(sorted(outcome_cols)) + "). Check this is the right file."))
    for c in present:
        vals = set(pd.Series(df[c]).dropna().unique().tolist())
        if not vals <= {0, 1, 0.0, 1.0, True, False}:
            out.append(("error", f"`{c}` should only contain 0 and 1 but also has: "
                                 f"{sorted(map(str, vals - {0, 1}))[:5]}."))
        nmiss = int(df[c].isna().sum())
        if nmiss:
            out.append(("warning", f"`{c}` has {nmiss:,} blank values; those rows cannot be modelled."))
    if "application_number" not in df.columns:
        out.append(("warning", "No `application_number` column: results will treat every row as "
                               "an independent applicant, so confidence intervals will be too narrow."))
    else:
        per = df.groupby("application_number").size()
        out.append(("ok", f"{len(df):,} rows from {len(per):,} applications "
                          f"(up to {int(per.max())} rows each); intervals are clustered by application."))
    have = [c for c in FALLBACK_CHARACTERISTICS.values() if c in df.columns]
    lost = [k for k, c in FALLBACK_CHARACTERISTICS.items() if c not in df.columns]
    out.append(("ok", f"{len(have)} of {len(FALLBACK_CHARACTERISTICS)} expected characteristic columns found."))
    if lost:
        out.append(("warning", "Not found, so not available: " + ", ".join(lost) + "."))
    dup = int(df.duplicated().sum())
    if dup:
        out.append(("warning", f"{dup:,} rows are exact duplicates of another row."))
    return out
