"""
utils/ame.py
============
Shared, Streamlit-free calculation code for "adjusted marginal effects" pages.

Every regression page on the site (e.g. 04_logistics_regression.py and
L2_logistics_regression.py) should call this module instead of carrying its own
copy of the maths, so all pages give the same answer for the same data.

METHOD (textbook)
-----------------
1. Ordinary (unpenalised) logistic regression, fitted by maximum likelihood
   with statsmodels. No shrinkage, no scaling, no randomness.
2. Average Marginal Effect (AME) for a category =
       mean predicted probability if EVERYONE were in that category
     - mean predicted probability if EVERYONE were in the reference category
   with every other characteristic left at its observed value.
   Reported in percentage points (pp).
3. 95% confidence intervals by the delta method, using the model's covariance
   matrix. If a cluster column is given (e.g. application_number) the covariance
   is cluster-robust, because several rows can belong to one application and
   those rows are not independent. A result is "Significant" when its interval
   excludes 0.

TYPICAL USE
-----------
    run = run_ame(df, "offer_as_at_30_june",
                  characteristics={"Gender": "gender", "Ethnicity": "ethnic_group"},
                  refs={"Gender": "Women", "Ethnicity": "White"},
                  group_col="application_number")
    run.results        # DataFrame, one row per category
    run.n_used, run.n_dropped, run.n_clusters, run.pseudo_r2

Pages that need the categories before choosing reference groups (to fill a
drop-down) call categorise() first, then fit_prepared().

All problems a manager could hit raise AmeError with a readable message.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.special import expit

NOT_RECORDED = "Not recorded"
SMALL_GROUP = "Other (small groups)"

# Values treated as "missing" (compared in lower case, after stripping spaces).
EXCLUDE = frozenset({"unknown", "not applicable", "unanswered", "not known",
                     "nonclassified", "don't know", "not assigned", "nan", "<na>",
                     "none", ""})

MIN_ROWS = 100          # fewer rows than this -> refuse to fit
UNSTABLE_COEF = 10.0    # |coefficient| above this usually means a tiny/separated group


class AmeError(ValueError):
    """Raised with a message that is safe to show to a manager.

    `labels` (optional) names the variables that caused a model-fit failure, so the caller
    can retry without them."""

    def __init__(self, message, labels=()):
        super().__init__(message)
        self.labels = list(labels)


@dataclass
class Prepared:
    """Data after cleaning: one text column '__<label>' per variable."""
    work: pd.DataFrame
    variables: dict
    target: str
    group_col: str | None
    n_base: int
    merged_notes: dict = field(default_factory=dict)


@dataclass
class AmeRun:
    results: pd.DataFrame
    n_used: int
    n_dropped: int
    n_clusters: int
    pseudo_r2: float
    outcome_mean: float
    skipped: list = field(default_factory=list)      # variables with <2 categories left
    refs: dict = field(default_factory=dict)         # reference category actually used
    dup_dropped: list = field(default_factory=list)  # [(dropped column, kept column)]
    merged_notes: dict = field(default_factory=dict)
    extreme: bool = False                            # any |coefficient| > UNSTABLE_COEF
    clustered: bool = False


# ---------------------------------------------------------------------------
# 1. Cleaning: text categories, missing handling, merging small groups
# ---------------------------------------------------------------------------
def _clean(series: pd.Series, exclude) -> pd.Series:
    s = series.astype("string").str.strip()
    return s.mask(s.isna() | s.str.lower().isin(exclude), NOT_RECORDED)


def categorise(df, target, variables, group_col=None, missing="drop",
               min_cat=0, exclude=EXCLUDE) -> Prepared:
    """Clean every variable into text categories.

    df         : the already-filtered dataframe (passed in explicitly, so results
                 can never be stale or mixed up with another filter)
    variables  : {display label: column name}  (characteristics AND controls)
    missing    : 'drop' removes rows with any missing/unknown value;
                 'keep' turns them into a 'Not recorded' category
    min_cat    : categories with fewer rows are merged into 'Other (small groups)'
    """
    need = [target] + list(variables.values()) + ([group_col] if group_col else [])
    for c in need:
        if c not in df.columns:
            raise AmeError(f"Column '{c}' is not in the data.")
    if missing not in ("drop", "keep"):
        raise AmeError("missing must be 'drop' or 'keep'.")

    work = df[list(dict.fromkeys(need))].copy()
    n_base = len(work)
    work = work[work[target].notna()]
    for label, col in variables.items():
        work[f"__{label}"] = _clean(work[col], exclude)

    if missing == "drop":
        for label in variables:
            work = work[work[f"__{label}"] != NOT_RECORDED]

    merged_notes = {}
    for label in variables:
        c = f"__{label}"
        counts = work[c].value_counts()
        small = counts[counts < min_cat].index.tolist()
        if small and len(counts) - len(small) >= 1:
            work[c] = work[c].where(~work[c].isin(small), SMALL_GROUP)
            merged_notes[label] = small

    y = work[target]
    if len(y) and not set(pd.unique(y.astype(float))) <= {0.0, 1.0}:
        raise AmeError(f"Outcome '{target}' must be coded 0/1 but contains other values.")
    if y.nunique() < 2:
        raise AmeError(f"Outcome '{target}' has only one value in this selection, "
                       "so there is nothing to explain. Broaden the filters.")
    if len(work) < MIN_ROWS:
        raise AmeError(f"Only {len(work):,} usable rows remain (minimum {MIN_ROWS}).")
    return Prepared(work=work, variables=dict(variables), target=target,
                    group_col=group_col, n_base=n_base, merged_notes=merged_notes)


# ---------------------------------------------------------------------------
# 2. Design matrix
# ---------------------------------------------------------------------------
def _build_design(frame, variables, refs):
    """Dummy columns for every non-reference category; drops duplicate columns."""
    blocks, meta = [], []
    for label in variables:
        c = f"__{label}"
        cats = [v for v in frame[c].value_counts().index if v != refs[label]]
        for cat in cats:
            blocks.append((frame[c] == cat).astype(float).rename(f"{label}={cat}"))
            meta.append((label, cat))
    X = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=frame.index)

    # Identical columns (typically 'Not recorded' across variables that go missing
    # together) make the model unidentifiable. Keep the first, report the rest.
    dropped, seen = [], {}
    for col in X.columns:
        key = X[col].values.tobytes()
        if key in seen:
            dropped.append((col, seen[key]))
        else:
            seen[key] = col
    if dropped:
        gone = {d for d, _ in dropped}
        keep = [i for i, col in enumerate(X.columns) if col not in gone]
        meta = [meta[i] for i in keep]
        X = X.drop(columns=list(gone))
    const = [col for col in X.columns if X[col].nunique() < 2]
    if const:
        keep = [i for i, col in enumerate(X.columns) if col not in const]
        meta = [meta[i] for i in keep]
        X = X.drop(columns=const)
    return X, meta, dropped


build_design = _build_design      # public name for pages that need the same dummy columns


# ---------------------------------------------------------------------------
# 3. Model + marginal effects
# ---------------------------------------------------------------------------
@dataclass
class FittedModel:
    """One fitted logit, shared by the effects table and the probability calculator."""
    prep: Prepared
    target: str
    controls: set
    refs_used: dict
    meta: list            # [(label, category)] aligned with Xc columns 1..k
    Xc: np.ndarray
    beta: np.ndarray
    cov: np.ndarray
    n: int
    n_clusters: int
    pseudo_r2: float
    skipped: list
    dup_dropped: list


def _collinearity_message(Xc, meta, with_labels: bool = False):
    """Name the variables that cannot be separated, and what each is tied to.

    Columns are added left to right; one that adds no new information is a linear
    combination of earlier ones. lstsq then shows which earlier columns make it up."""
    redundant, idx = [], []
    rank = 0
    for j in range(Xc.shape[1]):
        r = np.linalg.matrix_rank(Xc[:, idx + [j]])
        if r > rank:
            idx.append(j)
            rank = r
        elif j > 0:
            coef = np.linalg.lstsq(Xc[:, idx], Xc[:, j], rcond=None)[0]
            tied = [meta[k - 1] for k, c in zip(idx, coef) if k > 0 and abs(c) > 1e-6]
            redundant.append((meta[j - 1], tied))
    lines = []
    for (label, cat), tied in redundant:
        tl = list(dict.fromkeys(l for l, _ in tied if l != label))
        lines.append(f"'{label}: {cat}' is exactly made up of " +
                     (", ".join(tl) if tl else "the other variables"))
    msg = ("Some characteristics overlap so completely that the model cannot separate them "
           "(collinearity). " + "; ".join(lines) + ". "
           "Typical causes: the same applicants are 'Not recorded' across several variables "
           "(so one 'Not recorded' group is the sum of others), or one variable is fully "
           "determined by others (College by School). Untick one of the variables named, "
           "or switch Missing values to 'Drop rows'.")
    if not with_labels:
        return msg
    names = []
    for (label, _), tied in redundant:
        names.append(label)
        names.extend(l for l, _ in tied)
    return msg, list(dict.fromkeys(names))


def _separation_labels(work, meta, yv) -> list:
    """Variables with a category where everyone had the same outcome, biggest group first."""
    found = []
    for label, cat in meta:
        m = (work[f"__{label}"] == cat).to_numpy()
        if m.sum() and float(yv[m].mean()) in (0.0, 1.0):
            found.append((int(m.sum()), label))
    found.sort(reverse=True)
    return list(dict.fromkeys(l for _, l in found))


def _separation_hint(work, meta, yv, limit: int = 6) -> str:
    """Name categories where every row has the same outcome (the usual reason a logit fails)."""
    found = []
    for label, cat in meta:
        m = (work[f"__{label}"] == cat).to_numpy()
        n = int(m.sum())
        if n == 0:
            continue
        rate = float(yv[m].mean())
        if rate in (0.0, 1.0):
            found.append((n, f"{label}: {cat} (all {n:,} had {'the outcome' if rate == 1 else 'no outcome'})"))
    if not found:
        return ""
    found.sort(reverse=True)
    return (" Categories where everyone had the same outcome: "
            + "; ".join(t for _, t in found[:limit]) + ".")


def fit_model(prep: Prepared, target: str, refs: dict, controls=()) -> FittedModel:
    """Fit the unpenalised, cluster-robust logit. Raises AmeError with a plain message."""
    work, variables = prep.work, prep.variables
    controls = set(controls)

    active, skipped = {}, []
    for label in variables:
        cats = list(work[f"__{label}"].unique())
        if len(cats) < 2:
            skipped.append(label)             # e.g. 'Domicile' after a UK-only filter
            continue
        ref = refs.get(label)
        if ref is None or str(ref) not in set(map(str, cats)):
            raise AmeError(
                f"Reference category '{ref}' for {label} is not present in the selected "
                f"data. Available: {', '.join(sorted(map(str, cats)))}.")
        active[label] = variables[label]
    refs_used = {k: str(refs[k]) for k in active}
    if not active:
        raise AmeError("No characteristic has two or more categories in this selection.")

    X_df, meta, dup_dropped = _build_design(work, active, refs_used)
    if not meta:
        raise AmeError("No usable categories remain after removing duplicates.")

    yv = work[target].astype(float).to_numpy()
    n = len(yv)
    if prep.group_col:
        groups = pd.factorize(work[prep.group_col].astype(str))[0]
    else:
        groups = np.arange(n)
    n_clusters = int(groups.max()) + 1

    Xc = np.column_stack([np.ones(n), X_df.to_numpy(dtype=np.float64)])
    if np.linalg.matrix_rank(Xc) < Xc.shape[1]:
        msg, names = _collinearity_message(Xc, meta, with_labels=True)
        raise AmeError(msg, labels=names)
    try:
        res = sm.Logit(yv, Xc).fit(disp=False, maxiter=200, cov_type="cluster",
                                   cov_kwds={"groups": groups})
    except Exception as e:                    # separation, singular matrix, ...
        raise AmeError("The model could not be fitted on this selection "
                       f"({type(e).__name__}). Try fewer characteristics or broader "
                       "filters." + _separation_hint(work, meta, yv),
                       labels=_separation_labels(work, meta, yv)) from e
    if not res.mle_retvals.get("converged", True):
        raise AmeError("The model did not converge on this selection. This is usually a "
                       "category where every applicant had the same outcome. Raise the "
                       "merge threshold or remove tiny categories." + _separation_hint(work, meta, yv),
                       labels=_separation_labels(work, meta, yv))

    return FittedModel(prep=prep, target=target, controls=controls, refs_used=refs_used,
                       meta=meta, Xc=Xc, beta=np.asarray(res.params),
                       cov=np.asarray(res.cov_params()), n=n, n_clusters=n_clusters,
                       pseudo_r2=float(res.prsquared), skipped=skipped,
                       dup_dropped=dup_dropped)


def fit_prepared(prep: Prepared, target: str, refs: dict, controls=()) -> AmeRun:
    """Fit the logit on prepared data and return effects for every category."""
    return effects_table(fit_model(prep, target, refs, controls))


def effects_table(m: FittedModel) -> AmeRun:
    prep, target, controls = m.prep, m.target, m.controls
    work, Xc, meta, beta, cov = prep.work, m.Xc, m.meta, m.beta, m.cov
    refs_used, n, n_clusters = m.refs_used, m.n, m.n_clusters
    skipped, dup_dropped = m.skipped, m.dup_dropped
    yv = work[target].astype(float).to_numpy()
    z = stats.norm.ppf(0.975)

    var_cols = {}                              # label -> column positions in Xc
    for j, (label, _) in enumerate(meta, start=1):
        var_cols.setdefault(label, []).append(j)

    rows = []
    for label, idxs in var_cols.items():
        Xg = Xc.copy()
        Xg[:, idxs] = 0.0                      # everyone at the reference category
        eta0 = Xg @ beta
        p0 = expit(eta0)
        g0 = ((p0 * (1 - p0)) @ Xg) / n
        for k in idxs:
            cat = meta[k - 1][1]
            p1 = expit(eta0 + beta[k])         # everyone at this category
            w1 = p1 * (1 - p1)
            g1 = (w1 @ Xg) / n
            g1[k] += w1.mean()
            grad = g1 - g0
            ame = float(p1.mean() - p0.mean())
            se = float(np.sqrt(max(grad @ cov @ grad, 0.0)))

            col = work[f"__{label}"]
            in_cat, in_ref = col == cat, col == refs_used[label]
            raw_gap = (work.loc[in_cat, target].astype(float).mean()
                       - work.loc[in_ref, target].astype(float).mean()) * 100
            rows.append({
                "_order": k,
                "Characteristic": label,
                "Category": cat,
                "Reference": refs_used[label],
                "Role": "Control" if label in controls else "Characteristic",
                "N in group": int(in_cat.sum()),
                "Rate %": float(work.loc[in_cat, target].astype(float).mean() * 100),
                "Unadjusted gap (pp)": float(raw_gap),
                "AME (pp)": ame * 100,
                "CI Lower (pp)": (ame - z * se) * 100,
                "CI Upper (pp)": (ame + z * se) * 100,
                "p": float(2 * stats.norm.sf(abs(ame / se))) if se > 0 else float("nan"),
                "Unstable": bool(abs(beta[k]) > UNSTABLE_COEF or not np.isfinite(se)),
            })

    results = (pd.DataFrame(rows).sort_values("_order").drop(columns="_order")
               .reset_index(drop=True))
    results["Significant"] = (results["CI Lower (pp)"] > 0) | (results["CI Upper (pp)"] < 0)
    return AmeRun(
        results=results, n_used=n, n_dropped=prep.n_base - n, n_clusters=n_clusters,
        pseudo_r2=m.pseudo_r2, outcome_mean=float(yv.mean()),
        skipped=skipped, refs=refs_used, dup_dropped=dup_dropped,
        merged_notes=prep.merged_notes, extreme=bool(np.any(np.abs(beta[1:]) > UNSTABLE_COEF)),
        clustered=prep.group_col is not None,
    )


@dataclass
class RobustFit:
    model: FittedModel
    prep: Prepared
    refs: dict
    variables: dict        # label -> column actually modelled
    controls: list
    min_cat: int
    steps: list            # plain-English record of every automatic change; [] if none


def default_refs(prep: Prepared) -> dict:
    """Largest real category of each variable (ignores 'Not recorded' / small-group merge)."""
    out = {}
    for label in prep.variables:
        counts = prep.work[f"__{label}"].value_counts()
        opts = [v for v in counts.index if v not in (NOT_RECORDED, SMALL_GROUP)] or list(counts.index)
        out[label] = str(opts[0])
    return out


def fit_robust(df, target, variables: dict, refs: dict, controls=(), group_col=None,
               missing="keep", min_cat=30, auto=True) -> RobustFit:
    """Fit the model; if it fails on a small selection, simplify step by step and say so.

    Order of repairs (each one is recorded in `steps`, never silent):
      1. merge small categories at a higher threshold (50, then 100);
      2. leave out the variable(s) the failure points at, characteristics before controls;
      3. merge at a still higher threshold (200, then 400).
    With auto=False the first failure is raised unchanged. Reference groups that vanish
    because their category was merged are replaced by the largest remaining group."""
    variables, controls, steps, m = dict(variables), list(controls), [], int(min_cat)
    last = None
    for _ in range(40):
        prep = categorise(df, target, variables, group_col=group_col, missing=missing, min_cat=m)
        use = default_refs(prep)
        for k, v in refs.items():
            if k in use and str(v) in set(prep.work[f"__{k}"].astype(str)):
                use[k] = str(v)
            elif k in use:
                steps.append(f"Reference group for {k} ('{v}') no longer exists after merging; "
                             f"using '{use[k]}' instead.")
        try:
            model = fit_model(prep, target, use, [c for c in controls if c in variables])
            steps = list(dict.fromkeys(steps))
            return RobustFit(model, prep, use, variables, [c for c in controls if c in variables],
                             m, steps)
        except AmeError as e:
            last = e
            if not auto:
                raise
            nxt = next((t for t in (50, 100) if t > m), None)
            if nxt is not None:
                steps.append(f"The model could not run, so categories smaller than {nxt} were merged "
                             f"(you chose {min_cat}).")
                m = nxt
                continue
            culprits = [l for l in e.labels if l in variables]
            culprits.sort(key=lambda l: (l in controls))        # characteristics first
            chars_left = [v for v in variables if v not in controls]
            culprits = [l for l in culprits if l in controls or len(chars_left) > 1]
            if culprits:
                gone = culprits[0]
                variables.pop(gone)
                controls = [c for c in controls if c != gone]
                steps.append(f"'{gone}' was left out because the model could not separate it on this "
                             "selection (a group where everyone had the same outcome, or an overlap "
                             "with another variable).")
                continue
            nxt = next((t for t in (200, 400) if t > m), None)
            if nxt is not None:
                steps.append(f"Categories smaller than {nxt} were merged.")
                m = nxt
                continue
            break
    raise AmeError(str(last) + " The page also tried merging small groups and leaving out the "
                   "problem variables: " + " ".join(dict.fromkeys(steps)), labels=getattr(last, "labels", ()))


def run_ame(df, target, characteristics, refs, controls=None, group_col=None,
            missing="drop", min_cat=0, exclude=EXCLUDE) -> AmeRun:
    """Clean, fit, return an AmeRun. The one call a simple page needs.

    characteristics : {label: column}   (reported as findings)
    controls        : {label: column}   (adjusted for; flagged Role='Control')
    refs            : {label: reference category as text} for every variable
    """
    controls = dict(controls or {})
    variables = {**characteristics, **controls}
    prep = categorise(df, target, variables, group_col=group_col, missing=missing,
                      min_cat=min_cat, exclude=exclude)
    return fit_prepared(prep, target, refs, controls=controls)


# ---------------------------------------------------------------------------
# 4. Helpers shared by pages
# ---------------------------------------------------------------------------
_UG_WORDS = {"undergraduate", "undergrad", "ug", "bachelor", "bachelors", "bsc", "ba", "beng",
             "foundation", "fda", "fdsc", "fdeng", "hnd", "hnc", "certhe", "diphe"}
# Integrated masters are undergraduate programmes even though the label says "Master".
# They win over the postgraduate words below.
_INTEGRATED_UG = {"meng", "msci", "mcomp", "mmath", "mphys", "mchem", "mbiol", "mgeol",
                  "mpharm", "march", "mgeog", "mears", "mfin"}
_PG_WORDS = {"postgraduate", "postgrad", "pg", "pgt", "pgr", "masters", "master",
             "mba", "msc", "ma", "mres", "phd", "doctorate"}


def looks_undergraduate(label) -> bool:
    """Whole-word test for an undergraduate degree-type label.

    Replaces the old substring test, which wrongly matched e.g. 'Postgraduate
    Taught' (it contains 'ug'). Any postgraduate word vetoes the match.
    """
    words = set(re.findall(r"[a-z]+", str(label).lower()))
    if words & _INTEGRATED_UG:
        return True
    if words & _PG_WORDS:
        return False
    return bool(words & _UG_WORDS)


def describe_filters(items: dict) -> str:
    """One readable line, e.g. 'Cycle year: 2023, 2024 · UK only: Yes'.
    Show it above every chart and put it in every download."""
    parts = []
    for key, value in items.items():
        if isinstance(value, (list, tuple, set, frozenset)):
            value = ", ".join(map(str, sorted(value, key=str))) or "none"
        parts.append(f"{key}: {value}")
    return " · ".join(parts)


def methods_text(refs: dict, n_used: int, n_dropped: int, clustered=False) -> str:
    """Plain-language methods note (markdown) so every page reports it identically."""
    ref_summary = ", ".join(f"{k}: '{v}'" for k, v in refs.items())
    se_text = ("Standard errors are clustered on application, because one application "
               "can appear under several course choices and those rows are not "
               "independent. " if clustered else "")
    return f"""
**Method.** Logistic regression (unpenalised maximum likelihood) with all selected
characteristics entered together. The adjusted effect for a category is the average
difference in predicted probability if everyone were in that category rather than the
reference category, with all other characteristics left as observed (an average marginal
effect, in percentage points). {se_text}95% confidence intervals come from the model's
standard errors (delta method), so results are identical every time the page is run.

**Reference categories:** {ref_summary}

**Rows:** {n_used:,} used; {n_dropped:,} excluded because a selected characteristic was
missing or unknown.

**Reading the chart.** Where the interval crosses zero the effect cannot be distinguished
from no difference. Results describe association in observational data, not causation.
"""