"""
utils/model_config.py
=====================
Settings shared by the regression pages (outcomes, characteristic columns, controls).
Edit them HERE so every page uses the same definitions.
"""

# outcome -> (numerator, denominator column or None for all rows)
TARGETS = {
    "Offer rate": ("offer_as_at_30_june", "application"),
    "Acceptance rate": ("acceptance", "application"),
    "Conversion rate": ("acceptance", "offer_as_at_30_june"),
    "Application rate": ("application", None),
}
DENOM_LABEL = {"application": "applications", "offer_as_at_30_june": "offers",
               None: "everyone in the UCAS system"}

# label -> candidate column names. The FIRST one present in the file is used, so the same
# label works on the main applications table and on the L2 table (which names things slightly
# differently). Add a name to a list to support another file layout.
CHARACTERISTICS = {
    "Age band": ["age_group_2band"],
    "Age band (5 levels)": ["age_band_5_lvls", "age_group_5band"],
    "Ethnic group (high level)": ["ethnic_group_summary_lvl"],
    "Gender": ["gender"],
    "Socio-economic group": ["sec_group", "socio_economic_group", "Socio-economic group"],
    "Free School Meals": ["fsm_indicator"],
    "Parental HE Indicator": ["parental_he_indicator"],
    "Domicile (high level)": ["app_domicile_high_lvl_3_lvls"],
    "Disability": ["disability_indicator"],
    "In Care": ["in_care_indicator"],
    "Estranged": ["estranged_indicator"],
    "Caring responsibilities": ["caring_responsibilities_indicator"],
    "Armed forces": ["served_in_the_armed_forces_indicator"],
    "Lives at home": ["live_at_home_indicator"],
    "TUNDRA quintile": ["tundra_msoa_quint"],
    "IMD quintile": ["imd_quint_2019"],
    "School": ["school"],
    "Priority Subject": ["priority_label"],
    "A-Level only": ["qual_alevel_only"],
    "BTEC only": ["qual_btec_only"],
    "Has A-level qualification": ["a_lvl_qual_indicator"],
    "Has BTEC qualification": ["btec_qual_indicator"],
    "English L2": ["English_L2", "English L2"],
    "Maths L2": ["Maths_L2", "Maths L2"],
    "Science L2": ["Science_L2", "Science L2"],
    "Any L2 qualification": ["L2_Qualifications", "L2 qualification"],
    "Multiple L2 qualifications": ["Multiple_L2_Qualifications", "Multiple L2 Qualifications"],
}

# Offered by default when present, in this order.
DEFAULT_ORDER = [
    "Ethnic group (high level)", "Gender", "Age band", "IMD quintile",
    "Parental HE Indicator", "Disability", "Tariff (18 YO, 3 A-levels)",
    "Maths L2", "English L2",
]

# Controls: things you adjust for but are not the question.
CONTROLS = {
    "Cycle year": ["cycle_year"],
    "Subject group": ["subject_group_summary_lvl"],
    "Department": ["department", "Course_Department_Latest"],
    "College": ["college", "Course_College_Short_Latest"],
}

# Available in the list but NOT ticked by default. The L2 summary columns overlap with the
# separate English / Maths / Science columns, so ticking them together can make the model fail.
DEFAULT_EXCLUDE = ["School", "Priority Subject", "Any L2 qualification",
                   "Multiple L2 qualifications"]

# Backwards-compatible views (first candidate name).
FALLBACK_CHARACTERISTICS = {k: v[0] for k, v in CHARACTERISTICS.items()}
CONTROL_CANDIDATES = {k: v[0] for k, v in CONTROLS.items()}


def first_present(df, candidates):
    """First candidate column found in df (exact name), else None."""
    return next((c for c in candidates if c in df.columns), None)


def resolve_controls(df) -> dict:
    out = {}
    for label, cands in CONTROLS.items():
        col = first_present(df, cands)
        if col:
            out[label] = col
    return out


def resolve_characteristics(df) -> dict:
    """{label: column} for every characteristic this file has (controls excluded)."""
    ctrl_cols = set(resolve_controls(df).values())
    out = {}
    for label, cands in CHARACTERISTICS.items():
        col = first_present(df, cands)
        if col and col not in ctrl_cols:
            out[label] = col
    return out


# Short names used on chart axes (full names stay in tables and downloads).
SHORT_LABELS = {
    "Ethnic group (high level)": "Ethnicity",
    "Age band": "Age group",
    "Age band (5 levels)": "Age group (5)",
    "Parental HE Indicator": "Parental HE",
    "Free School Meals": "FSM",
    "Socio-economic group": "Socioeconomic group",
    "IMD quintile": "IMD Quintile",
    "TUNDRA quintile": "TUNDRA Quintile",
    "Domicile (high level)": "Domicile",
    "Caring responsibilities": "Carer",
    "Armed forces": "Armed forces",
    "Lives at home": "Lives at home",
    "Has A-level qualification": "A-Level qual",
    "Has BTEC qualification": "BTEC qual",
    "Multiple L2 qualifications": "Multiple L2",
    "Any L2 qualification": "Any L2",
}


def short_label(label: str) -> str:
    return SHORT_LABELS.get(label, label)