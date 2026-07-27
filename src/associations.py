"""
Adjusted associations -- survey-weighted multivariable logistic regression
complementing the machine-learning risk model.

The locked pipeline (src/modeling.py) and its SHAP attribution (src/interpret_shap.py)
describe how a *fitted model* uses its inputs; they are not adjusted epidemiological
associations and do not report effect sizes. This module fits a single survey-weighted,
reference-coded multivariable logistic regression over the primary risk-factor scope
(src.preprocess.primary_feature_cols) on the full primary cohort, except for the two
variables deterministically or near-deterministically redundant with birth order
(first-birth status and children ever born). It pools all four rounds, both
"development" and 2022 births, since this is a population-descriptive association
model rather than a temporal transport evaluation, and reports adjusted odds ratios
with cluster-bootstrap 95% confidence intervals.

Design choices, deliberately different from the ML preprocessing pipeline:
  * Reference-level dummy coding (not one-hot without a dropped level), so every
    coefficient has a directly interpretable adjusted OR relative to a named reference
    category -- the standard convention for an adjusted-associations table.
  * Continuous predictors enter on their natural scale (per year, per birth, per month),
    not standardised, so each OR has a concrete real-world unit.
  * Survey round is included as a covariate (reference 2011) to adjust every other
    association for the secular mortality decline documented in src.trends.
  * Same unpenalised weighted-logistic + PSU-within-stratum cluster bootstrap already
    used in src.trends, for methodological consistency and because statsmodels'
    freq_weights + cluster covariance is unreliable on this data (see src/trends.py).

Produces:
  results/table_adjusted_or.csv  -- predictor, level, reference, adjusted OR, 95% CI

Run: python -m src.associations
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.harmonize import harmonize
from src.trends import N_BOOT, _boot_round_indices, _coef

# --------------------------------------------------------------------------- #
# Reference-coded design
# --------------------------------------------------------------------------- #
# (column, pretty label, unit) for continuous/binary terms entered directly.
_CONTINUOUS = [
    ("mother_age", "Maternal age at birth", "per year"),
    ("birth_order", "Birth order", "per additional birth"),
    ("birth_interval", "Preceding birth interval", "per 12 months"),
]
# First-birth status is exactly derived from birth order and children ever born
# correlates with birth order at r=0.984 (VIF 32, far above the project's VIF>10
# flag). Entering either alongside birth order creates redundant or compensating,
# uninterpretable coefficients. Birth order is retained as the birth-time-anchored
# parity measure; the two excluded fields remain in the ML feature set (a different
# design purpose -- see src/preprocess.py), where collinearity affects coefficient
# interpretability but not the declared prediction performance.
_BINARY = [
    ("multiple_birth", "Multiple birth", "yes vs. no"),
    ("water_improved", "Improved drinking water", "yes vs. no"),
    ("sanitation_improved", "Improved sanitation", "yes vs. no"),
    ("media_exposure", "Any media exposure", "yes vs. no"),
]
# (column, pretty name, ordered levels, reference level)
_CATEGORICAL = [
    ("mother_edu", "Maternal education", ["none", "primary", "secondary", "higher"], "none"),
    ("wealth", "Household wealth quintile",
     ["poorest", "poorer", "middle", "richer", "richest"], "poorest"),
    ("sex", "Infant sex", ["female", "male"], "female"),
    ("religion", "Religion", ["islam", "hinduism", "other"], "islam"),
    ("residence", "Residence", ["urban", "rural"], "urban"),
    ("division", "Division",
     ["dhaka", "chattogram", "barishal", "khulna", "rajshahi", "rangpur", "sylhet"], "dhaka"),
]
_ROUND_REF = 2011


def _build_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str], list[dict]]:
    """Reference-coded design matrix + column metadata for OR reporting."""
    n = len(df)
    cols: list[np.ndarray] = []
    meta: list[dict] = []

    for col, label, unit in _CONTINUOUS:
        x = pd.to_numeric(df[col], errors="coerce")
        missing = x.isna().to_numpy()
        x = x.fillna(x.median())
        if col == "birth_interval":
            x = x / 12.0  # per 12 months, defined among non-first births
        cols.append(x.to_numpy(float))
        meta.append({"predictor": label, "level": unit, "reference": "", "kind": "continuous"})
        if missing.any():
            cols.append(missing.astype(float))
            meta.append({"predictor": label, "level": "missing indicator",
                        "reference": "observed", "kind": "indicator"})

    for col, label, unit in _BINARY:
        x = pd.to_numeric(df[col], errors="coerce")
        missing = x.isna().to_numpy()
        x = x.fillna(0.0)
        cols.append(x.to_numpy(float))
        meta.append({"predictor": label, "level": unit, "reference": "no", "kind": "binary"})
        if missing.any():
            cols.append(missing.astype(float))
            meta.append({"predictor": label, "level": "missing indicator",
                        "reference": "observed", "kind": "indicator"})

    for col, label, levels, ref in _CATEGORICAL:
        x = df[col].astype("object").where(df[col].notna(), None)
        for level in levels:
            if level == ref:
                continue
            cols.append((x == level).astype(float).to_numpy())
            meta.append({"predictor": label, "level": level, "reference": ref,
                        "kind": "categorical"})

    years = sorted(df[config.YEAR_COL].unique())
    for year in years:
        if year == _ROUND_REF:
            continue
        cols.append((df[config.YEAR_COL] == year).astype(float).to_numpy())
        meta.append({"predictor": "Survey round", "level": str(year),
                    "reference": str(_ROUND_REF), "kind": "adjustment"})

    X = np.column_stack(cols) if cols else np.empty((n, 0))
    names = [f"{m['predictor']}|{m['level']}" for m in meta]
    return X, names, meta


def fit_adjusted_associations(force: bool = False) -> pd.DataFrame:
    out = config.RESULTS / "table_adjusted_or.csv"
    if out.exists() and not force:
        return pd.read_csv(out)

    df = harmonize(module_only=False)
    y = df[config.TARGET].to_numpy(float)
    w = df[config.WEIGHT_COL].to_numpy(float)
    X, names, meta = _build_matrix(df)

    intercept, coef = _coef(X, y, w)
    adjusted_or = np.exp(coef)

    rng = np.random.default_rng(config.SEED)
    boot_or = np.empty((N_BOOT, len(coef)))
    failures = 0
    for b in range(N_BOOT):
        idx = _boot_round_indices(df, rng)
        try:
            _, cb = _coef(X[idx], y[idx], w[idx])
            boot_or[b] = np.exp(cb)
        except Exception:
            boot_or[b] = np.nan
            failures += 1
    ci_lo, ci_hi = np.nanpercentile(boot_or, [2.5, 97.5], axis=0)

    tab = pd.DataFrame(meta)
    tab["adjusted_OR"] = np.round(adjusted_or, 3)
    tab["ci_low"] = np.round(ci_lo, 3)
    tab["ci_high"] = np.round(ci_hi, 3)
    tab["n"] = len(df)
    tab["events"] = int(y.sum())
    tab["bootstrap_successes"] = N_BOOT - failures
    tab["bootstrap_failures"] = failures
    tab = tab[tab["kind"] != "indicator"].drop(columns="kind").reset_index(drop=True)
    tab.to_csv(out, index=False)
    print(f"  Adjusted associations: {len(tab)} terms, {len(df):,} births / "
          f"{int(y.sum())} deaths, all four rounds pooled, year-adjusted -> {out}")
    return tab


def run(force: bool = False) -> pd.DataFrame:
    return fit_adjusted_associations(force=force)


if __name__ == "__main__":
    tab = run(force=True)
    print(tab.to_string(index=False))
