"""Cohort characteristics for the main article and supplement.

Main Table 1 compares the pooled development rounds with the separated 2022
evaluation cohort using survey-weighted summaries and no design-ignoring
significance tests.  Outcome-stratified pooled descriptives are supplementary.

Run: python -m src.descriptives
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.harmonize import harmonize

CATEGORICAL = ["sex", "firstborn", "multiple_birth", "mother_edu", "wealth",
               "religion", "residence", "division", "delivery_place", "csection",
               "skilled_attendant", "anc_4plus", "mother_working",
               "water_improved", "sanitation_improved", "media_exposure"]
CONTINUOUS = ["mother_age", "birth_order", "children_ever_born",
              "birth_interval", "anc_visits"]

_LABELS = {0: "survived", 1: "neonatal death"}


def _wmean_sd(x, w):
    m = np.average(x, weights=w)
    sd = np.sqrt(np.average((x - m) ** 2, weights=w))
    return m, sd


def build_outcome_stratified() -> pd.DataFrame:
    df = harmonize()
    y = df[config.TARGET]
    w = df[config.WEIGHT_COL].to_numpy(float)
    rows = []

    # Survey-weighted summaries by outcome group; deliberately descriptive.
    for var in CONTINUOUS:
        s = pd.to_numeric(df[var], errors="coerce")
        m0 = (y == 0) & s.notna()
        m1 = (y == 1) & s.notna()
        wm0, wsd0 = _wmean_sd(s[m0].to_numpy(), w[m0.to_numpy()])
        wm1, wsd1 = _wmean_sd(s[m1].to_numpy(), w[m1.to_numpy()])
        rows.append({
            "variable": var, "level": "mean (SD)",
            _LABELS[0]: f"{wm0:.1f} ({wsd0:.1f})",
            _LABELS[1]: f"{wm1:.1f} ({wsd1:.1f})",
        })

    for var in CATEGORICAL:
        s = df[var]
        ct = pd.crosstab(s, y)                              # unweighted, for the chi-square test
        wct = df.assign(_w=w).pivot_table(index=var, columns=config.TARGET,
                                          values="_w", aggfunc="sum", fill_value=0.0)
        wtot0 = wct[0].sum() if 0 in wct.columns else 1.0
        wtot1 = wct[1].sum() if 1 in wct.columns else 1.0
        for level in ct.index:
            n0 = int(ct.loc[level, 0]) if 0 in ct.columns else 0
            n1 = int(ct.loc[level, 1]) if 1 in ct.columns else 0
            w0 = float(wct.loc[level, 0]) if (0 in wct.columns and level in wct.index) else 0.0
            w1 = float(wct.loc[level, 1]) if (1 in wct.columns and level in wct.index) else 0.0
            rows.append({
                "variable": var, "level": str(level),
                _LABELS[0]: f"{n0} ({100*w0/wtot0:.1f}%)",
                _LABELS[1]: f"{n1} ({100*w1/wtot1:.1f}%)",
            })

    tab = pd.DataFrame(rows)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    header = pd.DataFrame([{
        "variable": "N", "level": "",
        _LABELS[0]: str(n0), _LABELS[1]: str(n1),
    }])
    tab = pd.concat([header, tab], ignore_index=True)
    out = config.RESULTS / "tableS1_outcome_descriptives.csv"
    tab.to_csv(out, index=False)
    print(f"  Table 1 -> {out}  ({n0} survived / {n1} neonatal deaths)")
    return tab


def _period_summary(df: pd.DataFrame, label: str) -> list[dict]:
    w = df[config.WEIGHT_COL].to_numpy(float)
    rows = [{"variable": "Live births", "level": "n", label: f"{len(df):,}"},
            {"variable": "Neonatal deaths", "level": "n",
             label: f"{int(df[config.TARGET].sum()):,}"}]
    nmr = np.average(df[config.TARGET], weights=w) * 1000
    rows.append({"variable": "Neonatal mortality", "level": "per 1,000",
                 label: f"{nmr:.1f}"})
    for var in CONTINUOUS:
        s = pd.to_numeric(df[var], errors="coerce")
        ok = s.notna().to_numpy()
        mean, sd = _wmean_sd(s[ok].to_numpy(), w[ok])
        rows.append({"variable": var, "level": "mean (SD)",
                     label: f"{mean:.1f} ({sd:.1f})"})
    for var in CATEGORICAL:
        for level in sorted(df[var].dropna().astype(str).unique()):
            mask = df[var].astype(str).eq(level).to_numpy()
            weighted_pct = 100 * w[mask].sum() / w[df[var].notna().to_numpy()].sum()
            rows.append({"variable": var, "level": level,
                         label: f"{int(mask.sum()):,} ({weighted_pct:.1f}%)"})
    return rows


def build() -> pd.DataFrame:
    df = harmonize()
    dev = df[df[config.YEAR_COL].isin(config.TRAIN_YEARS)]
    eva = df[df[config.YEAR_COL] == config.TEST_YEAR]
    dev_label = "Development (2011-2017/18)"
    eva_label = "Evaluation (2022)"
    left = pd.DataFrame(_period_summary(dev, dev_label))
    right = pd.DataFrame(_period_summary(eva, eva_label))
    tab = left.merge(right, on=["variable", "level"], how="outer")
    tab.to_csv(config.RESULTS / "table1_cohort_characteristics.csv", index=False)
    build_outcome_stratified()
    print(f"  Table 1 -> table1_cohort_characteristics.csv ({len(dev):,} vs {len(eva):,})")
    return tab


def _fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


if __name__ == "__main__":
    tab = build()
    print(tab.to_string(index=False))
