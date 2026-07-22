"""
Table 1 -- baseline characteristics of the analytic sample by neonatal-death status.

Categorical predictors: n (%) within each outcome group + chi-square p-value.
Continuous predictors: mean (SD) by group + Welch t-test p-value.

Output: results/table1_descriptives.csv

Run: python -m src.descriptives
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

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


def build() -> pd.DataFrame:
    df = harmonize()
    y = df[config.TARGET]
    w = df[config.WEIGHT_COL].to_numpy(float)
    rows = []

    # Survey-weighted summaries by outcome group; p-values from unweighted
    # association tests (design-based percentages, unweighted significance tests).
    for var in CONTINUOUS:
        s = pd.to_numeric(df[var], errors="coerce")
        m0 = (y == 0) & s.notna()
        m1 = (y == 1) & s.notna()
        wm0, wsd0 = _wmean_sd(s[m0].to_numpy(), w[m0.to_numpy()])
        wm1, wsd1 = _wmean_sd(s[m1].to_numpy(), w[m1.to_numpy()])
        p = stats.ttest_ind(s[m0].dropna(), s[m1].dropna(), equal_var=False).pvalue
        rows.append({
            "variable": var, "level": "mean (SD)",
            _LABELS[0]: f"{wm0:.1f} ({wsd0:.1f})",
            _LABELS[1]: f"{wm1:.1f} ({wsd1:.1f})",
            "p_value": _fmt_p(p),
        })

    for var in CATEGORICAL:
        s = df[var]
        ct = pd.crosstab(s, y)                              # unweighted, for the chi-square test
        wct = df.assign(_w=w).pivot_table(index=var, columns=config.TARGET,
                                          values="_w", aggfunc="sum", fill_value=0.0)
        try:
            p = stats.chi2_contingency(ct)[1]
        except ValueError:
            p = np.nan
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
                "p_value": _fmt_p(p) if level == ct.index[0] else "",
            })

    tab = pd.DataFrame(rows)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    header = pd.DataFrame([{
        "variable": "N", "level": "",
        _LABELS[0]: str(n0), _LABELS[1]: str(n1), "p_value": "",
    }])
    tab = pd.concat([header, tab], ignore_index=True)
    out = config.RESULTS / "table1_descriptives.csv"
    tab.to_csv(out, index=False)
    print(f"  Table 1 -> {out}  ({n0} survived / {n1} neonatal deaths)")
    return tab


def _fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


if __name__ == "__main__":
    tab = build()
    print(tab.to_string(index=False))
