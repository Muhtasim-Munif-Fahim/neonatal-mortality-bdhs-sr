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
    # Care-module cohort so antenatal/delivery associations are on a populated
    # denominator; documents the care sensitivity cohort by outcome (supplement).
    df = harmonize(module_only=True)
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

    def continuous(section, label, column, left=dev, right=eva, mask_left=None,
                   mask_right=None):
        a = pd.to_numeric(left[column], errors="coerce")
        b = pd.to_numeric(right[column], errors="coerce")
        ma = a.notna() if mask_left is None else a.notna() & mask_left
        mb = b.notna() if mask_right is None else b.notna() & mask_right
        am, asd = _wmean_sd(a[ma].to_numpy(), left.loc[ma, config.WEIGHT_COL].to_numpy())
        bm, bsd = _wmean_sd(b[mb].to_numpy(), right.loc[mb, config.WEIGHT_COL].to_numpy())
        pooled = np.sqrt((asd ** 2 + bsd ** 2) / 2)
        smd = (bm - am) / pooled if pooled > 0 else np.nan
        return {"section": section, "characteristic": label, "statistic": "mean (SD)",
                dev_label: f"{am:.1f} ({asd:.1f})", eva_label: f"{bm:.1f} ({bsd:.1f})",
                "weighted_SMD": round(smd, 3)}

    def binary(section, label, left_values, right_values):
        a = pd.to_numeric(left_values, errors="coerce")
        b = pd.to_numeric(right_values, errors="coerce")
        oka, okb = a.notna(), b.notna()
        wa = dev.loc[oka, config.WEIGHT_COL].to_numpy()
        wb = eva.loc[okb, config.WEIGHT_COL].to_numpy()
        pa = np.average(a[oka], weights=wa)
        pb = np.average(b[okb], weights=wb)
        pooled = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
        smd = (pb - pa) / pooled if pooled > 0 else np.nan
        return {"section": section, "characteristic": label,
                "statistic": "n (weighted %)",
                dev_label: f"{int(a[oka].sum()):,} ({100 * pa:.1f}%)",
                eva_label: f"{int(b[okb].sum()):,} ({100 * pb:.1f}%)",
                "weighted_SMD": round(smd, 3)}

    rows = [
        {"section": "Cohort", "characteristic": "Survey rounds", "statistic": "years",
         dev_label: "2011, 2014, 2017-18", eva_label: "2022", "weighted_SMD": np.nan},
        {"section": "Cohort", "characteristic": "Live births", "statistic": "n",
         dev_label: f"{len(dev):,}", eva_label: f"{len(eva):,}", "weighted_SMD": np.nan},
        {"section": "Cohort", "characteristic": "Neonatal deaths", "statistic": "n",
         dev_label: f"{int(dev[config.TARGET].sum()):,}",
         eva_label: f"{int(eva[config.TARGET].sum()):,}", "weighted_SMD": np.nan},
        {"section": "Cohort", "characteristic": "Neonatal mortality",
         "statistic": "weighted per 1,000",
         dev_label: f"{1000 * np.average(dev[config.TARGET], weights=dev[config.WEIGHT_COL]):.1f}",
         eva_label: f"{1000 * np.average(eva[config.TARGET], weights=eva[config.WEIGHT_COL]):.1f}",
         "weighted_SMD": np.nan},
        continuous("Infant and maternal characteristics", "Maternal age at birth, years", "mother_age"),
        continuous("Infant and maternal characteristics", "Birth order", "birth_order"),
        continuous("Infant and maternal characteristics", "Preceding birth interval, months",
                   "birth_interval", mask_left=dev["firstborn"].eq(0),
                   mask_right=eva["firstborn"].eq(0)),
        continuous("Infant and maternal characteristics", "Children ever born", "children_ever_born"),
        binary("Infant and maternal characteristics", "First birth", dev["firstborn"], eva["firstborn"]),
        binary("Infant and maternal characteristics", "Male infant", dev["sex"].eq("male").astype(int),
               eva["sex"].eq("male").astype(int)),
        binary("Infant and maternal characteristics", "Multiple birth", dev["multiple_birth"],
               eva["multiple_birth"]),
        binary("Household socioeconomic characteristics", "Mother completed secondary or higher education",
               dev["mother_edu"].isin(["secondary", "higher"]).astype(int),
               eva["mother_edu"].isin(["secondary", "higher"]).astype(int)),
        binary("Household socioeconomic characteristics", "Mother currently working",
               dev["mother_working"], eva["mother_working"]),
        binary("Household socioeconomic characteristics", "Poorest wealth quintile",
               dev["wealth"].eq("poorest").astype(int), eva["wealth"].eq("poorest").astype(int)),
        binary("Household socioeconomic characteristics", "Rural residence",
               dev["residence"].eq("rural").astype(int), eva["residence"].eq("rural").astype(int)),
        binary("Household socioeconomic characteristics", "Improved drinking water",
               dev["water_improved"], eva["water_improved"]),
        binary("Household socioeconomic characteristics", "Improved sanitation",
               dev["sanitation_improved"], eva["sanitation_improved"]),
        binary("Household socioeconomic characteristics", "Any media exposure",
               dev["media_exposure"], eva["media_exposure"]),
    ]
    tab = pd.DataFrame(rows)
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
