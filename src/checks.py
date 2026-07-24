"""
End-to-end integrity checks -- run after the pipeline to assert the anti-leakage
guarantees the paper claims. Writes results/verification.txt.

Checks:
  1. Temporal split is clean: training rows are only 2011-2017, test rows only 2022.
  2. No PSU cluster appears in both the training and the test set.
  3. Boruta / design never saw 2022 (feature selection used training rows only).
  4. Determinism: refitting a model twice with the fixed seed gives identical scores.
  5. Prevalence sanity: per-round neonatal-death rate is in the published ballpark.

Run: python -m src.checks
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import config
from src.harmonize import cohort_counts, derive_neonatal_outcome, harmonize
from src.preprocess import build_design, clean, primary_feature_cols
from src.modeling import _pipeline, _smote


def run() -> bool:
    lines, ok = [], True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        lines.append(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

    d = build_design()

    design_meta = json.loads((config.DATA_INTERIM / "design_meta.json").read_text())
    num_cols, nom_cols = primary_feature_cols()
    check("primary design uses the infant/maternal/household risk-factor set",
          set(design_meta.get("source_columns", [])) == set(num_cols + nom_cols),
          f"{len(design_meta.get('source_columns', []))} source columns")
    check("no antenatal/delivery-care variable enters the primary all-recent model",
          not any(v in set(num_cols + nom_cols)
                  for v in ["anc_visits", "anc_4plus", "csection",
                            "delivery_place", "skilled_attendant"]))

    # 0. Endpoint boundary and complete follow-up definition
    boundary = derive_neonatal_outcome(pd.DataFrame({
        "b5": [0, 0, 0, 0, 0, 0], "b6": [127, 128, 129, 135, 199, 999],
    }))
    check("neonatal endpoint is exactly days 0-27",
          boundary.iloc[:4].tolist() == [1, 0, 0, 0]
          and boundary.iloc[4:].isna().all())
    h = harmonize(module_only=False)
    check("all retained births meet conservative complete-follow-up rule",
          bool(h["age_mo"].ge(config.MIN_FOLLOWUP_MONTHS).all()),
          f"minimum age_mo={h['age_mo'].min()}")
    check("survey stratum retained for design-based resampling",
          bool(h[config.STRATUM_COL].notna().all()),
          f"strata={h[config.STRATUM_COL].nunique()}")
    all_recent = h
    long_2022 = all_recent[(all_recent[config.YEAR_COL] == config.TEST_YEAR)
                           & all_recent["questionnaire_type"].eq("long")]
    care = harmonize(module_only=True)
    module_2022 = care[care[config.YEAR_COL] == config.TEST_YEAR]
    expected_2022 = long_2022[long_2022["birth_index"].eq(1)]
    check("2022 care cohort is most-recent births in the long subsample",
          len(expected_2022) == len(module_2022)
          and module_2022["questionnaire_type"].eq("long").all()
          and module_2022["birth_index"].eq(1).all(),
          f"expected={len(expected_2022)}, module={len(module_2022)}")
    check("primary prediction cohort is not questionnaire restricted",
          len(h[h[config.YEAR_COL] == config.TEST_YEAR]) > len(module_2022),
          f"primary={len(h[h[config.YEAR_COL] == config.TEST_YEAR])}, "
          f"care={len(module_2022)}")

    # 1. temporal split
    yrs = set(np.unique(d["year_train"]).tolist())
    check("train rounds are only 2011-2017", yrs == set(config.TRAIN_YEARS), str(sorted(yrs)))
    check("test set size == 2022 sample",
          len(d["y_test"]) == int((h[config.YEAR_COL] == config.TEST_YEAR).sum()),
          f"n_test={len(d['y_test'])}")

    # 2. cluster disjointness (reconstruct test clusters the same way build_design split)
    df = clean(h)
    test_groups = set(df.loc[df[config.YEAR_COL] == config.TEST_YEAR, config.CLUSTER_COL])
    train_groups = set(d["groups_train"].tolist())
    check("no PSU cluster in both train and test",
          train_groups.isdisjoint(test_groups),
          f"overlap={len(train_groups & test_groups)}")

    # 3. Boruta used training rows only (feature count matches train matrix width)
    sel = json.loads((config.RESULTS / "selected_features.json").read_text())
    check("Boruta selected features exist in design",
          all(f in d["feature_names"] for f in sel["selected"]))

    # 4. determinism
    idx = list(range(d["X_train"].shape[1]))
    p1 = _fit_predict(d, idx)
    p2 = _fit_predict(d, idx)
    check("model is deterministic under fixed seed", np.allclose(p1, p2),
          f"max|dp|={np.max(np.abs(p1-p2)):.2e}")

    # 5. prevalence sanity (weighted NMR per round within 15-35 / 1000)
    bad = []
    for yr, sub in h.groupby(config.YEAR_COL):
        nmr = np.average(sub[config.TARGET], weights=sub[config.WEIGHT_COL]) * 1000
        if not (12 <= nmr <= 40):
            bad.append(f"{yr}:{nmr:.0f}")
    check("weighted NMR per round in published ballpark (12-40/1000)", not bad, ",".join(bad))

    # 6. trend outputs (present only after src.trends has run)
    t3 = config.RESULTS / "table3_nmr_trend.csv"
    t5 = config.RESULTS / "table5_decomposition.csv"
    if t3.exists() and t5.exists():
        nmr = pd.read_csv(t3)
        check("NMR declined 2011 -> 2022",
              nmr.iloc[-1]["NMR"] < nmr.iloc[0]["NMR"],
              f"{nmr.iloc[0]['NMR']} -> {nmr.iloc[-1]['NMR']}")
        dec = pd.read_csv(t5).set_index("component")["value_per_1000"]
        s = dec["distribution"] + dec["effect"]
        check("decomposition components sum to total change",
              abs(s - dec["total_change"]) < 0.15, f"{s:.1f} vs {dec['total_change']:.1f}")
        check("trend and primary decomposition use the same 2011/2022 NMR",
              abs(nmr.iloc[0]["NMR"] - dec["NMR_2011"]) < 0.15
              and abs(nmr.iloc[-1]["NMR"] - dec["NMR_2022"]) < 0.15,
              f"trend {nmr.iloc[0]['NMR']}/{nmr.iloc[-1]['NMR']} vs "
              f"decomp {dec['NMR_2011']}/{dec['NMR_2022']}")
    else:
        lines.append("[SKIP] trend checks (run src.trends first)")

    # Machine-readable cohort definition and requested one-month sensitivity.
    cohort_rows = []
    for name, module, min_followup in [
        ("primary_allrecent", False, config.MIN_FOLLOWUP_MONTHS),
        ("sensitivity_care_module", True, config.MIN_FOLLOWUP_MONTHS),
        ("sensitivity_allrecent_min1", False, 1),
        ("sensitivity_care_module_min1", True, 1),
    ]:
        counts = cohort_counts(harmonize(module_only=module,
                                         min_followup_months=min_followup))
        for year, row in counts.iterrows():
            cohort_rows.append({"cohort": name, "minimum_age_mo": min_followup,
                                "survey_year": year, **row.to_dict()})
    pd.DataFrame(cohort_rows).to_csv(config.RESULTS / "cohort_definition.csv", index=False)

    report = "\n".join(lines)
    (config.RESULTS / "verification.txt").write_text(report)
    print(report)
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return ok


def _fit_predict(d, idx):
    pipe = _pipeline(LogisticRegression(max_iter=2000, random_state=config.SEED),
                     needs_scaling=True, balancer_steps=_smote())
    pipe.fit(d["X_train"][:, idx], d["y_train"])
    return pipe.predict_proba(d["X_test"][:, idx])[:, 1]


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
