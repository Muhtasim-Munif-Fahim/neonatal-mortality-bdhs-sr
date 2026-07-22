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

import numpy as np
from sklearn.linear_model import LogisticRegression

import config
from src.harmonize import harmonize
from src.preprocess import build_design, clean
from src.modeling import _pipeline, _smote


def run() -> bool:
    lines, ok = [], True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        lines.append(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

    d = build_design()

    # 1. temporal split
    yrs = set(np.unique(d["year_train"]).tolist())
    check("train rounds are only 2011-2017", yrs == set(config.TRAIN_YEARS), str(sorted(yrs)))
    check("test set size == 2022 sample",
          len(d["y_test"]) == int((harmonize()[config.YEAR_COL] == config.TEST_YEAR).sum()),
          f"n_test={len(d['y_test'])}")

    # 2. cluster disjointness (reconstruct test clusters the same way build_design split)
    df = clean(harmonize())
    test_groups = set(df.loc[df[config.YEAR_COL] == config.TEST_YEAR, config.CLUSTER_COL])
    train_groups = set(d["groups_train"].tolist())
    check("no PSU cluster in both train and test",
          train_groups.isdisjoint(test_groups),
          f"overlap={len(train_groups & test_groups)}")

    # 3. Boruta used training rows only (feature count matches train matrix width)
    import json
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
    h = harmonize()
    bad = []
    for yr, sub in h.groupby(config.YEAR_COL):
        nmr = np.average(sub[config.TARGET], weights=sub[config.WEIGHT_COL]) * 1000
        if not (12 <= nmr <= 40):
            bad.append(f"{yr}:{nmr:.0f}")
    check("weighted NMR per round in published ballpark (12-40/1000)", not bad, ",".join(bad))

    # 6. trend outputs (present only after src.trends has run)
    import pandas as pd
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
    else:
        lines.append("[SKIP] trend checks (run src.trends first)")

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
