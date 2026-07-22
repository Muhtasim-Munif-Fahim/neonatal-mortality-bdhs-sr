"""
ML robustness add-ons (run after modeling/evaluate):

  A. Balancing sensitivity -- retrain the best model under SMOTE vs ADASYN vs
     class-weight vs no resampling; compare discrimination AND calibration (raw +
     isotonic-recalibrated). Confirms the resampler choice on the honest criterion
     (calibration), not accuracy.                       -> table6, fig10
  B. Cluster-bootstrap 95% CIs on the 2022 test metrics (resample 2022 PSUs) for
     every model -- turns point estimates into interval estimates.  -> table7
  C. Subgroup performance of the best model (residence / sex / wealth). -> table8

No new dependency (ADASYN ships with imbalanced-learn).

Run: python -m src.robustness
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold

import config
from src import viz
from src.harmonize import harmonize
from src.preprocess import build_design

N_BOOT = 1000


# --------------------------------------------------------------------------- #
# Align test-row metadata (cluster / subgroups) to the saved test predictions
# --------------------------------------------------------------------------- #
def _test_meta(preds) -> pd.DataFrame:
    """2022 rows in build_design order; asserted to match the saved predictions.

    Uses harmonize() (readable string categories) NOT clean() -- clean() ordinal-
    encodes wealth/education to integers. Row order is identical (clean preserves
    order), and the assert below guards alignment.
    """
    df = harmonize()
    test = df[df[config.YEAR_COL] == config.TEST_YEAR].reset_index(drop=True)
    assert np.array_equal(test[config.TARGET].to_numpy(), preds["y_test"].to_numpy()), \
        "test-row order mismatch — cannot align clusters/subgroups"
    return test


def _feature_idx(names, feature_set):
    if feature_set == "boruta":
        sel = json.loads((config.RESULTS / "selected_features.json").read_text())["selected"]
        return [names.index(f) for f in sel]
    return list(range(len(names)))


def _oof_isotonic(pipe, Xtr, ytr, groups, raw_test_p):
    """Recalibrate: isotonic fit on grouped out-of-fold training predictions."""
    oof = np.zeros(len(ytr))
    for tr, va in GroupKFold(5).split(Xtr, ytr, groups):
        oof[va] = clone(pipe).fit(Xtr[tr], ytr[tr]).predict_proba(Xtr[va])[:, 1]
    return IsotonicRegression(out_of_bounds="clip").fit(oof, ytr).transform(raw_test_p)


# --------------------------------------------------------------------------- #
# A. Balancing sensitivity
# --------------------------------------------------------------------------- #
def balancing_sensitivity() -> pd.DataFrame:
    best = json.loads((config.RESULTS / "best_model.json").read_text())
    d = build_design()
    idx = _feature_idx(list(d["feature_names"]), best["feature_set"])
    Xtr, ytr, groups = d["X_train"][:, idx], d["y_train"], d["groups_train"]
    Xte, yte = d["X_test"][:, idx], d["y_test"]

    base = joblib.load(config.MODELS / f"{best['model']}__{best['feature_set']}.joblib")
    clf = base.named_steps["clf"]           # tuned estimator

    strategies = {
        "SMOTE": ImbPipeline([("balance", SMOTE(random_state=config.SEED)),
                              ("clf", clone(clf))]),
        "ADASYN": ImbPipeline([("balance", ADASYN(random_state=config.SEED)),
                               ("clf", clone(clf))]),
        "class_weight": clone(clf).set_params(class_weight="balanced"),
        "none": clone(clf),
    }

    rows = []
    curves = {}
    for name, pipe in strategies.items():
        pipe.fit(Xtr, ytr)
        raw = pipe.predict_proba(Xte)[:, 1]
        cal = _oof_isotonic(pipe, Xtr, ytr, groups, raw)
        curves[name] = raw
        rows.append({
            "balancing": name,
            "ROC_AUC": round(roc_auc_score(yte, raw), 3),
            "PR_AUC": round(average_precision_score(yte, raw), 3),
            "Brier_raw": round(brier_score_loss(yte, raw), 4),
            "Brier_recalibrated": round(brier_score_loss(yte, cal), 4),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(config.RESULTS / "table6_balancing.csv", index=False)
    winner = tab.loc[tab["Brier_recalibrated"].idxmin(), "balancing"]
    print(f"  A. balancing ({best['model']}): best calibration = {winner}")
    _fig_balancing(yte, curves)
    return tab


def _fig_balancing(y, curves):
    from sklearn.calibration import calibration_curve
    fig, ax = viz.plt.subplots(figsize=(6.5, 5.5))
    for i, (name, p) in enumerate(curves.items()):
        frac, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", color=viz.PALETTE[i], lw=1.6,
                label=f"{name} (Brier {brier_score_loss(y, p):.3f})")
    ax.plot([0, 0.2], [0, 0.2], "k--", lw=0.8)
    ax.set(xlabel="mean predicted risk (raw)", ylabel="observed frequency",
           title="Balancing strategy vs raw calibration (2022)")
    ax.legend(fontsize=8)
    viz.save(fig, "fig10_balancing_calibration.png")


# --------------------------------------------------------------------------- #
# B. Cluster-bootstrap CIs on test metrics
# --------------------------------------------------------------------------- #
def _boot_ci(y, p, clusters, rng):
    uniq = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    roc, pr, br = [], [], []
    for _ in range(N_BOOT):
        rows = np.concatenate([idx_by[c] for c in rng.choice(uniq, len(uniq), replace=True)])
        yy, pp = y[rows], p[rows]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        roc.append(roc_auc_score(yy, pp))
        pr.append(average_precision_score(yy, pp))
        br.append(brier_score_loss(yy, pp))

    def ci(a):
        return f"{np.percentile(a, 2.5):.3f}-{np.percentile(a, 97.5):.3f}"
    return ci(roc), ci(pr), ci(br)


def bootstrap_cis() -> pd.DataFrame:
    preds = pd.read_parquet(config.RESULTS / "test_predictions.parquet")
    meta = _test_meta(preds)
    clusters = meta[config.CLUSTER_COL].to_numpy()
    y = preds["y_test"].to_numpy()
    rng = np.random.default_rng(config.SEED)

    rows = []
    for col in [c for c in preds.columns if "|" in c]:
        model, fset = col.split("|")
        p = preds[col].to_numpy()
        roc_ci, pr_ci, br_ci = _boot_ci(y, p, clusters, rng)
        rows.append({
            "model": model, "feature_set": fset,
            "ROC_AUC": round(roc_auc_score(y, p), 3), "ROC_AUC_95CI": roc_ci,
            "PR_AUC": round(average_precision_score(y, p), 3), "PR_AUC_95CI": pr_ci,
            "Brier": round(brier_score_loss(y, p), 4), "Brier_95CI": br_ci,
        })
    tab = pd.DataFrame(rows).sort_values(["feature_set", "PR_AUC"], ascending=[True, False])
    tab.to_csv(config.RESULTS / "table7_bootstrap_ci.csv", index=False)
    print(f"  B. bootstrap 95% CIs ({N_BOOT} reps) for {len(tab)} models -> table7")
    return tab


# --------------------------------------------------------------------------- #
# C. Subgroup performance (best model)
# --------------------------------------------------------------------------- #
def subgroup_performance() -> pd.DataFrame:
    best = json.loads((config.RESULTS / "best_model.json").read_text())
    preds = pd.read_parquet(config.RESULTS / "test_predictions.parquet")
    meta = _test_meta(preds)
    y = preds["y_test"].to_numpy()
    p = preds[f"{best['model']}|{best['feature_set']}"].to_numpy()

    meta = meta.assign(
        wealth_group=np.where(meta["wealth"].isin(["poorest", "poorer"]), "poor",
                              np.where(meta["wealth"] == "middle", "middle", "rich")),
    )
    rows = []
    for var in ["residence", "sex", "wealth_group"]:
        for level, m in meta.groupby(var).groups.items():
            m = np.asarray(m)
            yy, pp = y[m], p[m]
            auc = roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else np.nan
            rows.append({"subgroup": f"{var}={level}", "n": len(m),
                         "deaths": int(yy.sum()),
                         "ROC_AUC": round(auc, 3) if np.isfinite(auc) else "NA"})
    tab = pd.DataFrame(rows)
    tab.to_csv(config.RESULTS / "table8_subgroups.csv", index=False)
    print(f"  C. subgroup performance ({best['model']}) -> table8")
    return tab


# --------------------------------------------------------------------------- #
# D. Missing-indicator sensitivity
# --------------------------------------------------------------------------- #
def _is_indicator(name: str) -> bool:
    return "missingindicator" in name or name.endswith("_missing")


def indicator_sensitivity() -> pd.DataFrame:
    """Refit the primary model WITHOUT missing-data indicators and compare.

    Answers the concern that a missing-data flag dominates the interpretation:
    does the model still perform once those features are removed?
    """
    from src.evaluate import _calibration
    best = json.loads((config.RESULTS / "best_model.json").read_text())
    d = build_design()
    names = list(d["feature_names"])
    idx_all = _feature_idx(names, best["feature_set"])
    idx_red = [i for i in idx_all if not _is_indicator(names[i])]

    base = joblib.load(config.MODELS / f"{best['model']}__{best['feature_set']}.joblib")
    Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]

    rows, fitted = [], {}
    for label, idx in [("with indicators (primary)", idx_all),
                       ("without indicators", idx_red)]:
        pipe = clone(base).fit(Xtr[:, idx], ytr)
        p = pipe.predict_proba(Xte[:, idx])[:, 1]
        slope, intercept = _calibration(yte, p)
        rows.append({
            "model": f"{best['model']} [{label}]", "n_features": len(idx),
            "ROC_AUC": round(roc_auc_score(yte, p), 3),
            "PR_AUC": round(average_precision_score(yte, p), 3),
            "Brier": round(brier_score_loss(yte, p), 4),
            "cal_slope": round(slope, 3), "cal_intercept": round(intercept, 3),
        })
        fitted[label] = (pipe, idx)

    # top features of the reduced model (tree models only)
    try:
        import shap
        from src.interpret_shap import _apply_transformers
        pipe, idx = fitted["without indicators"]
        Xt = _apply_transformers(pipe, Xte[:, idx])
        sv = shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[1]
        elif getattr(sv, "ndim", 2) == 3:
            sv = sv[:, :, 1]
        imp = np.abs(np.asarray(sv)).mean(axis=0)
        top = [names[idx[i]].replace("num__", "").replace("cat__", "")
               for i in np.argsort(imp)[::-1][:5]]
        print(f"  D. top-5 SHAP without indicators: {', '.join(top)}")
    except Exception as exc:
        print(f"  D. SHAP on reduced model skipped ({type(exc).__name__})")

    tab = pd.DataFrame(rows)
    tab.to_csv(config.RESULTS / "table9_indicator_sensitivity.csv", index=False)
    print(f"  D. missing-indicator sensitivity -> table9")
    return tab


def run():
    balancing_sensitivity()
    bootstrap_cis()
    subgroup_performance()
    indicator_sensitivity()


if __name__ == "__main__":
    run()
