"""
Step 05 -- temporal validation: score every model on the held-out 2022 survey.

All metrics are computed on the NATIVE (imbalanced, ~2%) 2022 prevalence -- never
on a resampled test set -- so they reflect real-world performance. Accuracy is
reported only for completeness; the honest headline metrics for a rare outcome are
PR-AUC, ROC-AUC, Brier score and calibration.

Outputs:
  * results/metrics_all.csv     (Table 2: every model x feature set)
  * figures/fig3_roc_pr.png     (ROC + PR curves)
  * figures/fig4_calibration_dca.png  (calibration + decision-curve analysis)
  * results/best_model.json     (CV-selected best model, for the SHAP step)

Run: python -m src.evaluate
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_recall_curve, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GroupKFold, cross_val_predict

import config
from src import viz
from src.preprocess import build_design


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def _calibration(y, p):
    """Calibration slope + intercept from logistic recalibration y ~ logit(p).

    Ideal: slope = 1, intercept = 0. Slope < 1 indicates over-confident (too extreme)
    predicted probabilities; a positive/negative intercept indicates over/under-prediction.
    """
    from sklearn.linear_model import LogisticRegression
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logit.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def _row(y, p, thr, w=None) -> dict:
    cal_slope, cal_intercept = _calibration(y, p)
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "ROC_AUC": round(roc_auc_score(y, p), 3),
        "ROC_AUC_wt": round(roc_auc_score(y, p, sample_weight=w), 3),
        "PR_AUC": round(average_precision_score(y, p), 3),
        "Brier": round(brier_score_loss(y, p), 4),
        "cal_slope": round(cal_slope, 3),
        "cal_intercept": round(cal_intercept, 3),
        "sensitivity": round(sens, 3),
        "specificity": round(spec, 3),
        "F1": round(f1_score(y, yhat, zero_division=0), 3),
        "threshold": round(thr, 4),
    }


def _oof_threshold(model, fset, d) -> float:
    """Youden threshold from grouped out-of-fold TRAINING predictions (no test peeking)."""
    idx = _feature_idx(list(d["feature_names"]), fset)
    pipe = clone(joblib.load(config.MODELS / f"{model}__{fset}.joblib"))
    oof = cross_val_predict(pipe, d["X_train"][:, idx], d["y_train"],
                            groups=d["groups_train"], cv=GroupKFold(5),
                            method="predict_proba")[:, 1]
    return _youden_threshold(d["y_train"], oof)


def _feature_idx(names, feature_set):
    if feature_set == "boruta":
        sel = json.loads((config.RESULTS / "selected_features.json").read_text())["selected"]
        return [names.index(f) for f in sel]
    return list(range(len(names)))


def metrics_table() -> pd.DataFrame:
    preds = pd.read_parquet(config.RESULTS / "test_predictions.parquet")
    y, w = preds["y_test"].to_numpy(), preds["weight"].to_numpy()
    d = build_design()
    rows = []
    for col in preds.columns:
        if "|" not in col:
            continue
        model, fset = col.split("|")
        thr = _oof_threshold(model, fset, d)          # threshold from TRAIN, applied to 2022
        rows.append({"model": model, "feature_set": fset,
                     **_row(y, preds[col].to_numpy(), thr, w)})
    tab = pd.DataFrame(rows).sort_values(["feature_set", "PR_AUC"], ascending=[True, False])
    out = config.RESULTS / "metrics_all.csv"
    tab.to_csv(out, index=False)
    print(f"  Table 2 -> {out}")
    return tab


# --------------------------------------------------------------------------- #
# Best model selection (by CV PR-AUC, NOT test -- avoids peeking)
# --------------------------------------------------------------------------- #
def best_model() -> dict:
    cv = pd.read_csv(config.RESULTS / "cv_results.csv")
    top = cv.sort_values("cv_pr_auc", ascending=False).iloc[0]
    info = {"model": top["model"], "feature_set": top["feature_set"],
            "cv_pr_auc": float(top["cv_pr_auc"])}
    (config.RESULTS / "best_model.json").write_text(json.dumps(info, indent=2))
    return info


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _fig_roc_pr(preds, feature_set):
    y = preds["y_test"].to_numpy()
    cols = [c for c in preds.columns if c.endswith(f"|{feature_set}")]
    fig, (ax1, ax2) = viz.plt.subplots(1, 2, figsize=(12, 5))
    for i, col in enumerate(cols):
        p = preds[col].to_numpy()
        model = col.split("|")[0]
        c = viz.PALETTE[i % len(viz.PALETTE)]
        fpr, tpr, _ = roc_curve(y, p)
        ax1.plot(fpr, tpr, color=c, lw=1.6,
                 label=f"{model} ({roc_auc_score(y, p):.2f})")
        prec, rec, _ = precision_recall_curve(y, p)
        ax2.plot(rec, prec, color=c, lw=1.6,
                 label=f"{model} ({average_precision_score(y, p):.2f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax1.set(xlabel="1 - specificity", ylabel="sensitivity",
            title=f"ROC -- 2022 test ({feature_set} features)")
    ax1.legend(fontsize=8, title="AUROC")
    ax2.axhline(y.mean(), ls="--", c="k", lw=0.8, label=f"prevalence {y.mean():.3f}")
    ax2.set(xlabel="recall", ylabel="precision",
            title=f"Precision-Recall -- 2022 test ({feature_set} features)")
    ax2.legend(fontsize=8, title="AP")
    viz.save(fig, "fig3_roc_pr.png")


def _net_benefit(y, p, thresholds):
    n = len(y)
    nb = []
    for pt in thresholds:
        yhat = p >= pt
        tp = np.sum(yhat & (y == 1))
        fp = np.sum(yhat & (y == 0))
        nb.append(tp / n - fp / n * (pt / (1 - pt)))
    return np.array(nb)


def _fig_calibration_dca(best):
    """Calibration + decision curves for the best model, raw vs isotonic-recalibrated."""
    d = build_design()
    names = list(d["feature_names"])
    if best["feature_set"] == "boruta":
        sel = json.loads((config.RESULTS / "selected_features.json").read_text())["selected"]
        idx = [names.index(f) for f in sel]
    else:
        idx = list(range(len(names)))

    import joblib
    pipe = joblib.load(config.MODELS / f"{best['model']}__{best['feature_set']}.joblib")
    Xtr, ytr, groups = d["X_train"][:, idx], d["y_train"], d["groups_train"]
    Xte, yte = d["X_test"][:, idx], d["y_test"]

    raw_p = pipe.predict_proba(Xte)[:, 1]
    # recalibrate: isotonic fit on OUT-OF-FOLD training predictions from grouped CV
    # (cluster-aware; SMOTE lives inside each cloned pipe so no leakage).
    oof = np.zeros(len(ytr))
    for tr, va in GroupKFold(5).split(Xtr, ytr, groups):
        oof[va] = clone(pipe).fit(Xtr[tr], ytr[tr]).predict_proba(Xtr[va])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, ytr)
    cal_p = iso.transform(raw_p)

    fig, (ax1, ax2) = viz.plt.subplots(1, 2, figsize=(12, 5))
    for p, lab, c in [(raw_p, "raw (SMOTE)", viz.PALETTE[4]),
                      (cal_p, "isotonic-recalibrated", viz.PALETTE[3])]:
        frac, mean_pred = calibration_curve(yte, p, n_bins=10, strategy="quantile")
        ax1.plot(mean_pred, frac, "o-", color=c, lw=1.6,
                 label=f"{lab} (Brier {brier_score_loss(yte, p):.3f})")
    ax1.plot([0, 0.2], [0, 0.2], "k--", lw=0.8)
    ax1.set(xlabel="mean predicted risk", ylabel="observed frequency",
            title=f"Calibration -- {best['model']} on 2022")
    ax1.legend(fontsize=8)

    thr = np.linspace(0.005, 0.15, 60)
    ax2.plot(thr, _net_benefit(yte, cal_p, thr), color=viz.PALETTE[3], lw=1.8,
             label=f"{best['model']} (recalibrated)")
    ax2.plot(thr, _net_benefit(yte, np.ones_like(yte, float), thr),
             color=viz.PALETTE[7], lw=1.0, ls="--", label="treat all")
    ax2.axhline(0, color="k", lw=1.0, label="treat none")
    ax2.set(xlabel="threshold probability", ylabel="net benefit",
            title="Decision-curve analysis -- 2022")
    ax2.set_ylim(-0.01, max(yte.mean() * 1.2, 0.01))
    ax2.legend(fontsize=8)
    viz.save(fig, "fig4_calibration_dca.png")


def run():
    tab = metrics_table()
    print(tab.to_string(index=False))
    best = best_model()
    print(f"\n  Best (CV PR-AUC): {best['model']} [{best['feature_set']}] "
          f"= {best['cv_pr_auc']:.3f}")
    preds = pd.read_parquet(config.RESULTS / "test_predictions.parquet")
    # plot ROC/PR for whichever feature set contains the best model
    _fig_roc_pr(preds, best["feature_set"])
    _fig_calibration_dca(best)
    return tab, best


if __name__ == "__main__":
    run()
