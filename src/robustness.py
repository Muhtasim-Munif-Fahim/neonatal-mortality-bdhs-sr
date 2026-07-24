"""
ML robustness add-ons (run after modeling/evaluate):

  A. Balancing sensitivity -- retrain the selected comparison model under SMOTE vs ADASYN vs
     class-weight vs no resampling; compare discrimination AND calibration (raw +
     isotonic-recalibrated). Confirms the resampler choice on the honest criterion
     (calibration), not accuracy.                       -> table6, fig10
  B. Cluster-bootstrap 95% CIs on the 2022 test metrics (resample 2022 PSUs) for
     every model -- turns point estimates into interval estimates.  -> table7
  C. Subgroup performance of the selected pipeline (residence / sex / wealth). -> table8

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
from src.preprocess import (BINARY, CARE_BINARY, CARE_NOMINAL, CARE_NUMERIC,
                            CARE_ORDINAL, CONTEXT_BINARY, CONTEXT_NOMINAL,
                            CONTEXT_NUMERIC, CONTEXT_ORDINAL, NOMINAL, NUMERIC,
                            _numeric_block, build_design, clean,
                            primary_feature_cols, primary_frame)

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
    """Exploratory comparison-model analysis on the cached development design.

    This is not the locked raw-data pipeline and must not be interpreted as a
    second pipeline-selection exercise.
    """
    best = json.loads((config.RESULTS / "best_model.json").read_text())
    d = build_design()
    idx = _feature_idx(list(d["feature_names"]), best["feature_set"])
    Xtr, ytr, groups = d["X_train"][:, idx], d["y_train"], d["groups_train"]
    Xte, yte = d["X_test"][:, idx], d["y_test"]

    base = joblib.load(config.MODELS / f"{best['model']}__{best['feature_set']}.joblib")
    clf = base.named_steps["clf"]           # tuned estimator

    if "auto_class_weights" in clf.get_params():
        weighted_clf = clone(clf).set_params(auto_class_weights="Balanced")
    elif "class_weight" in clf.get_params():
        weighted_clf = clone(clf).set_params(class_weight="balanced")
    else:
        ratio = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        weighted_clf = clone(clf).set_params(scale_pos_weight=ratio)

    strategies = {
        "SMOTE": ImbPipeline([("balance", SMOTE(random_state=config.SEED)),
                              ("clf", clone(clf))]),
        "ADASYN": ImbPipeline([("balance", ADASYN(random_state=config.SEED)),
                               ("clf", clone(clf))]),
        "class_weight": weighted_clf,
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
            "analysis_scope": "exploratory fixed-design comparison model",
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
def _boot_ci(y, p, clusters, rng, strata=None):
    strata = np.zeros(len(y), dtype=int) if strata is None else np.asarray(strata)
    stratum_groups = []
    for stratum in np.unique(strata):
        sub = np.where(strata == stratum)[0]
        uniq = np.unique(clusters[sub])
        idx_by = {c: sub[clusters[sub] == c] for c in uniq}
        stratum_groups.append((uniq, idx_by))
    roc, pr, br = [], [], []
    for _ in range(N_BOOT):
        rows = np.concatenate([
            np.concatenate([idx_by[c] for c in rng.choice(uniq, len(uniq), replace=True)])
            for uniq, idx_by in stratum_groups
        ])
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
        roc_ci, pr_ci, br_ci = _boot_ci(
            y, p, clusters, rng, meta[config.STRATUM_COL].to_numpy())
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


def primary_uncertainty() -> pd.DataFrame:
    """Stratified-PSU bootstrap CIs for raw and recalibrated headline metrics."""
    from src.evaluate import _calibration
    pred = pd.read_parquet(config.RESULTS / "primary_predictions.parquet")
    meta = _test_meta(pd.DataFrame({"y_test": pred["y_test"]}))
    y = pred["y_test"].to_numpy()
    clusters = meta[config.CLUSTER_COL].to_numpy()
    strata = meta[config.STRATUM_COL].to_numpy()
    rng = np.random.default_rng(config.SEED)

    groups = []
    for stratum in np.unique(strata):
        sub = np.where(strata == stratum)[0]
        uniq = np.unique(clusters[sub])
        groups.append((uniq, {c: sub[clusters[sub] == c] for c in uniq}))

    rows = []
    for label in ["raw", "recalibrated"]:
        p = pred[label].to_numpy()
        boot = {k: [] for k in ["ROC_AUC", "PR_AUC", "Brier",
                                "cal_slope", "cal_intercept"]}
        for _ in range(N_BOOT):
            take = np.concatenate([
                np.concatenate([idx[c] for c in rng.choice(uniq, len(uniq), replace=True)])
                for uniq, idx in groups
            ])
            yy, pp = y[take], p[take]
            if yy.sum() == 0 or yy.sum() == len(yy):
                continue
            try:
                slope, intercept = _calibration(yy, pp)
            except Exception:
                continue
            boot["ROC_AUC"].append(roc_auc_score(yy, pp))
            boot["PR_AUC"].append(average_precision_score(yy, pp))
            boot["Brier"].append(brier_score_loss(yy, pp))
            boot["cal_slope"].append(slope)
            boot["cal_intercept"].append(intercept)
        slope, intercept = _calibration(y, p)
        point = {
            "ROC_AUC": roc_auc_score(y, p),
            "PR_AUC": average_precision_score(y, p),
            "Brier": brier_score_loss(y, p),
            "cal_slope": slope,
            "cal_intercept": intercept,
        }
        row = {"prediction": label, **point,
               "observed_prevalence": y.mean(), "mean_predicted_risk": p.mean(),
               "observed_to_expected": y.sum() / p.sum(),
               "bootstrap_successes": len(boot["Brier"]),
               "bootstrap_failures": N_BOOT - len(boot["Brier"])}
        for metric, values in boot.items():
            lo, hi = np.percentile(values, [2.5, 97.5])
            row[f"{metric}_95CI"] = f"{lo:.3f}-{hi:.3f}"
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(config.RESULTS / "primary_performance_uncertainty.csv", index=False)
    print("  B2. primary raw/recalibrated uncertainty -> primary_performance_uncertainty")
    return tab


# --------------------------------------------------------------------------- #
# C. Subgroup performance (selected pipeline)
# --------------------------------------------------------------------------- #
def subgroup_performance() -> pd.DataFrame:
    best = json.loads((config.RESULTS / "best_model.json").read_text())
    preds = pd.read_parquet(config.RESULTS / "primary_predictions.parquet")
    meta = _test_meta(preds)
    y = preds["y_test"].to_numpy()
    p = preds["raw"].to_numpy()

    meta = meta.assign(
        wealth_group=np.where(meta["wealth"].isin(["poorest", "poorer"]), "poor",
                              np.where(meta["wealth"] == "middle", "middle", "rich")),
    )
    from src.evaluate import _calibration
    rows = []
    for var in ["residence", "sex", "wealth_group"]:
        for level, m in meta.groupby(var).groups.items():
            m = np.asarray(m)
            yy, pp = y[m], p[m]
            valid = 0 < yy.sum() < len(yy)
            auc = roc_auc_score(yy, pp) if valid else np.nan
            pr = average_precision_score(yy, pp) if valid else np.nan
            brier = brier_score_loss(yy, pp) if valid else np.nan
            roc_ci, pr_ci, br_ci = _boot_ci(
                yy, pp, meta.iloc[m][config.CLUSTER_COL].to_numpy(),
                np.random.default_rng(config.SEED + len(rows)),
                meta.iloc[m][config.STRATUM_COL].to_numpy(),
            ) if valid else ("NA", "NA", "NA")
            slope, intercept = (np.nan, np.nan)
            if valid and yy.sum() >= 20:
                try:
                    slope, intercept = _calibration(yy, pp)
                except Exception:
                    pass
            rows.append({
                "subgroup": f"{var}={level}", "n": len(m),
                "deaths": int(yy.sum()),
                "ROC_AUC": round(auc, 3) if np.isfinite(auc) else "NA",
                "ROC_AUC_95CI": roc_ci,
                "PR_AUC": round(pr, 3) if np.isfinite(pr) else "NA",
                "PR_AUC_95CI": pr_ci,
                "Brier": round(brier, 4) if np.isfinite(brier) else "NA",
                "Brier_95CI": br_ci,
                "cal_slope": round(slope, 3) if np.isfinite(slope) else "NA",
                "cal_intercept": round(intercept, 3) if np.isfinite(intercept) else "NA",
                "analysis_label": "exploratory; raw selected-pipeline predictions",
            })
    tab = pd.DataFrame(rows)
    tab.to_csv(config.RESULTS / "table8_subgroups.csv", index=False)
    print(f"  C. subgroup performance ({best['model']}) -> table8")
    return tab


# --------------------------------------------------------------------------- #
# E. Survey-weighted, native-class training sensitivity
# --------------------------------------------------------------------------- #
def survey_weighted_native_sensitivity() -> pd.DataFrame:
    """Compare native-class fits with and without survey training weights."""
    from src.evaluate import _calibration
    from sklearn.linear_model import LogisticRegression

    def weighted_calibration(y, p, w):
        clipped = np.clip(p, 1e-6, 1 - 1e-6)
        logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        lr.fit(logit, y, sample_weight=w)
        return float(lr.coef_[0][0]), float(lr.intercept_[0])
    best = json.loads((config.RESULTS / "best_model.json").read_text())
    df = primary_frame()
    numeric_cols, nominal_cols = primary_feature_cols()
    feature_cols = numeric_cols + nominal_cols
    train = df[df[config.YEAR_COL].isin(config.TRAIN_YEARS)]
    test = df[df[config.YEAR_COL] == config.TEST_YEAR]
    Xtr, ytr = train[feature_cols], train[config.TARGET].to_numpy()
    Xte, yte = test[feature_cols], test[config.TARGET].to_numpy()
    w_train = train[config.WEIGHT_COL].to_numpy()
    w_test = test[config.WEIGHT_COL].to_numpy()
    base = joblib.load(config.MODELS / "locked_pipeline.joblib")
    native_steps = [(name, clone(step)) for name, step in base.steps
                    if not hasattr(step, "fit_resample")]

    rows = []
    for label, weights in [("native_unweighted", None),
                           ("native_survey_weighted", w_train)]:
        pipe = ImbPipeline([(name, clone(step)) for name, step in native_steps])
        if weights is None:
            pipe.fit(Xtr, ytr)
        else:
            try:
                pipe.fit(Xtr, ytr, clf__sample_weight=weights)
            except TypeError:
                # KNN and other estimators without sample_weight use a
                # deterministic probability-proportional weighted resample.
                rng = np.random.default_rng(config.SEED)
                take = rng.choice(len(ytr), len(ytr), replace=True,
                                  p=weights / weights.sum())
                pipe.fit(Xtr.iloc[take], ytr[take])
        p = pipe.predict_proba(Xte)[:, 1]
        slope, intercept = _calibration(yte, p)
        slope_w, intercept_w = weighted_calibration(yte, p, w_test)
        rows.append({
            "training": label,
            "ROC_AUC": roc_auc_score(yte, p),
            "PR_AUC": average_precision_score(yte, p),
            "Brier": brier_score_loss(yte, p),
            "cal_slope": slope,
            "cal_intercept": intercept,
            "ROC_AUC_weighted": roc_auc_score(yte, p, sample_weight=w_test),
            "PR_AUC_weighted": average_precision_score(
                yte, p, sample_weight=w_test),
            "Brier_weighted": brier_score_loss(yte, p, sample_weight=w_test),
            "cal_slope_weighted": slope_w,
            "cal_intercept_weighted": intercept_w,
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(config.RESULTS / "table10_survey_weighted_native.csv", index=False)
    print("  E. survey-weighted native-class sensitivity -> table10")
    return tab


# --------------------------------------------------------------------------- #
# F. Predictor-timing and questionnaire-scope sensitivities
# --------------------------------------------------------------------------- #
def predictor_scope_sensitivity() -> pd.DataFrame:
    """Evaluate enriched scopes without altering primary pipeline selection.

    The primary model contains only birth-history variables anchored at or
    before birth. This function holds the selected model family and tuned
    hyperparameters fixed, then adds (1) survey-time contextual variables in
    the all-recent cohort and (2) maternity-care variables in the most-recent
    birth/long-questionnaire cohort. Grouped development OOF predictions are
    used solely to fit isotonic recalibration for each sensitivity.
    """
    from src.modeling import _models, _raw_pipeline

    lock = json.loads((config.RESULTS / "pipeline_lock.json").read_text())
    est, _, scale, balancer = _models()[lock["model"]]
    fixed_params = lock.get("final_best_params", {})
    scopes = [
        ("birth_anchored_only", False,
         NUMERIC + BINARY, NOMINAL,
         "variables anchored at or before birth; prospective-valid but limited discrimination"),
        ("survey_context_enriched", False,
         CONTEXT_NUMERIC + CONTEXT_BINARY + CONTEXT_ORDINAL, CONTEXT_NOMINAL,
         "primary model; survey-time socioeconomic context is a cross-sectional risk factor, not birth-time state"),
        ("care_enriched_most_recent", True,
         CARE_NUMERIC + CARE_BINARY + CARE_ORDINAL, CARE_NOMINAL,
         "most-recent birth; 2022 long questionnaire; selection can depend on later fertility"),
    ]
    rows = []
    for label, module_only, numeric_cols, nominal_cols, limitation in scopes:
        df = clean(harmonize(module_only=module_only))
        feature_cols = numeric_cols + nominal_cols
        train = df[df[config.YEAR_COL].isin(config.TRAIN_YEARS)].copy()
        test = df[df[config.YEAR_COL] == config.TEST_YEAR].copy()
        ytr = train[config.TARGET].to_numpy(int)
        yte = test[config.TARGET].to_numpy(int)
        groups = train[config.CLUSTER_COL].to_numpy()
        pipe = _raw_pipeline(est, scale, balancer, numeric_cols, nominal_cols,
                             add_missing_indicators=True)
        if fixed_params:
            pipe.set_params(**fixed_params)

        oof = np.zeros(len(train), dtype=float)
        gkf = GroupKFold(n_splits=5)
        for tr_idx, va_idx in gkf.split(train[feature_cols], ytr, groups):
            fitted = clone(pipe).fit(train.iloc[tr_idx][feature_cols], ytr[tr_idx])
            oof[va_idx] = fitted.predict_proba(train.iloc[va_idx][feature_cols])[:, 1]
        fitted = clone(pipe).fit(train[feature_cols], ytr)
        raw_p = fitted.predict_proba(test[feature_cols])[:, 1]
        calibrated = IsotonicRegression(out_of_bounds="clip").fit(oof, ytr).transform(raw_p)
        null_brier = brier_score_loss(yte, np.full(len(yte), yte.mean()))
        for prediction, p in [("raw", raw_p), ("isotonic_recalibrated", calibrated)]:
            from src.evaluate import _calibration
            slope, intercept = _calibration(yte, p)
            brier = brier_score_loss(yte, p)
            rows.append({
                "scope": label,
                "prediction": prediction,
                "n_development": len(train),
                "events_development": int(ytr.sum()),
                "n_evaluation": len(test),
                "events_evaluation": int(yte.sum()),
                "source_predictors": len(feature_cols),
                "ROC_AUC": roc_auc_score(yte, p),
                "PR_AUC": average_precision_score(yte, p),
                "PR_AUC_baseline": float(yte.mean()),
                "Brier": brier,
                "Brier_null_evaluation": null_brier,
                "Brier_skill": 1 - brier / null_brier,
                "cal_slope": slope,
                "cal_intercept": intercept,
                "interpretive_limit": limitation,
            })
        print(f"  F. {label}: {len(train):,}/{int(ytr.sum())} development, "
              f"{len(test):,}/{int(yte.sum())} evaluation")
    tab = pd.DataFrame(rows)
    tab.to_csv(config.RESULTS / "table11_predictor_scope_sensitivity.csv", index=False)
    return tab


# --------------------------------------------------------------------------- #
# D. Missing-indicator sensitivity
# --------------------------------------------------------------------------- #
def _is_indicator(name: str) -> bool:
    return "missingindicator" in name or name.endswith("_missing")


def indicator_sensitivity() -> pd.DataFrame:
    """Refit a fixed-design comparison model without missing indicators.

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
    comparisons = ([ ("no automatic indicators; explicit firstborn flag", idx_all) ]
                   if idx_all == idx_red else
                   [("with indicators (comparison)", idx_all),
                    ("without indicators", idx_red)])
    for label, idx in comparisons:
        pipe = clone(base).fit(Xtr[:, idx], ytr)
        p = pipe.predict_proba(Xte[:, idx])[:, 1]
        slope, intercept = _calibration(yte, p)
        rows.append({
            "analysis_scope": "exploratory fixed-design comparison model",
            "model": f"{best['model']} [{label}]", "n_features": len(idx),
            "ROC_AUC": round(roc_auc_score(yte, p), 3),
            "PR_AUC": round(average_precision_score(yte, p), 3),
            "Brier": round(brier_score_loss(yte, p), 4),
            "cal_slope": round(slope, 3), "cal_intercept": round(intercept, 3),
        })
        fitted[label] = (pipe, idx)

    # Top features of a reduced tree comparison, when such a reduction exists.
    try:
        if idx_all == idx_red:
            raise ValueError("primary design has no automatic missing indicators")
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
    primary_uncertainty()
    subgroup_performance()
    indicator_sensitivity()
    survey_weighted_native_sensitivity()
    predictor_scope_sensitivity()


if __name__ == "__main__":
    run()
