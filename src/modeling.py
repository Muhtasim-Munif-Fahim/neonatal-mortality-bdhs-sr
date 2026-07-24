"""
Step 04 -- train the seven classifiers with leakage-safe resampling + tuning.

Key rigor points:
  * the model family is locked by forward-chaining development-only evaluation
    (2011->2014 and 2011-2014->2017-18), with grouped inner tuning,
  * each model is an imblearn Pipeline  [scale?] -> SMOTE -> estimator, so the
    resampler (and scaler) only ever see TRAINING-fold rows,
  * hyper-parameters are tuned with GROUP K-fold, grouped by DHS cluster (v001),
    so the same PSU never lands in both a train and a validation fold,
  * tuning uses average-precision (PR-AUC) -- the right criterion for a ~2.5%
    outcome; accuracy is never optimised,
  * the 2022 test survey is only ever used for final prediction (in evaluate.py),
    never for tuning.

Each model is run on two feature sets: the Boruta-selected subset and the full
matrix, so the paper can report whether selection costs anything.

Outputs:
  * results/cv_results.csv            (best params + CV PR-AUC + fit time)
  * results/test_predictions.parquet  (per-model predicted risk on 2022)
  * models/<model>__<featureset>.joblib

Run: python -m src.modeling
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from src.features_boruta import run as boruta_run
from src.harmonize import harmonize
from src.preprocess import (NOMINAL, PRIMARY_ADD_INDICATORS, _numeric_block,
                            build_design, clean, make_preprocessor,
                            primary_feature_cols, primary_frame)

warnings.filterwarnings("ignore", category=UserWarning)

CV_FOLDS = 5
SCORING = "average_precision"    # PR-AUC
NEAR_TIE_PR_AUC = 0.005
_SIMPLICITY = {"LR": 0, "DT": 1, "RF": 2, "KNN": 3,
               "XGB": 4, "CatBoost": 5, "SVM": 6}


def _smote():
    return [("balance", SMOTE(random_state=config.SEED))]


def _hybrid():
    # undersample majority to 10x minority, then SMOTE to 1:1 -> a balanced set of
    # ~6-7k rows. Used for rbf-SVM only, where full SMOTE (~29k) is intractable
    # (a single fit exceeds 5 min). Documented as hybrid resampling.
    return [("under", RandomUnderSampler(sampling_strategy=0.1, random_state=config.SEED)),
            ("smote", SMOTE(random_state=config.SEED))]


def _models() -> dict:
    """name -> (estimator, param_grid, needs_scaling, balancer_steps)."""
    seed = config.SEED
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    return {
        "LR": (LogisticRegression(max_iter=2000, random_state=seed),
               {"clf__C": [0.1, 1.0, 10.0]}, True, _smote()),
        "DT": (DecisionTreeClassifier(random_state=seed),
               {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [20, 50]}, False, _smote()),
        "RF": (RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
               {"clf__max_depth": [6, 10], "clf__min_samples_leaf": [10, 20]}, False, _smote()),
        "KNN": (KNeighborsClassifier(),
                {"clf__n_neighbors": [15, 31, 51]}, True, _smote()),
        "SVM": (SVC(probability=True, kernel="rbf", cache_size=600, random_state=seed),
                {"clf__C": [0.5, 1.0, 2.0]}, True, _hybrid()),
        "XGB": (XGBClassifier(n_estimators=300, eval_metric="logloss",
                              tree_method="hist", random_state=seed, n_jobs=-1),
                {"clf__max_depth": [3, 5], "clf__learning_rate": [0.05, 0.1]}, False, _smote()),
        "CatBoost": (CatBoostClassifier(iterations=300, random_state=seed, verbose=0),
                     {"clf__depth": [4, 6]}, False, _smote()),
    }


def _pipeline(estimator, needs_scaling: bool, balancer_steps: list) -> ImbPipeline:
    steps = []
    if needs_scaling:
        steps.append(("scale", StandardScaler()))
    steps += balancer_steps
    steps.append(("clf", estimator))
    return ImbPipeline(steps)


def _raw_pipeline(estimator, needs_scaling: bool,
                  balancer_steps: list, numeric_cols: list[str] | None = None,
                  nominal_cols: list[str] | None = None,
                  add_missing_indicators: bool = False) -> ImbPipeline:
    """Preprocessing-to-estimator pipeline used in chronological validation."""
    numeric_cols = numeric_cols or _numeric_block(pd.DataFrame())
    nominal_cols = nominal_cols or list(NOMINAL)
    steps = [("preprocess", make_preprocessor(
        numeric_cols, nominal_cols, add_missing_indicators=add_missing_indicators))]
    if needs_scaling:
        steps.append(("scale", StandardScaler()))
    steps += balancer_steps
    steps.append(("clf", estimator))
    return ImbPipeline(steps)


def forward_validate(force: bool = False) -> tuple[pd.DataFrame, dict]:
    """Lock a model family using development surveys only.

    Mean chronological validation PR-AUC is the ranking criterion. Candidates
    within 0.005 of the highest mean are resolved by lower validation Brier
    score, then by a declared model-simplicity order.
    """
    out = config.RESULTS / "forward_validation.csv"
    lock_path = config.RESULTS / "pipeline_lock.json"
    if out.exists() and lock_path.exists() and not force:
        return pd.read_csv(out), json.loads(lock_path.read_text())

    df = primary_frame()
    numeric_cols, nominal_cols = primary_feature_cols()
    feature_cols = numeric_cols + nominal_cols
    windows = [([2011], 2014), ([2011, 2014], 2017)]
    rows = []
    for development_years, validation_year in windows:
        tr = df[df[config.YEAR_COL].isin(development_years)].copy()
        va = df[df[config.YEAR_COL] == validation_year].copy()
        n_splits = min(CV_FOLDS, tr[config.CLUSTER_COL].nunique())
        inner = GroupKFold(n_splits=n_splits)
        for mname, (est, grid, scale, balancer) in _models().items():
            search = GridSearchCV(
                _raw_pipeline(est, scale, balancer, numeric_cols, nominal_cols,
                              add_missing_indicators=PRIMARY_ADD_INDICATORS),
                grid, scoring=SCORING,
                cv=inner, n_jobs=-1, refit=True, error_score="raise",
            )
            search.fit(tr[feature_cols], tr[config.TARGET],
                       groups=tr[config.CLUSTER_COL])
            p = search.predict_proba(va[feature_cols])[:, 1]
            pr = average_precision_score(va[config.TARGET], p)
            brier_raw = brier_score_loss(va[config.TARGET], p)
            # A Brier tie-breaker on SMOTE probabilities would mostly compare
            # resampling-induced calibration distortion. Fit an isotonic map on
            # PSU-grouped OOF predictions from the development years only, then
            # apply that map to the forward validation round.
            oof = cross_val_predict(
                clone(search.best_estimator_), tr[feature_cols], tr[config.TARGET],
                groups=tr[config.CLUSTER_COL], cv=inner, method="predict_proba",
                n_jobs=-1,
            )[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip").fit(
                oof, tr[config.TARGET].to_numpy())
            brier = brier_score_loss(va[config.TARGET], iso.transform(p))
            rows.append({
                "model": mname,
                "development_years": "+".join(map(str, development_years)),
                "validation_year": validation_year,
                "n_validation": len(va),
                "events_validation": int(va[config.TARGET].sum()),
                "PR_AUC": pr,
                "Brier": brier,
                "Brier_raw": brier_raw,
                "Brier_definition": "validation Brier after development-only grouped-OOF isotonic recalibration",
                "best_params": json.dumps(search.best_params_, sort_keys=True),
            })
            print(f"  forward {development_years}->{validation_year} {mname}: "
                  f"PR-AUC={pr:.3f}, recalibrated Brier={brier:.4f}")

    detail = pd.DataFrame(rows)
    summary = (detail.groupby("model", as_index=False)
               .agg(mean_PR_AUC=("PR_AUC", "mean"),
                    mean_Brier=("Brier", "mean")))
    top_pr = float(summary["mean_PR_AUC"].max())
    summary["near_tie"] = summary["mean_PR_AUC"] >= top_pr - NEAR_TIE_PR_AUC
    summary["simplicity_rank"] = summary["model"].map(_SIMPLICITY)
    chosen = (summary[summary["near_tie"]]
              .sort_values(["mean_Brier", "simplicity_rank", "model"])
              .iloc[0])
    summary["selected"] = summary["model"].eq(chosen["model"])
    detail.merge(summary, on="model", how="left").to_csv(out, index=False)
    lock = {
        "model": str(chosen["model"]),
        "feature_set": "full",
        "selection_rule": (
            "highest mean forward-validation PR-AUC; within 0.005, lower "
            "mean development-OOF-recalibrated Brier score; then model simplicity"
        ),
        "mean_pr_auc": float(chosen["mean_PR_AUC"]),
        "mean_brier": float(chosen["mean_Brier"]),
        "development_windows": ["2011->2014", "2011+2014->2017"],
    }
    lock_path.write_text(json.dumps(lock, indent=2))
    print(f"  Pipeline selected using 2011-2017/18 only: {lock['model']} [full]")
    return detail.merge(summary, on="model", how="left"), lock


def fit_locked_pipeline(lock: dict) -> np.ndarray:
    """Tune/final-fit the locked raw-data pipeline with fold-local preprocessing.

    The returned 2022 predictions replace the corresponding comparison-model
    column so all headline evaluation uses the fully nested pipeline. Grouped
    OOF development predictions are saved for thresholding and recalibration.
    """
    df = primary_frame()
    numeric_cols, nominal_cols = primary_feature_cols()
    feature_cols = numeric_cols + nominal_cols
    train = df[df[config.YEAR_COL].isin(config.TRAIN_YEARS)].copy()
    test = df[df[config.YEAR_COL] == config.TEST_YEAR].copy()
    est, grid, scale, balancer = _models()[lock["model"]]
    grouped_cv = GroupKFold(n_splits=CV_FOLDS)
    # Fully nested OOF predictions for thresholding/recalibration: each outer
    # validation fold is predicted by hyperparameters selected only within the
    # remaining outer-training PSUs.
    y_train = train[config.TARGET].to_numpy()
    groups = train[config.CLUSTER_COL].to_numpy()
    oof = np.zeros(len(train), dtype=float)
    for fold, (outer_tr, outer_va) in enumerate(
            grouped_cv.split(train[feature_cols], y_train, groups), start=1):
        inner = GroupKFold(n_splits=CV_FOLDS)
        nested = GridSearchCV(
            _raw_pipeline(est, scale, balancer, numeric_cols, nominal_cols,
                          add_missing_indicators=PRIMARY_ADD_INDICATORS),
            grid, scoring=SCORING,
            cv=inner, n_jobs=-1, refit=True, error_score="raise",
        )
        nested.fit(train.iloc[outer_tr][feature_cols], y_train[outer_tr],
                   groups=groups[outer_tr])
        oof[outer_va] = nested.predict_proba(
            train.iloc[outer_va][feature_cols])[:, 1]
        print(f"  nested calibration OOF fold {fold}/{CV_FOLDS} complete")

    # Once OOF predictions are sealed, tune on all development data and refit
    # the final pipeline for the single temporally separated 2022 evaluation.
    search = GridSearchCV(
        _raw_pipeline(est, scale, balancer, numeric_cols, nominal_cols,
                      add_missing_indicators=PRIMARY_ADD_INDICATORS),
        grid, scoring=SCORING,
        cv=grouped_cv, n_jobs=-1, refit=True, error_score="raise",
    )
    search.fit(train[feature_cols], y_train, groups=groups)
    locked = search.best_estimator_
    pd.DataFrame({
        "y_train": y_train,
        "oof_probability": oof,
    }).to_parquet(config.RESULTS / "locked_oof_predictions.parquet", index=False)
    joblib.dump(locked, config.MODELS / "locked_pipeline.joblib")
    details = {**lock, "final_best_params": search.best_params_,
               "final_group_cv_pr_auc": float(search.best_score_),
               "preprocessing": "inside every grouped CV fold"}
    (config.RESULTS / "pipeline_lock.json").write_text(json.dumps(details, indent=2))
    return locked.predict_proba(test[feature_cols])[:, 1]


def train_all(force: bool = False) -> pd.DataFrame:
    pred_path = config.RESULTS / "test_predictions.parquet"
    cv_path = config.RESULTS / "cv_results.csv"
    lock_path = config.RESULTS / "pipeline_lock.json"
    if pred_path.exists() and cv_path.exists() and lock_path.exists() and not force:
        return pd.read_csv(cv_path)

    _, lock = forward_validate(force=force)

    d = build_design()
    names = list(d["feature_names"])
    sel = boruta_run()["selected"]
    sel_idx = [names.index(f) for f in sel]
    feature_sets = {"full": list(range(len(names))), "boruta": sel_idx}

    Xtr, ytr = d["X_train"], d["y_train"]
    groups = d["groups_train"]
    Xte, yte = d["X_test"], d["y_test"]
    gkf = GroupKFold(n_splits=CV_FOLDS)

    preds = {"y_test": yte, "weight": d["w_test"]}
    cv_rows = []

    for fset, idx in feature_sets.items():
        Xtr_f, Xte_f = Xtr[:, idx], Xte[:, idx]
        for mname, (est, grid, scale, balancer) in _models().items():
            t0 = time.time()
            search = GridSearchCV(
                _pipeline(est, scale, balancer), grid, scoring=SCORING,
                cv=gkf, n_jobs=-1, refit=True, error_score="raise",
            )
            try:
                search.fit(Xtr_f, ytr, groups=groups)
            except Exception as exc:            # keep the run alive; log the failure
                print(f"  [FAIL] {mname}|{fset}: {type(exc).__name__}: {exc}")
                continue
            dt = time.time() - t0
            proba = search.predict_proba(Xte_f)[:, 1]
            preds[f"{mname}|{fset}"] = proba
            joblib.dump(search.best_estimator_,
                        config.MODELS / f"{mname}__{fset}.joblib")
            cv_rows.append({
                "model": mname, "feature_set": fset,
                "cv_pr_auc": round(search.best_score_, 4),
                "best_params": json.dumps(search.best_params_),
                "fit_seconds": round(dt, 1),
            })
            print(f"  {mname:9s} {fset:6s}  CV PR-AUC={search.best_score_:.3f}  ({dt:.0f}s)")

    preds[f"{lock['model']}|full"] = fit_locked_pipeline(lock)
    pd.DataFrame(preds).to_parquet(pred_path, index=False)
    cv = pd.DataFrame(cv_rows).sort_values(["feature_set", "cv_pr_auc"], ascending=[True, False])
    cv.to_csv(cv_path, index=False)
    print(f"\nCV summary -> {cv_path}\nTest predictions -> {pred_path}")
    return cv


if __name__ == "__main__":
    print("Training 7 models x 2 feature sets (grouped-CV, SMOTE inside folds) ...")
    cv = train_all(force=True)
    print("\n" + cv.to_string(index=False))
