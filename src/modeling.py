"""
Step 04 -- train the seven classifiers with leakage-safe resampling + tuning.

Key rigor points:
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
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config
from src.features_boruta import run as boruta_run
from src.preprocess import build_design

warnings.filterwarnings("ignore", category=UserWarning)

CV_FOLDS = 5
SCORING = "average_precision"    # PR-AUC


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


def train_all(force: bool = False) -> pd.DataFrame:
    pred_path = config.RESULTS / "test_predictions.parquet"
    cv_path = config.RESULTS / "cv_results.csv"
    if pred_path.exists() and cv_path.exists() and not force:
        return pd.read_csv(cv_path)

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

    pd.DataFrame(preds).to_parquet(pred_path, index=False)
    cv = pd.DataFrame(cv_rows).sort_values(["feature_set", "cv_pr_auc"], ascending=[True, False])
    cv.to_csv(cv_path, index=False)
    print(f"\nCV summary -> {cv_path}\nTest predictions -> {pred_path}")
    return cv


if __name__ == "__main__":
    print("Training 7 models x 2 feature sets (grouped-CV, SMOTE inside folds) ...")
    cv = train_all(force=True)
    print("\n" + cv.to_string(index=False))
