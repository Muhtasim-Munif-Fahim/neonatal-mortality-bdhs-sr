"""
Step 02 -- clean the analytic table and build leakage-safe design matrices.

Design:
  * clean()            -- deterministic ops only (outlier caps, ordinal mapping,
                          duplicate + multicollinearity REPORTING). No leakage.
  * make_preprocessor()-- a ColumnTransformer (median-impute + missing-indicator +
                          one-hot) used both here and, refit per fold, in modeling.
  * build_design()     -- split by survey year (train = past rounds, test = 2022),
                          FIT the preprocessor on TRAIN ONLY, transform both, cache
                          X_train/X_test (+ y, groups, weights) to data/interim.

Scaling and SMOTE are deliberately NOT done here -- they live inside each model's
per-fold pipeline in modeling.py so they only ever see training-fold rows.

Run: python -m src.preprocess
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import config
from src.harmonize import harmonize

# --------------------------------------------------------------------------- #
# Feature typing
# --------------------------------------------------------------------------- #
NUMERIC = ["mother_age", "birth_order", "birth_interval"]
BINARY = ["multiple_birth", "firstborn"]
ORDINAL = {
    "mother_edu": ["none", "primary", "secondary", "higher"],
    "wealth": ["poorest", "poorer", "middle", "richer", "richest"],
}
NOMINAL = ["sex"]

# Sensitivity scopes. These are never used to select or fit the primary model.
CONTEXT_NUMERIC = NUMERIC + ["children_ever_born"]
CONTEXT_BINARY = BINARY + ["water_improved", "sanitation_improved", "media_exposure"]
CONTEXT_ORDINAL = ["mother_edu", "wealth"]
CONTEXT_NOMINAL = NOMINAL + ["religion", "residence", "division"]
CARE_NUMERIC = CONTEXT_NUMERIC + ["anc_visits"]
CARE_BINARY = CONTEXT_BINARY + ["anc_4plus", "skilled_attendant", "csection",
                                "mother_working"]
CARE_ORDINAL = CONTEXT_ORDINAL
CARE_NOMINAL = CONTEXT_NOMINAL + ["delivery_place"]

# Domain clip bounds for outlier control (applied in clean()).
_CLIP = {
    "mother_age": (12, 49), "birth_order": (1, 15), "children_ever_born": (1, 15),
    "birth_interval": (0, 300), "anc_visits": (0, 20),
}

_VIF_THRESHOLD = 10.0   # numeric features above this are flagged, not removed


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # duplicates: report only (rows are distinct births even if covariates match).
    dups = int(df.duplicated().sum())
    print(f"  exact duplicate rows (kept, distinct births): {dups}")

    # outlier caps
    for col, (lo, hi) in _CLIP.items():
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lo, hi)

    # ordinal -> integer codes (NaN preserved)
    for col, order in ORDINAL.items():
        mapping = {lab: i for i, lab in enumerate(order)}
        df[col] = df[col].map(mapping)

    # binaries -> float (NaN preserved)
    for col in set(BINARY + CARE_BINARY):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _numeric_block(df: pd.DataFrame) -> list[str]:
    return NUMERIC + BINARY


def report_multicollinearity(train: pd.DataFrame) -> list[str]:
    """Report VIF on the numeric block (train only); no feature is removed."""
    cols = _numeric_block(train)
    X = train[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    # VIF = diagonal of the inverse correlation matrix (pandas .corr is NaN-safe).
    corr = X.corr().to_numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    try:
        vif = np.diag(np.linalg.pinv(corr))
    except np.linalg.LinAlgError:
        vif = np.full(len(cols), np.nan)
    vif_tab = (pd.DataFrame({"feature": cols, "VIF": vif})
               .sort_values("VIF", ascending=False))
    out = config.RESULTS / "multicollinearity_vif.csv"
    vif_tab.to_csv(out, index=False)
    flagged = vif_tab.loc[vif_tab["VIF"] > _VIF_THRESHOLD, "feature"].tolist()
    print(f"  VIF report -> {out}; flagged only (VIF>{_VIF_THRESHOLD}): "
          f"{flagged or 'none'}")
    return flagged


# --------------------------------------------------------------------------- #
# Preprocessor
# --------------------------------------------------------------------------- #
def make_preprocessor(numeric_cols: list[str], nominal_cols: list[str],
                      add_missing_indicators: bool = False) -> ColumnTransformer:
    numeric = SimpleImputer(strategy="median", add_indicator=add_missing_indicators)
    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=25)),
    ])
    return ColumnTransformer(
        [("num", numeric, numeric_cols), ("cat", nominal, nominal_cols)],
        remainder="drop", verbose_feature_names_out=True,
    )


# --------------------------------------------------------------------------- #
# Build design matrices
# --------------------------------------------------------------------------- #
def build_design(force: bool = False) -> dict:
    cache = config.DATA_INTERIM / "design.npz"
    meta_path = config.DATA_INTERIM / "design_meta.json"
    if cache.exists() and meta_path.exists() and not force:
        return _load_design(cache, meta_path)

    df = clean(harmonize())
    train_mask = df[config.YEAR_COL].isin(config.TRAIN_YEARS)
    test_mask = df[config.YEAR_COL] == config.TEST_YEAR
    train, test = df[train_mask].copy(), df[test_mask].copy()

    # Report collinearity, but do not use it as a prediction-feature filter.
    # VIF concerns coefficient precision rather than predictive validity, and
    # a fixed candidate set is needed across chronological validation windows.
    report_multicollinearity(train)
    numeric_cols = _numeric_block(df)
    nominal_cols = list(NOMINAL)

    # Birth interval is structurally undefined for first births, which are
    # already represented by an explicit firstborn indicator. Adding an
    # imputer-generated flag would duplicate that information exactly.
    pre = make_preprocessor(numeric_cols, nominal_cols, add_missing_indicators=False)
    Xtr = pre.fit_transform(train[numeric_cols + nominal_cols])   # FIT ON TRAIN ONLY
    Xte = pre.transform(test[numeric_cols + nominal_cols])
    names = list(pre.get_feature_names_out())

    design = {
        "X_train": np.asarray(Xtr, dtype=float),
        "X_test": np.asarray(Xte, dtype=float),
        "y_train": train[config.TARGET].to_numpy(int),
        "y_test": test[config.TARGET].to_numpy(int),
        "groups_train": train[config.CLUSTER_COL].to_numpy(),
        "w_train": train[config.WEIGHT_COL].to_numpy(float),
        "w_test": test[config.WEIGHT_COL].to_numpy(float),
        "year_train": train[config.YEAR_COL].to_numpy(int),
        "feature_names": names,
    }
    np.savez_compressed(
        cache,
        X_train=design["X_train"], X_test=design["X_test"],
        y_train=design["y_train"], y_test=design["y_test"],
        w_train=design["w_train"], w_test=design["w_test"],
        year_train=design["year_train"],
        groups_train=design["groups_train"].astype(str),
    )
    meta_path.write_text(json.dumps({
        "feature_names": names,
        "feature_scope": "birth-history variables anchored at or before birth",
        "source_columns": numeric_cols + nominal_cols,
    }, indent=2))
    print(f"Design cached -> {cache}")
    print(f"  X_train {design['X_train'].shape}  X_test {design['X_test'].shape}  "
          f"({len(names)} features)")
    print(f"  train positives {design['y_train'].sum()}/{len(design['y_train'])}  "
          f"test positives {design['y_test'].sum()}/{len(design['y_test'])}")
    return design


def _load_design(cache, meta_path) -> dict:
    z = np.load(cache, allow_pickle=True)
    names = json.loads(meta_path.read_text())["feature_names"]
    return {k: z[k] for k in z.files} | {"feature_names": names}


if __name__ == "__main__":
    d = build_design(force=True)
    print("\nFeature names:")
    for n in d["feature_names"]:
        print("  ", n)
