"""
Step 06 -- descriptive SHAP attribution and round-specific profiles.

  * Fig 5: SHAP beeswarm + mean|SHAP| bar for the best TREE-based model on 2022.
  * Fig 6: mean|SHAP| per feature computed SEPARATELY within each survey round
           (2011, 2014, 2017, 2022) using the one model trained on 2011-2017 --
           showing which predictors persist vs fade across a decade.

SHAP uses the best tree-based model (TreeExplainer is exact and fast). If the
selected model is not tree-based, we still explain the selected tree model and
say so, because that is what SHAP can attribute reliably.

Run: python -m src.interpret_shap
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import shap

import config
from src import viz
from src.harmonize import harmonize
from src.preprocess import NOMINAL, _numeric_block, build_design, clean

TREE_MODELS = {"DT", "RF", "XGB", "CatBoost"}
TOP_N = 12


def _best_tree() -> dict:
    lock_path = config.RESULTS / "pipeline_lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if lock["model"] in TREE_MODELS:
            return {"model": lock["model"], "feature_set": lock["feature_set"],
                    "locked": True}
    cv = pd.read_csv(config.RESULTS / "cv_results.csv")
    cv = cv[cv["model"].isin(TREE_MODELS)].sort_values("cv_pr_auc", ascending=False)
    top = cv.iloc[0]
    return {"model": top["model"], "feature_set": top["feature_set"],
            "locked": False}


def _feature_idx_and_names(d):
    names = list(d["feature_names"])
    best = _best_tree()
    if best["feature_set"] == "boruta":
        sel = json.loads((config.RESULTS / "selected_features.json").read_text())["selected"]
        idx = [names.index(f) for f in sel]
    else:
        idx = list(range(len(names)))
    pretty = [names[i].replace("num__", "").replace("cat__", "") for i in idx]
    return best, idx, pretty


def _apply_transformers(pipe, X):
    """Apply only genuine transformers (e.g. scaler); skip samplers (SMOTE/RUS)."""
    Xt = X
    for _name, step in pipe.steps[:-1]:
        if hasattr(step, "fit_resample"):     # a sampler -> no-op at inference
            continue
        Xt = step.transform(Xt)
    return Xt


def _shap_values(pipe, X):
    """Return a (n, n_features) SHAP matrix for the positive class."""
    Xt = _apply_transformers(pipe, X)
    clf = pipe.named_steps["clf"]
    sv = shap.TreeExplainer(clf).shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    return np.asarray(sv), Xt


def run():
    best = _best_tree()
    if best["locked"]:
        pipe = joblib.load(config.MODELS / "locked_pipeline.joblib")
        df = clean(harmonize())
        feature_cols = _numeric_block(df) + list(NOMINAL)
        transformed_names = list(
            pipe.named_steps["preprocess"].get_feature_names_out())
        pretty = [n.replace("num__", "").replace("cat__", "")
                  for n in transformed_names]
        train = df[df[config.YEAR_COL].isin(config.TRAIN_YEARS)]
        test = df[df[config.YEAR_COL] == config.TEST_YEAR]
        Xte = test[feature_cols]
    else:
        d = build_design()
        best, idx, pretty = _feature_idx_and_names(d)
        pipe = joblib.load(config.MODELS / f"{best['model']}__{best['feature_set']}.joblib")
        Xte = d["X_test"][:, idx]
    print(f"  SHAP on best tree model: {best['model']} [{best['feature_set']}]")

    # ---- Fig 5: beeswarm + bar on 2022 ----------------------------------- #
    sv, Xt = _shap_values(pipe, Xte)
    shap.summary_plot(sv, Xt, feature_names=pretty, show=False, max_display=TOP_N)
    fig = viz.plt.gcf()
    fig.suptitle(f"SHAP -- {best['model']} in the 2022 temporal evaluation", y=1.02)
    viz.save(fig, "fig5_shap_beeswarm.png")

    # ---- Fig 6: temporal stability of mean|SHAP| ------------------------- #
    per_round = {}
    for yr in config.TRAIN_YEARS:
        if best["locked"]:
            rows = train.loc[train[config.YEAR_COL] == yr, feature_cols]
        else:
            rows = d["X_train"][d["year_train"] == yr][:, idx]
        sv_y, _ = _shap_values(pipe, rows)
        per_round[yr] = np.abs(sv_y).mean(axis=0)
    per_round[config.TEST_YEAR] = np.abs(sv).mean(axis=0)

    imp = pd.DataFrame(per_round, index=pretty)
    imp = imp.div(imp.sum(axis=0), axis=1)                  # share of importance per round
    order = imp.mean(axis=1).sort_values(ascending=False).index[:TOP_N]
    imp = imp.loc[order]
    imp.to_csv(config.RESULTS / "shap_temporal.csv")

    fig, ax = viz.plt.subplots(figsize=(9, 6))
    for i, yr in enumerate(list(config.TRAIN_YEARS) + [config.TEST_YEAR]):
        ax.plot(imp[yr].values, range(len(imp))[::-1], "o-", color=viz.PALETTE[i],
                lw=1.4, label=str(yr))
    ax.set_yticks(range(len(imp))[::-1])
    ax.set_yticklabels(imp.index, fontsize=8)
    ax.set_xlabel("share of mean|SHAP| within round")
    ax.set_title(f"Round-specific SHAP attribution profiles ({best['model']})")
    ax.legend(title="BDHS round", fontsize=8)
    viz.save(fig, "fig6_shap_temporal.png")


if __name__ == "__main__":
    run()
