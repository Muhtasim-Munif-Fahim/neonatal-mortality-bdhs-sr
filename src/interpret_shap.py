"""Step 06 -- descriptive attribution for the locked prediction pipeline.

SHAP values are always computed for the model family locked by chronological
development-only validation. LinearExplainer is used for logistic regression;
TreeExplainer is used for supported tree models. A single development-derived
background/reference distribution is retained across survey rounds so temporal
profiles are comparable.

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
from src.preprocess import NOMINAL, _numeric_block, clean

TREE_MODELS = {"DT", "RF", "XGB", "CatBoost"}
TOP_N = 12


def _apply_transformers(pipe, X):
    """Apply fitted inference-time transformers while skipping samplers."""
    Xt = X
    for _name, step in pipe.steps[:-1]:
        if hasattr(step, "fit_resample"):
            continue
        Xt = step.transform(Xt)
    return np.asarray(Xt)


def _make_explainer(model_name, clf, background):
    if model_name == "LR":
        return shap.LinearExplainer(clf, background)
    if model_name in TREE_MODELS:
        return shap.TreeExplainer(clf)
    return shap.Explainer(clf.predict_proba, background)


def _shap_values(explainer, Xt):
    """Return a (n, n_features) attribution matrix for the positive class."""
    try:
        values = explainer.shap_values(Xt)
    except AttributeError:
        values = explainer(Xt).values
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def run():
    lock = json.loads((config.RESULTS / "pipeline_lock.json").read_text())
    model_name = lock["model"]
    pipe = joblib.load(config.MODELS / "locked_pipeline.joblib")
    df = clean(harmonize(module_only=False))
    feature_cols = _numeric_block(df) + list(NOMINAL)
    train = df[df[config.YEAR_COL].isin(config.TRAIN_YEARS)]
    test = df[df[config.YEAR_COL] == config.TEST_YEAR]
    names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    pretty = [n.replace("num__", "").replace("cat__", "") for n in names]

    transformed_train = _apply_transformers(pipe, train[feature_cols])
    rng = np.random.default_rng(config.SEED)
    take = rng.choice(len(transformed_train), min(1000, len(transformed_train)),
                      replace=False)
    explainer = _make_explainer(model_name, pipe.named_steps["clf"],
                                transformed_train[take])
    transformed_test = _apply_transformers(pipe, test[feature_cols])
    test_values = _shap_values(explainer, transformed_test)
    print(f"  SHAP on locked model: {model_name} [{lock['feature_set']}]")

    shap.summary_plot(test_values, transformed_test, feature_names=pretty,
                      show=False, max_display=TOP_N)
    fig = viz.plt.gcf()
    fig.suptitle(f"Attribution in the 2022 temporal evaluation ({model_name})", y=1.02)
    viz.save(fig, "fig5_shap_beeswarm.png")

    per_round = {}
    for year in config.TRAIN_YEARS:
        rows = train.loc[train[config.YEAR_COL] == year, feature_cols]
        values = _shap_values(explainer, _apply_transformers(pipe, rows))
        per_round[year] = np.abs(values).mean(axis=0)
    per_round[config.TEST_YEAR] = np.abs(test_values).mean(axis=0)

    importance = pd.DataFrame(per_round, index=pretty)
    importance = importance.div(importance.sum(axis=0), axis=1)
    order = importance.mean(axis=1).sort_values(ascending=False).index[:TOP_N]
    importance = importance.loc[order]
    importance.index.name = "feature"
    importance.to_csv(config.RESULTS / "shap_temporal.csv")

    fig, ax = viz.plt.subplots(figsize=(9, 6))
    for i, year in enumerate(list(config.TRAIN_YEARS) + [config.TEST_YEAR]):
        ax.plot(importance[year].values, range(len(importance))[::-1], "o-",
                color=viz.PALETTE[i], lw=1.4, label=str(year))
    ax.set_yticks(range(len(importance))[::-1])
    ax.set_yticklabels(importance.index, fontsize=8)
    ax.set_xlabel("share of mean absolute SHAP value within round")
    ax.set_title(f"Round-specific attribution profiles ({model_name})")
    ax.legend(title="BDHS round", fontsize=8)
    viz.save(fig, "fig6_shap_temporal.png")


if __name__ == "__main__":
    run()
