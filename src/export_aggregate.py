"""Export non-disclosive aggregate tables underlying the five main figures.

No record-level predictions, DHS microdata, cluster identifiers, or fitted model
objects are written to the release directory.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

import config

OUT = config.RESULTS / "aggregate_release"


def _curve_table(pred: pd.DataFrame) -> pd.DataFrame:
    grid = np.linspace(0, 1, 101)
    rows = []
    y = pred["y_test"].to_numpy()
    for label in ["raw", "recalibrated"]:
        p = pred[label].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        for x, value in zip(grid, np.interp(grid, fpr, tpr)):
            rows.append({"prediction": label, "curve": "ROC",
                         "x": round(float(x), 3), "y": round(float(value), 6)})
        precision, recall, _ = precision_recall_curve(y, p)
        order = np.argsort(recall)
        for x, value in zip(grid, np.interp(grid, recall[order], precision[order])):
            rows.append({"prediction": label, "curve": "precision-recall",
                         "x": round(float(x), 3), "y": round(float(value), 6)})
    return pd.DataFrame(rows)


def _calibration_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = pred["y_test"].to_numpy()
    for label in ["raw", "recalibrated"]:
        obs, mean = calibration_curve(y, pred[label], n_bins=10, strategy="quantile")
        for i, (m, o) in enumerate(zip(mean, obs), 1):
            rows.append({"prediction": label, "quantile_bin": i,
                         "mean_prediction": round(float(m), 6),
                         "observed_fraction": round(float(o), 6)})
    return pd.DataFrame(rows)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(config.RESULTS / "primary_predictions.parquet")
    _curve_table(pred).to_csv(OUT / "figure2_discrimination_curves.csv", index=False)
    _calibration_table(pred).to_csv(OUT / "figure2_calibration_bins.csv", index=False)

    curated = [
        "cohort_definition.csv", "forward_validation.csv", "pipeline_lock.json",
        "primary_performance_uncertainty.csv", "primary_recalibration.csv",
        "table1_cohort_characteristics.csv", "shap_temporal.csv",
        "table3_nmr_trend.csv", "table3b_prevalence.csv",
        "table5_decomposition.csv", "table5b_decomposition_care_module.csv",
        "table8_subgroups.csv", "table10_survey_weighted_native.csv",
    ]
    for name in curated:
        src = config.RESULTS / name
        if src.exists():
            shutil.copy2(src, OUT / name)

    contents = sorted(p.name for p in OUT.iterdir() if p.is_file())
    (OUT / "README.txt").write_text(
        "Non-disclosive aggregate source tables for the Scientific Reports "
        "submission. Record-level DHS data, predictions, identifiers and fitted "
        "model objects are deliberately excluded.\n\nFiles:\n- " +
        "\n- ".join(contents) + "\n", encoding="utf-8")
    print(f"Aggregate figure/source tables -> {OUT}")


if __name__ == "__main__":
    run()
