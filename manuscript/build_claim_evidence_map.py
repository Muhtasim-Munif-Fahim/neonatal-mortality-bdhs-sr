"""Generate a machine-readable map from headline claims to regenerated outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "manuscript" / "scientific_reports" / "review"


def main() -> None:
    cohort = pd.read_csv(R / "cohort_definition.csv")
    perf = pd.read_csv(R / "primary_performance_uncertainty.csv").set_index("prediction")
    recal = pd.read_csv(R / "primary_recalibration.csv").set_index("prediction")
    trend = pd.read_csv(R / "table3_nmr_trend.csv").set_index("year")
    dec = pd.read_csv(R / "table5_decomposition.csv").set_index("component")
    care = pd.read_csv(R / "table5b_decomposition_care_module.csv").set_index("component")
    lock = json.loads((R / "pipeline_lock.json").read_text())

    allr = cohort[cohort.cohort.eq("primary_allrecent")]
    raw, cal = perf.loc["raw"], perf.loc["recalibrated"]
    calp = recal.loc["isotonic_recalibrated"]
    claims = [
        {"claim": f"Primary socioeconomic risk-factor cohort: {int(allr.n.sum()):,} births and {int(allr.deaths.sum())} neonatal deaths.",
         "source": "results/cohort_definition.csv", "fields": "primary_allrecent n,deaths", "displays": "Abstract; Results; Figure 1; Table 1"},
        {"claim": "The same all-recent complete-follow-up cohort is used for the population mortality trend and primary decomposition.",
         "source": "results/cohort_definition.csv; results/table3_nmr_trend.csv; results/table5_decomposition.csv", "fields": "primary_allrecent; 2011/2022 NMR; decomposition", "displays": "Results; Figures 4-5; Table 3"},
        {"claim": "Neonatal death is completed days 0-27 (b6 100-127); days 28-29 are non-neonatal and invalid ages are excluded.",
         "source": "src/harmonize.py; tests/test_harmonize.py", "fields": "derive_neonatal_outcome and boundary tests", "displays": "Methods; Figure 1"},
        {"claim": f"{lock['model']} was selected using 2011-2017/18 forward validation only (mean AP {lock['mean_pr_auc']:.3f}; mean Brier {lock['mean_brier']:.4f}).",
         "source": "results/pipeline_lock.json; results/forward_validation.csv", "fields": "model,mean_pr_auc,mean_brier", "displays": "Abstract; Results; Methods"},
        {"claim": f"Raw 2022 AUROC {raw.ROC_AUC:.3f} ({raw.ROC_AUC_95CI}), AP {raw.PR_AUC:.3f} ({raw.PR_AUC_95CI}), Brier {raw.Brier:.4f} ({raw.Brier_95CI}).",
         "source": "results/primary_performance_uncertainty.csv", "fields": "prediction=raw", "displays": "Results; Figure 2; Table 2"},
        {"claim": f"Recalibrated 2022 AUROC {cal.ROC_AUC:.3f} ({cal.ROC_AUC_95CI}), AP {cal.PR_AUC:.3f} ({cal.PR_AUC_95CI}), Brier {cal.Brier:.4f} ({cal.Brier_95CI}).",
         "source": "results/primary_performance_uncertainty.csv", "fields": "prediction=recalibrated", "displays": "Abstract; Results; Figure 2; Table 2"},
        {"claim": f"Recalibrated Brier skill over the evaluation-prevalence null is {calp.Brier_skill:.3f}.",
         "source": "results/primary_recalibration.csv", "fields": "prediction=isotonic_recalibrated,Brier_skill", "displays": "Abstract; Results; Table 2"},
        {"claim": f"All-recent mortality proportion changed from {trend.loc[2011].NMR:.1f} ({trend.loc[2011].ci_low:.1f}-{trend.loc[2011].ci_high:.1f}) to {trend.loc[2022].NMR:.1f} ({trend.loc[2022].ci_low:.1f}-{trend.loc[2022].ci_high:.1f}) per 1,000.",
         "source": "results/table3_nmr_trend.csv", "fields": "survey_year 2011 and 2022", "displays": "Abstract; Results; Figure 4"},
        {"claim": f"Primary modeled change {dec.loc['total_change'].value_per_1000:.1f} per 1,000: composition {dec.loc['distribution'].value_per_1000:.1f}, residual coefficient-associated {dec.loc['effect'].value_per_1000:.1f}.",
         "source": "results/table5_decomposition.csv", "fields": "total_change,distribution,effect", "displays": "Abstract; Results; Figure 5; Table 3"},
        {"claim": "The care-variable decomposition is exploratory, module-restricted, and includes explicit missingness indicators.",
         "source": "src/trends.py; results/table5b_decomposition_care_module.csv", "fields": "care_module sensitivity", "displays": "Results; Methods; Figure 5; Table 3; Supplement"},
    ]
    payload = {"generated_from_regenerated_outputs": True, "claims": claims}
    (OUT / "claim_evidence_map.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = ["# Claim-evidence map", "", "| Claim | Machine-readable source | Fields | Manuscript/display |", "|---|---|---|---|"]
    for c in claims:
        rows.append(f"| {c['claim'].replace('|', '/')} | `{c['source']}` | {c['fields']} | {c['displays']} |")
    (OUT / "claim_evidence_map.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(OUT / "claim_evidence_map.json")


if __name__ == "__main__":
    main()
