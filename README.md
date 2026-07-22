# Neonatal mortality prediction from pooled BDHS (2011–2022)

Machine-learning prediction of **neonatal death (first 28 days of life)** in Bangladesh,
pooling four Demographic and Health Survey rounds (2011, 2014, 2017‑18, 2022) and
**temporally validating** on the most recent survey (train on 2011–2017 → test on 2022).

The pipeline: harmonise 4 rounds → derive the outcome → restrict to recent births with the
maternal‑care module → preprocess (impute/encode, VIF) → **Boruta** feature selection →
train 7 classifiers (LR, DT, RF, SVM, KNN, XGBoost, CatBoost) with **SMOTE inside grouped
cross‑validation** → score on held‑out **2022** with ROC/PR‑AUC, Brier, **calibration** and
**decision‑curve analysis** → **SHAP** interpretation incl. temporal stability across rounds.

It also runs a **design‑based trend analysis** (survey‑weighted, PSU cluster bootstrap): the NMR
trend 2011–2022 with a linear‑trend test, risk‑factor prevalence trends, association‑shift
(predictor × year), and a decomposition of the NMR decline into *distribution* (exposure change)
vs *effect* (coefficient change) components. This is the population‑level companion to the
individual ML risk model — *predict who is at risk* + *explain why mortality fell*.

## Public-release scope

This repository contains **analysis code and reproducibility configuration only**. It deliberately
does not contain DHS microdata, derived records, cached intermediates, fitted models, output tables,
figures, or manuscript files. DHS microdata are access-controlled and non-redistributable: each user
must obtain the four Bangladesh Births Recode files directly from The DHS Program before running the
pipeline. The code uses only paths relative to the repository root, so no machine-specific paths are
needed.

---

## 1. Get the data (each user downloads their own — DHS data may not be shared)

The DHS Program licenses the microdata per user. Register at
<https://dhsprogram.com>, request access to the **Bangladesh Standard DHS** surveys, and
download the **Births Recode (BR), Stata (.dta)** file for each round. Place the four zips in
the **repo root** (do not unzip — the code does that):

| Round | File to download | Zip name expected in repo root |
|-------|------------------|--------------------------------|
| BDHS 2011    | Births Recode → Stata (.dta) | `BDBR61DT.zip` |
| BDHS 2014    | Births Recode → Stata (.dta) | `BDBR72DT.zip` |
| BDHS 2017‑18 | Births Recode → Stata (.dta) | `BDBR7RDT.zip` |
| BDHS 2022    | Births Recode → Stata (.dta) | `BD_2022_DHS_*.zip` (the bundled 2022 download) |

If your 2022 file has a different name, edit the `2022` entry in [`config.py`](config.py)
(`ROUNDS`). The 2022 bundle contains `BDBR81DT/BDBR81FL.DTA`; the loader finds it automatically.

## 2. Set up the environment (any OS)

Python 3.11–3.13. From the repo root:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

(Conda users may instead run `conda env create -f environment.yml && conda activate neonatal-bdhs`.)

## 3. Run everything

```bash
python run_all.py
```

Steps cache to `data/interim/`, so re‑runs are fast. Force a clean rebuild with
`python run_all.py --force`. You can also run any single step, e.g. `python -m src.evaluate`.

## 4. Outputs

`results/` (CSV) and `figures/` (PNG, 300 dpi):

| File | Content |
|------|---------|
| `figures/fig0_flow.png` | TRIPOD participant-flow diagram |
| `figures/fig1_workflow.png` | methodology diagram |
| `results/table1_descriptives.csv` | Table 1 — baseline characteristics by outcome |
| `figures/fig2_boruta.png` | Boruta feature selection |
| `results/cv_results.csv` | cross‑validated PR‑AUC + tuned hyper‑parameters |
| `results/metrics_all.csv` | Table 2 — 2022 test metrics (ROC/PR-AUC, Brier, calibration slope/intercept, sens/spec/F1) |
| `figures/fig3_roc_pr.png` | ROC + precision‑recall curves |
| `figures/fig4_calibration_dca.png` | calibration + decision‑curve analysis |
| `figures/fig5_shap_beeswarm.png` | SHAP for the best model |
| `figures/fig6_shap_temporal.png` | temporal stability of the risk signature |
| `figures/fig7_nmr_trend.png` | NMR trend 2011-2022 + 95% CI + SDG-3.2 target |
| `figures/fig8_riskfactor_trends.png` | risk-factor / coverage prevalence trends |
| `figures/fig9_decomposition.png` | NMR change: distribution vs effect components |
| `results/table3_nmr_trend.csv`, `table3b_prevalence.csv` | NMR + prevalence trends |
| `results/table4_association_shift.csv` | predictor x year interactions (+ BH-FDR q) |
| `results/table5_decomposition.csv` | NMR-decline decomposition (all vars) |
| `results/table5b_decomposition_noncare.csv`, `figures/fig9b_decomposition_noncare.png` | decomposition excluding care variables (confounding-by-indication sensitivity) |
| `figures/fig10_balancing_calibration.png` | SMOTE/ADASYN/class-weight/none vs calibration |
| `results/table6_balancing.csv` | balancing-strategy sensitivity |
| `results/table7_bootstrap_ci.csv` | 95% CIs on 2022 test metrics (cluster bootstrap) |
| `results/table8_subgroups.csv` | best-model AUROC by residence/sex/wealth |
| `results/table9_indicator_sensitivity.csv` | performance with vs without missing-data indicators |
| `results/variable_availability.csv`, `results/multicollinearity_vif.csv` | audit tables |

## 5. Method notes / defensible choices

- **Outcome:** `neonatal_death = (b5==0) & (b7==0)` — died in completed month 0.
- **Sample:** live births in the 36 months before each survey **with the maternal‑care module
  observed**, so care variables (ANC, delivery, C‑section, attendant) are measured
  consistently across rounds — a temporal model is not fed round‑dependent missingness.
- **Dropped predictors:** `birth_size` (100% missing in 2017‑18 BR) and `cooking fuel` /
  `mother_bmi` (structurally/largely missing in 2022 BR) — kept only variables comparable
  across all four rounds.
- **Region** (`v024`) and (considered) **cooking fuel** are mapped by value **label** because
  their numeric codes were renumbered between rounds (2022 added Mymensingh → folded into Dhaka).
- **No leakage:** Boruta and tuning never see 2022; SMOTE runs **inside** each CV fold; folds
  are split by **PSU cluster** so a cluster never appears in both train and validation.
- **Rare outcome (~2.5%):** models are tuned and compared on **PR‑AUC**; accuracy is reported
  only for completeness. Probabilities are recalibrated (isotonic) for the decision analysis.
- **Weights:** DHS sample weights (`v005/1e6`, normalised within round) are used for the
  weighted NMR sanity check and a weighted AUROC column; models are trained unweighted (standard
  for prediction), with cluster‑aware CV providing the design‑awareness.

## 6. Repo layout

```
config.py          paths, seed, round + variable definitions
run_all.py         end-to-end entry point
src/load.py        00 unzip + read the 4 Births Recode files
src/harmonize.py   01 cross-round harmonisation + outcome + sample restriction
src/preprocess.py  02 clean, VIF, build leakage-safe design matrices
src/features_boruta.py  03 Boruta selection (train only)
src/modeling.py    04 7 models, SMOTE-in-CV, grouped tuning
src/evaluate.py    05 temporal metrics, ROC/PR, calibration, DCA
src/interpret_shap.py   06 SHAP + temporal stability
src/descriptives.py     Table 1
src/workflow_fig.py     Fig 1
data/  results/  figures/  models/     (all git-ignored)
```

Data files and generated outputs are git‑ignored; only code is version‑controlled.
