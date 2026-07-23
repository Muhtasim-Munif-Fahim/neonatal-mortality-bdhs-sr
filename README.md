# Neonatal mortality prediction from pooled BDHS (2011–2022)

This repository contains a reproducible, code-only analysis of neonatal death (completed days 0–27) in four Bangladesh Demographic and Health Survey rounds: 2011, 2014, 2017–18 and 2022.

The prediction analysis harmonises a maternal-care-module cohort, selects one of seven model families using historical forward validation (2011→2014 and 2011+2014→2017–18), and performs a retrospective temporal external evaluation in 2022. The 2022 data are excluded from fitting and selection in the corrected analysis. Evaluation reports AUROC, average precision, Brier score, prevalence-only null benchmarks, calibration and design-aware uncertainty. Decision-curve and subgroup analyses are exploratory.

A separate population analysis estimates survey-weighted recent-birth mortality proportions and service-coverage trends. Its primary symmetric nonlinear decomposition uses the identical all-recent cohort and six common non-care covariates. The residual coefficient-associated component is not causal: it also absorbs confounding, unmeasured composition, non-collapsibility and model misspecification.

## Public-release boundary

This repository contains analysis code and reproducibility configuration only. It deliberately excludes DHS microdata, derived record-level data, predictions, cached intermediates, fitted models, output tables, figures and manuscript files. DHS microdata are controlled access and cannot be redistributed.

Each user must request the Bangladesh Standard DHS surveys from [The DHS Program](https://dhsprogram.com/data/available-datasets.cfm), download the Births Recode Stata archives, and place them in the repository root:

| Round | Expected archive |
|---|---|
| BDHS 2011 | `BDBR61DT.zip` |
| BDHS 2014 | `BDBR72DT.zip` |
| BDHS 2017–18 | `BDBR7RDT.zip` |
| BDHS 2022 | `BD_2022_DHS_*.zip` containing `BDBR81FL.DTA` |

The loader auto-detects the Stata file inside each archive. All paths are relative to the repository root.

## Environment

Python 3.11–3.13 is supported.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip check
```

Conda users may instead run:

```bash
conda env create -f environment.yml
conda activate neonatal-bdhs
```

## Run and verify

```bash
python run_all.py --force
python -m src.checks
python -m unittest discover -s tests -v
```

The forced run rebuilds both the prediction-module and all-recent cohorts, models, uncertainty analyses, aggregate tables and figures. Subsequent runs may reuse schema-validated caches.

## Core analysis definitions

- Outcome: `b5==0` with exact DHS age-at-death codes `b6=100–127` (completed days 0–27).
- Boundary handling: deaths on days 28–29 are non-neonatal; special, missing or invalid age-at-death codes are excluded.
- Complete follow-up: primary eligibility requires calendar-month age 2–35. A one-month difference can be fewer than 28 elapsed days, so age 1–35 is sensitivity only.
- Prediction cohort: births with the harmonised maternal-care module. In 2022 this is the random long-questionnaire household subsample, not a most-recent-birth approximation.
- Population cohort: all recent births with complete follow-up, regardless of care-module administration.
- Model selection: seven raw-data pipelines; fold-local imputation/encoding and resampling; PSU-grouped inner validation; mean average precision ranking with a declared computational near-tie rule (lower Brier, then simplicity).
- Calibration: fully nested PSU-grouped out-of-fold development predictions; 2022 remains native prevalence.
- Survey design: round-normalised DHS weights; PSU resampling within round-specific strata for primary uncertainty.
- Intended estimand: primary unweighted fitting targets individual prediction in the observed cohort. A native-class survey-weighted training/evaluation sensitivity is also generated.
- Interpretation: SHAP is descriptive and distribution-dependent. Decomposition components are non-causal.

## Outputs

Generated outputs are ignored by Git.

| Output | Content |
|---|---|
| `results/cohort_definition.csv` | primary and follow-up-sensitivity cohort arithmetic |
| `results/forward_validation.csv`, `pipeline_lock.json` | historical pipeline-selection evidence |
| `results/primary_recalibration.csv` | raw/recalibrated metrics, null Brier and weighted metrics |
| `results/primary_performance_uncertainty.csv` | 1,000-replicate stratified-PSU intervals |
| `results/table1_cohort_characteristics.csv` | development-versus-2022 characteristics |
| `results/table3_nmr_trend.csv`, `table3b_prevalence.csv` | population trends |
| `results/table5_decomposition.csv` | primary all-recent/non-care decomposition |
| `results/table5b_decomposition_care_module.csv` | exploratory care-module sensitivity |
| `results/table8_subgroups.csv` | exploratory subgroup metrics and uncertainty |
| `results/table10_survey_weighted_native.csv` | native-class survey-weighted sensitivity |
| `results/aggregate_release/` | non-disclosive aggregate figure/source tables |
| `figures/*.png`, `figures/*.pdf` | raster previews and vector source figures |

## Repository layout

```text
config.py                 paths, variables, seed and cohort constants
run_all.py                end-to-end entry point
src/load.py               controlled-access BR loading
src/harmonize.py          exact outcome, cohorts and cross-round recoding
src/preprocess.py         cleaning and fold-local preprocessor definition
src/features_boruta.py    supplementary feature-selection sensitivity
src/modeling.py           forward selection, nested OOF calibration and refit
src/evaluate.py           temporal metrics, null benchmarks and plots
src/interpret_shap.py     descriptive SHAP outputs
src/trends.py             weighted trends and non-causal decompositions
src/robustness.py         uncertainty and sensitivity analyses
src/descriptives.py       development/evaluation Table 1
src/checks.py             integrity and cohort-closure checks
src/export_aggregate.py   non-disclosive aggregate release tables
tests/                    boundary and cohort-arithmetic unit tests
```

Code is licensed under the MIT License. Documentation is licensed under CC BY 4.0. DHS data remain subject to The DHS Program’s access terms.
