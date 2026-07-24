"""
Central configuration: paths, seed, round definitions, variable groups.

Every path is derived from this file's location, so the project runs unchanged on
any computer as long as the repo folder is kept intact. NO absolute paths anywhere
else in the codebase -- always import from here.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (all relative to the repo root = the folder containing this file)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent

DATA_RAW = ROOT / "data" / "raw"          # extracted DHS .DTA files (git-ignored)
DATA_INTERIM = ROOT / "data" / "interim"  # cached parquet between steps (git-ignored)
RESULTS = ROOT / "results"                # tables / csv outputs (git-ignored)
FIGURES = ROOT / "figures"                # plots (git-ignored)
MODELS = ROOT / "models"                  # fitted estimators (git-ignored)

# The raw DHS zips live in the repo root (already downloaded by the authors).
RAW_ZIP_DIR = ROOT

for _d in (DATA_RAW, DATA_INTERIM, RESULTS, FIGURES, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# BDHS survey rounds -> Births Recode (BR) files.
# `zip` is the archive in the repo root; `dta` is the inner Stata file basename
# (we glob for it after extraction, so nested folders inside the zip are fine).
# --------------------------------------------------------------------------- #
ROUNDS: dict[int, dict[str, str]] = {
    2011: {"zip": "BDBR61DT.zip", "dta": "BDBR61FL.DTA"},
    2014: {"zip": "BDBR72DT.zip", "dta": "BDBR72FL.DTA"},
    2017: {"zip": "BDBR7RDT.zip", "dta": "BDBR7RFL.DTA"},   # 2017-18
    2022: {"zip": "BD_2022_DHS_07192026_1239_236549.zip", "dta": "BDBR81FL.DTA"},
}

# Temporal evaluation split: fit/select on past rounds; evaluate in 2022.
TRAIN_YEARS = [2011, 2014, 2017]
TEST_YEAR = 2022

# Analytic sample: births in the N months before the survey (maternal-care module
# is only collected for recent births). 36 = last 3 years (DHS standard).
RECENCY_MONTHS = 36
# CMC month differences do not contain interview/birth day.  A difference of
# one calendar month can therefore represent fewer than 28 completed days;
# two months is the conservative cross-round threshold for complete neonatal
# outcome ascertainment.  The one-month rule is retained only as a sensitivity.
MIN_FOLLOWUP_MONTHS = 2

# Canonical column names produced by harmonize.py.
TARGET = "neonatal_death"
YEAR_COL = "survey_year"
CLUSTER_COL = "cluster"       # from v001, made unique per round
STRATUM_COL = "stratum"       # from v022, made unique per round
WEIGHT_COL = "weight"         # normalized survey weight (mean 1 within round)
# Non-predictor bookkeeping columns carried alongside the features.
META_COLS = [YEAR_COL, CLUSTER_COL, STRATUM_COL, WEIGHT_COL, "age_mo", "age_completed_mo",
             "birth_index", "questionnaire_type", TARGET]

# --------------------------------------------------------------------------- #
# Variable groups (DHS recode names). Harmonisation maps these to a common
# scheme across rounds in src/harmonize.py.
# --------------------------------------------------------------------------- #
# Survey-design columns (needed for weights / clustering, NOT used as predictors).
DESIGN_VARS = ["v001", "v005", "v021", "v022", "v023", "v024", "v025", "sqtype"]

# Columns required to build the outcome + the recency restriction.
OUTCOME_SOURCE_VARS = ["bidx", "b3", "b5", "b6", "b7", "b19", "v008", "v011"]

# Candidate predictor source columns (subset kept if present in a given round).
PREDICTOR_SOURCE_VARS = [
    # child
    "b0",      # multiple birth (twin)
    "b4",      # sex of child
    "b11",     # preceding birth interval (months)
    "bord",    # birth order number
    "m18",     # size of child at birth (mother's report)
    "m19",     # birth weight (grams; often missing)
    # maternal
    "v012",    # respondent current age
    "v106",    # highest education level
    "v149",    # educational attainment (detailed)
    "v445",    # BMI (x100)
    "v714",    # currently working
    "v201",    # total children ever born
    # pregnancy / health service (recent births only)
    "m14",     # number of ANC visits
    "m15",     # place of delivery
    "m17",     # caesarean section
    "m3a",     # assistance at delivery: doctor
    "m3b",     # assistance at delivery: nurse/midwife (with m3a -> skilled attendant)
    "m3c",     # assistance: family welfare visitor
    "m3d",     # assistance: community skilled birth attendant
    "m3e",     # assistance: medical assistant / SACMO
    # household / SES
    "v190",    # wealth index (quintile)
    "v130",    # religion
    "v113",    # source of drinking water
    "v116",    # type of toilet facility
    "v161",    # cooking fuel
    "v157",    # frequency reading newspaper (media)
    "v158",    # frequency listening radio
    "v159",    # frequency watching tv
    "v024",    # region / division
    "v025",    # type of place of residence (urban/rural)
]
