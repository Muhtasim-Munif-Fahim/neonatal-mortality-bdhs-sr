"""
Step 01 -- harmonise the pooled raw births into a common analytic table.

What it does:
  * derives the outcome exactly as death on completed days 0--27 (DHS b6
    codes 100--127), excluding deaths with an invalid/missing age-at-death code,
  * restricts to live births with conservative complete neonatal follow-up in
    calendar-month differences 2--35
    before each survey,
  * recodes every predictor to a scheme that is CONSISTENT across the four DHS
    phases (two variables -- cooking fuel v161 and region v024 -- are mapped by
    their Stata value LABEL because their numeric codes were renumbered between
    rounds; everything else is stable enough to map by code),
  * builds a unique cluster id and a within-round-normalised survey weight,
  * writes data/interim/analytic.parquet.

Run: python -m src.harmonize
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyreadstat

import config
from src.load import find_dta, load_all


# --------------------------------------------------------------------------- #
# Per-round value-label maps (needed for label-based harmonisation)
# --------------------------------------------------------------------------- #
def _value_labels(year: int, var: str) -> dict:
    """{code: label_string} for one variable in one round."""
    dta = find_dta(year)
    _, meta = pyreadstat.read_dta(str(dta), metadataonly=True)
    return meta.variable_value_labels.get(var, {})


# --------------------------------------------------------------------------- #
# Recode helpers
# --------------------------------------------------------------------------- #
# Improved drinking water (JMP), by v113 code (codes stable across these rounds).
_WATER_IMPROVED = {10, 11, 12, 13, 14, 20, 21, 31, 41, 51, 61, 62, 71, 72}
# Improved sanitation (JMP, ignoring the shared-facility downgrade), by v116 code.
_SANITATION_IMPROVED = {10, 11, 12, 13, 21, 22, 41}

_DIVISION_CANON = {
    "barisal": "barishal", "barishal": "barishal",
    "chittagong": "chattogram", "chattogram": "chattogram",
    "dhaka": "dhaka",
    "mymensingh": "dhaka",       # carved out of Dhaka in 2015; fold back for comparability
    "khulna": "khulna",
    "rajshahi": "rajshahi",
    "rangpur": "rangpur",
    "sylhet": "sylhet",
}


def _map_by_label(series: pd.Series, code2label: dict, classifier) -> pd.Series:
    """Map numeric codes -> label -> harmonised value via `classifier(label)`."""
    out = pd.Series(index=series.index, dtype="object")
    for code, label in code2label.items():
        out[series == code] = classifier(str(label).strip().lower())
    return out


def _division_classifier(label: str) -> str:
    for key, canon in _DIVISION_CANON.items():
        if key in label:
            return canon
    return np.nan


def derive_neonatal_outcome(raw: pd.DataFrame) -> pd.Series:
    """Return a nullable neonatal-death indicator from DHS ``b5``/``b6``.

    DHS ``b6`` encodes age at death as 100 + days, 200 + months, or
    300 + years.  Neonatal deaths are therefore exactly codes 100--127.
    Deaths on days 28--29 (128--129) and later valid ages are non-neonatal.
    A dead child with a missing or non-DHS age code cannot be classified and
    is returned as ``NA`` so the record can be excluded from the outcome
    cohort.  Living children do not require an age-at-death code.
    """
    status = pd.to_numeric(raw["b5"], errors="coerce")
    age_code = pd.to_numeric(raw["b6"], errors="coerce")
    outcome = pd.Series(pd.NA, index=raw.index, dtype="Int64")
    outcome.loc[status == 1] = 0
    dead = status == 0
    # DHS reserves last-two-digit values above 90 for special/unknown
    # responses (for example 199 = days, number missing).  Only exact numeric
    # ages are classifiable here.
    valid_age = (age_code.between(100, 190, inclusive="both")
                 | age_code.between(201, 290, inclusive="both")
                 | age_code.between(301, 390, inclusive="both"))
    outcome.loc[dead & valid_age] = age_code.loc[dead & valid_age].between(
        100, 127, inclusive="both"
    ).astype(int)
    return outcome


def complete_followup_mask(age_mo: pd.Series,
                           min_months: int = config.MIN_FOLLOWUP_MONTHS) -> pd.Series:
    """Eligibility for a fully observed 28-day outcome within the 3-year window."""
    age = pd.to_numeric(age_mo, errors="coerce")
    return age.ge(min_months) & age.lt(config.RECENCY_MONTHS)


def cohort_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Unweighted per-round cohort arithmetic used by tests and verification."""
    grouped = df.groupby(config.YEAR_COL, sort=True)[config.TARGET]
    return pd.DataFrame({"n": grouped.size(), "deaths": grouped.sum().astype(int)})


def recode_anc_4plus(anc: pd.Series) -> pd.Series:
    """ANC 4+ indicator that preserves missing exposure status."""
    values = pd.to_numeric(anc, errors="coerce")
    return pd.Series(np.where(values.isna(), pd.NA, values >= 4),
                     index=anc.index, dtype="Int64")


def derive_skilled_attendant(raw: pd.DataFrame) -> pd.Series:
    """Any survey-labelled skilled delivery cadre, preserving missingness."""
    cols = [c for c in ["m3a", "m3b", "m3c", "m3d", "m3e"] if c in raw]
    assistance = raw[cols].apply(pd.to_numeric, errors="coerce")
    any_recorded = assistance.notna().any(axis=1)
    skilled = assistance.fillna(0).eq(1).any(axis=1)
    return pd.Series(np.where(any_recorded, skilled, pd.NA),
                     index=raw.index, dtype="Int64")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def harmonize(force: bool = False, module_only: bool = True,
              min_followup_months: int = config.MIN_FOLLOWUP_MONTHS) -> pd.DataFrame:
    """Harmonised analytic table.

    module_only=True  (default, used by the ML pipeline): recent births WITH the
      maternal-care module, so care variables are consistent across rounds.
    module_only=False (used by trends.py for a representative NMR denominator):
      all recent births regardless of the module.
    """
    cohort = "analytic" if module_only else "analytic_allrecent"
    suffix = "" if min_followup_months == config.MIN_FOLLOWUP_MONTHS \
        else f"_minfollowup{min_followup_months}"
    cache = config.DATA_INTERIM / f"{cohort}{suffix}.parquet"
    if cache.exists() and not force:
        cached = pd.read_parquet(cache)
        required = set(config.META_COLS) | {config.TARGET, "age_mo"}
        if required.issubset(cached.columns):
            return cached
        missing = sorted(required - set(cached.columns))
        print(f"  invalidating {cache.name}; missing columns: {missing}")

    raw = load_all()
    out = pd.DataFrame(index=raw.index)

    # ---- bookkeeping ------------------------------------------------------ #
    year = raw["survey_year"]
    out[config.YEAR_COL] = year
    # unique cluster id across rounds: e.g. 2011_045
    out[config.CLUSTER_COL] = (
        year.astype(str) + "_" + raw["v001"].astype("Int64").astype(str)
    )
    out[config.STRATUM_COL] = (
        year.astype(str) + "_" + raw["v022"].astype("Int64").astype(str)
    )
    out["age_mo"] = raw["v008"] - raw["b3"]           # months since birth at interview
    if "b19" in raw:
        # Day-sensitive completed age, available only in newer DHS rounds.
        # Retained for follow-up diagnostics, never used as a predictor.
        out["age_completed_mo"] = pd.to_numeric(raw["b19"], errors="coerce")
    if "sqtype" in raw:
        out["questionnaire_type"] = pd.to_numeric(raw["sqtype"], errors="coerce").map(
            {1: "long", 2: "short"}
        )

    # ---- outcome ---------------------------------------------------------- #
    out[config.TARGET] = derive_neonatal_outcome(raw)

    # ---- child predictors ------------------------------------------------- #
    out["sex"] = raw["b4"].map({1: "male", 2: "female"})
    out["multiple_birth"] = (raw["b0"].fillna(0) > 0).astype(int)
    out["birth_order"] = pd.to_numeric(raw["bord"], errors="coerce")
    bi = pd.to_numeric(raw["b11"], errors="coerce")
    out["firstborn"] = raw["b11"].isna().astype(int)
    out["birth_interval"] = bi                          # NaN for firstborns (flagged above)
    # NOTE: birth_size (m18) is DROPPED -- 100% missing in the 2017-18 BR file, so it
    # cannot be used consistently across the training rounds (>90%-missing rule).

    # ---- maternal --------------------------------------------------------- #
    age_birth = (raw["b3"] - raw["v011"]) / 12.0        # age at THIS birth, years
    out["mother_age"] = age_birth.round(1)
    edu = pd.to_numeric(raw["v106"], errors="coerce")
    out["mother_edu"] = edu.map({0: "none", 1: "primary", 2: "secondary", 3: "higher"})
    # NOTE: mother_bmi (v445) is DROPPED -- anthropometry is not part of the birth
    # module and is ~50% missing in 2022 vs ~2% in earlier rounds (>90%-... no, but a
    # large round-dependent shift that would leak survey identity into a temporal model).
    out["children_ever_born"] = pd.to_numeric(raw["v201"], errors="coerce")
    out["mother_working"] = pd.to_numeric(raw["v714"], errors="coerce")

    # ---- pregnancy / delivery services ------------------------------------ #
    anc = pd.to_numeric(raw["m14"], errors="coerce")
    anc = anc.where(anc <= 20)                          # 98='don't know' -> NaN
    out["anc_visits"] = anc
    out["anc_4plus"] = recode_anc_4plus(anc)
    m15 = pd.to_numeric(raw["m15"], errors="coerce")
    out["delivery_place"] = np.select(
        [m15.isin([10, 11]), (m15 >= 20) & (m15 < 96)],
        ["home", "facility"], default=None,
    )
    out["csection"] = pd.to_numeric(raw["m17"], errors="coerce").map({0: 0, 1: 1})
    out["skilled_attendant"] = derive_skilled_attendant(raw)

    # ---- household / SES -------------------------------------------------- #
    out["wealth"] = pd.to_numeric(raw["v190"], errors="coerce").map(
        {1: "poorest", 2: "poorer", 3: "middle", 4: "richer", 5: "richest"}
    )
    rel = pd.to_numeric(raw["v130"], errors="coerce")
    out["religion"] = np.select(
        [rel == 1, rel == 2], ["islam", "hinduism"], default="other"
    )
    water = pd.to_numeric(raw["v113"], errors="coerce")
    out["water_improved"] = np.where(
        water.isna(), np.nan, water.isin(_WATER_IMPROVED).astype(float)
    )
    toilet = pd.to_numeric(raw["v116"], errors="coerce")
    out["sanitation_improved"] = np.where(
        toilet.isna(), np.nan, toilet.isin(_SANITATION_IMPROVED).astype(float)
    )
    # media exposure: any of newspaper/radio/tv at least "less than once a week"
    media = raw[["v157", "v158", "v159"]].apply(pd.to_numeric, errors="coerce")
    out["media_exposure"] = (media.fillna(0).max(axis=1) >= 1).astype(int)
    out["residence"] = pd.to_numeric(raw["v025"], errors="coerce").map({1: "urban", 2: "rural"})

    # ---- label-based (codes renumbered across rounds) --------------------- #
    # region v024: 2022 added Mymensingh (code 5) shifting later codes, so we map by
    # LABEL to a consistent 7-division scheme (Mymensingh folded into Dhaka).
    # (Cooking fuel v161 was considered but is 100% missing in the 2022 BR -> dropped.)
    division = pd.Series(np.nan, index=raw.index, dtype="object")
    for yr in config.ROUNDS:
        mask = year == yr
        division.loc[mask] = _map_by_label(
            raw.loc[mask, "v024"], _value_labels(yr, "v024"), _division_classifier
        )
    out["division"] = division

    # ---- survey weight (normalise to mean 1 WITHIN each round) ------------ #
    wt = pd.to_numeric(raw["v005"], errors="coerce") / 1_000_000.0
    out[config.WEIGHT_COL] = wt / wt.groupby(year).transform("mean")

    # Module marker: was the maternal-care module administered for this birth?
    # In 2022 this is exactly the long-questionnaire subsample, not merely a
    # most-recent-birth restriction. The standard women's weight (v005) is the
    # only individual weight supplied and is normalized within survey round.
    out["_module"] = raw["m15"].notna().to_numpy()

    # ---- sample restriction ---------------------------------------------- #
    # Births in the last RECENCY_MONTHS AND with the maternal-care module observed
    # In 2022 this selects the random long-questionnaire household subsample.
    # The module restriction makes the
    # care variables (ANC, delivery, C-section, attendant) consistently measured
    # across all four rounds, so a temporal model is not fed round-dependent
    # missingness. See docs / plan for the rationale.
    before = len(out)
    keep = (complete_followup_mask(out["age_mo"], min_followup_months)
            & out[config.TARGET].notna())
    if module_only:
        keep = keep & out["_module"]
    out = out[keep].drop(columns="_module").copy()
    out[config.TARGET] = out[config.TARGET].astype(int)
    print(f"Complete follow-up ({min_followup_months}-{config.RECENCY_MONTHS - 1} mo)"
          f"{' + care-module' if module_only else ''} filter: "
          f"{before:,} -> {len(out):,} births")

    _sanity_checks(out)
    out.reset_index(drop=True, inplace=True)
    out.to_parquet(cache, index=False)
    print(f"Analytic table -> {cache}  ({out.shape[0]:,} rows x {out.shape[1]} cols)")
    return out


def _sanity_checks(df: pd.DataFrame) -> None:
    assert df[config.TARGET].isin([0, 1]).all(), "outcome not binary"
    assert df[config.CLUSTER_COL].notna().all(), "missing cluster id"
    assert df[config.STRATUM_COL].notna().all(), "missing stratum id"
    # harmonised categoricals should have no unexpected leftovers
    assert set(df["division"].dropna().unique()) <= set(_DIVISION_CANON.values())
    print("Sanity checks passed.")


if __name__ == "__main__":
    df = harmonize(force=True)
    print("\nNeonatal deaths per round (unweighted count / n / rate per 1000):")
    g = df.groupby(config.YEAR_COL)[config.TARGET]
    tab = pd.DataFrame({"n": g.size(), "deaths": g.sum()})
    tab["nmr_per_1000"] = (tab["deaths"] / tab["n"] * 1000).round(1)
    print(tab.to_string())

    # weighted NMR sanity (should track published ~17-30/1000)
    w = df[config.WEIGHT_COL]
    for yr, sub in df.groupby(config.YEAR_COL):
        ws = sub[config.WEIGHT_COL]
        nmr_w = np.average(sub[config.TARGET], weights=ws) * 1000
        print(f"  {yr} weighted NMR = {nmr_w:.1f}/1000")

    print("\nMissingness (%) by harmonised predictor:")
    pred = [c for c in df.columns if c not in config.META_COLS]
    miss = (df[pred].isna().mean() * 100).round(1).sort_values(ascending=False)
    print(miss.to_string())
