"""
Step 00 -- load the four BDHS Births Recode (.DTA) files.

Responsibilities:
  * unzip each round's archive into data/raw/<year>/ (idempotent),
  * read the Stata file with pyreadstat (keeps value labels + metadata),
  * keep only the columns we need, tag `survey_year`,
  * cache the pooled raw subset to data/interim/raw_pooled.parquet,
  * (when run as a script) write a variable-availability report that drives
    harmonisation in src/harmonize.py.

Run: python -m src.load
"""
from __future__ import annotations

import zipfile

import pandas as pd
import pyreadstat

import config


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract_round(year: int) -> None:
    """Unzip the round's archive into data/raw/<year>/ if not already done."""
    spec = config.ROUNDS[year]
    zip_path = config.RAW_ZIP_DIR / spec["zip"]
    out_dir = config.DATA_RAW / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    if find_dta(year) is not None:
        return  # already extracted
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Missing DHS archive for {year}: {zip_path}\n"
            f"Download the Births Recode (Stata) '{spec['zip']}' from dhsprogram.com."
        )
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def find_dta(year: int):
    """Return the path to the round's .DTA (searched recursively), or None."""
    target = config.ROUNDS[year]["dta"].lower()
    out_dir = config.DATA_RAW / str(year)
    if not out_dir.exists():
        return None
    for p in out_dir.rglob("*"):
        if p.is_file() and p.name.lower() == target:
            return p
    # fall back: any .DTA in the folder
    hits = [p for p in out_dir.rglob("*") if p.suffix.lower() == ".dta"]
    return hits[0] if hits else None


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _wanted_columns() -> list[str]:
    seen: dict[str, None] = {}
    for col in (
        config.DESIGN_VARS
        + config.OUTCOME_SOURCE_VARS
        + config.PREDICTOR_SOURCE_VARS
    ):
        seen.setdefault(col, None)
    return list(seen)


def read_round(year: int, metadataonly: bool = False):
    """Read one round. Returns (DataFrame, pyreadstat metadata)."""
    extract_round(year)
    dta = find_dta(year)
    if dta is None:
        raise FileNotFoundError(f"No .DTA found for {year} under {config.DATA_RAW/str(year)}")

    # First read metadata only to learn which of our wanted columns exist.
    _, meta = pyreadstat.read_dta(str(dta), metadataonly=True)
    available = set(meta.column_names)
    usecols = [c for c in _wanted_columns() if c in available]

    if metadataonly:
        return None, meta

    df, meta = pyreadstat.read_dta(str(dta), usecols=usecols)
    df["survey_year"] = year
    return df, meta


def load_all(force: bool = False) -> pd.DataFrame:
    """Load + pool all four rounds; cache to data/interim/raw_pooled.parquet."""
    cache = config.DATA_INTERIM / "raw_pooled.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    frames = []
    for year in config.ROUNDS:
        df, _ = read_round(year)
        frames.append(df)
        print(f"  {year}: {len(df):,} birth records, {df.shape[1]} cols")

    pooled = pd.concat(frames, ignore_index=True)
    pooled.to_parquet(cache, index=False)
    print(f"Pooled: {len(pooled):,} rows -> {cache}")
    return pooled


# --------------------------------------------------------------------------- #
# Variable-availability report (harmonisation aid)
# --------------------------------------------------------------------------- #
def availability_report() -> pd.DataFrame:
    """For each wanted column, which rounds contain it + its Stata label."""
    wanted = _wanted_columns()
    rows = []
    labels_by_round = {}
    for year in config.ROUNDS:
        _, meta = read_round(year, metadataonly=True)
        present = set(meta.column_names)
        col_labels = dict(zip(meta.column_names, meta.column_labels))
        labels_by_round[year] = (present, col_labels)

    for col in wanted:
        row = {"variable": col}
        label = ""
        for year in config.ROUNDS:
            present, col_labels = labels_by_round[year]
            row[str(year)] = "yes" if col in present else "-"
            if not label and col in col_labels and col_labels[col]:
                label = col_labels[col]
        row["label"] = label
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Extracting + loading four BDHS Births Recode files ...")
    pooled = load_all(force=True)

    print("\nBirths per round:")
    print(pooled["survey_year"].value_counts().sort_index().to_string())

    rep = availability_report()
    out = config.RESULTS / "variable_availability.csv"
    rep.to_csv(out, index=False)
    print(f"\nVariable-availability report -> {out}")
    print(rep.to_string(index=False))
