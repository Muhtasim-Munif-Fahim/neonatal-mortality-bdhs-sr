"""
TRIPOD-style participant flow diagram (rendered from the actual analytic table).

Counts are read from the pooled raw and harmonised analytic extracts,
extract, so the figure stays consistent with the pipeline.

Run: python -m src.flow_fig
"""
from __future__ import annotations

import pandas as pd
from matplotlib.patches import FancyBboxPatch

import config
from src import viz
from src.harmonize import derive_neonatal_outcome, harmonize


def _counts():
    raw = pd.read_parquet(config.DATA_INTERIM / "raw_pooled.parquet")
    age = raw["v008"] - raw["b3"]
    window = age.ge(config.MIN_FOLLOWUP_MONTHS) & age.lt(config.RECENCY_MONTHS)
    outcome = derive_neonatal_outcome(raw)
    invalid_in_window = int((window & outcome.isna()).sum())
    analytic = harmonize(module_only=False)     # primary all-recent cohort
    care = harmonize(module_only=True)          # most-recent care sensitivity
    tr = analytic[analytic[config.YEAR_COL].isin(config.TRAIN_YEARS)]
    te = analytic[analytic[config.YEAR_COL] == config.TEST_YEAR]
    return {
        "pooled": len(raw),
        "outside_window": int((~window).sum()),
        "invalid_outcome": invalid_in_window,
        "recent": len(analytic),
        "analytic": len(analytic), "analytic_d": int(analytic[config.TARGET].sum()),
        "care": len(care), "care_d": int(care[config.TARGET].sum()),
        "train": len(tr), "train_d": int(tr[config.TARGET].sum()),
        "test": len(te), "test_d": int(te[config.TARGET].sum()),
    }


def run():
    c = _counts()
    pd.DataFrame([
        {"stage": "Pooled Births Recode records", "n": c["pooled"], "deaths": pd.NA},
        {"stage": "Outside complete-follow-up window", "n": c["outside_window"], "deaths": pd.NA},
        {"stage": "Unclassifiable death age within window", "n": c["invalid_outcome"], "deaths": pd.NA},
        {"stage": "Primary all-recent cohort", "n": c["analytic"], "deaths": c["analytic_d"]},
        {"stage": "Development rounds", "n": c["train"], "deaths": c["train_d"]},
        {"stage": "2022 temporal evaluation", "n": c["test"], "deaths": c["test_d"]},
        {"stage": "Care-enriched sensitivity", "n": c["care"], "deaths": c["care_d"]},
    ]).to_csv(config.RESULTS / "figure1_flow.csv", index=False)
    boxes = [
        (f"Pooled Births Recode, 4 BDHS rounds\n{c['pooled']:,} live births", 0.35, 0.90),
        (f"Complete neonatal follow-up:\ncalendar-month age 2-35;\nvalid b6 if deceased\n{c['recent']:,}", 0.35, 0.66),
        (f"Primary prediction cohort: all eligible births\n"
         f"{c['analytic']:,} ({c['analytic_d']} neonatal deaths)", 0.35, 0.40),
    ]
    excl = [
        (f"Excluded: outside age 2-35\ncalendar-month window\n({c['outside_window']:,})", 0.80, 0.82),
        (f"Excluded: unclassifiable\nage at death\n({c['invalid_outcome']:,})", 0.80, 0.70),
        (f"Care-enriched sensitivity:\nmost recent birth; 2022 long form\n"
         f"{c['care']:,} ({c['care_d']} deaths)",
         0.80, 0.53),
    ]
    splits = [
        (f"Training rounds\n2011, 2014, 2017-18\n{c['train']:,} ({c['train_d']} deaths)", 0.17, 0.12),
        (f"Evaluation round 2022\n(excluded from fitting/selection)\n{c['test']:,} ({c['test_d']} deaths)", 0.56, 0.12),
    ]

    fig, ax = viz.plt.subplots(figsize=(8, 9))
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def box(text, x, y, colour, w=0.42, h=0.1):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                     boxstyle="round,pad=0.01,rounding_size=0.02",
                     linewidth=1.2, edgecolor=colour, facecolor=colour + "22"))
        ax.text(x, y, text, ha="center", va="center", fontsize=9)

    for t, x, y in boxes:
        box(t, x, y, viz.PALETTE[0])
    for t, x, y in excl:
        box(t, x, y, viz.PALETTE[7], w=0.34, h=0.09)
    for t, x, y in splits:
        box(t, x, y, viz.PALETTE[2], w=0.3, h=0.12)

    # vertical arrows down the main column
    for y0, y1 in [(0.85, 0.71), (0.61, 0.45)]:
        ax.annotate("", xy=(0.35, y1), xytext=(0.35, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.3))
    # exclusion connectors
    for (_, _, y) in excl:
        ax.annotate("", xy=(0.63, y), xytext=(0.35, y),
                    arrowprops=dict(arrowstyle="-|>", color="#999", lw=1.0))
    # split arrows
    for x1 in (0.17, 0.56):
        ax.annotate("", xy=(x1, 0.19), xytext=(0.35, 0.35),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.3))

    # in-image title removed for journal style; caption carries it
    viz.save(fig, "fig0_flow.png")


if __name__ == "__main__":
    run()
