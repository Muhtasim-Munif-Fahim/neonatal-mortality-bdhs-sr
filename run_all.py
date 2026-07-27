"""
Single entry point -- runs the whole pipeline end to end and writes every table
and figure into results/ and figures/.

Usage (from the repo root, with the venv active):
    python run_all.py

Prerequisite: the four BDHS Births Recode zips sit in the repo root (see README).
Each step caches to data/interim/, so re-runs are fast; pass --force to rebuild.
"""
from __future__ import annotations

import argparse
import sys
import time


def main(force: bool = False) -> None:
    from src import (associations, checks, descriptives, evaluate, export_aggregate,
                     features_boruta, figures_main, flow_fig, harmonize, interpret_shap,
                     load, modeling, preprocess, robustness, trends, workflow_fig)

    steps = [
        ("00 load",        lambda: load.load_all(force=force)),
        ("01 harmonize",   lambda: (
            harmonize.harmonize(force=force, module_only=False),
            harmonize.harmonize(force=force, module_only=True),
        )),
        ("02 preprocess",  lambda: preprocess.build_design(force=force)),
        ("03 boruta",      lambda: features_boruta.run(force=force)),
        ("04 modeling",    lambda: modeling.train_all(force=force)),
        ("05 evaluate",    lambda: evaluate.run()),
        ("06 shap",        lambda: interpret_shap.run()),
        ("07 trends",      lambda: trends.run()),
        ("08 robustness",  lambda: robustness.run()),
        ("09 associations", lambda: associations.run(force=force)),
        ("   table1",      lambda: descriptives.build()),
        ("   fig1",        lambda: workflow_fig.run()),
        ("   flow",        lambda: flow_fig.run()),
        ("   aggregate",   lambda: export_aggregate.run()),
        ("   figures_main", lambda: figures_main.run()),
        ("   checks",      lambda: checks.run()),
    ]
    for name, fn in steps:
        print(f"\n{'='*66}\n>>> {name}\n{'='*66}")
        t0 = time.time()
        fn()
        print(f"<<< {name} done in {time.time()-t0:.0f}s")

    print("\nAll steps complete. See results/ and figures/.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="rebuild every cached artifact from scratch")
    args = ap.parse_args()
    try:
        main(force=args.force)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
