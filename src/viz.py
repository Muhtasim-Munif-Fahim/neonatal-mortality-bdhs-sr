"""Shared plotting setup: headless backend + a consistent house style."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")            # no display needed; write files only
import matplotlib.pyplot as plt

# A restrained, publication-leaning palette (colour-blind safe).
PALETTE = ["#2c3e50", "#e67e22", "#2980b9", "#27ae60", "#c0392b",
           "#8e44ad", "#16a085", "#7f8c8d"]

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})


def save(fig, name: str):
    from config import FIGURES
    path = FIGURES / name
    fig.savefig(path)
    # Preserve a vector twin for journal assembly and source-data review.
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  figure -> {path} + {pdf_path.name}")
    return path
