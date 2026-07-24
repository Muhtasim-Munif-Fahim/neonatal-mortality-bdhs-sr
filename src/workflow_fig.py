"""
Fig 1 -- methodology workflow diagram (rendered programmatically, reproducible).

Run: python -m src.workflow_fig
"""
from __future__ import annotations

from matplotlib.patches import FancyBboxPatch

from src import viz

BOXES = [
    ("4 BDHS Births Recode files\n(2011, 2014, 2017-18, 2022)", 0.5, 0.94, viz.PALETTE[0]),
    ("Outcome: death on days 0-27\n(b6=100-127); invalid ages excluded", 0.5, 0.82, viz.PALETTE[2]),
    ("Complete follow-up: age 2-35 mo\nall eligible births; birth-time predictors", 0.5, 0.70, viz.PALETTE[2]),
    ("Forward selection: 2011->2014 and\n2011+2014->2017-18", 0.5, 0.58, viz.PALETTE[2]),
    ("Fold-local imputation/encoding\n+ PSU-grouped inner tuning", 0.5, 0.46, viz.PALETTE[1]),
    ("Rank 7 pipelines by mean AP;\nnear tie -> Brier -> simplicity\nselection uses 2011-2017/18 only",
     0.5, 0.32, viz.PALETTE[5]),
    ("Retrospective temporal evaluation on 2022:\nAP/AUROC, Brier, calibration", 0.5, 0.17, viz.PALETTE[4]),
    ("Exploratory SHAP and subgroup\nanalyses; DCA in supplement", 0.5, 0.05, viz.PALETTE[3]),
]


def run():
    fig, ax = viz.plt.subplots(figsize=(6.5, 9))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for text, x, y, colour in BOXES:
        box = FancyBboxPatch((x - 0.34, y - 0.04), 0.68, 0.075,
                             boxstyle="round,pad=0.01,rounding_size=0.02",
                             linewidth=1.2, edgecolor=colour, facecolor=colour + "22")
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=9, color="black")
    for i in range(len(BOXES) - 1):
        y0 = BOXES[i][2] - 0.04
        y1 = BOXES[i + 1][2] + 0.035
        ax.annotate("", xy=(0.5, y1), xytext=(0.5, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.3))
    # in-image title removed for journal style; caption carries it
    viz.save(fig, "fig1_workflow.png")


if __name__ == "__main__":
    run()
