"""
Main display items (Figures 1-5) for the Scientific Reports submission.

This module is the live, pipeline-integrated successor to the one-off
``manuscript/scientific_reports/figure_studio/render_all.py`` prototype, whose
design system it reuses in full (Arial, 7 pt base type, Okabe-Ito palette, bold
lowercase panel labels, an exact 183 mm double-column canvas, and a save routine
that deliberately avoids ``bbox_inches="tight"`` because a tight bounding box
silently changes the physical canvas width and breaks the journal specification).
That prototype was written against the superseded 6-variable birth-anchored
primary model; this module re-points every panel at the current socioeconomic
risk-factor model (context scope, all-recent cohort) and adds the panels the
prototype did not have: forward-validation model selection, the predictor-scope
comparison, the adjusted odds-ratio forest plot, and the association-shift
slope graph.

Every figure is written in five formats: vector PDF and SVG/EPS (editable text,
not path-outlined), 600 dpi TIFF, and PNG for the human-readable reading copy.

Run: python -m src.figures_main
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

import config

WIDTH_2COL = 7.205     # 183 mm, Scientific Reports double column
WIDTH_1COL = 3.464     # 88 mm, single column

# --------------------------------------------------------------------------- #
# Okabe-Ito colourblind-safe palette + semantic roles
# --------------------------------------------------------------------------- #
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
BLACK = "#222222"
GREY = "#7A7A7A"
LIGHT = "#F3F5F7"

# One colour per recurring concept so the same thing means the same colour in
# every figure, not just within one panel.
ROLE = {
    "selected_model": BLUE,
    "comparison_model": GREY,
    "reference": GREY,
    "scope_birth": SKY,
    "scope_context": BLUE,
    "scope_care": PURPLE,
    "raw": VERMILION,
    "recalibrated": BLUE,
    "strengthening": VERMILION,
    "weakening": BLUE,
}

# Every current primary (context-scope) predictor, plus the sensitivity-scope
# extras, in reader-facing form. Extended from the birth-anchored-only prototype.
PRETTY = {
    "mother_age": "Maternal age at birth",
    "birth_order": "Birth order",
    "birth_interval": "Preceding birth interval",
    "children_ever_born": "Children ever born",
    "firstborn": "First birth",
    "multiple_birth": "Multiple birth",
    "sex": "Infant sex",
    "water_improved": "Improved drinking water",
    "sanitation_improved": "Improved sanitation",
    "media_exposure": "Any media exposure",
    "mother_edu": "Maternal education",
    "wealth": "Household wealth quintile",
    "religion": "Religion",
    "residence": "Residence",
    "division": "Division",
    "anc_visits": "Antenatal visits",
    "anc_4plus": "Four or more antenatal visits",
    "delivery_place": "Delivery place",
    "csection": "Caesarean delivery",
    "skilled_attendant": "Skilled birth attendant",
    "mother_working": "Mother currently working",
}

RESULTS = config.RESULTS
FIGURES = config.FIGURES
SR_FIGURES = config.ROOT / "manuscript" / "scientific_reports" / "figures"


# --------------------------------------------------------------------------- #
# Style / save
# --------------------------------------------------------------------------- #
def _style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.1,
        "pdf.fonttype": 42,     # embed TrueType, not Type-3 paths -> editable PDF/EPS text
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # keep real <text> nodes in SVG, not outlined glyphs
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _panel(ax, label, title=None):
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top")
    if title:
        ax.set_title(title, loc="left", fontweight="bold", pad=5)


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=3)


def _box(ax, xy, width, height, text, edge=BLUE, face=LIGHT, fontsize=7):
    x, y = xy
    patch = FancyBboxPatch((x, y), width, height,
                           boxstyle="round,pad=0.008,rounding_size=0.012",
                           linewidth=0.9, edgecolor=edge, facecolor=face)
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=fontsize, linespacing=1.15)


def _opaque(hex_rgba: str) -> str:
    """Blend an 8-digit RGBA hex fill against white for EPS (no alpha support)."""
    if len(hex_rgba) != 9:
        return hex_rgba
    r, g, b = (int(hex_rgba[i:i + 2], 16) for i in (1, 3, 5))
    a = int(hex_rgba[7:9], 16) / 255.0
    blend = lambda c: round(c * a + 255 * (1 - a))
    return f"#{blend(r):02x}{blend(g):02x}{blend(b):02x}"


def save_figure(fig, name: str) -> dict:
    """Write PDF, SVG, EPS, 600-dpi TIFF, and PNG. Returns per-format paths.

    ``bbox_inches`` is deliberately left at its default (not "tight"): a tight
    bounding box silently changes the exported physical width, which would
    break the 183 mm / 88 mm journal canvas guarantee this module targets.
    """
    fig.tight_layout(pad=0.9)
    paths = {}
    for fmt in ("pdf", "svg"):
        p = FIGURES / f"{name}.{fmt}"
        fig.savefig(p)
        paths[fmt] = p
    tiff = FIGURES / f"{name}.tiff"
    fig.savefig(tiff, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    paths["tiff"] = tiff
    png = FIGURES / f"{name}.png"
    fig.savefig(png, dpi=300)
    paths["png"] = png
    # EPS: PostScript has no alpha channel. Every fill in this module's panels
    # is fully opaque (_box() uses solid hex fills, no RGBA/alpha=) precisely so
    # EPS export needs no colour conversion; this is checked, not assumed, by
    # the figure QA pass, which verifies each .eps opens and matches the PDF.
    try:
        eps = FIGURES / f"{name}.eps"
        fig.savefig(eps)
        paths["eps"] = eps
    except Exception as exc:  # pragma: no cover - defensive; verified in QA pass
        print(f"  [WARN] EPS export failed for {name}: {exc}")
    plt.close(fig)
    print(f"  figure -> {name}: " + ", ".join(f"{k}" for k in paths))
    return paths


# --------------------------------------------------------------------------- #
# Figure 1 -- study design
# --------------------------------------------------------------------------- #
def figure_1():
    flow = pd.read_csv(RESULTS / "figure1_flow.csv").set_index("stage")
    lock = json.loads((RESULTS / "pipeline_lock.json").read_text())
    num_cols, nom_cols = _primary_cols()
    n_predictors = len(num_cols) + len(nom_cols)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH_2COL, 4.85),
                                   gridspec_kw={"width_ratios": [0.92, 1.08]})
    for ax in (ax1, ax2):
        ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    _panel(ax1, "a", "Participant flow")
    pooled = int(flow.loc["Pooled Births Recode records", "n"])
    excluded = int(flow.loc["Outside complete-follow-up window", "n"])
    invalid = int(flow.loc["Unclassifiable death age within window", "n"])
    primary = flow.loc["Primary all-recent cohort"]
    dev = flow.loc["Development rounds"]
    eva = flow.loc["2022 temporal evaluation"]
    care = flow.loc["Care-enriched sensitivity"]
    _box(ax1, (0.10, 0.79), 0.73, 0.11,
         f"Four BDHS Births Recode files\n{pooled:,} birth records")
    _box(ax1, (0.10, 0.57), 0.73, 0.13,
         f"Complete 28-day follow-up and classifiable outcome\n"
         f"{int(primary.n):,} births | {int(primary.deaths):,} deaths", edge=GREEN)
    ax1.annotate("", (0.465, 0.70), (0.465, 0.79),
                 arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": GREY})
    ax1.text(0.78, 0.745, f"Excluded: {excluded:,} outside window;\n"
             f"{invalid:,} unclassifiable", ha="center", va="center",
             color=GREY, fontsize=6.0)
    _box(ax1, (0.02, 0.31), 0.43, 0.13,
         f"Development\n2011, 2014, 2017-18\n{int(dev.n):,} | {int(dev.deaths):,} deaths",
         edge=BLUE)
    _box(ax1, (0.51, 0.31), 0.43, 0.13,
         f"Temporal evaluation\n2022\n{int(eva.n):,} | {int(eva.deaths):,} deaths",
         edge=ORANGE)
    for x in (0.235, 0.725):
        ax1.annotate("", (x, 0.44), (0.465, 0.57),
                     arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": GREY})
    _box(ax1, (0.18, 0.07), 0.59, 0.11,
         f"Care-module sensitivity\nmost recent birth; 2022 long form\n"
         f"{int(care.n):,} | {int(care.deaths):,} deaths", edge=PURPLE)
    ax1.annotate("", (0.475, 0.18), (0.475, 0.31),
                 arrowprops={"arrowstyle": "-|>", "lw": 0.8,
                             "linestyle": "--", "color": PURPLE})

    _panel(ax2, "b", "Chronological analysis and locked evaluation")
    y_positions = [0.82, 0.65, 0.48, 0.31, 0.12]
    labels = [
        f"Candidate predictors\n{n_predictors} infant, maternal, and household variables",
        "Forward window 1\ntrain 2011 -> validate 2014",
        "Forward window 2\ntrain 2011 + 2014 -> validate 2017-18",
        f"Lock pipeline\n{lock['model']} | mean forward AP {lock['mean_pr_auc']:.3f}",
        "Single 2022 evaluation\nAUROC, AP, Brier, calibration and uncertainty",
    ]
    colours = [BLUE, SKY, SKY, GREEN, ORANGE]
    for y, label, colour in zip(y_positions, labels, colours):
        _box(ax2, (0.12, y), 0.76, 0.11, label, edge=colour)
    for y0, y1 in zip(y_positions[:-1], y_positions[1:]):
        ax2.annotate("", (0.50, y1 + 0.11), (0.50, y0),
                     arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": GREY})
    ax2.text(0.50, 0.015,
             "Birth-anchored-only and care-enriched models were sensitivity analyses.",
             ha="center", va="bottom", fontsize=6.4, color=GREY)
    fig.subplots_adjust(wspace=0.18)
    return save_figure(fig, "Figure1")


# --------------------------------------------------------------------------- #
# Figure 2 -- model selection and temporal performance
# --------------------------------------------------------------------------- #
def figure_2():
    fwd = pd.read_csv(RESULTS / "forward_validation.csv").drop_duplicates("model")
    fwd = fwd.sort_values("mean_PR_AUC", ascending=True)
    curves = pd.read_csv(RESULTS / "aggregate_release" / "figure2_discrimination_curves.csv")
    bins = pd.read_csv(RESULTS / "aggregate_release" / "figure2_calibration_bins.csv")
    unc = pd.read_csv(RESULTS / "primary_performance_uncertainty.csv").set_index("prediction")

    fig = plt.figure(figsize=(WIDTH_2COL, 6.6))
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1, 1.05])
    ax_sel = fig.add_subplot(gs[0, 0:2])
    ax_roc = fig.add_subplot(gs[0, 2:4])
    ax_pr = fig.add_subplot(gs[0, 4:6])
    ax_raw = fig.add_subplot(gs[1, 0:3])
    ax_cal = fig.add_subplot(gs[1, 3:6])

    colours = [ROLE["selected_model"] if sel else ROLE["comparison_model"]
              for sel in fwd["selected"]]
    ax_sel.barh(fwd["model"], fwd["mean_PR_AUC"], color=colours, height=0.65)
    ax_sel.set_xlabel("Mean forward-validation\naverage precision")
    for y, (val, sel) in enumerate(zip(fwd["mean_PR_AUC"], fwd["selected"])):
        ax_sel.text(val + 0.0015, y, "selected" if sel else "", va="center",
                    fontsize=6, color=BLUE, fontweight="bold")
    _panel(ax_sel, "a", "Development-only model selection")
    _clean(ax_sel)

    styles = {"raw": (ROLE["raw"], "--"), "recalibrated": (ROLE["recalibrated"], "-")}
    for prediction in ["raw", "recalibrated"]:
        sub = curves[(curves.prediction == prediction) & (curves.curve == "ROC")]
        ax_roc.plot(sub.x, sub.y, color=styles[prediction][0],
                    ls=styles[prediction][1], label=prediction)
    ax_roc.plot([0, 1], [0, 1], color=GREY, ls=":", lw=0.9)
    ax_roc.set(xlabel="1 - specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1))
    row = unc.loc["recalibrated"]
    ax_roc.text(0.97, 0.05, f"AUROC {row.ROC_AUC:.3f}\n95% CI {row.ROC_AUC_95CI}",
                ha="right", va="bottom", fontsize=6.5)
    ax_roc.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 0.20))
    _panel(ax_roc, "b", "Receiver operating characteristic")
    _clean(ax_roc)

    for prediction in ["raw", "recalibrated"]:
        sub = curves[(curves.prediction == prediction) & (curves.curve == "precision-recall")]
        ax_pr.plot(sub.x, sub.y, color=styles[prediction][0],
                   ls=styles[prediction][1], label=prediction)
    prevalence = float(row.observed_prevalence)
    ax_pr.axhline(prevalence, color=GREY, ls=":", lw=0.9)
    ax_pr.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1))
    ax_pr.set_ylim(0, max(0.18, curves[curves.curve == "precision-recall"].y.max() * 1.1))
    ax_pr.text(0.97, 0.95, f"AP {row.PR_AUC:.3f}\n95% CI {row.PR_AUC_95CI}\n"
               f"Baseline {prevalence:.3f}", transform=ax_pr.transAxes,
               ha="right", va="top", fontsize=6.5)
    _panel(ax_pr, "c", "Precision-recall")
    _clean(ax_pr)

    sub = bins[bins.prediction == "raw"]
    ax_raw.plot(sub.mean_prediction, sub.observed_fraction, "o-", color=ROLE["raw"],
                ms=3, label="Quantile bins")
    lim = max(0.55, sub.mean_prediction.max() * 1.08)
    ax_raw.plot([0, lim], [0, lim], color=GREY, ls=":", lw=0.9)
    ax_raw.set(xlabel="Mean predicted probability", ylabel="Observed frequency",
               xlim=(0, lim), ylim=(0, max(0.08, sub.observed_fraction.max() * 1.3)))
    ax_raw.text(0.97, 0.92, f"Brier {unc.loc['raw'].Brier:.3f}\n"
                f"O:E {unc.loc['raw'].observed_to_expected:.2f}",
                transform=ax_raw.transAxes, ha="right", va="top", fontsize=6.5)
    _panel(ax_raw, "d", "Raw probability calibration")
    _clean(ax_raw)

    sub = bins[bins.prediction == "recalibrated"]
    ax_cal.plot(sub.mean_prediction, sub.observed_fraction, "o-", color=ROLE["recalibrated"], ms=3)
    lim = max(0.06, sub[["mean_prediction", "observed_fraction"]].to_numpy().max() * 1.15)
    ax_cal.plot([0, lim], [0, lim], color=GREY, ls=":", lw=0.9)
    ax_cal.set(xlabel="Mean predicted probability", ylabel="Observed frequency",
               xlim=(0, lim), ylim=(0, lim))
    skill = 1 - row.Brier / (prevalence * (1 - prevalence))
    ax_cal.text(0.97, 0.92, f"Brier {row.Brier:.5f}\nSkill {skill:+.3f}",
                transform=ax_cal.transAxes, ha="right", va="top", fontsize=6.5)
    _panel(ax_cal, "e", "Development-recalibrated probabilities")
    _clean(ax_cal)

    fig.subplots_adjust(wspace=0.9, hspace=0.42)
    return save_figure(fig, "Figure2")


# --------------------------------------------------------------------------- #
# Figure 3 -- where the mortality signal lies
# --------------------------------------------------------------------------- #
def _primary_cols():
    from src.preprocess import primary_feature_cols
    return primary_feature_cols()


def _group_shap(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in table.iterrows():
        feature = str(row.feature)
        for prefix, key in [("sex_", "sex"), ("religion_", "religion"),
                            ("residence_", "residence"), ("division_", "division"),
                            ("mother_edu", "mother_edu"), ("wealth", "wealth"),
                            ("missingindicator_", None)]:
            if feature.startswith(prefix):
                if key is None:
                    key = None  # missing indicators folded into their source below
                    base = feature[len(prefix):]
                    key = base
                break
        else:
            key = feature
        rows.append({"feature": key, **{y: row[y] for y in ["2011", "2014", "2017", "2022"]}})
    grouped = pd.DataFrame(rows).groupby("feature", as_index=False)[
        ["2011", "2014", "2017", "2022"]].sum()
    grouped["label"] = grouped.feature.map(PRETTY).fillna(grouped.feature)
    return grouped.sort_values("2022", ascending=True).tail(12)


def figure_3():
    scope = pd.read_csv(RESULTS / "table11_predictor_scope_sensitivity.csv")
    scope = scope[scope.prediction == "isotonic_recalibrated"].set_index("scope")
    scope_order = ["birth_anchored_only", "survey_context_enriched", "care_enriched_most_recent"]
    scope_labels = ["Birth-anchored\nonly", "+ Household\nsocioeconomic\n(primary)",
                    "Care-enriched\n(most-recent birth)"]
    scope_colours = [ROLE["scope_birth"], ROLE["scope_context"], ROLE["scope_care"]]

    orr = pd.read_csv(RESULTS / "table_adjusted_or.csv")
    orr = orr[orr["predictor"] != "Survey round"].copy()
    orr["term"] = np.where(orr["level"].isin(["yes vs. no", "per year", "per additional birth",
                                              "per 12 months", "per additional child"]),
                           orr["predictor"], orr["predictor"] + ": " + orr["level"])
    orr = orr.reindex(orr["adjusted_OR"].sub(1).abs().sort_values(ascending=True).index).tail(14)

    shap_tab = _group_shap(pd.read_csv(RESULTS / "shap_temporal.csv"))

    fig = plt.figure(figsize=(WIDTH_2COL, 5.6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.62, 1.35, 0.85])
    ax_scope = fig.add_subplot(gs[0, 0])
    ax_or = fig.add_subplot(gs[0, 1])
    ax_shap = fig.add_subplot(gs[0, 2])

    for i, (key, label, colour) in enumerate(zip(scope_order, scope_labels, scope_colours)):
        row = scope.loc[key]
        ax_scope.errorbar([row.ROC_AUC], [i], xerr=[[0], [0]], fmt="o", color=colour, ms=7)
        ax_scope.text(row.ROC_AUC + 0.012, i, f"{row.ROC_AUC:.3f}", va="center", fontsize=6.3)
    ax_scope.axvline(0.5, color=GREY, ls=":", lw=0.9)
    ax_scope.set_yticks(range(3)); ax_scope.set_yticklabels(scope_labels, fontsize=6.3)
    ax_scope.set_xlabel("Recalibrated 2022 AUROC")
    ax_scope.set_xlim(0.45, 0.75)
    ax_scope.invert_yaxis()
    _panel(ax_scope, "a", "Predictor-scope comparison")
    _clean(ax_scope)

    ypos = np.arange(len(orr))
    xerr = np.vstack([orr["adjusted_OR"] - orr["ci_low"], orr["ci_high"] - orr["adjusted_OR"]])
    xerr = np.clip(xerr, 0, None)
    colours_or = [VERMILION if v > 1 else BLUE for v in orr["adjusted_OR"]]
    ax_or.errorbar(orr["adjusted_OR"], ypos, xerr=xerr, fmt="none", ecolor=BLACK,
                   capsize=2.5, lw=0.8, zorder=2)
    ax_or.scatter(orr["adjusted_OR"], ypos, c=colours_or, s=22, zorder=3,
                  edgecolor="white", linewidth=0.4)
    ax_or.axvline(1, color=GREY, ls=":", lw=0.9)
    ax_or.set_yticks(ypos); ax_or.set_yticklabels(orr["term"], fontsize=6.2)
    ax_or.set_xlabel("Adjusted odds ratio (95% CI), log scale")
    ax_or.set_xscale("log")
    ax_or.set_xlim(max(0.05, orr["ci_low"].min() * 0.7), orr["ci_high"].max() * 1.3)
    _panel(ax_or, "b", "Adjusted associations")
    _clean(ax_or)

    ypos2 = np.arange(len(shap_tab))
    ax_shap.barh(ypos2, 100 * shap_tab["2022"], color=BLUE, height=0.65)
    ax_shap.set_yticks(ypos2); ax_shap.set_yticklabels(shap_tab.label, fontsize=6.3)
    ax_shap.set_xlabel("Share of mean absolute\nSHAP value (%)")
    for y, value in zip(ypos2, 100 * shap_tab["2022"]):
        ax_shap.text(value + 0.5, y, f"{value:.1f}", va="center", fontsize=6.0)
    _panel(ax_shap, "c", "Model attribution")
    _clean(ax_shap)

    fig.subplots_adjust(wspace=0.75)
    return save_figure(fig, "Figure3")


# --------------------------------------------------------------------------- #
# Figure 4 -- mortality and service trends
# --------------------------------------------------------------------------- #
def figure_4():
    trend = pd.read_csv(RESULTS / "table3_nmr_trend.csv")
    coverage = pd.read_csv(RESULTS / "table3b_prevalence.csv")
    services = ["ANC 4+ visits", "facility delivery", "skilled attendant", "caesarean section"]
    coverage = coverage[coverage.exposure.isin(services)].set_index("exposure").loc[services]
    shift = pd.read_csv(RESULTS / "table4_association_shift.csv")
    years = [2011, 2014, 2017, 2022]
    cols = [f"y{year}" for year in years]

    fig = plt.figure(figsize=(WIDTH_2COL, 4.3))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.85, 1.1, 1.05])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    ax1.errorbar(trend.year, trend.NMR,
                 yerr=[trend.NMR - trend.ci_low, trend.ci_high - trend.NMR],
                 color=BLUE, marker="o", ms=4, capsize=3, lw=1.2)
    ax1.axhline(12, color=GREY, ls=":", lw=0.9)
    ax1.text(2021.8, 12.8, "SDG 3.2 target", ha="right", fontsize=6.2, color=GREY)
    for _, row in trend.iterrows():
        ax1.text(row.year, row.ci_high + 1.2, f"{row.NMR:.1f}", ha="center", fontsize=6.3)
    ax1.set(xlabel="BDHS survey round", ylabel="Neonatal deaths per 1,000 live births",
            xticks=years, ylim=(0, max(trend.ci_high) + 5))
    ax1.set_xticklabels(["2011", "2014", "2017-18", "2022"])
    _panel(ax1, "a", "Weighted neonatal mortality")
    _clean(ax1)

    colours = [ORANGE, GREEN, BLUE, PURPLE]
    for (label, row), colour in zip(coverage.iterrows(), colours):
        values = row[cols].to_numpy(float)
        ax2.plot(years, values, marker="o", ms=3.5, color=colour, label=label)
        ax2.text(2022.15, values[-1], f"{values[-1]:.1f}", va="center", fontsize=6.2,
                 color=colour)
    ax2.set(xlabel="BDHS survey round", ylabel="Survey-weighted prevalence (%)",
            xticks=years, ylim=(0, 80), xlim=(2010.5, 2023.6))
    ax2.set_xticklabels(["2011", "2014", "2017-18", "2022"])
    ax2.legend(frameon=False, loc="upper left")
    _panel(ax2, "b", "Maternity-service coverage")
    _clean(ax2)

    shift = shift.sort_values("OR_2011")
    ypos = np.arange(len(shift))
    or_2011 = shift["OR_2011"].to_numpy()
    or_2022 = or_2011 * shift["OR_ratio_per_year"].to_numpy() ** (2022 - 2011)
    for y, (o1, o2, sig, direction) in enumerate(
            zip(or_2011, or_2022, shift["interaction_q_bh"] < 0.05, shift["direction"])):
        colour = ROLE[direction]
        lw = 1.6 if sig else 0.8
        ax3.plot([o1, o2], [y, y], color=colour, lw=lw, zorder=2)
        ax3.scatter([o1], [y], marker="o", s=16, color=colour, zorder=3)
        ax3.scatter([o2], [y], marker=">", s=20, color=colour, zorder=3)
    ax3.axvline(1, color=GREY, ls=":", lw=0.9)
    ax3.set_xscale("log")
    ax3.set_yticks(ypos); ax3.set_yticklabels(shift["predictor"], fontsize=6.2)
    ax3.set_xlabel("Cross-sectional OR, 2011 (o) -> 2022 (>)\nbold = FDR q<0.05")
    _panel(ax3, "c", "Association shift over time")
    _clean(ax3)

    fig.subplots_adjust(wspace=0.75)
    return save_figure(fig, "Figure4")


# --------------------------------------------------------------------------- #
# Figure 5 -- decomposition
# --------------------------------------------------------------------------- #
def _decomp_panel(ax, table, title, label):
    order = ["distribution", "effect", "total_change"]
    names = ["Observed\ncomposition", "Residual\ncoefficient-associated", "Total\nchange"]
    sub = table.set_index("component").loc[order]
    ypos = np.arange(3)
    values = sub.value_per_1000.to_numpy()
    errors = np.vstack([values - sub.ci_low.to_numpy(), sub.ci_high.to_numpy() - values])
    colours = [GREEN, ORANGE, BLUE]
    ax.errorbar(values, ypos, xerr=errors, fmt="none", ecolor=BLACK, capsize=3, lw=0.9)
    ax.scatter(values, ypos, c=colours, s=30, zorder=3, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=GREY, ls=":", lw=0.9)
    ax.set_yticks(ypos); ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Contribution to change (deaths per 1,000)")
    ax.set_xlim(-19, 7)
    for x, y in zip(values, ypos):
        label_y = y - 0.18 if y == ypos.max() else y + 0.18
        ax.text(x + (0.7 if x >= 0 else -0.7), label_y, f"{x:+.1f}",
                ha="left" if x >= 0 else "right", fontsize=6.3)
    _panel(ax, label, title)
    _clean(ax)


def figure_5():
    primary = pd.read_csv(RESULTS / "table5_decomposition.csv")
    care = pd.read_csv(RESULTS / "table5b_decomposition_care_module.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH_2COL, 3.55))
    _decomp_panel(ax1, primary, "Representative all-recent cohort (primary)", "a")
    _decomp_panel(ax2, care, "Care-module sensitivity", "b")
    fig.subplots_adjust(wspace=0.55, bottom=0.22)
    return save_figure(fig, "Figure5")


RENDERERS = {
    "Figure1": figure_1, "Figure2": figure_2, "Figure3": figure_3,
    "Figure4": figure_4, "Figure5": figure_5,
}


def run():
    _style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    SR_FIGURES.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, fn in RENDERERS.items():
        manifest[name] = {k: str(v) for k, v in fn().items()}
        # Also place PDF/TIFF copies where the manuscript expects submission files.
        src_pdf = FIGURES / f"{name}.pdf"
        (SR_FIGURES / f"{name}.pdf").write_bytes(src_pdf.read_bytes())
        src_tiff = FIGURES / f"{name}.tiff"
        (SR_FIGURES / f"{name}.tiff").write_bytes(src_tiff.read_bytes())
    (RESULTS / "figures_main_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  5 main figures rendered -> {FIGURES} and {SR_FIGURES}")


if __name__ == "__main__":
    run()
