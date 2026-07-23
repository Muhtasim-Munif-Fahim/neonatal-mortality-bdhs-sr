"""
Trend analysis -- the population-level companion to the individual ML risk model.

Design-based (DHS is a weighted, clustered sample): survey weights on every point
estimate + a PSU **cluster bootstrap, stratified by survey round**, for all inference
(CIs and p-values). Logistic fits use sklearn with penalty=None (unpenalised MLE) so
weighting is exact and fast; statsmodels' freq_weights + cluster cov is unreliable.

Produces:
  A. NMR trend 2011->2022 with 95% CI + linear-trend test        -> fig7,  table3_nmr_trend
  B. Risk-factor prevalence trends (+ trend p)                   -> fig8,  table3b_prevalence
  C. Association shift (predictor x survey-year interactions)    ->        table4_association_shift
  D. Decomposition of the NMR change: distribution vs effect     -> fig9,  table5_decomposition

NMR trend (A) and the primary decomposition (D) use the same representative
all-recent-births, complete-follow-up cohort and the same six non-care
covariates.  Care coverage and association-shift analyses use the maternal-care
module; the ten-covariate decomposition is a module-restricted sensitivity.

Run: python -m src.trends
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from sklearn.linear_model import LogisticRegression

import config
from src import viz
from src.harmonize import harmonize

SDG_NMR_TARGET = 12          # SDG-3.2: <=12 neonatal deaths / 1000 live births by 2030
N_BOOT = 1000
_ROUNDS = [2011, 2014, 2017, 2022]


# --------------------------------------------------------------------------- #
# Core numerics
# --------------------------------------------------------------------------- #
def _coef(X, y, w):
    """Unpenalised weighted logistic -> (intercept, coef vector)."""
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    m.fit(X, y, sample_weight=w)
    return m.intercept_[0], m.coef_[0]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _wmean(y, w):
    return float(np.average(y, weights=w))


def _boot_round_indices(df, rng):
    """Resample PSUs with replacement within survey round and stratum."""
    clusters = df[config.CLUSTER_COL].to_numpy()
    strata = df[config.STRATUM_COL].to_numpy()
    parts = []
    for stratum in np.unique(strata):
        sub = np.where(strata == stratum)[0]
        cl = clusters[sub]
        uniq = np.unique(cl)
        idx_by = {c: sub[cl == c] for c in uniq}
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        parts.append(np.concatenate([idx_by[c] for c in samp]))
    return np.concatenate(parts)


def _boot_pvalue(stats: np.ndarray) -> float:
    """Two-sided bootstrap p for H0: statistic == 0."""
    stats = stats[np.isfinite(stats)]
    if len(stats) == 0:
        return np.nan
    frac_pos = np.mean(stats > 0)
    p = min(1.0, 2 * min(frac_pos, 1 - frac_pos))
    return float(max(p, 1.0 / (len(stats) + 1)))   # floor: cannot resolve below 1/(B+1)


# --------------------------------------------------------------------------- #
# A. NMR trend
# --------------------------------------------------------------------------- #
def nmr_trend() -> pd.DataFrame:
    df = harmonize(module_only=False)
    rng = np.random.default_rng(config.SEED)
    y = df[config.TARGET].to_numpy(float)
    w = df[config.WEIGHT_COL].to_numpy(float)
    yr = df[config.YEAR_COL].to_numpy(float)

    # per-round NMR + bootstrap CI (reuse the same replicate indices for all rounds)
    boot_nmr = {r: [] for r in _ROUNDS}
    boot_beta = []
    for _ in range(N_BOOT):
        idx = _boot_round_indices(df, rng)
        yb, wb, yrb = y[idx], w[idx], yr[idx]
        for r in _ROUNDS:
            m = yrb == r
            boot_nmr[r].append(_wmean(yb[m], wb[m]) * 1000)
        boot_beta.append(_coef((yrb - 2011).reshape(-1, 1), yb, wb)[1][0])

    rows = []
    for r in _ROUNDS:
        m = yr == r
        nmr = _wmean(y[m], w[m]) * 1000
        lo, hi = np.percentile(boot_nmr[r], [2.5, 97.5])
        rows.append({"year": int(r), "n": int(m.sum()), "deaths": int(y[m].sum()),
                     "NMR": round(nmr, 1), "ci_low": round(lo, 1), "ci_high": round(hi, 1)})
    tab = pd.DataFrame(rows)

    beta = _coef((yr - 2011).reshape(-1, 1), y, w)[1][0]
    p = _boot_pvalue(np.array(boot_beta))
    apc = (np.exp(beta) - 1) * 100
    tab.attrs["trend"] = {"log_or_per_year": beta, "annual_pct_change_odds": apc, "p": p}
    tab.to_csv(config.RESULTS / "table3_nmr_trend.csv", index=False)
    print(f"  A. NMR {tab['NMR'].iloc[0]:.1f} -> {tab['NMR'].iloc[-1]:.1f}/1000; "
          f"odds {apc:+.1f}%/yr, p={p:.3f}")
    _fig_nmr(tab)
    return tab


def _fig_nmr(tab):
    fig, ax = viz.plt.subplots(figsize=(7.5, 5))
    ax.errorbar(tab["year"], tab["NMR"],
                yerr=[tab["NMR"] - tab["ci_low"], tab["ci_high"] - tab["NMR"]],
                fmt="o-", color=viz.PALETTE[4], lw=2, capsize=4, label="weighted NMR (95% CI)")
    ax.axhline(SDG_NMR_TARGET, ls="--", color=viz.PALETTE[3], lw=1.5,
               label=f"SDG-3.2 target ({SDG_NMR_TARGET}/1000 by 2030)")
    for _, r in tab.iterrows():
        ax.annotate(f"{r['NMR']:.0f}", (r["year"], r["NMR"]),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
    ax.set(xlabel="BDHS survey year", ylabel="neonatal deaths / 1000 live births",
           title="Neonatal mortality trend, Bangladesh 2011-2022")
    ax.set_xticks(_ROUNDS)
    ax.set_ylim(0, max(tab["ci_high"]) * 1.15)
    ax.legend(fontsize=9)
    viz.save(fig, "fig7_nmr_trend.png")


# --------------------------------------------------------------------------- #
# Exposures (compact, binary, low-missing) for B / C / D
# --------------------------------------------------------------------------- #
EXPOSURE_LABELS = {
    "male": "male infant", "multiple_birth": "multiple birth",
    "adolescent_mother": "mother <20 y", "edu_secondary_plus": "mother >= secondary edu",
    "poorest": "poorest quintile", "rural": "rural residence",
    "anc_4plus": "ANC 4+ visits", "facility_delivery": "facility delivery",
    "skilled_attendant": "skilled attendant", "csection": "caesarean section",
}


def _exposures(df: pd.DataFrame) -> pd.DataFrame:
    e = pd.DataFrame(index=df.index)
    e["male"] = (df["sex"] == "male").astype(float)
    e["multiple_birth"] = df["multiple_birth"].astype(float)
    e["adolescent_mother"] = (df["mother_age"] < 20).astype(float)
    e["edu_secondary_plus"] = df["mother_edu"].isin(["secondary", "higher"]).astype(float)
    e["poorest"] = (df["wealth"] == "poorest").astype(float)
    e["rural"] = (df["residence"] == "rural").astype(float)
    e["anc_4plus"] = pd.to_numeric(df["anc_4plus"], errors="coerce")
    place = df["delivery_place"]
    e["facility_delivery"] = pd.Series(
        np.where(place.isna(), np.nan, place.eq("facility").astype(float)),
        index=df.index, dtype=float)
    e["skilled_attendant"] = df["skilled_attendant"].astype(float)
    e["csection"] = pd.to_numeric(df["csection"], errors="coerce")
    # Retain the full module cohort in the care-variable decomposition. A
    # zero-filled value is interpreted jointly with its explicit missingness
    # indicator, rather than silently selecting complete cases.
    for col in ["anc_4plus", "facility_delivery", "skilled_attendant", "csection"]:
        e[f"{col}_missing"] = e[col].isna().astype(float)
        e[col] = e[col].fillna(0.0)
    return e


# --------------------------------------------------------------------------- #
# B. Prevalence trends
# --------------------------------------------------------------------------- #
def prevalence_trends() -> pd.DataFrame:
    df = harmonize(module_only=True)
    E = _exposures(df)
    yr = df[config.YEAR_COL].to_numpy(float)
    w = df[config.WEIGHT_COL].to_numpy(float)
    rng = np.random.default_rng(config.SEED)

    # bootstrap replicate indices reused across exposures
    boot_idx = [_boot_round_indices(df, rng) for _ in range(N_BOOT)]

    rows = []
    for col in EXPOSURE_LABELS:
        x = E[col].to_numpy(float)
        ok = np.isfinite(x)
        prev = {r: 100 * _wmean(x[ok & (yr == r)], w[ok & (yr == r)]) for r in _ROUNDS}
        betas = []
        for idx in boot_idx:
            xb, wb, yb = x[idx], w[idx], yr[idx]
            m = np.isfinite(xb)
            if len(np.unique(xb[m])) < 2:
                continue
            betas.append(_coef((yb[m] - 2011).reshape(-1, 1), xb[m], wb[m])[1][0])
        p = _boot_pvalue(np.array(betas))
        rows.append({"exposure": EXPOSURE_LABELS[col],
                     **{f"y{r}": round(prev[r], 1) for r in _ROUNDS},
                     "change": round(prev[2022] - prev[2011], 1),
                     "trend_p": p})
    tab = pd.DataFrame(rows)
    tab["trend_q_bh"] = false_discovery_control(tab["trend_p"].to_numpy(), method="bh")
    tab[["trend_p", "trend_q_bh"]] = tab[["trend_p", "trend_q_bh"]].round(3)
    tab.to_csv(config.RESULTS / "table3b_prevalence.csv", index=False)
    print(f"  B. prevalence trends for {len(tab)} exposures (+BH FDR) -> table3b")
    _fig_prevalence(df, E)
    return tab


def _fig_prevalence(df, E):
    yr = df[config.YEAR_COL].to_numpy(float)
    w = df[config.WEIGHT_COL].to_numpy(float)
    fig, ax = viz.plt.subplots(figsize=(8.5, 6))
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]
    for i, col in enumerate(EXPOSURE_LABELS):
        x = E[col].to_numpy(float)
        ok = np.isfinite(x)
        prev = [100 * _wmean(x[ok & (yr == r)], w[ok & (yr == r)]) for r in _ROUNDS]
        ax.plot(_ROUNDS, prev, marker=markers[i], linestyle=linestyles[i],
                color=viz.PALETTE[i % len(viz.PALETTE)], lw=1.6,
                label=EXPOSURE_LABELS[col])
    ax.set(xlabel="BDHS survey year", ylabel="weighted prevalence (%)",
           title="Risk-factor / coverage prevalence trends, 2011-2022")
    ax.set_xticks(_ROUNDS)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    viz.save(fig, "fig8_riskfactor_trends.png")


# --------------------------------------------------------------------------- #
# C. Association shift (predictor x year interaction)
# --------------------------------------------------------------------------- #
def association_shift() -> pd.DataFrame:
    df = harmonize(module_only=True)
    E = _exposures(df)
    y = df[config.TARGET].to_numpy(float)
    w = df[config.WEIGHT_COL].to_numpy(float)
    yr_c = df[config.YEAR_COL].to_numpy(float) - 2011
    rng = np.random.default_rng(config.SEED)
    boot_idx = [_boot_round_indices(df, rng) for _ in range(N_BOOT)]

    rows = []
    for col in EXPOSURE_LABELS:
        x = E[col].to_numpy(float)
        ok = np.isfinite(x)
        X = np.column_stack([x, yr_c, x * yr_c])[ok]
        b = _coef(X, y[ok], w[ok])[1]
        main_or, inter_or = np.exp(b[0]), np.exp(b[2])
        inter_betas = []
        for idx in boot_idx:
            m = idx[np.isfinite(x[idx])]
            Xb = np.column_stack([x[m], yr_c[m], x[m] * yr_c[m]])
            if len(np.unique(x[m])) < 2:
                continue
            inter_betas.append(_coef(Xb, y[m], w[m])[1][2])
        p = _boot_pvalue(np.array(inter_betas))
        rows.append({"predictor": EXPOSURE_LABELS[col],
                     "OR_2011": round(main_or, 2),
                     "OR_ratio_per_year": round(inter_or, 3),
                     "direction": "strengthening" if inter_or > 1 else "weakening",
                     "interaction_p": p})
    tab = pd.DataFrame(rows)
    tab["interaction_q_bh"] = false_discovery_control(
        tab["interaction_p"].to_numpy(), method="bh")
    tab[["interaction_p", "interaction_q_bh"]] = tab[
        ["interaction_p", "interaction_q_bh"]].round(3)
    tab = tab.sort_values("interaction_p")
    tab.to_csv(config.RESULTS / "table4_association_shift.csv", index=False)
    print(f"  C. association-shift for {len(tab)} predictors (+BH FDR) -> table4")
    return tab


# --------------------------------------------------------------------------- #
# D. Decomposition of the NMR change (distribution vs effect)
# --------------------------------------------------------------------------- #
_DECOMP_COLS = list(EXPOSURE_LABELS)
_CARE_MISSING_COLS = [f"{c}_missing" for c in
                      ["anc_4plus", "facility_delivery", "skilled_attendant", "csection"]]
# Sensitivity: drop the care variables that are confounded by indication
# (facility/skilled/C-section/ANC), keeping only less-confounded socio-demographic
# + infant composition -> a cleaner distribution-vs-effect read.
_NONCARE_COLS = ["male", "multiple_birth", "adolescent_mother",
                 "edu_secondary_plus", "poorest", "rural"]


def _decomp_once(Xa, wa, Xb, wb, ya, yb):
    ia, ca = _coef(Xa, ya, wa)
    ib, cb = _coef(Xb, yb, wb)

    def P(X, w, i, c):
        return _wmean(_sigmoid(i + X @ c), w)

    Na, Nb = P(Xa, wa, ia, ca), P(Xb, wb, ib, cb)
    dist = 0.5 * ((P(Xb, wb, ia, ca) - P(Xa, wa, ia, ca))
                  + (P(Xb, wb, ib, cb) - P(Xa, wa, ib, cb)))
    effect = 0.5 * ((P(Xa, wa, ib, cb) - P(Xa, wa, ia, ca))
                    + (P(Xb, wb, ib, cb) - P(Xb, wb, ia, ca)))
    return np.array([Na * 1000, Nb * 1000, (Nb - Na) * 1000, dist * 1000, effect * 1000])


def decomposition(cols=None, tag="primary", module_only=False) -> pd.DataFrame:
    """Symmetric nonlinear decomposition for a declared cohort/covariate set.

    The ``effect`` component is a residual coefficient-associated change.  It
    is not a causal mechanism and can absorb unmeasured composition,
    confounding, model misspecification, and coefficient change.
    """
    cols = cols or _NONCARE_COLS
    df = harmonize(module_only=module_only)
    E = _exposures(df)
    keep = E[cols].notna().all(axis=1).to_numpy()
    df, E = df[keep], E[keep]
    yr = df[config.YEAR_COL].to_numpy()
    y = df[config.TARGET].to_numpy(float)
    w = df[config.WEIGHT_COL].to_numpy(float)
    X = E[cols].to_numpy(float)

    a, b = yr == 2011, yr == 2022
    point = _decomp_once(X[a], w[a], X[b], w[b], y[a], y[b])

    rng = np.random.default_rng(config.SEED)
    boots = []
    sub = df.copy()
    for _ in range(N_BOOT):
        idx = _boot_round_indices(sub, rng)  # note: resamples all 4 rounds; we use 2011/2022
        yrb = yr[idx]
        aa, bb = yrb == 2011, yrb == 2022
        try:
            boots.append(_decomp_once(X[idx][aa], w[idx][aa], X[idx][bb], w[idx][bb],
                                      y[idx][aa], y[idx][bb]))
        except Exception:
            continue
    boots = np.array(boots)
    ci = np.percentile(boots, [2.5, 97.5], axis=0)

    labels = ["NMR_2011", "NMR_2022", "total_change", "distribution", "effect"]
    tab = pd.DataFrame({
        "component": labels,
        "value_per_1000": np.round(point, 1),
        "ci_low": np.round(ci[0], 1),
        "ci_high": np.round(ci[1], 1),
    })
    tab["bootstrap_successes"] = len(boots)
    tab["bootstrap_failures"] = N_BOOT - len(boots)
    total = point[2]
    tab["pct_of_change"] = [np.nan, np.nan, 100.0,
                            round(100 * point[3] / total, 1),
                            round(100 * point[4] / total, 1)]
    fname = "table5_decomposition.csv" if tag == "primary" else f"table5b_decomposition_{tag}.csv"
    tab.to_csv(config.RESULTS / fname, index=False)
    print(f"  D[{tag}]. NMR change {total:+.1f}/1000 = distribution {point[3]:+.1f} "
          f"+ effect {point[4]:+.1f}  ({len(cols)} vars)")
    _fig_decomposition(point, ci, tag)
    return tab


def _fig_decomposition(point, ci, tag="primary"):
    _, _, total, dist, effect = point
    fig, ax = viz.plt.subplots(figsize=(6.5, 5))
    values = np.array([dist, effect, total])
    lows = np.array([ci[0, 3], ci[0, 4], ci[0, 2]])
    highs = np.array([ci[1, 3], ci[1, 4], ci[1, 2]])
    bars = ax.bar(["observed\ncomposition",
                   "residual\ncoefficient-associated",
                   "total\nchange"], values,
                  yerr=np.vstack([values - lows, highs - values]), capsize=4,
                  color=[viz.PALETTE[2], viz.PALETTE[1], viz.PALETTE[0]])
    for bar, val in zip(bars, values):
        ax.annotate(f"{val:+.1f}", (bar.get_x() + bar.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 6 if val >= 0 else -14),
                    ha="center", fontsize=10)
    ax.axhline(0, color="k", lw=0.8)
    suffix = "" if tag == "primary" else " (care-module sensitivity)"
    ax.set(ylabel="contribution to NMR change (per 1000)",
           title=f"Decomposition of the 2011->2022 NMR decline{suffix}")
    fname = "fig9_decomposition.png" if tag == "primary" else f"fig9b_decomposition_{tag}.png"
    viz.save(fig, fname)


def run():
    nmr_trend()
    prevalence_trends()
    association_shift()
    decomposition(_NONCARE_COLS, "primary", module_only=False)
    decomposition(_DECOMP_COLS + _CARE_MISSING_COLS, "care_module", module_only=True)


if __name__ == "__main__":
    run()
