#!/usr/bin/env python3
"""ROI-level Age x hypopnea-index interaction for the predefined spindle ROIs.

Unlike the channel-wise/cluster Age x HI analysis (``age_hi_interaction.py``),
this tests whether age moderates the HI association *within each predefined
region of interest* (the behavioral-averaging ROIs in
``roi_values_corrected.csv``). Because each ROI is a single averaged series, the
test is a single OLS interaction model per ROI -- no spatial permutation is
needed, so it is not affected by cluster-null assumptions.

For each ROI:

    roi ~ age_c + logHI_c + age_c:logHI_c + sex      (N = 62)

We report the interaction coefficient (beta, t, p), the simple slope of
log(HI + 1) at ages 6, 8, and 10 years, and emit a two-panel figure with the
age-stratified HI -> duration curves for the two anterior duration ROIs.

This directly tests the developmental-moderation hypothesis: if still-maturing
anterior circuits are more vulnerable to respiratory disruption, the negative
HI -> duration slope should be steeper in younger children.

Outputs:
- output/tables/age_hi_roi_interaction.csv
- output/Fig_S3.png / output/Fig_S3.pdf      : Supplementary Figure S3
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from spindle_common import COV_PATH, PROJECT, ROI_PATH, TABLE_DIR, ensure_dir, load_covariates

FIG_S3 = PROJECT / "output/Fig_S3"  # stem; saved as .png and .pdf at 600 dpi
OUT_TABLE = TABLE_DIR / "age_hi_roi_interaction.csv"

# (column in roi_values_corrected.csv, label, unit, ylabel for figure or None)
ROIS = [
    ("ant_fast_dur", "Anterior fast spindle duration", "s", "Predicted fast spindle duration (s)"),
    ("ant_slow_dur", "Anterior slow spindle duration", "s", "Predicted slow spindle duration (s)"),
    ("pos_slow_amp", "Posterior slow spindle amplitude", "µV", None),
    ("ant_slow_peakfreq", "Anterior slow spindle peak frequency", "Hz", None),
]
FIG_ROIS = ["ant_fast_dur", "ant_slow_dur"]
# Simple slopes are probed at mean and mean +/- 1 SD of age (Aiken & West),
# i.e. data-grounded points rather than arbitrary integer ages.
PROBE_TAGS = ["lo", "mid", "hi"]
PROBE_LABELS = {"lo": "−1 SD", "mid": "Mean", "hi": "+1 SD"}


def fit_interaction(roi_values: np.ndarray, cov: pd.DataFrame):
    """OLS interaction model for one ROI; returns (fit, df)."""
    df = cov[["age_years", "age_c", "logHI", "logHI_c", "sex"]].copy()
    df["roi"] = roi_values
    df = df.dropna()
    df["ageXHI"] = df["age_c"] * df["logHI_c"]
    fit = sm.OLS.from_formula("roi ~ age_c + logHI_c + ageXHI + sex", df).fit()
    return fit, df


def simple_slope_at_age(fit, df, age_value: float) -> tuple[float, float, float]:
    """Slope of logHI_c at a given age, with t and p, via a linear contrast.

    d(roi)/d(logHI_c) = b_logHI + b_ageXHI * (age_value - mean_age).
    """
    age_c_val = age_value - df["age_years"].mean()
    params = fit.params
    cov_b = fit.cov_params()
    slope = params["logHI_c"] + params["ageXHI"] * age_c_val
    var = (
        cov_b.loc["logHI_c", "logHI_c"]
        + age_c_val**2 * cov_b.loc["ageXHI", "ageXHI"]
        + 2 * age_c_val * cov_b.loc["logHI_c", "ageXHI"]
    )
    se = float(np.sqrt(var))
    t = slope / se
    from scipy import stats

    p = 2 * stats.t.sf(abs(t), df=int(fit.df_resid))
    return float(slope), float(t), float(p)


def make_figure(results_for_fig: dict, probes: list[tuple[float, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    panel_letters = ["(a)", "(b)"]
    for ax, letter, col in zip(axes, panel_letters, FIG_ROIS):
        fit, df, label, ylabel = (
            results_for_fig[col]["fit"],
            results_for_fig[col]["df"],
            results_for_fig[col]["label"],
            results_for_fig[col]["ylabel"],
        )
        age_mean = df["age_years"].mean()
        loghi_mean = df["logHI"].mean()
        loghi_raw = np.linspace(df["logHI"].min(), df["logHI"].max(), 100)
        sex_mean = df["sex"].mean()
        for (age_value, tag), color in zip(probes, ["#2b8cbe", "#4d4d4d", "#e34a33"]):
            pred = pd.DataFrame(
                {
                    "age_c": age_value - age_mean,
                    "logHI_c": loghi_raw - loghi_mean,
                    "sex": sex_mean,
                },
                index=np.arange(len(loghi_raw)),
            )
            pred["ageXHI"] = pred["age_c"] * pred["logHI_c"]
            frame = fit.get_prediction(pred).summary_frame(alpha=0.05)
            ax.plot(
                loghi_raw, frame["mean"], color=color, lw=2.2,
                label=f"{tag} ({age_value:.1f} y)",
            )
            ax.fill_between(
                loghi_raw,
                frame["mean_ci_lower"].to_numpy(dtype=float),
                frame["mean_ci_upper"].to_numpy(dtype=float),
                color=color, alpha=0.15, linewidth=0,
            )
        ax.set_xlabel(r"Hypopnea index  [$\log_{10}$(HI + 1)]", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{letter} {label}", loc="left", fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)
    leg = axes[0].legend(frameon=False, title="Age", fontsize=10)
    leg.get_title().set_fontsize(10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG_S3}.{ext}", dpi=600)
    plt.close(fig)


def main() -> None:
    cov = load_covariates(COV_PATH)
    cov["sex"] = cov["gender_bin"]
    roi_df = pd.read_csv(ROI_PATH)
    cov = cov.merge(roi_df, on="id", how="left")

    age_mean = cov["age_years"].mean()
    age_sd = cov["age_years"].std(ddof=1)
    probe_ages = {"lo": age_mean - age_sd, "mid": age_mean, "hi": age_mean + age_sd}
    probes = [(probe_ages[t], PROBE_LABELS[t]) for t in PROBE_TAGS]
    print(f"[SAMPLE] N = {len(cov)} | age mean={age_mean:.2f}, SD={age_sd:.2f}")
    print(
        "[PROBES] simple slopes at mean +/- 1 SD: "
        + ", ".join(f"{PROBE_LABELS[t]}={probe_ages[t]:.1f}y" for t in PROBE_TAGS)
        + "\n"
    )

    rows = []
    fig_inputs = {}
    for col, label, unit, ylabel in ROIS:
        fit, df = fit_interaction(cov[col].to_numpy(dtype=float), cov)
        b = fit.params["ageXHI"]
        t = fit.tvalues["ageXHI"]
        p = fit.pvalues["ageXHI"]
        slopes = {tag: simple_slope_at_age(fit, df, probe_ages[tag]) for tag in PROBE_TAGS}
        rows.append(
            {
                "roi": col,
                "label": label,
                "unit": unit,
                "n": int(fit.nobs),
                "age_mean": round(age_mean, 2),
                "age_sd": round(age_sd, 2),
                "interaction_beta": round(b, 5),
                "interaction_t": round(t, 3),
                "interaction_p": round(p, 4),
                **{f"slope_{tag}": round(slopes[tag][0], 5) for tag in PROBE_TAGS},
                **{f"slope_p_{tag}": round(slopes[tag][2], 4) for tag in PROBE_TAGS},
            }
        )
        print(f"[{label}]")
        print(f"    Age x HI interaction: beta={b:+.4f} {unit}/log-unit, t={t:+.2f}, p={p:.4f}")
        for tag in PROBE_TAGS:
            s, st, sp = slopes[tag]
            print(
                f"    HI slope @ {PROBE_LABELS[tag]} ({probe_ages[tag]:.1f}y): "
                f"{s:+.4f} {unit}/log-unit (t={st:+.2f}, p={sp:.4f})"
            )
        print()
        if col in FIG_ROIS:
            fig_inputs[col] = {"fit": fit, "df": df, "label": label, "ylabel": ylabel}

    ensure_dir(TABLE_DIR)
    pd.DataFrame(rows).to_csv(OUT_TABLE, index=False)
    print(f"[SAVED] {OUT_TABLE}")

    make_figure(fig_inputs, probes)
    print(f"[FIG S3] {FIG_S3}.png / .pdf")


if __name__ == "__main__":
    main()
