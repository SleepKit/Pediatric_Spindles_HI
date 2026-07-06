#!/usr/bin/env python3
"""Age x hypopnea-index interaction analysis for sleep spindle metrics.

Channel-wise test of whether age moderates the association between the hypopnea
index (HI) and each spindle metric on the canonical N=62 analytic sample:

    metric ~ age_c + logHI_c + age_c:logHI_c + sex

Age and log10(HI + 1) are mean-centered before forming the interaction term. The
interaction is assessed with a Freedman-Lane residual permutation test (reduced
model: age_c + logHI_c + sex) and cluster correction over the channel adjacency.

Outputs:
- output/tables/age_hi_interaction_clusters.csv : per-metric cluster summary
- output/Fig_S2.png / output/Fig_S2.pdf       : Supplementary Figure S2
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import statsmodels.api as sm

from spindle_common import (
    CHAN_PATH,
    COV_PATH,
    PROJECT,
    SPINDLE_DIR,
    TABLE_DIR,
    data_matrix_for_metric,
    load_channel_info,
    load_covariates,
    load_spindle_matrices,
)

N_PERM = 5000
RANDOM_SEED = 20260512
MIN_N = 40
# Channel-forming |t| threshold, applied identically to the observed and the
# permutation-null t-maps and matching the main HI cluster test
# (cluster_permutation.py, T_THRESH = 2.0); |t| > 2.0 is the two-sided p = 0.05
# critical value at the interaction model's residual df.
T_THRESH = 2.0

FIG_S2 = PROJECT / "output/Fig_S2"  # stem; saved as .png and .pdf at 600 dpi
CLUSTER_TABLE = TABLE_DIR / "age_hi_interaction_clusters.csv"

# (band, Metric, label, set). The manuscript reports only the primary slow/fast
# duration/count/density question plus slow amplitude; frequency and fast
# amplitude are computed for completeness and documented in the table.
METRICS = [
    ("slow", "Density", "Slow density", "primary"),
    ("slow", "Count", "Slow count", "primary"),
    ("slow", "Duration", "Slow duration", "primary"),
    ("slow", "Amplitude", "Slow amplitude", "primary"),
    ("fast", "Density", "Fast density", "primary"),
    ("fast", "Count", "Fast count", "primary"),
    ("fast", "Duration", "Fast duration", "primary"),
    ("fast", "Amplitude", "Fast amplitude", "secondary"),
    ("slow", "Frequency", "Slow peak frequency", "secondary"),
    ("fast", "Frequency", "Fast peak frequency", "secondary"),
]
FIGURE_METRIC = ("slow", "Amplitude")  # the metric shown in Figure S2


def prepare_covariates() -> pd.DataFrame:
    """Canonical N=62 covariates with interaction predictors added."""
    cov = load_covariates(COV_PATH)  # pins to sample_ids_N62, no useable filter
    cov["sex"] = cov["gender_bin"]
    cov["ageXHI"] = cov["age_c"] * cov["logHI_c"]
    return cov


def metric_matrix(matrices, band: str, metric: str) -> np.ndarray:
    data_dict = matrices.slow if band == "slow" else matrices.fast
    return data_matrix_for_metric(data_dict, metric, matrices.sleep_mins)


def channel_interaction(y_mat: np.ndarray, cov: pd.DataFrame):
    """Per-channel OLS interaction coefficient (beta, t, p) for age_c:logHI_c."""
    base = cov[["age_c", "logHI_c", "ageXHI", "sex"]].to_numpy(dtype=float)
    n_ch = y_mat.shape[1]
    beta = np.full(n_ch, np.nan)
    tval = np.full(n_ch, np.nan)
    pval = np.full(n_ch, np.nan)
    for ch in range(n_ch):
        y = y_mat[:, ch].astype(float)
        valid = ~np.isnan(y) & ~np.isnan(base).any(axis=1)
        if valid.sum() < MIN_N:
            continue
        fit = sm.OLS(y[valid], sm.add_constant(base[valid], has_constant="add")).fit()
        beta[ch], tval[ch], pval[ch] = fit.params[3], fit.tvalues[3], fit.pvalues[3]
    return beta, tval, pval


def cluster_components(supra: np.ndarray, adjacency: np.ndarray) -> list[list[int]]:
    nodes = set(np.where(supra)[0].tolist())
    visited: set[int] = set()
    out: list[list[int]] = []
    for seed in sorted(nodes):
        if seed in visited:
            continue
        stack, comp = [seed], []
        visited.add(seed)
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb in np.where(adjacency[node])[0]:
                if nb in nodes and nb not in visited:
                    visited.add(int(nb))
                    stack.append(int(nb))
        out.append(sorted(comp))
    return out


def cluster_correct(y_mat, cov, obs_t, obs_p, adjacency):
    """Freedman-Lane cluster correction for the age_c:logHI_c interaction.

    Returns the observed clusters (sorted by mass) with permutation p-values.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_ch = y_mat.shape[1]
    full = sm.add_constant(
        cov[["age_c", "logHI_c", "ageXHI", "sex"]].to_numpy(dtype=float), has_constant="add"
    )
    reduced = sm.add_constant(
        cov[["age_c", "logHI_c", "sex"]].to_numpy(dtype=float), has_constant="add"
    )
    x_ok = ~np.isnan(full).any(axis=1) & ~np.isnan(reduced).any(axis=1)

    fitted_red, resid_red, x_full, valids = [], [], [], []
    for ch in range(n_ch):
        y = y_mat[:, ch].astype(float)
        valid = x_ok & ~np.isnan(y)
        if valid.sum() < MIN_N:
            fitted_red.append(None)
            resid_red.append(None)
            x_full.append(None)
            valids.append(None)
            continue
        rfit = sm.OLS(y[valid], reduced[valid]).fit()
        fitted_red.append(np.asarray(rfit.fittedvalues))
        resid_red.append(np.asarray(rfit.resid))
        x_full.append(full[valid])
        valids.append(valid)

    obs_supra = np.isfinite(obs_t) & (np.abs(obs_t) > T_THRESH)
    clusters = []
    for comp in cluster_components(obs_supra, adjacency):
        vals = obs_t[comp]
        clusters.append(
            {
                "n_channels": len(comp),
                "channels_0idx": comp,
                "mass_abs": float(np.nansum(np.abs(vals))),
                "mean_t": float(np.nanmean(vals)),
                "min_p": float(np.nanmin(obs_p[comp])),
            }
        )

    # Freedman-Lane null. Draw ONE global subject permutation per iteration and
    # apply the same subject ordering across all channels (restricted to each
    # channel's valid rows, preserving relative order). Permuting each channel
    # independently would make the null t-maps spatially white and collapse the
    # null cluster masses, inflating significance -- the same correctness
    # requirement enforced in cluster_permutation.py.
    n_sub = y_mat.shape[0]
    max_masses = np.zeros(N_PERM)
    for p in range(N_PERM):
        glob = rng.permutation(n_sub)
        t_perm = np.full(n_ch, np.nan)
        for ch in range(n_ch):
            if resid_red[ch] is None:
                continue
            order_local = np.argsort(np.argsort(glob[valids[ch]]))
            y_perm = fitted_red[ch] + resid_red[ch][order_local]
            try:
                t_perm[ch] = sm.OLS(y_perm, x_full[ch]).fit().tvalues[3]
            except Exception:
                t_perm[ch] = np.nan
        supra = np.isfinite(t_perm) & (np.abs(t_perm) > T_THRESH)
        comps = cluster_components(supra, adjacency)
        max_masses[p] = max(
            [float(np.nansum(np.abs(t_perm[c]))) for c in comps] or [0.0]
        )

    for cl in clusters:
        cl["p_cluster"] = float((np.sum(max_masses >= cl["mass_abs"]) + 1) / (N_PERM + 1))
    clusters.sort(key=lambda d: d["mass_abs"], reverse=True)
    return clusters


def fit_roi_model(roi_values, cov):
    df = cov[["age_years", "age_c", "logHI", "logHI_c", "ageXHI", "sex"]].copy()
    df["roi"] = roi_values
    df = df.dropna()
    fit = sm.OLS.from_formula("roi ~ age_c + logHI_c + ageXHI + sex", df).fit()
    return fit, df


def make_figure_s2(channels, t_slow_amp, cluster_channels, roi_fit, roi_df):
    """Two-panel SI Figure S2: interaction t-map (a) and age-stratified curves (b)."""
    info = channels.info
    fig, (ax_map, ax_slope) = plt.subplots(1, 2, figsize=(10.4, 4.3))

    # (a) Age x HI interaction t-map with the surviving cluster marked.
    t_lim = max(2.0, float(np.nanmax(np.abs(t_slow_amp))))
    im, _ = mne.viz.plot_topomap(
        t_slow_amp, info, axes=ax_map, show=False, cmap="RdBu_r",
        vlim=(-t_lim, t_lim), sensors=True, contours=0,
    )
    if cluster_channels:
        xy = np.asarray([info["chs"][i]["loc"][:2] for i in cluster_channels])
        ax_map.scatter(xy[:, 0], xy[:, 1], s=14, c="k", marker="o", linewidths=0, zorder=10)
    cbar = fig.colorbar(im, ax=ax_map, shrink=0.7)
    cbar.set_label("Age × HI interaction (t)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    ax_map.set_title("(a)", loc="left", fontsize=12, fontweight="bold")

    # (b) Model-estimated amplitude across HI at mean and mean +/- 1 SD of age
    # (Aiken & West simple-slope probing; data-grounded rather than arbitrary).
    age_mean = roi_df["age_years"].mean()
    age_sd = roi_df["age_years"].std(ddof=1)
    loghi_mean = roi_df["logHI"].mean()
    loghi_raw = np.linspace(roi_df["logHI"].min(), roi_df["logHI"].max(), 100)
    sex_mean = roi_df["sex"].mean()
    probes = [
        (age_mean - age_sd, "−1 SD"),
        (age_mean, "Mean"),
        (age_mean + age_sd, "+1 SD"),
    ]
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
        frame = roi_fit.get_prediction(pred).summary_frame(alpha=0.05)
        ax_slope.plot(
            loghi_raw, frame["mean"], color=color, lw=2.2,
            label=f"{tag} ({age_value:.1f} y)",
        )
        ax_slope.fill_between(
            loghi_raw,
            frame["mean_ci_lower"].to_numpy(dtype=float),
            frame["mean_ci_upper"].to_numpy(dtype=float),
            color=color, alpha=0.15, linewidth=0,
        )
    ax_slope.set_xlabel(r"Hypopnea index  [$\log_{10}$(HI + 1)]", fontsize=11)
    ax_slope.set_ylabel("Predicted slow spindle amplitude (µV)", fontsize=11)
    leg = ax_slope.legend(frameon=False, title="Age", fontsize=10)
    leg.get_title().set_fontsize(10)
    ax_slope.spines["top"].set_visible(False)
    ax_slope.spines["right"].set_visible(False)
    ax_slope.tick_params(labelsize=10)
    ax_slope.set_title("(b)", loc="left", fontsize=12, fontweight="bold")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG_S2}.{ext}", dpi=600)
    plt.close(fig)


def main() -> None:
    mne.set_log_level("ERROR")
    channels = load_channel_info(CHAN_PATH)
    n_ch = len(channels.labels)
    adjacency_sparse, _ = mne.channels.find_ch_adjacency(channels.info, "eeg")
    adjacency = adjacency_sparse.toarray().astype(bool)

    cov = prepare_covariates()
    matrices = load_spindle_matrices(cov, n_ch, SPINDLE_DIR, include_frequency=True)
    if matrices.missing_subjects:
        raise SystemExit(f"missing spindle files: {matrices.missing_subjects}")
    print(f"[SAMPLE] N = {len(cov)} | missing spindle files: none")

    summary_rows = []
    fig_inputs = {}
    for band, metric, label, group in METRICS:
        y_mat = metric_matrix(matrices, band, metric)
        beta, tval, pval = channel_interaction(y_mat, cov)
        clusters = cluster_correct(y_mat, cov, tval, pval, adjacency)
        corrected = [c for c in clusters if c["p_cluster"] < 0.05]
        best = clusters[0] if clusters else None
        summary_rows.append(
            {
                "band": band,
                "metric": metric,
                "label": label,
                "set": group,
                "n_channels_p05": int(np.nansum(np.isfinite(pval) & (pval < 0.05))),
                "best_cluster_n": best["n_channels"] if best else 0,
                "best_cluster_mass": round(best["mass_abs"], 3) if best else np.nan,
                "best_cluster_mean_t": round(best["mean_t"], 3) if best else np.nan,
                "best_cluster_p": best["p_cluster"] if best else np.nan,
                "cluster_corrected": bool(corrected),
            }
        )
        print(
            f"[{label:>20}] p<.05 ch={summary_rows[-1]['n_channels_p05']:>3} | "
            + (
                f"cluster {best['n_channels']} ch, p={best['p_cluster']:.4f}, "
                f"mean t={best['mean_t']:+.2f}"
                if best
                else "no suprathreshold cluster"
            )
        )
        if (band, metric) == FIGURE_METRIC:
            fig_inputs = {"t": tval, "clusters": clusters, "y_mat": y_mat}

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(CLUSTER_TABLE, index=False)
    print(f"[SAVED] {CLUSTER_TABLE}")

    # Figure S2 visualizes the largest slow-amplitude interaction cluster as an
    # exploratory pattern. After the permutation fix this cluster is marginal and
    # does NOT survive correction; the figure/text report the corrected p as-is.
    largest = fig_inputs["clusters"][0]
    cluster_channels = largest["channels_0idx"]
    roi_values = np.nanmean(fig_inputs["y_mat"][:, cluster_channels], axis=1)
    roi_fit, roi_df = fit_roi_model(roi_values, cov)
    make_figure_s2(channels, fig_inputs["t"], cluster_channels, roi_fit, roi_df)
    print(
        f"[FIG S2] {FIG_S2}\n"
        f"         largest cluster {len(cluster_channels)} ch, p={largest['p_cluster']:.4f}, "
        f"mean t={largest['mean_t']:+.2f} | "
        f"ROI N={int(roi_fit.nobs)}, beta={roi_fit.params['ageXHI']:.3g}, "
        f"t={roi_fit.tvalues['ageXHI']:.2f}, p={roi_fit.pvalues['ageXHI']:.4f}"
    )


def figure_only() -> None:
    """Regenerate Figure S2 WITHOUT the permutation test (fast; seconds).

    The 5000-permutation cluster test only produces the corrected cluster p-value,
    which is already persisted in age_hi_interaction_clusters.csv. Figure S2 needs
    only the OBSERVED interaction t-map and its largest cluster's membership (both
    derived from the observed data), so this path skips permutations entirely —
    keeping figure generation separate from the (slow) analysis.
    """
    mne.set_log_level("ERROR")
    channels = load_channel_info(CHAN_PATH)
    n_ch = len(channels.labels)
    adjacency = mne.channels.find_ch_adjacency(channels.info, "eeg")[0].toarray().astype(bool)
    cov = prepare_covariates()
    matrices = load_spindle_matrices(cov, n_ch, SPINDLE_DIR, include_frequency=True)
    band, metric = FIGURE_METRIC
    y_mat = metric_matrix(matrices, band, metric)
    _beta, tval, _pval = channel_interaction(y_mat, cov)
    supra = np.isfinite(tval) & (np.abs(tval) > T_THRESH)
    comps = cluster_components(supra, adjacency)
    comps.sort(key=lambda c: float(np.nansum(np.abs(tval[c]))), reverse=True)
    cluster_channels = comps[0] if comps else []
    roi_values = np.nanmean(y_mat[:, cluster_channels], axis=1)
    roi_fit, roi_df = fit_roi_model(roi_values, cov)
    make_figure_s2(channels, tval, cluster_channels, roi_fit, roi_df)
    print(f"[FIG S2 only] {FIG_S2}.png / .pdf | observed cluster {len(cluster_channels)} ch "
          f"(corrected p read separately from {CLUSTER_TABLE.name})")


if __name__ == "__main__":
    import sys
    if "--figure-only" in sys.argv:
        figure_only()
    else:
        main()
