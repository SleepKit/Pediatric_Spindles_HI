#!/usr/bin/env python3
"""Cluster-based permutation test with covariate-adjusted regression.

Implements a statistically correct family-wise-error-corrected cluster test for
the per-channel OLS effect of logHI_c in the model:

    metric ~ logHI_c + age_c + gender_bin

MNE's built-in 1-sample / F permutation tests do NOT support covariate-adjusted
regression, so they would be statistically wrong here. Instead we use:

- Adjacency: MNE Delaunay triangulation on the montage
  (``mne.channels.find_ch_adjacency``).
- Observed statistic: channel-wise OLS t-value for logHI_c (predictor index 1 in
  the design matrix [const, logHI_c, age_c, gender_bin]). Cluster-forming
  threshold |t| > 2.0. A cluster is a connected component (over the adjacency
  graph) of suprathreshold channels; cluster mass = sum of |t| over the
  component.
- Null distribution: Freedman-Lane residual permutation. The reduced model
  (age_c + gender_bin, dropping logHI_c) is fit, its residuals are permuted and
  added back to the reduced fitted values, the FULL model is refit, and the
  logHI_c t-map is recomputed. The MAX cluster mass per permutation is recorded.
- p-value for an observed cluster = (#{null_max_mass >= obs_mass} + 1) / (N + 1).

The vectorized Freedman-Lane math is ported from
``docs/followup_analyses/rerun_with_subject80.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spindle_common import (
    PREDICTORS,
    TABLE_DIR,
    channel_regression,
    data_matrix_for_metric,
    ensure_dir,
    load_channel_info,
    load_covariates,
    load_spindle_matrices,
)

T_THRESH = 2.0
N_PERM = 5000
SEED = 20260607
MIN_N = 30


def cluster_components(supra: np.ndarray, adjacency: np.ndarray) -> list[list[int]]:
    """Connected components of suprathreshold channels over the adjacency graph."""
    nodes = set(np.where(supra)[0].tolist())
    visited: set[int] = set()
    out: list[list[int]] = []
    for s in sorted(nodes):
        if s in visited:
            continue
        stack = [s]
        visited.add(s)
        comp: list[int] = []
        while stack:
            n = stack.pop()
            comp.append(n)
            for nb in np.where(adjacency[n])[0]:
                nb = int(nb)
                if nb in nodes and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        out.append(sorted(comp))
    return out


def largest_cluster_channels(
    data_matrix: np.ndarray,
    cov_df: pd.DataFrame,
    adjacency: np.ndarray,
    predictors: list[str] | None = None,
    t_thresh: float = T_THRESH,
) -> list[int]:
    """Channel indices of the top observed cluster for the logHI_c effect.

    Membership only (no permutation): runs the channel-wise covariate-adjusted
    regression, thresholds the logHI_c t-map at |t| > ``t_thresh``, and returns the
    connected component (over ``adjacency``) with the greatest cluster mass
    (sum of |t|). This is the SAME observed t-map, adjacency, and cluster-forming
    rule that ``cluster_permutation_test`` uses, so the returned cluster is exactly
    the one whose corrected p-value is reported in cluster_permutation_results.csv.

    Shared by ``compute_rois_corrected`` (to build the cluster-corrected peak-
    frequency ROI) and ``generate_manuscript_figures`` (topomap cluster overlays)
    so cluster membership is defined in one place.
    """
    predictors = predictors or PREDICTORS
    _, t, _ = channel_regression(data_matrix, cov_df, predictors)
    supra = np.isfinite(t) & (np.abs(t) > t_thresh)
    comps = cluster_components(supra, adjacency)
    if not comps:
        return []
    comps.sort(key=lambda c: float(np.nansum(np.abs(t[c]))), reverse=True)
    return comps[0]


def _channel_t_maps(
    data_matrix: np.ndarray,
    cov: np.ndarray,
    rng: np.random.Generator,
    n_perm: int,
    t_thresh: float,
    min_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (observed t-map (nCh,), null t-maps (nCh x n_perm)).

    ``cov`` is (nSub x 3) with columns ordered [logHI_c, age_c, gender_bin] so the
    full design matrix is [const, logHI_c, age_c, gender_bin] and the effect of
    interest is coefficient index 1.
    """
    n_sub, n_ch = data_matrix.shape
    obs_t = np.full(n_ch, np.nan)
    null_t = np.full((n_ch, n_perm), np.nan)
    base_ok = ~np.isnan(cov).any(axis=1)

    # Cluster inference requires the SAME subject relabeling across all channels
    # within a permutation, otherwise the null t-maps become spatially white and
    # null cluster masses collapse (inflating significance). Draw one global
    # permutation of the full subject set per iteration and derive each channel's
    # within-valid-subject ordering from that shared relabeling.
    global_orders = np.argsort(rng.random((n_perm, n_sub)), axis=1)
    ranks = np.empty((n_perm, n_sub), dtype=np.int64)
    np.put_along_axis(
        ranks, global_orders, np.broadcast_to(np.arange(n_sub), (n_perm, n_sub)), axis=1
    )

    for ch in range(n_ch):
        y = data_matrix[:, ch]
        valid = base_ok & ~np.isnan(y)
        nv = int(valid.sum())
        if nv < min_n:
            continue
        yv = y[valid]
        # Full design: const, logHI_c, age_c, gender_bin
        Xf = np.column_stack([np.ones(nv), cov[valid, 0], cov[valid, 1], cov[valid, 2]])
        # Reduced design (Freedman-Lane): const, age_c, gender_bin
        Xr = np.column_stack([np.ones(nv), cov[valid, 1], cov[valid, 2]])
        XtX = Xf.T @ Xf
        try:
            A = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            continue
        P = A @ Xf.T               # (4 x nv) pseudo-inverse
        dof = nv - 4
        if dof <= 0:
            continue
        A11 = A[1, 1]

        # Observed t for logHI_c
        beta = P @ yv
        rss = float(yv @ yv - beta @ (Xf.T @ yv))
        se1 = np.sqrt(rss / dof * A11)
        obs_t[ch] = beta[1] / se1 if se1 > 0 else np.nan

        # Reduced fit for Freedman-Lane
        Ar = np.linalg.inv(Xr.T @ Xr)
        beta_r = Ar @ (Xr.T @ yv)
        fitted_r = Xr @ beta_r
        resid_r = yv - fitted_r

        # Vectorized permutations of reduced residuals, using the shared global
        # relabeling restricted to this channel's valid subjects. argsort of the
        # global ranks over the valid columns gives the within-valid ordering that
        # is consistent with every other channel in the same permutation.
        order = np.argsort(ranks[:, valid], axis=1)        # (n_perm x nv)
        Yp = fitted_r[None, :] + resid_r[order]            # (n_perm x nv)
        beta1 = Yp @ P[1]                                  # (n_perm,)
        XtY = Yp @ Xf                                       # (n_perm x 4)
        betas = XtY @ A.T                                   # (n_perm x 4)
        yty = np.einsum("pi,pi->p", Yp, Yp)
        rss_p = yty - np.einsum("pj,pj->p", betas, XtY)
        se1_p = np.sqrt(np.maximum(rss_p, 1e-12) / dof * A11)
        null_t[ch] = beta1 / se1_p

    return obs_t, null_t


def cluster_permutation_test(
    data_matrix: np.ndarray,
    cov_df: pd.DataFrame,
    adjacency: np.ndarray,
    predictors: list[str] | None = None,
    t_thresh: float = T_THRESH,
    n_perm: int = N_PERM,
    seed: int = SEED,
    min_n: int = MIN_N,
) -> list[dict]:
    """Cluster-based permutation test for the logHI_c effect.

    Parameters
    ----------
    data_matrix : ndarray (nSub x nCh)
        Metric values, built with ``data_matrix_for_metric``.
    cov_df : DataFrame
        Covariate frame from ``load_covariates``. Predictor columns are pulled via
        ``predictors``; the first column (index 1 in the design matrix) is the
        effect of interest.
    adjacency : ndarray (nCh x nCh) bool
        Channel adjacency (MNE Delaunay).
    predictors : list[str]
        Defaults to PREDICTORS = ["logHI_c", "age_c", "gender_bin"].

    Returns
    -------
    list[dict]
        Observed clusters sorted by mass descending. Each dict has keys
        ``channels``, ``n``, ``mass``, ``peak_t``, ``p``.
    """
    predictors = predictors or PREDICTORS
    cov = cov_df[predictors].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    obs_t, null_t = _channel_t_maps(
        data_matrix, cov, rng, n_perm, t_thresh, min_n
    )

    # Observed clusters
    supra = np.isfinite(obs_t) & (np.abs(obs_t) > t_thresh)
    comps = cluster_components(supra, adjacency)
    clusters: list[dict] = []
    for comp in comps:
        mass = float(np.nansum(np.abs(obs_t[comp])))
        peak_t = float(obs_t[comp][np.nanargmax(np.abs(obs_t[comp]))])
        clusters.append(
            {"channels": comp, "n": len(comp), "mass": mass, "peak_t": peak_t}
        )

    # Null max-mass distribution
    max_masses = np.zeros(n_perm)
    for p in range(n_perm):
        tp = null_t[:, p]
        sp = np.isfinite(tp) & (np.abs(tp) > t_thresh)
        cc = cluster_components(sp, adjacency)
        max_masses[p] = max(
            [float(np.nansum(np.abs(tp[c]))) for c in cc] or [0.0]
        )

    for cl in clusters:
        cl["p"] = float((np.sum(max_masses >= cl["mass"]) + 1) / (n_perm + 1))

    clusters.sort(key=lambda d: d["mass"], reverse=True)
    return clusters


# Metrics for the canonical stats table.
# (metric_label, band, metric)
METRICS = [
    ("Fast duration", "fast", "Duration"),       # HEADLINE
    ("Slow peak frequency", "slow", "Frequency"),
    ("Slow amplitude", "slow", "Amplitude"),
    ("Slow duration", "slow", "Duration"),
    ("Slow density", "slow", "Density"),
    ("Fast density", "fast", "Density"),
]


def main() -> None:
    import mne

    mne.set_log_level("ERROR")

    channels = load_channel_info()
    info = channels.info
    n_ch = len(channels.labels)

    adj_sparse, _ = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    adjacency = adj_sparse.toarray().astype(bool)

    cov = load_covariates()
    matrices = load_spindle_matrices(cov, n_ch, include_frequency=True)
    sleep_mins = matrices.sleep_mins
    band_data = {"slow": matrices.slow, "fast": matrices.fast}

    rows = []
    print("\n" + "=" * 86)
    print(f"{'Metric':<22}{'band':<6}{'metric':<11}{'n_ch':>5}{'mass':>11}"
          f"{'peak_t':>9}{'p':>10}")
    print("=" * 86)
    for metric_label, band, metric in METRICS:
        data_matrix = data_matrix_for_metric(band_data[band], metric, sleep_mins)
        clusters = cluster_permutation_test(data_matrix, cov, adjacency)
        top = clusters[0] if clusters else None
        if top is None:
            rows.append(
                {
                    "metric_label": metric_label,
                    "band": band,
                    "metric": metric,
                    "n_channels": 0,
                    "cluster_mass": np.nan,
                    "peak_t": np.nan,
                    "p_value": np.nan,
                }
            )
            print(f"{metric_label:<22}{band:<6}{metric:<11}{'-':>5}"
                  f"{'no suprathreshold cluster':>40}")
            continue
        star = "*" if top["p"] < 0.05 else " "
        rows.append(
            {
                "metric_label": metric_label,
                "band": band,
                "metric": metric,
                "n_channels": top["n"],
                "cluster_mass": round(top["mass"], 4),
                "peak_t": round(top["peak_t"], 4),
                "p_value": round(top["p"], 6),
            }
        )
        print(f"{metric_label:<22}{band:<6}{metric:<11}{top['n']:>5}"
              f"{top['mass']:>11.2f}{top['peak_t']:>9.2f}"
              f"{top['p']:>9.4f}{star}")
    print("=" * 86)

    ensure_dir(TABLE_DIR)
    out_path = TABLE_DIR / "cluster_permutation_results.csv"
    pd.DataFrame(
        rows,
        columns=[
            "metric_label",
            "band",
            "metric",
            "n_channels",
            "cluster_mass",
            "peak_t",
            "p_value",
        ],
    ).to_csv(out_path, index=False)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
