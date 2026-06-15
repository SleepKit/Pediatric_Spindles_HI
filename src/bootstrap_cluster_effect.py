#!/usr/bin/env python3
"""Subject-level bootstrap of the cluster-mean HI effect on fast spindle duration."""
from __future__ import annotations

import mne
import numpy as np
import pandas as pd

from cluster_permutation import cluster_permutation_test
from spindle_common import (
    PREDICTORS,
    TABLE_DIR,
    data_matrix_for_metric,
    load_channel_info,
    load_covariates,
    load_spindle_matrices,
)

N_BOOT = 10000
SEED = 20260615
CI_ALPHA = 0.05
OUT = TABLE_DIR / "bootstrap_fast_duration_effect.csv"
DRAWS = TABLE_DIR / "bootstrap_fast_duration_draws.npy"


def _std_beta(y: np.ndarray, cov: np.ndarray) -> float:
    valid = np.isfinite(y) & np.isfinite(cov).all(axis=1)
    if valid.sum() < 10:
        return np.nan
    yv, Xv = y[valid], cov[valid]
    if yv.std() == 0:
        return np.nan
    yz = (yv - yv.mean()) / yv.std()
    cols = [(c - c.mean()) / c.std() if c.std() > 0 else c - c.mean() for c in Xv.T]
    Xz = np.column_stack([np.ones(valid.sum()), *cols])
    beta = np.linalg.lstsq(Xz, yz, rcond=None)[0]
    return float(beta[1])


def main() -> None:
    mne.set_log_level("ERROR")
    channels = load_channel_info()
    n_ch = len(channels.labels)
    adjacency = mne.channels.find_ch_adjacency(channels.info, "eeg")[0].toarray().astype(bool)

    cov_df = load_covariates()
    matrices = load_spindle_matrices(cov_df, n_ch, include_frequency=True)
    data = data_matrix_for_metric(matrices.fast, "Duration", matrices.sleep_mins)

    cluster_ch = cluster_permutation_test(data, cov_df, adjacency, n_perm=200)[0]["channels"]
    dur_cluster = np.nanmean(data[:, cluster_ch], axis=1)
    cov = cov_df[PREDICTORS].to_numpy(dtype=float)

    obs = _std_beta(dur_cluster, cov)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(dur_cluster), size=(N_BOOT, len(dur_cluster)))
    boot = np.array([_std_beta(dur_cluster[ix], cov[ix]) for ix in idx])
    boot = boot[np.isfinite(boot)]

    lo = float(np.percentile(boot, 100 * CI_ALPHA / 2))
    hi = float(np.percentile(boot, 100 * (1 - CI_ALPHA / 2)))
    p_dir = float(np.mean(boot < 0))

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "n_cluster_channels": len(cluster_ch),
        "std_beta_observed": obs,
        "boot_median": float(np.median(boot)),
        "ci_lo": lo,
        "ci_hi": hi,
        "prop_negative": p_dir,
        "n_boot": len(boot),
    }]).to_csv(OUT, index=False)
    np.save(DRAWS, boot)
    print(f"beta={obs:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  dir={p_dir:.3f}")


if __name__ == "__main__":
    main()
