#!/usr/bin/env python3
"""Leave-one-subject-out sensitivity for the fast-duration cluster."""
from __future__ import annotations

import mne
import numpy as np
import pandas as pd

from cluster_permutation import cluster_permutation_test
from spindle_common import (
    TABLE_DIR,
    data_matrix_for_metric,
    load_channel_info,
    load_covariates,
    load_spindle_matrices,
)

OUT = TABLE_DIR / "loo_fast_duration_sensitivity.csv"


def main() -> None:
    mne.set_log_level("ERROR")
    channels = load_channel_info()
    n_ch = len(channels.labels)
    adjacency = mne.channels.find_ch_adjacency(channels.info, "eeg")[0].toarray().astype(bool)

    cov = load_covariates()
    matrices = load_spindle_matrices(cov, n_ch, include_frequency=True)
    data = data_matrix_for_metric(matrices.fast, "Duration", matrices.sleep_mins)
    ids = cov["id"].to_numpy()
    n_sub = len(ids)

    base = cluster_permutation_test(data, cov, adjacency)
    base_p = base[0]["p"] if base else np.nan

    rows = []
    for i in range(n_sub):
        keep = np.arange(n_sub) != i
        clusters = cluster_permutation_test(data[keep], cov.iloc[keep].reset_index(drop=True), adjacency)
        top = clusters[0] if clusters else None
        rows.append({
            "dropped_id": ids[i],
            "dropped_hi": float(cov.iloc[i]["overall_hi"]),
            "top_cluster_p": top["p"] if top else np.nan,
            "top_cluster_nch": top["n"] if top else 0,
            "peak_t": top["peak_t"] if top else np.nan,
        })

    df = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"full p={base_p:.4f}  LOO p range {df.top_cluster_p.min():.4f}-{df.top_cluster_p.max():.4f}"
          f"  sig {(df.top_cluster_p < 0.05).sum()}/{n_sub}")


if __name__ == "__main__":
    main()
