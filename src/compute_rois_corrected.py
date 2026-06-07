#!/usr/bin/env python3
"""Compute corrected spindle ROI values for behavioral models.

The pipeline keeps subjects without spindle files as missing rows, while valid
subject/channel combinations with no detected spindles get count=0.

Output:
  datasets/original/roi_values_corrected.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cluster_permutation import largest_cluster_channels
from spindle_common import (
    CHAN_PATH,
    COV_PATH,
    PREDICTORS,
    ROI_PATH,
    SPINDLE_DIR,
    channel_regression,
    data_matrix_for_metric,
    ensure_dir,
    load_channel_info,
    load_covariates,
    load_spindle_matrices,
)


ADJACENCY_DIST = 0.035


def find_roi_channels(
    data_dict: dict[str, np.ndarray],
    metric: str,
    cov_df: pd.DataFrame,
    sleep_mins: np.ndarray,
    adjacency: np.ndarray,
    points_x: np.ndarray,
) -> list[int]:
    """Find bilateral data-driven ROI cluster from channel-wise HI regression."""
    data_matrix = data_matrix_for_metric(data_dict, metric, sleep_mins)
    _, _, p_vals = channel_regression(data_matrix, cov_df, PREDICTORS)

    sig_seeds = np.where(p_vals < 0.05)[0]
    seeds_meeting_contiguity = [
        ch for ch in sig_seeds
        if len([n for n in np.where(adjacency[ch])[0] if p_vals[n] < 0.05]) >= 2
    ]

    final_roi = set(seeds_meeting_contiguity)
    for ch in seeds_meeting_contiguity:
        neighbors = np.where(adjacency[ch])[0]
        final_roi.update([n for n in neighbors if 0.05 <= p_vals[n] < 0.08])

    roi_list = sorted(final_roi)
    if roi_list:
        left = np.sum(points_x[roi_list] < -0.015)
        right = np.sum(points_x[roi_list] > 0.015)
        if left < 2 or right < 2:
            roi_list = []

    return roi_list


def main() -> None:
    import mne

    mne.set_log_level("ERROR")

    channels = load_channel_info(CHAN_PATH)
    cov = load_covariates(COV_PATH)
    # include_frequency=True so the slow-spindle Frequency matrix is available for
    # the cluster-corrected peak-frequency ROI below.
    matrices = load_spindle_matrices(
        cov, len(channels.labels), SPINDLE_DIR, include_frequency=True
    )
    adjacency = (channels.distance_matrix < ADJACENCY_DIST) & (channels.distance_matrix > 0)

    print(f"[INFO] Loaded {len(cov)} subjects, {len(channels.labels)} channels")
    if matrices.missing_subjects:
        print(f"[INFO] Missing spindle files: {matrices.missing_subjects}")

    # Three exploratory ROIs defined from the uncorrected channel-wise map via
    # distance-based contiguity (find_roi_channels).
    tasks = [
        ("Slow", matrices.slow, "Amplitude", "pos_slow_amp"),
        ("Slow", matrices.slow, "Duration", "ant_slow_dur"),
        ("Fast", matrices.fast, "Duration", "ant_fast_dur"),
    ]

    results = pd.DataFrame({"id": cov["id"]})

    print("\n=== ROI Identification ===")
    for band, data_dict, metric, col_name in tasks:
        roi_channels = find_roi_channels(
            data_dict,
            metric,
            cov,
            matrices.sleep_mins,
            adjacency,
            channels.x,
        )

        if roi_channels:
            data_matrix = data_matrix_for_metric(data_dict, metric, matrices.sleep_mins)
            roi_values = np.full(len(cov), np.nan)
            roi_matrix = data_matrix[:, roi_channels]
            valid_rows = ~np.all(np.isnan(roi_matrix), axis=1)
            roi_values[valid_rows] = np.nanmean(roi_matrix[valid_rows], axis=1)
            results[col_name] = roi_values
            print(f"  {col_name}: {len(roi_channels)} channels - INCLUDED ({band} {metric})")
        else:
            print(f"  {col_name}: 0 channels - EXCLUDED ({band} {metric})")

    # Fourth ROI: anterior slow-spindle peak frequency. Unlike the three ROIs
    # above, this is the cluster-CORRECTED cluster (largest |t|>2.0 component over
    # MNE Delaunay adjacency, the same membership reported in
    # cluster_permutation_results.csv), averaged over the slow Frequency matrix.
    adj_mne = mne.channels.find_ch_adjacency(channels.info, ch_type="eeg")[0].toarray().astype(bool)
    freq_roi = largest_cluster_channels(
        data_matrix_for_metric(matrices.slow, "Frequency", matrices.sleep_mins),
        cov,
        adj_mne,
        PREDICTORS,
    )
    if freq_roi:
        freq_matrix = matrices.slow["Frequency"][:, freq_roi]
        roi_values = np.full(len(cov), np.nan)
        valid_rows = ~np.all(np.isnan(freq_matrix), axis=1)
        roi_values[valid_rows] = np.nanmean(freq_matrix[valid_rows], axis=1)
        results["ant_slow_peakfreq"] = roi_values
        print(f"  ant_slow_peakfreq: {len(freq_roi)} channels - INCLUDED (Slow Frequency, cluster-corrected)")
    else:
        print("  ant_slow_peakfreq: 0 channels - EXCLUDED (Slow Frequency)")

    ensure_dir(ROI_PATH.parent)
    results.to_csv(ROI_PATH, index=False)
    cols_with_data = [c for c in results.columns if c != "id" and results[c].notna().any()]
    print(f"\n[SAVED] {ROI_PATH}")
    print(f"  ROIs included: {cols_with_data}")
    print(f"  N subjects: {len(results)}")


if __name__ == "__main__":
    main()
