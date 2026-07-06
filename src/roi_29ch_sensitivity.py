#!/usr/bin/env python3
"""Sensitivity check for the anterior fast-duration behavioral ROI.

The primary behavioral models average fast spindle duration over the 47-channel
anterior ROI (the full spatially coherent uncorrected effect). A reviewer may ask
why the behavioral ROI is broader than the 29-channel cluster that survived
permutation correction. This script re-runs the accuracy-related TOVA models using
fast duration averaged over ONLY the 29-channel corrected cluster (the exact
cluster whose corrected p = 0.017 is reported in cluster_permutation_results.csv),
and compares the sign/significance pattern to the 47-channel ROI.

Output: prints a side-by-side comparison and writes
  output/tables/roi_29ch_sensitivity.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from behavioral_models import (
    TOVA_OUTCOMES,
    fit_roi_model,
    load_behavioral_dataset,
)
from cluster_permutation import largest_cluster_channels
from spindle_common import (
    CHAN_PATH,
    PREDICTORS,
    TABLE_DIR,
    data_matrix_for_metric,
    ensure_dir,
    load_channel_info,
    load_covariates,
    load_spindle_matrices,
)


def compute_fast_dur_29(cov: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Average fast spindle duration over the 29-channel corrected cluster."""
    import mne

    mne.set_log_level("ERROR")

    channels = load_channel_info(CHAN_PATH)
    adj = mne.channels.find_ch_adjacency(channels.info, ch_type="eeg")[0].toarray().astype(bool)
    matrices = load_spindle_matrices(cov, len(channels.labels), include_frequency=False)

    fast_dur = data_matrix_for_metric(matrices.fast, "Duration", matrices.sleep_mins)
    cluster = largest_cluster_channels(fast_dur, cov, adj, PREDICTORS)

    sub = fast_dur[:, cluster]
    vals = np.full(len(cov), np.nan)
    valid = ~np.all(np.isnan(sub), axis=1)
    vals[valid] = np.nanmean(sub[valid], axis=1)
    return pd.DataFrame({"id": cov["id"], "ant_fast_dur_29": vals}), len(cluster)


def main() -> None:
    cov = load_covariates()
    roi29, n_cluster = compute_fast_dur_29(cov)
    print(f"[INFO] corrected fast-duration cluster size = {n_cluster} channels "
          f"(expected 29 from cluster_permutation_results.csv)")

    data = load_behavioral_dataset()          # 47-ch ROI + all preprocessing/outlier removal
    data = data.merge(roi29, on="id", how="left")

    rows = []
    print("\n" + "=" * 82)
    print(f"{'Outcome':<18}{'ROI':<10}{'beta':>9}{'SE':>8}{'p':>9}  sig")
    print("=" * 82)
    for tova_label, tova_col in TOVA_OUTCOMES:
        for roi_label, roi_col in [("47-ch", "ant_fast_dur"), ("29-ch", "ant_fast_dur_29")]:
            fit, roi_z_col, n = fit_roi_model(data, tova_col, roi_col)
            beta = fit.params[roi_z_col]
            se = fit.bse[roi_z_col]
            p = fit.pvalues[roi_z_col]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            rows.append({"outcome": tova_label, "roi": roi_label, "n": n,
                         "beta": round(beta, 4), "SE": round(se, 4), "p": round(p, 4), "sig": sig})
            print(f"{tova_label:<18}{roi_label:<10}{beta:>9.3f}{se:>8.3f}{p:>9.4f}  {sig}")
        print("-" * 82)

    df = pd.DataFrame(rows)
    ensure_dir(TABLE_DIR)
    out = TABLE_DIR / "roi_29ch_sensitivity.csv"
    df.to_csv(out, index=False)
    print(f"[SAVED] {out}")

    # Verdict: does the accuracy-outcome pattern (sign + p<0.05) match across ROIs?
    print("\n=== PATTERN AGREEMENT (accuracy outcomes) ===")
    agree = True
    for tova_label, _ in TOVA_OUTCOMES:
        sub = df[df["outcome"] == tova_label]
        r47 = sub[sub["roi"] == "47-ch"].iloc[0]
        r29 = sub[sub["roi"] == "29-ch"].iloc[0]
        same_sign = np.sign(r47["beta"]) == np.sign(r29["beta"])
        same_sig = (r47["p"] < 0.05) == (r29["p"] < 0.05)
        ok = same_sign and same_sig
        if tova_label != "Reaction time" and not ok:
            agree = False
        print(f"  {tova_label:<18} 47-ch p={r47['p']:.4f} ({r47['sig']})  |  "
              f"29-ch p={r29['p']:.4f} ({r29['sig']})  -> "
              f"{'MATCH' if ok else 'DIFFERS'}")
    print(f"\nSAME PATTERN OF RESULTS (accuracy outcomes): {agree}")


if __name__ == "__main__":
    main()
