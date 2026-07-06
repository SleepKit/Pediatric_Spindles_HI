#!/usr/bin/env python3
"""ANALYSIS step for the main manuscript figures.

Computes every quantity the Figure 2-5 plots need (normative topographies,
channel-wise HI regression maps, ROI / permutation-cluster channel lists,
partial-residual scatter series, and the Figure 5 forest-plot coefficients) and
pickles them to output/figure_data/main_figure_data.pkl.

This is deliberately SEPARATE from plotting: generate_manuscript_figures.py only
loads this artifact and draws. Re-run this whenever the underlying data or models
change; never run analysis from the figure script.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import mne

from cluster_permutation import largest_cluster_channels
from spindle_common import (
    COV_PATH,
    PREDICTORS,
    ROI_PATH,
    TABLE_DIR,
    channel_regression as run_channel_regression,
    data_matrix_for_metric,
    ensure_dir,
    load_channel_info,
    load_covariates,
    load_spindle_matrices,
)
from spindle_common import PROJECT

CACHE_DIR = PROJECT / "output/figure_data"
CACHE_PATH = CACHE_DIR / "main_figure_data.pkl"
CLUSTER_T_THRESH = 2.0


def load() -> dict:
    """Load the precomputed figure data; raise if the analysis step has not run."""
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"{CACHE_PATH} not found. Run `python scripts/compute_figure_data.py` "
            "first — figures are generated from this artifact, not from raw analysis."
        )
    with open(CACHE_PATH, "rb") as fh:
        return pickle.load(fh)


def _find_roi_channels(data_dict, metric, cov, sleep_mins, dist_matrix, points_x):
    """Data-driven bilateral ROI: p<0.05 seeds (>=2 sig neighbours) + p<0.08 trending."""
    data_matrix = data_matrix_for_metric(data_dict, metric, sleep_mins)
    _, _t, p_vals = run_channel_regression(data_matrix, cov, PREDICTORS)
    sig_seeds = np.where(p_vals < 0.05)[0]
    seeds_contiguous = []
    for ch in sig_seeds:
        neighbors = np.where((dist_matrix[ch] < 0.035) & (dist_matrix[ch] > 0))[0]
        if len([n for n in neighbors if p_vals[n] < 0.05]) >= 2:
            seeds_contiguous.append(ch)
    final_roi = set(seeds_contiguous)
    for ch in seeds_contiguous:
        neighbors = np.where((dist_matrix[ch] < 0.035) & (dist_matrix[ch] > 0))[0]
        final_roi.update(n for n in neighbors if 0.05 <= p_vals[n] < 0.08)
    roi_list = sorted(final_roi)
    if roi_list:
        left = np.sum(points_x[roi_list] < -0.015)
        right = np.sum(points_x[roi_list] > 0.015)
        if left < 2 or right < 2:
            return []
    return roi_list


def _scatter_series(data_matrix, roi_channels, cov):
    """Partial-residual scatter (ROI metric vs logHI | age, sex) — precomputed."""
    if not roi_channels:
        return None
    roi_avg = np.nanmean(data_matrix[:, roi_channels], axis=1)
    hi = cov["logHI"].values
    age = cov["age_c"].values
    sex = cov["gender_bin"].values
    mask = ~(np.isnan(roi_avg) | np.isnan(hi) | np.isnan(age) | np.isnan(sex))
    cov_mat = sm.add_constant(np.column_stack([age[mask], sex[mask]]))
    y_resid = roi_avg[mask] - sm.OLS(roi_avg[mask], cov_mat).fit().predict(cov_mat)
    x_resid = hi[mask] - sm.OLS(hi[mask], cov_mat).fit().predict(cov_mat)
    slope, intercept, r_val, _p, _se = stats.linregress(x_resid, y_resid)
    return {"x": x_resid, "y": y_resid, "slope": slope, "intercept": intercept, "r": r_val}


def _fig5_results():
    """Figure 5 forest-plot coefficients (same inline model the figure used)."""
    cov = pd.read_csv(COV_PATH)
    roi = pd.read_csv(ROI_PATH)
    df = cov.merge(roi, on="id", how="left").dropna(subset=["DPRIMEQ1"])
    df["RT_log"] = np.log(df["RTMEANQ1"])
    df["OM_sqrt"] = np.sqrt(df["OMPERQ1"])
    df["COM_sqrt"] = np.sqrt(df["COMPERQ1"])
    for col in ["RT_log", "OM_sqrt"]:
        z = np.abs(stats.zscore(df[col].dropna()))
        valid_idx = df[col].dropna().index[z < 3]
        df = df.loc[df.index.isin(valid_idx)]
    roi_cols = ["ant_fast_dur", "ant_slow_peakfreq", "ant_slow_dur", "pos_slow_amp"]
    for col in ["DPRIMEQ1", "OM_sqrt", "COM_sqrt"] + roi_cols:
        df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std()
    outcomes = ["DPRIMEQ1_z", "OM_sqrt_z", "COM_sqrt_z"]
    roi_vars = ["ant_fast_dur_z", "ant_slow_peakfreq_z", "ant_slow_dur_z", "pos_slow_amp_z"]
    results = {}
    for ri, roi_var in enumerate(roi_vars):
        for oi, out_var in enumerate(outcomes):
            sub = df.dropna(subset=[out_var, roi_var, "age_years", "gender", "overall_hi"])
            y = sub[out_var].values
            X = sm.add_constant(sub[[roi_var, "age_years", "gender", "overall_hi"]].values)
            model = sm.OLS(y, X).fit()
            ci = model.conf_int(alpha=0.05)
            results[(ri, oi)] = (model.params[1], ci[1, 0], ci[1, 1], model.pvalues[1])
    return {"results": results, "n": len(df)}


def _ctag_map():
    """(band, metric) -> formatted corrected cluster-p tag, from the results CSV."""
    csv = TABLE_DIR / "cluster_permutation_results.csv"
    out = {}
    if csv.exists():
        tbl = pd.read_csv(csv)
        for _, r in tbl.iterrows():
            p = r["p_value"]
            if pd.isna(p):
                tag = "(cluster p = n/a)"
            else:
                star = "*" if p < 0.05 else ""
                p_txt = f"{p:.3f}" if p >= 0.001 else f"{p:.4f}"
                tag = f"(cluster p = {p_txt}{star})"
            out[(r["band"], r["metric"])] = tag
    return out


def compute() -> dict:
    mne.set_log_level("ERROR")
    channels = load_channel_info()
    info = channels.info
    dist_matrix = channels.distance_matrix
    points_x = channels.x
    nCh = len(channels.labels)

    cov = load_covariates(COV_PATH)
    matrices = load_spindle_matrices(cov, nCh, include_frequency=True)
    slow, fast, sleep_mins = matrices.slow, matrices.fast, matrices.sleep_mins
    band_dict = {"slow": slow, "fast": fast}

    adj = mne.channels.find_ch_adjacency(info, ch_type="eeg")[0].toarray().astype(bool)

    # Figure 2: normative per-channel topographies
    topo = {
        "den_slow": np.nanmean(slow["Count"] / sleep_mins, axis=0),
        "den_fast": np.nanmean(fast["Count"] / sleep_mins, axis=0),
        "dur_slow": np.nanmean(slow["Duration"], axis=0),
        "dur_fast": np.nanmean(fast["Duration"], axis=0),
    }

    # Data-driven ROIs and permutation clusters
    roi = {
        "ddroi_slow_amp": _find_roi_channels(slow, "Amplitude", cov, sleep_mins, dist_matrix, points_x),
        "ddroi_slow_dur": _find_roi_channels(slow, "Duration", cov, sleep_mins, dist_matrix, points_x),
        "ddroi_fast_dur": _find_roi_channels(fast, "Duration", cov, sleep_mins, dist_matrix, points_x),
        "perm_fast_dur": largest_cluster_channels(
            data_matrix_for_metric(fast, "Duration", sleep_mins), cov, adj, PREDICTORS, CLUSTER_T_THRESH),
        "perm_slow_freq": largest_cluster_channels(
            data_matrix_for_metric(slow, "Frequency", sleep_mins), cov, adj, PREDICTORS, CLUSTER_T_THRESH),
    }

    # Figures 3-4: channel-wise HI regression maps and scatter series.
    # ("slow", "Density") and ("fast", "Density") are added for Supplementary
    # Figure S5: it documents the non-significant slow- and fast-density HI trends
    # as topomaps only (neither reaches a suprathreshold cluster, so there is no
    # data-driven ROI and no scatter series).
    metrics = [("fast", "Duration"), ("slow", "Frequency"), ("slow", "Amplitude"),
               ("slow", "Duration"), ("slow", "Density"), ("fast", "Density")]
    scatter_roi = {
        ("fast", "Duration"): roi["ddroi_fast_dur"],
        ("slow", "Frequency"): roi["perm_slow_freq"],
        ("slow", "Amplitude"): roi["ddroi_slow_amp"],
        ("slow", "Duration"): roi["ddroi_slow_dur"],
        ("slow", "Density"): [],  # topomap-only figure; no ROI scatter
        ("fast", "Density"): [],  # topomap-only figure; no ROI scatter
    }
    reg, scatter = {}, {}
    for band, metric in metrics:
        dm = data_matrix_for_metric(band_dict[band], metric, sleep_mins)
        beta, t, p = run_channel_regression(dm, cov, PREDICTORS)
        reg[(band, metric)] = {"beta": beta, "t": t, "p": p}
        scatter[(band, metric)] = _scatter_series(dm, scatter_roi[(band, metric)], cov)

    data = {
        "topo": topo,
        "roi": roi,
        "reg": reg,
        "scatter": scatter,
        "ctag": _ctag_map(),
        "fig5": _fig5_results(),
        "channel_labels": channels.labels,
    }
    return data


def main() -> None:
    data = compute()
    ensure_dir(CACHE_DIR)
    with open(CACHE_PATH, "wb") as fh:
        pickle.dump(data, fh)
    print(f"[SAVED] {CACHE_PATH}")
    print(f"  ROIs: " + ", ".join(f"{k}={len(v)}" for k, v in data["roi"].items()))
    print(f"  Figure 5 N = {data['fig5']['n']}")


if __name__ == "__main__":
    main()
