#!/usr/bin/env python3
"""Shared data-loading and modeling helpers for manuscript spindle scripts."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import statsmodels.api as sm
from scipy.spatial.distance import cdist


PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "temp/cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT / "temp/cache/xdg"))
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MNE_HOME", str(PROJECT / "temp/cache/mne"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MNE_HOME"]).mkdir(parents=True, exist_ok=True)

import mne

SPINDLE_DIR = PROJECT / "datasets/original/spindles_individual_data"
COV_PATH = PROJECT / "datasets/clean/analysis_sample_complete_N62.csv"
SAMPLE_IDS_PATH = PROJECT / "datasets/clean/sample_ids_N62.csv"
CHAN_PATH = PROJECT / "assets/inside172.mat"
ROI_PATH = PROJECT / "datasets/original/roi_values_corrected.csv"
FIGURE_DIR = PROJECT / "output/figures"
TABLE_DIR = PROJECT / "output/tables"

SPHERE = 0.095
SLOW_RANGE = (10.0, 12.0)  # [10, 12)
FAST_RANGE = (12.0, 16.0)  # [12, 16]
PREDICTORS = ["logHI_c", "age_c", "gender_bin"]


@dataclass(frozen=True)
class ChannelInfo:
    labels: list[str]
    x: np.ndarray
    y: np.ndarray
    info: mne.Info
    distance_matrix: np.ndarray


@dataclass(frozen=True)
class SpindleMatrices:
    slow: dict[str, np.ndarray]
    fast: dict[str, np.ndarray]
    sleep_mins: np.ndarray
    missing_subjects: list[int | float | str]


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_channel_info(chan_path: str | Path = CHAN_PATH, sphere: float = SPHERE) -> ChannelInfo:
    mat = sio.loadmat(chan_path)
    chanlocs = mat["inside172"].flatten()

    theta_deg, radius_raw, labels = [], [], []
    for ch in chanlocs:
        theta_deg.append(float(ch["theta"].item()))
        radius_raw.append(float(ch["radius"].item()))
        lbl = ch["labels"]
        if isinstance(lbl, np.ndarray):
            lbl = "".join(str(x).strip() for x in lbl.flatten())
        else:
            lbl = str(lbl).strip()
        labels.append(lbl)

    theta_deg = np.asarray(theta_deg)
    radius_raw = np.asarray(radius_raw)
    radius_scaled = radius_raw * (sphere / np.max(radius_raw))
    x = radius_scaled * np.sin(np.radians(theta_deg))
    y = radius_scaled * np.cos(np.radians(theta_deg))

    ch_pos = {name: [float(px), float(py), 0.0] for name, px, py in zip(labels, x, y)}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info = mne.create_info(ch_names=labels, sfreq=100, ch_types="eeg")
    info.set_montage(montage)
    distance_matrix = cdist(np.column_stack([x, y]), np.column_stack([x, y]))
    return ChannelInfo(labels=labels, x=x, y=y, info=info, distance_matrix=distance_matrix)


def load_covariates(cov_path: str | Path = COV_PATH) -> pd.DataFrame:
    cov = pd.read_csv(cov_path)
    # Pin to the canonical N=62 analytic sample. Do NOT filter on the legacy
    # all.night.useable flag: 2 canonical subjects have useable==0 and would be
    # wrongly dropped. Membership is defined solely by SAMPLE_IDS_PATH.
    canon_ids = pd.read_csv(SAMPLE_IDS_PATH)["id"]
    cov = cov[cov["id"].isin(canon_ids)].reset_index(drop=True)
    assert len(cov) == 62, f"expected 62 subjects, got {len(cov)}"
    cov["logHI"] = np.log10(cov["overall_hi"] + 1)
    cov["logHI_c"] = cov["logHI"] - cov["logHI"].mean()
    cov["age_c"] = cov["age_years"] - cov["age_years"].mean()
    cov["gender_bin"] = cov["gender"]
    return cov


def match_spindle_file(files: list[str], sid: int | float | str) -> str | None:
    sid_str = str(int(sid)) if isinstance(sid, (int, float)) and not pd.isna(sid) else str(sid)
    for padded in [sid_str, sid_str.zfill(2), sid_str.zfill(3)]:
        prefix = f"{padded}_"
        for file_path in files:
            if os.path.basename(file_path).startswith(prefix):
                return file_path
    return None


def load_spindle_matrices(
    cov: pd.DataFrame,
    n_channels: int,
    spindle_dir: str | Path = SPINDLE_DIR,
    include_frequency: bool = False,
) -> SpindleMatrices:
    n_subjects = len(cov)
    slow = {
        "Count": np.full((n_subjects, n_channels), np.nan),
        "Amplitude": np.full((n_subjects, n_channels), np.nan),
        "Duration": np.full((n_subjects, n_channels), np.nan),
    }
    fast = {
        "Count": np.full((n_subjects, n_channels), np.nan),
        "Amplitude": np.full((n_subjects, n_channels), np.nan),
        "Duration": np.full((n_subjects, n_channels), np.nan),
    }
    if include_frequency:
        slow["Frequency"] = np.full((n_subjects, n_channels), np.nan)
        fast["Frequency"] = np.full((n_subjects, n_channels), np.nan)

    files = glob.glob(str(Path(spindle_dir) / "*_all_spindles_detailed.csv"))
    missing_subjects = []

    for row_idx, sid in enumerate(cov["id"].tolist()):
        file_path = match_spindle_file(files, sid)
        if file_path is None:
            missing_subjects.append(sid)
            continue

        tbl = pd.read_csv(file_path)
        tbl["IdxChannel"] = pd.to_numeric(tbl["IdxChannel"], errors="coerce").fillna(-1).astype(int)
        tbl["Frequency"] = pd.to_numeric(tbl["Frequency"], errors="coerce")
        tbl["Amplitude"] = pd.to_numeric(tbl["Amplitude"], errors="coerce")
        tbl["Duration"] = pd.to_numeric(tbl["Duration"], errors="coerce")

        for band_data, freq_mask in [
            (slow, (tbl["Frequency"] >= SLOW_RANGE[0]) & (tbl["Frequency"] < SLOW_RANGE[1])),
            (fast, (tbl["Frequency"] >= FAST_RANGE[0]) & (tbl["Frequency"] <= FAST_RANGE[1])),
        ]:
            band_tbl = tbl.loc[freq_mask]
            idx = band_tbl["IdxChannel"]
            for ch in range(n_channels):
                ch_tbl = band_tbl.loc[idx == ch]
                band_data["Count"][row_idx, ch] = len(ch_tbl)
                if len(ch_tbl) == 0:
                    continue
                band_data["Amplitude"][row_idx, ch] = ch_tbl["Amplitude"].mean()
                band_data["Duration"][row_idx, ch] = ch_tbl["Duration"].mean()
                if include_frequency:
                    band_data["Frequency"][row_idx, ch] = ch_tbl["Frequency"].mean()

    sleep_mins = cov["cleanN2N3_min"].to_numpy(dtype=float)[:, np.newaxis]
    return SpindleMatrices(
        slow=slow,
        fast=fast,
        sleep_mins=sleep_mins,
        missing_subjects=missing_subjects,
    )


def data_matrix_for_metric(
    data_dict: dict[str, np.ndarray],
    metric: str,
    sleep_mins: np.ndarray,
) -> np.ndarray:
    if metric == "Density":
        return data_dict["Count"] / sleep_mins
    return data_dict[metric]


def channel_regression(
    data_matrix: np.ndarray,
    cov: pd.DataFrame,
    predictors: list[str] | None = None,
    min_n: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictors = predictors or PREDICTORS
    valid_rows = cov[predictors].notna().all(axis=1).to_numpy()
    x_arr = sm.add_constant(cov.loc[valid_rows, predictors].to_numpy(dtype=float))
    y_all = data_matrix[valid_rows, :]

    n_channels = data_matrix.shape[1]
    betas = np.full(n_channels, np.nan)
    t_stats = np.full(n_channels, np.nan)
    p_vals = np.ones(n_channels)

    for ch in range(n_channels):
        y = y_all[:, ch]
        ok = ~np.isnan(y)
        if ok.sum() < min_n:
            continue
        model = sm.OLS(y[ok], x_arr[ok]).fit()
        betas[ch] = model.params[1]
        t_stats[ch] = model.tvalues[1]
        p_vals[ch] = model.pvalues[1]

    return betas, t_stats, p_vals
