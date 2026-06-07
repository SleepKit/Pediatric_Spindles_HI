#!/usr/bin/env python3
"""
Generate publication-quality figures for the pediatric sleep spindles manuscript.
Figures 1-3 per Stephanie Jones's guidance and the code audit fixes.

Author: automated pipeline
"""

import os
import warnings
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "temp/cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT / "temp/cache/xdg"))
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MNE_HOME", str(PROJECT / "temp/cache/mne"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MNE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Patch
from scipy import stats
from scipy.signal import butter, filtfilt, hilbert
import mne

from cluster_permutation import largest_cluster_channels
from spindle_common import (
    COV_PATH,
    FIGURE_DIR,
    PREDICTORS,
    ROI_PATH,
    SPHERE,
    channel_regression as run_channel_regression,
    data_matrix_for_metric,
    ensure_dir,
    load_channel_info,
    load_covariates,
    load_spindle_matrices,
)

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
mne.set_log_level('ERROR')

OUT_DIR = FIGURE_DIR
ensure_dir(OUT_DIR)

# ============================================================
# STYLING CONSTANTS (Nature figure bible)
# ============================================================
FULL_WIDTH_IN = 7.2          # 183 mm
DPI = 300
FONT_FAMILY = 'Arial'
FONT_LABEL = 6               # axis labels 5-7pt
FONT_TITLE = 7
FONT_PANEL = 14              # panel labels (70% larger than 8pt base)
AXES_LW = 0.6

plt.rcParams.update({
    'font.family': FONT_FAMILY,
    'font.size': FONT_LABEL,
    'axes.labelsize': FONT_LABEL,
    'axes.titlesize': FONT_TITLE,
    'axes.linewidth': AXES_LW,
    'xtick.major.width': AXES_LW,
    'ytick.major.width': AXES_LW,
    'xtick.labelsize': FONT_LABEL,
    'ytick.labelsize': FONT_LABEL,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ============================================================
# 1-3. LOAD CHANNELS, COVARIATES, AND SPINDLE MATRICES
# ============================================================
channels = load_channel_info()
chan_labels = channels.labels
points_x = channels.x
points_y = channels.y
info = channels.info
nCh = len(chan_labels)

print(f'[INFO] {nCh} channels loaded, coordinate range x=[{points_x.min():.4f}, {points_x.max():.4f}]')

covUse = load_covariates(COV_PATH)
subject_list = covUse['id'].tolist()
nSub = len(subject_list)
print(f'[INFO] {nSub} subjects in covariate table')

matrices = load_spindle_matrices(covUse, nCh, include_frequency=True)
if matrices.missing_subjects:
    print(f'  [WARN] Missing spindle files for subjects: {matrices.missing_subjects}')
print(f'[INFO] Spindle extraction complete for {nSub} subjects')

slow_data = matrices.slow
fast_data = matrices.fast
sleep_mins = matrices.sleep_mins

CountMat_slow = slow_data['Count']
AmpMat_slow = slow_data['Amplitude']
DurMat_slow = slow_data['Duration']
FreqMat_slow = slow_data['Frequency']
CountMat_fast = fast_data['Count']
AmpMat_fast = fast_data['Amplitude']
DurMat_fast = fast_data['Duration']

# ============================================================
# 4. ROI CLUSTER EXTRACTION  (replicating notebook logic)
# ============================================================
predictors = PREDICTORS
dist_matrix = channels.distance_matrix


def find_roi_channels(data_dict, metric, cov_df):
    """Return list of channel indices forming a significant bilateral ROI cluster."""
    data_matrix = data_matrix_for_metric(data_dict, metric, sleep_mins)
    _, t_stats, p_vals = run_channel_regression(data_matrix, cov_df, predictors)

    sig_seeds = np.where(p_vals < 0.05)[0]
    seeds_contiguous = []
    for ch in sig_seeds:
        neighbors = np.where((dist_matrix[ch] < 0.035) & (dist_matrix[ch] > 0))[0]
        sig_neighbors = [n for n in neighbors if p_vals[n] < 0.05]
        if len(sig_neighbors) >= 2:
            seeds_contiguous.append(ch)

    final_roi = set(seeds_contiguous)
    for ch in seeds_contiguous:
        neighbors = np.where((dist_matrix[ch] < 0.035) & (dist_matrix[ch] > 0))[0]
        final_roi.update(n for n in neighbors if 0.05 <= p_vals[n] < 0.08)

    roi_list = sorted(final_roi)
    if roi_list:
        left_count = np.sum(points_x[roi_list] < -0.015)
        right_count = np.sum(points_x[roi_list] > 0.015)
        if left_count < 2 or right_count < 2:
            return [], t_stats, p_vals
    return roi_list, t_stats, p_vals


# Pre-compute all ROI clusters using MNE Delaunay adjacency.
# NOTE: This adjacency only defines spatial neighbours for forming connected
# components of suprathreshold channels. The family-wise-error-corrected cluster
# p-values come from cluster_permutation.cluster_permutation_test (covariate-
# adjusted Freedman-Lane residual permutation), NOT from any MNE permutation test
# (MNE's 1-sample/F tests do not support covariate-adjusted regression).
CLUSTER_T_THRESH = 2.0
adj_mne_sparse, _ = mne.channels.find_ch_adjacency(info, ch_type='eeg')
adj_mne_dense = adj_mne_sparse.toarray().astype(bool)
print(f'[INFO] MNE Delaunay adjacency: mean neighbors={adj_mne_dense.sum(axis=1).mean():.1f}')


def find_permutation_clusters(data_dict, metric, cov_df):
    """Channel indices of the top cluster, for drawing the membership overlay.

    Thin wrapper over cluster_permutation.largest_cluster_channels (the single
    shared definition used by compute_rois_corrected.py too): the largest |t|>2.0
    connected component over the MNE Delaunay adjacency, ranked by cluster mass.
    Used purely for the topomap overlay; the corrected cluster p-value is computed
    separately by cluster_permutation_test.
    """
    data_matrix = data_matrix_for_metric(data_dict, metric, sleep_mins)
    return largest_cluster_channels(
        data_matrix, cov_df, adj_mne_dense, predictors, CLUSTER_T_THRESH
    )


# Data-driven ROIs (p < 0.05 + trending neighbors) — used for behavioral models
ddroi_slow_count, _, _ = find_roi_channels(slow_data, 'Count', covUse)
ddroi_slow_amp, _, _   = find_roi_channels(slow_data, 'Amplitude', covUse)
ddroi_slow_dur, _, _   = find_roi_channels(slow_data, 'Duration', covUse)
ddroi_fast_dur, _, _   = find_roi_channels(fast_data, 'Duration', covUse)

print(f'[INFO] Data-driven ROIs: slow_count={len(ddroi_slow_count)}, slow_amp={len(ddroi_slow_amp)}, '
      f'slow_dur={len(ddroi_slow_dur)}, fast_dur={len(ddroi_fast_dur)}')

# Permutation clusters (|t| > 2.0, Delaunay adjacency) — matches MNE correction
perm_slow_count = find_permutation_clusters(slow_data, 'Density', covUse)
perm_slow_amp   = find_permutation_clusters(slow_data, 'Amplitude', covUse)
perm_slow_dur   = find_permutation_clusters(slow_data, 'Duration', covUse)
perm_fast_dur   = find_permutation_clusters(fast_data, 'Duration', covUse)
perm_slow_freq  = find_permutation_clusters(slow_data, 'Frequency', covUse)

print(f'[INFO] Perm clusters (Delaunay): slow_count={len(perm_slow_count)}, slow_amp={len(perm_slow_amp)}, '
      f'slow_dur={len(perm_slow_dur)}, fast_dur={len(perm_fast_dur)}, slow_freq={len(perm_slow_freq)}')


# ============================================================
# 5. REAL cluster-corrected p-values
# ============================================================
# Family-wise-error-corrected cluster p-values are computed by the covariate-
# adjusted Freedman-Lane permutation test in cluster_permutation.py. To keep the
# figure build fast we read the precomputed table when present, and only run the
# (slow) permutation test on the fly when it is missing.
from spindle_common import TABLE_DIR  # noqa: E402

_CLUSTER_P_CSV = Path(TABLE_DIR) / 'cluster_permutation_results.csv'


def _load_cluster_pvalues():
    """Return {(band, metric): p_value} of corrected top-cluster p-values."""
    if _CLUSTER_P_CSV.exists():
        tbl = pd.read_csv(_CLUSTER_P_CSV)
        return {(r['band'], r['metric']): r['p_value'] for _, r in tbl.iterrows()}

    print('[INFO] cluster_permutation_results.csv missing; computing p-values '
          'on the fly (slow)...')
    from cluster_permutation import METRICS, cluster_permutation_test
    band_data = {'slow': slow_data, 'fast': fast_data}
    out = {}
    for _label, band, metric in METRICS:
        data_matrix = data_matrix_for_metric(band_data[band], metric, sleep_mins)
        clusters = cluster_permutation_test(data_matrix, covUse, adj_mne_dense)
        out[(band, metric)] = clusters[0]['p'] if clusters else float('nan')
    return out


_CLUSTER_P = _load_cluster_pvalues()


def _ctag(band, metric):
    """Format a corrected cluster p-value tag '(cluster p = X*)'.

    Appends '*' when the FWE-corrected cluster p-value is < 0.05.
    """
    p = _CLUSTER_P.get((band, metric))
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return '(cluster p = n/a)'
    star = '*' if p < 0.05 else ''
    p_txt = f'{p:.3f}' if p >= 0.001 else f'{p:.4f}'
    return f'(cluster p = {p_txt}{star})'


# ============================================================
# HELPER: channel-wise regression returning beta, t, p arrays
# ============================================================
def channel_regression(data_dict, metric, cov_df):
    """Run channel-wise OLS: metric ~ logHI_c + age_c + gender_bin.
    Returns beta, tstat, pval arrays (nCh,) for the logHI_c predictor.
    """
    data_matrix = data_matrix_for_metric(data_dict, metric, sleep_mins)
    return run_channel_regression(data_matrix, cov_df, predictors)


# ============================================================
# HELPER: scatter plot of ROI metric vs HI
# ============================================================
def _scatter_roi_vs_hi(ax, data_matrix, roi_channels, cov_df, ylabel, color='#1a5276'):
    """Partial-residual scatter: ROI metric vs logHI, adjusted for age and sex.

    Both ROI average and logHI are residualized on age_c + gender_bin so the
    scatter reflects the same partial association shown by the regression models.
    """
    if not roi_channels:
        ax.text(0.5, 0.5, 'No ROI', transform=ax.transAxes,
                ha='center', va='center', fontsize=FONT_LABEL, color='grey')
        return
    roi_avg = np.nanmean(data_matrix[:, roi_channels], axis=1)
    hi_vals = cov_df['logHI'].values
    age_vals = cov_df['age_c'].values
    sex_vals = cov_df['gender_bin'].values

    mask = ~(np.isnan(roi_avg) | np.isnan(hi_vals) | np.isnan(age_vals) | np.isnan(sex_vals))
    roi_m = roi_avg[mask]
    hi_m = hi_vals[mask]
    cov_mat = sm.add_constant(np.column_stack([age_vals[mask], sex_vals[mask]]))

    # Residualize both variables on age + sex
    y_resid = roi_m - sm.OLS(roi_m, cov_mat).fit().predict(cov_mat)
    x_resid = hi_m  - sm.OLS(hi_m,  cov_mat).fit().predict(cov_mat)

    ax.scatter(x_resid, y_resid, s=12, alpha=0.55, color=color,
               edgecolors='white', linewidths=0.3, zorder=5)
    slope, intercept, r_val, p_val, _ = stats.linregress(x_resid, y_resid)
    x_fit = np.linspace(x_resid.min(), x_resid.max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, color=color, linewidth=1.0, zorder=6)

    ax.text(0.95, 0.95, f'r = {r_val:.2f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=5, color='#555555')
    ax.set_xlabel('logHI | age, sex', fontsize=FONT_LABEL)
    ax.set_ylabel(f'{ylabel} | age, sex', fontsize=FONT_LABEL)
    ax.tick_params(axis='both', length=2, width=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(AXES_LW)
    ax.spines['bottom'].set_linewidth(AXES_LW)


# ============================================================
# FIGURE 1 HELPERS — Synthetic Illustration Panels
# ============================================================
def _make_panel_hypnogram(ax):
    """Panel a: Synthetic pediatric hypnogram with N2/N3 analysis window highlighted."""
    # Stage encoding: W=0, REM=1, N1=2, N2=3, N3=4  (inverted y -> W at top)
    transitions = [
        (0, 10), (2, 5), (3, 20), (4, 35), (3, 15), (1, 10),
        (0, 3),  (2, 3), (3, 25), (4, 30), (3, 20), (1, 18),
        (2, 3),  (3, 25), (4, 18), (3, 20), (1, 22),
        (0, 2),  (3, 30), (4, 8),  (3, 20), (1, 28),
        (0, 2),  (2, 3),  (3, 25), (1, 30), (0, 5),
    ]
    times_min, stages = [], []
    t_acc = 0
    for stage, dur in transitions:
        times_min.append(t_acc)
        stages.append(stage)
        t_acc += dur
    times_min.append(t_acc)
    stages.append(stages[-1])
    hours = np.array(times_min) / 60.0

    ax.step(hours, stages, where='post', color='#333333', linewidth=0.8, zorder=3)
    for i in range(len(times_min) - 1):
        if stages[i] in (3, 4):
            c = '#6baed6' if stages[i] == 3 else '#2171b5'
            a_val = 0.25 if stages[i] == 3 else 0.30
            ax.axvspan(hours[i], hours[i + 1], alpha=a_val, color=c, lw=0, zorder=1)

    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(['W', 'REM', 'N1', 'N2', 'N3'])
    ax.set_xlabel('Time from lights-out (hours)')
    ax.set_xlim(0, hours[-1])
    ax.set_ylim(-0.5, 4.5)
    ax.invert_yaxis()
    ax.legend(handles=[Patch(fc='#6baed6', alpha=0.4, label='N2'),
                       Patch(fc='#2171b5', alpha=0.4, label='N3')],
              loc='lower right', fontsize=5, framealpha=0.9, edgecolor='#cccccc',
              fancybox=False, title='Analysis window', title_fontsize=5)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(AXES_LW)
    ax.spines['bottom'].set_linewidth(AXES_LW)
    ax.tick_params(length=2, width=0.4)


def _make_panel_spindle_detection(ax_raw, ax_filt):
    """Panel b: Synthetic N2 EEG with detected spindle and annotated parameters."""
    fs_syn = 256
    t = np.arange(0, 4.0, 1 / fs_syn)
    n = len(t)
    np.random.seed(42)

    # Background: coloured noise + slow oscillation (N2-like)
    white = np.random.randn(n + 1000)
    b_hp, a_hp = butter(2, 0.5, btype='high', fs=fs_syn)
    b_lp, a_lp = butter(3, 30, btype='low', fs=fs_syn)
    bg = filtfilt(b_lp, a_lp, filtfilt(b_hp, a_hp, np.cumsum(white)[:n]))
    bg += 5 * np.sin(2 * np.pi * 2.0 * t)
    bg = bg / np.std(bg) * 8

    # Embedded spindle: 12 Hz, ~1 s visible duration, centred at 2.0 s
    sp_env = np.exp(-0.5 * ((t - 2.0) / 0.22) ** 2)
    spindle = sp_env * 30 * np.sin(2 * np.pi * 12 * t)
    raw = bg + spindle

    # Sigma bandpass 10-16 Hz
    b_bp, a_bp = butter(4, [10, 16], btype='band', fs=fs_syn)
    sigma = filtfilt(b_bp, a_bp, raw)

    # Amplitude envelope via Hilbert transform
    env = np.abs(hilbert(sigma))
    b_sm, a_sm = butter(2, 4, btype='low', fs=fs_syn)
    env_smooth = filtfilt(b_sm, a_sm, env)

    # Detection threshold and boundaries
    thr = 1.5 * np.median(env_smooth)
    above = env_smooth > thr
    d_above = np.diff(above.astype(int))
    starts = np.where(d_above == 1)[0] + 1
    ends = np.where(d_above == -1)[0] + 1
    if above[0]:
        starts = np.concatenate([[0], starts])
    if above[-1]:
        ends = np.concatenate([ends, [n - 1]])

    det_s = det_e = None
    for s0, e0 in zip(starts, ends):
        dur_det = t[e0] - t[s0]
        if 0.3 <= dur_det <= 2.5 and t[s0] < 2.5 and t[e0] > 1.5:
            det_s, det_e = s0, e0
            break
    if det_s is None:
        det_s, det_e = np.argmin(np.abs(t - 1.55)), np.argmin(np.abs(t - 2.45))

    ts, te = t[det_s], t[det_e]
    det_dur = te - ts

    # ---- Upper trace: raw EEG ----
    ax_raw.plot(t, raw, color='#666666', linewidth=0.4, zorder=2)
    ax_raw.axvspan(ts, te, alpha=0.15, color='#2171b5', zorder=1)
    mask = (t >= ts) & (t <= te)
    ax_raw.plot(t[mask], raw[mask], color='#2171b5', linewidth=0.6, zorder=3)

    ax_raw.set_ylabel('\u00b5V', labelpad=6)
    for sp in ('top', 'right', 'bottom'):
        ax_raw.spines[sp].set_visible(False)
    ax_raw.tick_params(axis='x', labelbottom=False, length=0)
    ax_raw.tick_params(axis='y', length=2, width=0.4)
    ax_raw.spines['left'].set_linewidth(AXES_LW)
    ax_raw.text(0.005, 0.93, 'Raw EEG (single channel, N2)',
                transform=ax_raw.transAxes, fontsize=5, color='#555555', va='top')

    # Duration bracket below the trace
    ylo, yhi = ax_raw.get_ylim()
    pad = 0.18 * (yhi - ylo)
    ax_raw.set_ylim(ylo - pad, yhi)
    y_bkt = ylo - pad * 0.3
    ax_raw.annotate('', xy=(ts, y_bkt), xytext=(te, y_bkt),
                    arrowprops=dict(arrowstyle='<->', color='#2171b5', lw=0.8,
                                    shrinkA=0, shrinkB=0))
    ax_raw.text((ts + te) / 2, y_bkt - 1,
                f'{det_dur:.2f} s  (valid range: 0.5\u20132.0 s)',
                ha='center', va='top', fontsize=5, color='#2171b5')

    # Parameter summary box (upper-right)
    params = ('YASA detection parameters\n'
              'Freq. range: 10\u201316 Hz\n'
              '  Slow: 10\u201312 Hz | Fast: 12\u201316 Hz\n'
              'Duration: 0.5\u20132.0 s\n'
              'Min. spacing: 500 ms\n'
              'Thresholds:\n'
              '  Rel. power \u2265 0.2\n'
              '  Correlation \u2265 0.65\n'
              '  RMS \u2265 1.5\u00d7 median')
    ax_raw.text(0.99, 0.95, params, transform=ax_raw.transAxes,
                fontsize=4, va='top', ha='right', family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', fc='#f7f7f7',
                          ec='#cccccc', alpha=0.95),
                linespacing=1.3)

    # ---- Lower trace: filtered + envelope ----
    ax_filt.plot(t, sigma, color='#6baed6', linewidth=0.4, zorder=2,
                 label='\u03c3-filtered (10\u201316 Hz)')
    ax_filt.plot(t, env_smooth, color='#d94801', linewidth=0.8, zorder=3,
                 label='Amplitude envelope')
    ax_filt.axhline(thr, color='#d94801', lw=0.5, ls='--', zorder=2, alpha=0.6)
    ax_filt.axvspan(ts, te, alpha=0.10, color='#2171b5', zorder=1)

    ax_filt.text(0.1, thr * 1.08,
                 'Threshold (1.5\u00d7 median RMS)',
                 fontsize=4.5, color='#d94801', va='bottom')
    ax_filt.set_ylabel('\u00b5V', labelpad=6)
    ax_filt.set_xlabel('Time (s)')
    for sp in ('top', 'right'):
        ax_filt.spines[sp].set_visible(False)
    ax_filt.spines['left'].set_linewidth(AXES_LW)
    ax_filt.spines['bottom'].set_linewidth(AXES_LW)
    ax_filt.tick_params(length=2, width=0.4)
    ax_filt.legend(loc='upper left', fontsize=4.5, framealpha=0.9,
                   edgecolor='#cccccc', fancybox=False)


# ============================================================
# FIGURE 1 — Methods Illustration (Sleep Architecture + Spindle Detection)
# ============================================================
def make_figure1():
    """Standalone methods figure: hypnogram (a) + spindle detection illustration (b)."""
    fig = plt.figure(figsize=(FULL_WIDTH_IN, FULL_WIDTH_IN * 0.7))
    gs = gridspec.GridSpec(2, 1,
                           height_ratios=[0.8, 1.2],
                           hspace=0.45,
                           left=0.10, right=0.96, top=0.95, bottom=0.08)

    # ---- panel a: hypnogram ----
    ax_hyp = fig.add_subplot(gs[0])
    _make_panel_hypnogram(ax_hyp)
    ax_hyp.text(-0.05, 1.15, 'a', transform=ax_hyp.transAxes,
                fontsize=FONT_PANEL, fontweight='bold', va='top')

    # ---- panel b: spindle detection (two sub-rows) ----
    gs_sp = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[1], hspace=0.08)
    ax_raw = fig.add_subplot(gs_sp[0])
    ax_filt = fig.add_subplot(gs_sp[1], sharex=ax_raw)
    _make_panel_spindle_detection(ax_raw, ax_filt)
    ax_raw.text(-0.05, 1.22, 'b', transform=ax_raw.transAxes,
                fontsize=FONT_PANEL, fontweight='bold', va='top')

    out = os.path.join(OUT_DIR, 'Figure1_methods_illustration.png')
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f'[SAVED] {out}')


# ============================================================
# FIGURE 2 — Normative Spindle Topography (2x2)
# ============================================================
def make_figure2():
    """2x2 grid: rows=slow,fast; cols=density,duration."""
    fig = plt.figure(figsize=(FULL_WIDTH_IN, FULL_WIDTH_IN * 0.55))
    gs = gridspec.GridSpec(2, 2, wspace=0.35, hspace=0.30,
                           left=0.04, right=0.96, top=0.88, bottom=0.06)

    den_slow = np.nanmean(CountMat_slow / sleep_mins, axis=0)
    den_fast = np.nanmean(CountMat_fast / sleep_mins, axis=0)
    dur_slow = np.nanmean(DurMat_slow, axis=0)
    dur_fast = np.nanmean(DurMat_fast, axis=0)

    den_vlim = (min(np.nanmin(den_slow), np.nanmin(den_fast)),
                max(np.nanmax(den_slow), np.nanmax(den_fast)))
    dur_vlim = (min(np.nanmin(dur_slow), np.nanmin(dur_fast)),
                max(np.nanmax(dur_slow), np.nanmax(dur_fast)))

    panels = [
        (0, 0, den_slow, den_vlim, 'Density (spindles/min)', 'a'),
        (0, 1, dur_slow, dur_vlim, 'Duration (s)',            'b'),
        (1, 0, den_fast, den_vlim, 'Density (spindles/min)', 'c'),
        (1, 1, dur_fast, dur_vlim, 'Duration (s)',            'd'),
    ]
    row_labels = ['Slow\n(10\u201312 Hz)', 'Fast\n(12\u201316 Hz)']

    for row, col, data, vlim, metric_label, panel_lbl in panels:
        ax = fig.add_subplot(gs[row, col])
        im, _ = mne.viz.plot_topomap(
            data, info, axes=ax, show=False,
            cmap='turbo', sphere=SPHERE, contours=0,
            vlim=vlim, sensors=False
        )
        cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02, aspect=15)
        cb.ax.tick_params(labelsize=5, width=0.4, length=2)
        cb.outline.set_linewidth(0.4)
        if row == 0:
            ax.set_title(metric_label, fontsize=FONT_TITLE, fontweight='bold', pad=6)
        if col == 0:
            ax.set_ylabel(row_labels[row], fontsize=FONT_TITLE, fontweight='bold',
                          labelpad=12, rotation=0, ha='right', va='center')
        ax.text(-0.15, 1.08, panel_lbl, transform=ax.transAxes,
                fontsize=FONT_PANEL, fontweight='bold', va='top')

    out = os.path.join(OUT_DIR, 'Figure2_normative_topography.png')
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f'[SAVED] {out}')


# ============================================================
# HELPER: topo-row with beta, t-stat, and scatter columns
# ============================================================
def _topo_scatter_row(fig, gs, r, row_label, corr_tag, ddict, metric,
                      overlay_channels, overlay_corrected, scatter_mat,
                      scatter_ylabel, panel_labels, pi,
                      scatter_roi=None, dd_roi=None):
    """Draw one row of the 3-column regression figure (beta | t-stat | scatter).

    Parameters
    ----------
    overlay_channels : list
        Channels drawn on the t-stat topomap (green if corrected, black if not).
    dd_roi : list or None
        Data-driven ROI channels (black open circles) drawn *in addition* to
        the green permutation cluster when overlay_corrected is True.
    scatter_roi : list or None
        Channels used for the scatter-plot ROI average.  Falls back to
        overlay_channels when not supplied.
    """
    betas, tstats, pvals = channel_regression(ddict, metric, covUse)
    display_label = f'{row_label}\n{corr_tag}'
    if scatter_roi is None:
        scatter_roi = overlay_channels

    # ----- COL 0: beta coefficients -----
    ax_b = fig.add_subplot(gs[r, 0])
    vlim_b = np.nanmax(np.abs(betas[np.isfinite(betas)])) if np.any(np.isfinite(betas)) else 1.0
    im_b, _ = mne.viz.plot_topomap(
        betas, info, axes=ax_b, show=False,
        cmap='PuOr_r', sphere=SPHERE, contours=0,
        vlim=(-vlim_b, vlim_b), sensors=False
    )
    cb_b = plt.colorbar(im_b, ax=ax_b, shrink=0.72, pad=0.02, aspect=15)
    cb_b.ax.tick_params(labelsize=5, width=0.4, length=2)
    cb_b.outline.set_linewidth(0.4)
    if r == 0:
        ax_b.set_title('Beta coefficient', fontsize=FONT_TITLE, fontweight='bold', pad=6)
    ax_b.set_ylabel(display_label, fontsize=FONT_LABEL, fontweight='bold',
                    labelpad=20, rotation=0, ha='right', va='center')
    ax_b.text(-0.08, 1.08, panel_labels[pi], transform=ax_b.transAxes,
              fontsize=FONT_PANEL, fontweight='bold', va='top')
    pi += 1

    # ----- COL 1: t-statistics + significance mask + cluster overlay -----
    ax_t = fig.add_subplot(gs[r, 1])
    vlim_t = np.nanmax(np.abs(tstats[np.isfinite(tstats)])) if np.any(np.isfinite(tstats)) else 2.5
    mask_sig = pvals < 0.05
    im_t, _ = mne.viz.plot_topomap(
        tstats, info, axes=ax_t, show=False,
        cmap='RdBu_r', sphere=SPHERE, contours=0,
        vlim=(-vlim_t, vlim_t), sensors=False,
        mask=mask_sig,
        mask_params=dict(marker='o', markerfacecolor='black',
                         markeredgecolor='black', markersize=2.5,
                         markeredgewidth=0.3)
    )
    # Data-driven ROI boundaries (black open circles) — always shown when provided
    if dd_roi:
        ax_t.scatter(points_x[dd_roi], points_y[dd_roi],
                     s=8, facecolors='none', edgecolors='black',
                     linewidths=0.5, zorder=8, alpha=0.6)
    elif not overlay_corrected and overlay_channels:
        # For uncorrected rows the overlay IS the dd ROI
        ax_t.scatter(points_x[overlay_channels], points_y[overlay_channels],
                     s=8, facecolors='none', edgecolors='black',
                     linewidths=0.5, zorder=8, alpha=0.6)
    # Permutation cluster (green open circles) — only for corrected effects
    if overlay_corrected and overlay_channels:
        ax_t.scatter(points_x[overlay_channels], points_y[overlay_channels],
                     s=14, facecolors='none', edgecolors='#2ca02c',
                     linewidths=0.9, zorder=9, alpha=0.9)
    cb_t = plt.colorbar(im_t, ax=ax_t, shrink=0.72, pad=0.02, aspect=15)
    cb_t.ax.tick_params(labelsize=5, width=0.4, length=2)
    cb_t.outline.set_linewidth(0.4)
    if r == 0:
        ax_t.set_title('t-statistic (p < .05 uncorr.)', fontsize=FONT_TITLE, fontweight='bold', pad=6)
    ax_t.text(-0.08, 1.08, panel_labels[pi], transform=ax_t.transAxes,
              fontsize=FONT_PANEL, fontweight='bold', va='top')
    pi += 1

    # ----- COL 2: partial-residual scatter -----
    ax_s = fig.add_subplot(gs[r, 2])
    _scatter_roi_vs_hi(ax_s, scatter_mat, scatter_roi, covUse, scatter_ylabel)
    if r == 0:
        ax_s.set_title('Partial residuals', fontsize=FONT_TITLE, fontweight='bold', pad=6)
    ax_s.text(-0.18, 1.08, panel_labels[pi], transform=ax_s.transAxes,
              fontsize=FONT_PANEL, fontweight='bold', va='top')
    pi += 1
    return pi


# ============================================================
# FIGURE 3 — Cluster-Corrected HI Associations (2 rows x 3 cols)
# ============================================================
def make_figure3():
    """2 rows (fast duration, slow peak freq) x 3 cols (beta, t-stat, scatter).
    Shows only effects that survived cluster-based permutation correction.
    """
    # rows_config: (label, corr_tag, data_dict, metric, perm_cluster, corrected,
    #               scatter_mat, ylabel, dd_roi_for_overlay, scatter_roi)
    rows_config = [
        ('Fast duration',   _ctag('fast', 'Duration'), fast_data, 'Duration',
         perm_fast_dur, True, DurMat_fast, 'Duration (s)', ddroi_fast_dur, ddroi_fast_dur),
        ('Slow peak freq.', _ctag('slow', 'Frequency'), slow_data, 'Frequency',
         perm_slow_freq, True, FreqMat_slow, 'Peak freq. (Hz)', [], perm_slow_freq),
    ]

    fig = plt.figure(figsize=(FULL_WIDTH_IN, FULL_WIDTH_IN * 0.55))
    gs = gridspec.GridSpec(2, 3, wspace=0.45, hspace=0.30,
                           left=0.10, right=0.96, top=0.88, bottom=0.08,
                           width_ratios=[1, 1, 0.9])
    panel_labels = ['a', 'b', 'c', 'd', 'e', 'f']
    pi = 0
    for r, (label, ctag, ddict, metric, overlay, corrected, smat, ylabel, dd_roi, s_roi) in enumerate(rows_config):
        pi = _topo_scatter_row(fig, gs, r, label, ctag, ddict, metric,
                               overlay, corrected, smat, ylabel, panel_labels, pi,
                               scatter_roi=s_roi, dd_roi=dd_roi)

    out = os.path.join(OUT_DIR, 'Figure3_HI_corrected_effects.png')
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f'[SAVED] {out}')


# ============================================================
# FIGURE 4 — Uncorrected HI Associations for TOVA ROIs (2 rows x 3 cols)
# ============================================================
def make_figure4():
    """2 rows (slow amplitude, slow duration) x 3 cols (beta, t-stat, scatter).
    Shows uncorrected effects used as data-driven ROIs for behavioral analyses.
    """
    rows_config = [
        ('Slow amplitude', _ctag('slow', 'Amplitude'), slow_data, 'Amplitude', ddroi_slow_amp, False, AmpMat_slow, 'Amplitude (\u00b5V)'),
        ('Slow duration',  _ctag('slow', 'Duration'),  slow_data, 'Duration',  ddroi_slow_dur, False, DurMat_slow, 'Duration (s)'),
    ]

    fig = plt.figure(figsize=(FULL_WIDTH_IN, FULL_WIDTH_IN * 0.55))
    gs = gridspec.GridSpec(2, 3, wspace=0.45, hspace=0.30,
                           left=0.10, right=0.96, top=0.88, bottom=0.08,
                           width_ratios=[1, 1, 0.9])
    panel_labels = ['a', 'b', 'c', 'd', 'e', 'f']
    pi = 0
    for r, (label, ctag, ddict, metric, overlay, corrected, smat, ylabel) in enumerate(rows_config):
        pi = _topo_scatter_row(fig, gs, r, label, ctag, ddict, metric,
                               overlay, corrected, smat, ylabel, panel_labels, pi)

    out = os.path.join(OUT_DIR, 'Figure4_HI_uncorrected_effects.png')
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f'[SAVED] {out}')


# ============================================================
# FIGURE 5 — Coefficient Forest Plot (Cognition)
# ============================================================
def make_figure5():
    """
    Rows = 3 ROI predictors (ant_fast_dur, ant_slow_dur, pos_slow_amp)
    Cols = 3 TOVA outcomes (d-prime, omissions, commissions)
    Standardized beta with 95% CI bars.
    Model: TOVA_z ~ ROI_z + age_years + gender + overall_hi
    """
    # --- Data preparation (replicating R pipeline) ---
    cov = pd.read_csv(COV_PATH)
    roi = pd.read_csv(ROI_PATH)
    # ant_slow_peakfreq now ships in roi_values_corrected.csv (cluster-corrected
    # ROI, written by compute_rois_corrected.py) rather than being recomputed
    # inline here, so the figure and behavioral_models.py share one ROI source.
    df = cov.merge(roi, on='id', how='left')

    # Filter to subjects with valid TOVA
    df = df.dropna(subset=['DPRIMEQ1'])

    # Transformations
    df['RT_log'] = np.log(df['RTMEANQ1'])
    df['OM_sqrt'] = np.sqrt(df['OMPERQ1'])
    df['COM_sqrt'] = np.sqrt(df['COMPERQ1'])

    # Remove outliers Z>3 on transformed RT and OM
    for col in ['RT_log', 'OM_sqrt']:
        z = np.abs(stats.zscore(df[col].dropna()))
        valid_idx = df[col].dropna().index[z < 3]
        df = df.loc[df.index.isin(valid_idx)]

    # Z-score DVs and ROI predictors
    dv_cols = ['DPRIMEQ1', 'OM_sqrt', 'COM_sqrt']
    roi_cols = ['ant_fast_dur', 'ant_slow_peakfreq', 'ant_slow_dur', 'pos_slow_amp']

    for col in dv_cols:
        df[col + '_z'] = stats.zscore(df[col].dropna())
        # Re-align with df
        z_series = (df[col] - df[col].mean()) / df[col].std()
        df[col + '_z'] = z_series

    for col in roi_cols:
        z_series = (df[col] - df[col].mean()) / df[col].std()
        df[col + '_z'] = z_series

    print(f'[INFO] Figure 4: N = {len(df)} after TOVA filter + outlier removal')

    # --- Run regressions ---
    outcomes = [
        ("d' (TOVA)", 'DPRIMEQ1_z'),
        ('Omission errors', 'OM_sqrt_z'),
        ('Commission errors', 'COM_sqrt_z'),
    ]
    roi_predictors = [
        ('Ant. fast duration\n(cluster-corrected*)', 'ant_fast_dur_z'),
        ('Ant. slow peak freq.\n(cluster-corrected*)', 'ant_slow_peakfreq_z'),
        ('Ant. slow duration\n(uncorrected)', 'ant_slow_dur_z'),
        ('Post. slow amplitude\n(uncorrected)', 'pos_slow_amp_z'),
    ]

    # Colors: primary (dark blue), secondary corrected (teal), uncorrected (muted grey)
    colors_primary   = '#1a5276'   # strong dark blue for cluster-corrected primary
    colors_secondary = ['#2e86c1', '#85929e', '#aab7b8']  # teal for 2nd corrected, muted grey for uncorrected

    # Collect results: rows = ROI predictors, cols = outcomes
    results = {}  # (roi_idx, out_idx) -> (beta, ci_lo, ci_hi, p)
    for ri, (roi_label, roi_var) in enumerate(roi_predictors):
        for oi, (out_label, out_var) in enumerate(outcomes):
            sub = df.dropna(subset=[out_var, roi_var, 'age_years', 'gender', 'overall_hi'])
            y = sub[out_var].values
            X = sm.add_constant(sub[[roi_var, 'age_years', 'gender', 'overall_hi']].values)
            model = sm.OLS(y, X).fit()
            beta = model.params[1]
            ci = model.conf_int(alpha=0.05)
            ci_lo, ci_hi = ci[1, 0], ci[1, 1]
            pval = model.pvalues[1]
            results[(ri, oi)] = (beta, ci_lo, ci_hi, pval)

    # --- Plot ---
    n_roi = len(roi_predictors)
    n_out = len(outcomes)

    fig, axes = plt.subplots(1, n_out, figsize=(FULL_WIDTH_IN, FULL_WIDTH_IN * 0.55))
    fig.subplots_adjust(wspace=0.35, left=0.22, right=0.85, top=0.82, bottom=0.10)

    y_positions = np.arange(n_roi)[::-1]  # top to bottom

    panel_labels = ['a', 'b', 'c']

    for oi, (out_label, out_var) in enumerate(outcomes):
        ax = axes[oi]
        for ri in range(n_roi):
            beta, ci_lo, ci_hi, pval = results[(ri, oi)]
            # Primary ROI (index 0) gets strong color; secondary get muted
            if ri == 0:
                color = colors_primary
                ms = 6
                elw = 0.8
            else:
                color = colors_secondary[ri - 1]
                ms = 5
                elw = 0.6
            marker = 'D' if pval < 0.05 else 'o'
            mfc = color if pval < 0.05 else 'white'

            ax.errorbar(beta, y_positions[ri],
                        xerr=[[beta - ci_lo], [ci_hi - beta]],
                        fmt=marker, color=color, markerfacecolor=mfc,
                        markeredgecolor=color, markersize=ms,
                        capsize=2.5, capthick=0.6, elinewidth=elw,
                        linewidth=elw, zorder=5)

            # p-value annotation
            if pval < 0.001:
                p_txt = 'p < .001'
            elif pval < 0.01:
                p_txt = f'p = {pval:.3f}'
            elif pval < 0.05:
                p_txt = f'p = {pval:.2f}'
            else:
                p_txt = f'p = {pval:.2f}'
            ax.annotate(p_txt, xy=(beta, y_positions[ri]),
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', va='bottom', fontsize=5, color='#555555',
                        clip_on=False)

        ax.axvline(0, color='grey', linewidth=0.4, linestyle='--', zorder=1)
        ax.set_xlabel('Standardized beta', fontsize=FONT_LABEL)
        ax.set_title(out_label, fontsize=FONT_TITLE, fontweight='bold', pad=14)
        ax.set_ylim(-0.7, n_roi - 0.3)
        ax.set_yticks(y_positions)
        if oi == 0:
            ax.set_yticklabels([lbl for lbl, _ in roi_predictors], fontsize=FONT_LABEL)
        else:
            ax.set_yticklabels(['' for _ in roi_predictors])
        ax.tick_params(axis='both', length=2, width=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(AXES_LW)
        ax.spines['bottom'].set_linewidth(AXES_LW)

        # Panel label
        ax.text(-0.08, 1.15, panel_labels[oi], transform=ax.transAxes,
                fontsize=FONT_PANEL, fontweight='bold', va='top')

    # No suptitle — title goes in the figure caption in the manuscript

    out = os.path.join(OUT_DIR, 'Figure5_cognition_coefficients.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(f'[SAVED] {out}')


# ============================================================
# SUPP FIGURE — Slow Spindle Peak Frequency Topography
# ============================================================
def make_supp_peakfreq():
    """Supplementary figure: slow spindle peak frequency regression maps.
    1 row x 2 cols: beta coefficients + t-statistics with cluster overlay.
    """
    betas, tstats, pvals = channel_regression(slow_data, 'Frequency', covUse)

    # Find ROI channels for peak freq (uncorrected clusters)
    roi_freq, _, _ = find_roi_channels(slow_data, 'Frequency', covUse)

    fig = plt.figure(figsize=(FULL_WIDTH_IN, FULL_WIDTH_IN * 0.35))
    gs = gridspec.GridSpec(1, 2, wspace=0.40,
                           left=0.06, right=0.94, top=0.82, bottom=0.08)

    # ----- LEFT: beta coefficients -----
    ax_b = fig.add_subplot(gs[0, 0])
    vlim_b = np.nanmax(np.abs(betas[np.isfinite(betas)])) if np.any(np.isfinite(betas)) else 0.1
    im_b, _ = mne.viz.plot_topomap(
        betas, info, axes=ax_b, show=False,
        cmap='PuOr_r', sphere=SPHERE, contours=0,
        vlim=(-vlim_b, vlim_b), sensors=False
    )
    cb_b = plt.colorbar(im_b, ax=ax_b, shrink=0.75, pad=0.02, aspect=15)
    cb_b.ax.tick_params(labelsize=5, width=0.4, length=2)
    cb_b.outline.set_linewidth(0.4)
    ax_b.set_title('Beta coefficient', fontsize=FONT_TITLE, fontweight='bold', pad=6)
    ax_b.set_ylabel(f"Slow peak freq.\n{_ctag('slow', 'Frequency')}", fontsize=FONT_LABEL,
                     fontweight='bold', labelpad=18, rotation=0, ha='right', va='center')
    ax_b.text(-0.08, 1.08, 'a', transform=ax_b.transAxes,
              fontsize=FONT_PANEL, fontweight='bold', va='top')

    # ----- RIGHT: t-statistics -----
    ax_t = fig.add_subplot(gs[0, 1])
    vlim_t = np.nanmax(np.abs(tstats[np.isfinite(tstats)])) if np.any(np.isfinite(tstats)) else 2.5
    mask_sig = pvals < 0.05
    im_t, _ = mne.viz.plot_topomap(
        tstats, info, axes=ax_t, show=False,
        cmap='RdBu_r', sphere=SPHERE, contours=0,
        vlim=(-vlim_t, vlim_t), sensors=False,
        mask=mask_sig,
        mask_params=dict(marker='o', markerfacecolor='black',
                         markeredgecolor='black', markersize=2.5,
                         markeredgewidth=0.3)
    )
    if roi_freq:
        ax_t.scatter(points_x[roi_freq], points_y[roi_freq],
                     s=8, facecolors='none', edgecolors='black',
                     linewidths=0.5, zorder=9, alpha=0.7)
    cb_t = plt.colorbar(im_t, ax=ax_t, shrink=0.75, pad=0.02, aspect=15)
    cb_t.ax.tick_params(labelsize=5, width=0.4, length=2)
    cb_t.outline.set_linewidth(0.4)
    ax_t.set_title('t-statistic (p < .05 uncorr.)', fontsize=FONT_TITLE, fontweight='bold', pad=6)
    ax_t.text(-0.08, 1.08, 'b', transform=ax_t.transAxes,
              fontsize=FONT_PANEL, fontweight='bold', va='top')

    out = os.path.join(OUT_DIR, 'Supp_Figure_peakfreq_topography.png')
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f'[SAVED] {out}')


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print('\n=== Generating Figure 1 (methods illustration) ===')
    make_figure1()
    print('\n=== Generating Figure 2 (normative topography) ===')
    make_figure2()
    print('\n=== Generating Figure 3 (corrected HI effects) ===')
    make_figure3()
    print('\n=== Generating Figure 4 (uncorrected HI effects) ===')
    make_figure4()
    print('\n=== Generating Figure 5 (cognition) ===')
    make_figure5()
    print('\n=== ALL FIGURES COMPLETE ===')
