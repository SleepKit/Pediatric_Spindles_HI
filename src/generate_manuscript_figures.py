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
from matplotlib.patches import Patch
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

# Final manuscript figures are written under output/ as BOTH png and pdf at 600 dpi
# (SLEEP submission format); see save_manuscript_fig().
PAPER_DIR = PROJECT / "output"


def save_manuscript_fig(fig, name, **kw):
    """Save a manuscript figure as PNG and PDF at 600 dpi under output/."""
    kw.setdefault("dpi", DPI)
    for ext in ("png", "pdf"):
        path = PAPER_DIR / f"{name}.{ext}"
        fig.savefig(path, **kw)
        print(f"[SAVED] {path}")

# Cluster-permutation pipeline schematic (Figure 1c): clean, label-free panel
# rasterized from output/figure1_panel-C.svg (poster artwork, internal labels removed).
TOPO_PATH = PROJECT / "output/figure1_panelC.png"

# ============================================================
# STYLING CONSTANTS (Nature figure bible)
# ============================================================
FULL_WIDTH_IN = 7.2          # 183 mm
DPI = 600                    # SLEEP submission resolution (png + pdf)
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
print(f'[INFO] {nCh} channels loaded (geometry only; this module runs no analysis)')

# ------------------------------------------------------------------
# Analysis and figure creation are SEPARATE steps. Every quantity below is
# precomputed by scripts/compute_figure_data.py and only loaded here; this
# module performs no raw-data loading, channel-wise regression, or permutation
# testing. Run `python scripts/compute_figure_data.py` first.
# ------------------------------------------------------------------
from compute_figure_data import load as load_figure_data  # noqa: E402
_FIG = load_figure_data()
_REG = _FIG['reg']          # (band, metric) -> {'beta','t','p'} channel maps
_SCATTER = _FIG['scatter']  # (band, metric) -> partial-residual scatter series
_TOPO = _FIG['topo']        # normative per-channel topographies (Figure 2)
_ROI = _FIG['roi']          # ROI / permutation-cluster channel index lists
_CTAG = _FIG['ctag']        # (band, metric) -> corrected cluster-p tag string
_FIG5 = _FIG['fig5']        # Figure 5 forest-plot coefficients

ddroi_slow_amp = _ROI['ddroi_slow_amp']
ddroi_slow_dur = _ROI['ddroi_slow_dur']
ddroi_fast_dur = _ROI['ddroi_fast_dur']
perm_fast_dur = _ROI['perm_fast_dur']
perm_slow_freq = _ROI['perm_slow_freq']

# ROI / permutation-cluster channel lists and regression maps come from the
# precomputed artifact loaded above; no analysis runs in this module.


def _ctag(band, metric):
    """Corrected cluster-p tag '(cluster p = X*)' from the precomputed artifact."""
    return _CTAG.get((band, metric), '(cluster p = n/a)')


# ============================================================
# HELPER: partial-residual scatter drawn from a precomputed series
# ============================================================
def _plot_scatter(ax, series, ylabel, color='#1a5276'):
    """Draw a precomputed partial-residual scatter (ROI metric vs logHI | age, sex)."""
    if series is None:
        ax.text(0.5, 0.5, 'No ROI', transform=ax.transAxes,
                ha='center', va='center', fontsize=FONT_LABEL, color='grey')
        return
    x_resid, y_resid = series['x'], series['y']
    ax.scatter(x_resid, y_resid, s=12, alpha=0.55, color=color,
               edgecolors='white', linewidths=0.3, zorder=5)
    x_fit = np.linspace(x_resid.min(), x_resid.max(), 100)
    ax.plot(x_fit, series['slope'] * x_fit + series['intercept'],
            color=color, linewidth=1.0, zorder=6)
    ax.text(0.95, 0.95, f"r = {series['r']:.2f}",
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
                fontsize=4, va='top', ha='right', ma='left', family='monospace',
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
    """Standalone methods figure: hypnogram (a) + spindle detection (b) +
    cluster-permutation pipeline schematic (c)."""
    LEFT, RIGHT = 0.10, 0.96
    PANEL_W = RIGHT - LEFT
    FIG_H = FULL_WIDTH_IN * 1.15

    # Panel c spans the full panel width (same as a/b); its height follows the
    # schematic's native aspect ratio so the image fills the width undistorted.
    topo_img = plt.imread(TOPO_PATH)
    img_aspect = topo_img.shape[1] / topo_img.shape[0]
    c_bottom = 0.05
    c_height = (PANEL_W * FULL_WIDTH_IN / img_aspect) / FIG_H

    fig = plt.figure(figsize=(FULL_WIDTH_IN, FIG_H))
    gs = gridspec.GridSpec(2, 1,
                           height_ratios=[0.8, 1.2],
                           hspace=0.45,
                           left=LEFT, right=RIGHT,
                           top=0.96, bottom=c_bottom + c_height + 0.07)

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

    # ---- panel c: cluster-permutation pipeline schematic ----
    ax_topo = fig.add_axes([LEFT, c_bottom, PANEL_W, c_height])
    ax_topo.imshow(topo_img, aspect='auto', interpolation='none')
    ax_topo.axis('off')
    ax_topo.text(-0.05, 1.08, 'c', transform=ax_topo.transAxes,
                 fontsize=FONT_PANEL, fontweight='bold', va='top')

    save_manuscript_fig(fig, 'Fig_1')
    plt.close(fig)


# ============================================================
# FIGURE 2 — Normative Spindle Topography (2x2)
# ============================================================
def make_figure2():
    """2x2 grid: rows=slow,fast; cols=density,duration."""
    fig = plt.figure(figsize=(FULL_WIDTH_IN, FULL_WIDTH_IN * 0.55))
    gs = gridspec.GridSpec(2, 2, wspace=0.35, hspace=0.30,
                           left=0.04, right=0.96, top=0.88, bottom=0.06)

    den_slow = _TOPO['den_slow']
    den_fast = _TOPO['den_fast']
    dur_slow = _TOPO['dur_slow']
    dur_fast = _TOPO['dur_fast']

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

    save_manuscript_fig(fig, 'Fig_2')
    plt.close(fig)


# ============================================================
# HELPER: topo-row with beta, t-stat, and scatter columns
# ============================================================
def _topo_scatter_row(fig, gs, r, row_label, corr_tag, band, metric,
                      overlay_channels, overlay_corrected,
                      scatter_ylabel, panel_labels, pi, dd_roi=None):
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
    reg = _REG[(band, metric)]
    betas, tstats, pvals = reg['beta'], reg['t'], reg['p']
    display_label = f'{row_label}\n{corr_tag}'

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
    _plot_scatter(ax_s, _SCATTER[(band, metric)], scatter_ylabel)
    if r == 0:
        ax_s.set_title('Partial residuals', fontsize=FONT_TITLE, fontweight='bold', pad=6)
    ax_s.text(-0.30, 1.08, panel_labels[pi], transform=ax_s.transAxes,
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
        ('Fast duration',   _ctag('fast', 'Duration'), 'fast', 'Duration',
         perm_fast_dur, True, 'Duration (s)', ddroi_fast_dur),
        ('Slow peak freq.', _ctag('slow', 'Frequency'), 'slow', 'Frequency',
         perm_slow_freq, True, 'Peak freq. (Hz)', []),
    ]

    fig = plt.figure(figsize=(FULL_WIDTH_IN, FULL_WIDTH_IN * 0.55))
    gs = gridspec.GridSpec(2, 3, wspace=0.45, hspace=0.30,
                           left=0.10, right=0.96, top=0.88, bottom=0.08,
                           width_ratios=[1, 1, 0.9])
    panel_labels = ['a', 'b', 'c', 'd', 'e', 'f']
    pi = 0
    for r, (label, ctag, band, metric, overlay, corrected, ylabel, dd_roi) in enumerate(rows_config):
        pi = _topo_scatter_row(fig, gs, r, label, ctag, band, metric,
                               overlay, corrected, ylabel, panel_labels, pi,
                               dd_roi=dd_roi)

    save_manuscript_fig(fig, 'Fig_3')
    plt.close(fig)


# ============================================================
# FIGURE 4 — Uncorrected HI Associations for TOVA ROIs (2 rows x 3 cols)
# ============================================================
def make_figure4():
    """2 rows (slow amplitude, slow duration) x 3 cols (beta, t-stat, scatter).
    Shows uncorrected effects used as data-driven ROIs for behavioral analyses.
    """
    rows_config = [
        ('Slow amplitude', _ctag('slow', 'Amplitude'), 'slow', 'Amplitude', ddroi_slow_amp, False, 'Amplitude (\u00b5V)'),
        ('Slow duration',  _ctag('slow', 'Duration'),  'slow', 'Duration',  ddroi_slow_dur, False, 'Duration (s)'),
    ]

    fig = plt.figure(figsize=(FULL_WIDTH_IN, FULL_WIDTH_IN * 0.55))
    gs = gridspec.GridSpec(2, 3, wspace=0.45, hspace=0.30,
                           left=0.10, right=0.96, top=0.88, bottom=0.08,
                           width_ratios=[1, 1, 0.9])
    panel_labels = ['a', 'b', 'c', 'd', 'e', 'f']
    pi = 0
    for r, (label, ctag, band, metric, overlay, corrected, ylabel) in enumerate(rows_config):
        pi = _topo_scatter_row(fig, gs, r, label, ctag, band, metric,
                               overlay, corrected, ylabel, panel_labels, pi)

    save_manuscript_fig(fig, 'Fig_4')
    plt.close(fig)


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
    # Forest-plot coefficients are precomputed by compute_figure_data.py; this
    # figure runs no regressions.
    results = _FIG5['results']  # (roi_idx, out_idx) -> (beta, ci_lo, ci_hi, p)

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
    colors_primary   = '#1a5276'
    colors_secondary = ['#2e86c1', '#85929e', '#aab7b8']

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

    save_manuscript_fig(fig, 'Fig_5', bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)


# ============================================================
# SUPPLEMENTARY FIGURE S5 — Spindle Density HI Trends (topomaps only)
# ============================================================
def _density_row(fig, gs, r, band, row_label, panel_labels):
    """Draw one density row (beta | t-stat topomaps) for band's HI association."""
    reg = _REG[(band, 'Density')]
    betas, tstats, pvals = reg['beta'], reg['t'], reg['p']

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
    ax_b.set_ylabel(f"{row_label}\n{_ctag(band, 'Density')}",
                    fontsize=FONT_LABEL, fontweight='bold',
                    labelpad=20, rotation=0, ha='right', va='center')
    ax_b.text(-0.08, 1.12, panel_labels[0], transform=ax_b.transAxes,
              fontsize=FONT_PANEL, fontweight='bold', va='top')

    # ----- COL 1: t-statistics + p < .05 significance mask -----
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
    cb_t = plt.colorbar(im_t, ax=ax_t, shrink=0.72, pad=0.02, aspect=15)
    cb_t.ax.tick_params(labelsize=5, width=0.4, length=2)
    cb_t.outline.set_linewidth(0.4)
    if r == 0:
        ax_t.set_title('t-statistic (p < .05 uncorr.)', fontsize=FONT_TITLE, fontweight='bold', pad=6)
    ax_t.text(-0.08, 1.12, panel_labels[1], transform=ax_t.transAxes,
              fontsize=FONT_PANEL, fontweight='bold', va='top')


def make_figure_s5():
    """Beta and t-statistic topomaps for the slow- and fast-density HI associations.

    Documents the density trends noted in the main text: neither band reaches a
    suprathreshold cluster, so unlike Figures 3-4 there is no data-driven ROI and
    no partial-residual scatter — only the channel-wise topographic maps are
    shown, each with the p < .05 significance mask, making the nulls explicit.
    Rows mirror Figure 2: slow density (top), fast density (bottom).
    """
    fig = plt.figure(figsize=(FULL_WIDTH_IN * 0.62, FULL_WIDTH_IN * 0.66))
    gs = gridspec.GridSpec(2, 2, wspace=0.45, hspace=0.30,
                           left=0.15, right=0.95, top=0.90, bottom=0.05)
    _density_row(fig, gs, 0, 'slow', 'Slow density', ['a', 'b'])
    _density_row(fig, gs, 1, 'fast', 'Fast density', ['c', 'd'])
    save_manuscript_fig(fig, 'Fig_S5')
    plt.close(fig)


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
    print('\n=== Generating Figure S5 (slow density HI trend) ===')
    make_figure_s5()
    print('\n=== ALL FIGURES COMPLETE ===')
