#!/usr/bin/env python3
"""
Schematic methods figure for the statistical pipeline:
channel-wise regression model -> observed t-map / cluster forming ->
Freedman-Lane residual permutation (x5,000) -> null distribution & corrected p.

Self-contained schematic (no participant data required). Visual style follows the
manuscript "Nature figure bible" used by generate_manuscript_figures.py.
"""

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "temp/cache/matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Polygon

# ----------------------------------------------------------------------
# Style (matches generate_manuscript_figures.py)
# ----------------------------------------------------------------------
DPI = 600
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 6,
    "axes.linewidth": 0.6,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
})

C_BLUE   = "#1a5276"   # primary dark blue
C_BLUE2  = "#2171b5"
C_LBLUE  = "#6baed6"
C_GREEN  = "#2ca02c"   # permutation cluster outline
C_ORANGE = "#d94801"
C_GREY   = "#555555"
C_LGREY  = "#cccccc"
C_BOX    = "#f4f6f7"

FONT_PANEL = 12
FONT_TITLE = 7
FONT_SMALL = 5
FONT_TINY  = 4.5

rng = np.random.default_rng(7)

# ----------------------------------------------------------------------
# Figure scaffold: 4 panels left -> right
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(7.2, 2.55))
gs = fig.add_gridspec(1, 4, left=0.045, right=0.985, top=0.83, bottom=0.09,
                      wspace=0.42)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])
axD = fig.add_subplot(gs[0, 3])

for ax in (axA, axB, axC, axD):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


LETTER_Y = 0.965   # figure coords — keeps labels aligned across panels
TITLE_Y = 0.895


def panel_label(ax, letter, title):
    pos = ax.get_position()
    fig.text(pos.x0 - 0.006, LETTER_Y, letter, fontsize=FONT_PANEL,
             fontweight="bold", va="top", ha="left")
    fig.text((pos.x0 + pos.x1) / 2, TITLE_Y, title, fontsize=FONT_TITLE,
             fontweight="bold", va="top", ha="center", color=C_BLUE)


def flow_arrow(x):
    """Horizontal arrow between panels (figure coords)."""
    fig.add_artist(FancyArrowPatch((x, 0.45), (x + 0.022, 0.45),
                   transform=fig.transFigure, arrowstyle="-|>",
                   mutation_scale=9, lw=1.1, color=C_GREY))


for x in (0.258, 0.503, 0.748):
    flow_arrow(x)

# ======================================================================
# Panel A — channel-wise regression model
# ======================================================================
panel_label(axA, "a", "Channel-wise model")
axA.set_xlim(0, 1); axA.set_ylim(0, 1)

# mini scatter (HI vs spindle metric at one electrode)
inset = axA.inset_axes([0.14, 0.40, 0.78, 0.42])
xh = rng.uniform(0, 1, 26)
yh = 0.75 - 0.55 * xh + rng.normal(0, 0.08, 26)
inset.scatter(xh, yh, s=7, color=C_BLUE, alpha=0.7, edgecolors="white", linewidths=0.25)
xf = np.array([0, 1])
inset.plot(xf, 0.75 - 0.55 * xf, color=C_ORANGE, lw=1.0)
inset.set_xlabel(r"log$_{10}$(HI+1)", fontsize=FONT_SMALL, labelpad=1)
inset.set_ylabel("spindle metric", fontsize=FONT_SMALL, labelpad=1)
inset.tick_params(length=1.5, labelsize=0)
inset.set_xticks([]); inset.set_yticks([])
for s in inset.spines.values():
    s.set_linewidth(0.6); s.set_color(C_GREY)
inset.text(0.97, 0.93, "one electrode", transform=inset.transAxes,
           ha="right", va="top", fontsize=FONT_TINY, color=C_GREY, style="italic")

# the model equation (sits below the scatter)
axA.text(0.5, 0.225,
         r"$Y = \beta_0 + \beta_1\,\log_{10}(\mathrm{HI}{+}1)$" "\n"
         r"$\quad\;+\,\beta_2\,\mathrm{Age} + \beta_3\,\mathrm{Sex} + \varepsilon$",
         ha="center", va="center", fontsize=6.2)
axA.text(0.5, 0.04,
         r"extract $t(\beta_1)$ at each of 172 channels",
         ha="center", va="center", fontsize=FONT_SMALL, color=C_BLUE,
         fontweight="bold")

# ======================================================================
# Helper — head schematic with electrode dots
# ======================================================================
def draw_head(ax, cx, cy, r):
    ax.add_patch(Circle((cx, cy), r, fill=False, lw=0.9, ec="#333333", zorder=3))
    # nose
    ax.add_patch(Polygon([[cx - r * 0.16, cy + r * 0.98],
                          [cx, cy + r * 1.18],
                          [cx + r * 0.16, cy + r * 0.98]],
                         closed=True, fill=False, lw=0.9, ec="#333333", zorder=3))
    # ears
    for sgn in (-1, 1):
        ax.add_patch(Circle((cx + sgn * r * 1.02, cy), r * 0.12,
                            fill=False, lw=0.9, ec="#333333", zorder=3))


def head_electrodes(r):
    """Return electrode x,y within unit head + an 'anterior cluster' mask."""
    pts = []
    for ring_r, n in [(0.0, 1), (0.32, 8), (0.62, 14), (0.88, 18)]:
        for k in range(n):
            ang = 2 * np.pi * k / max(n, 1) + (0.2 if ring_r > 0 else 0)
            pts.append((ring_r * np.cos(ang), ring_r * np.sin(ang)))
    pts = np.array(pts)
    pts = pts[np.argsort(-pts[:, 1])]  # top (anterior) first
    return pts


# ======================================================================
# Panel B — observed t-map + cluster forming
# ======================================================================
panel_label(axB, "b", "Observed $t$-map → cluster")
axB.set_xlim(-1.4, 1.4); axB.set_ylim(-1.45, 1.55)
axB.set_aspect("equal")
draw_head(axB, 0, 0.05, 1.12)

pts = head_electrodes(1.0) * 1.0
ex, ey = pts[:, 0], pts[:, 1] + 0.05
# Single localized anterior-negative effect: a compact blob over frontal
# channels, with the rest of the scalp near zero so only ONE cluster reads.
focus_x, focus_y = 0.0, 0.92
dist = np.hypot(ex - focus_x, ey - focus_y)
tvals = -3.5 * np.exp(-(dist / 0.55) ** 2) + rng.normal(0, 0.12, len(ex))
norm = plt.Normalize(-3.5, 3.5)
cmap = plt.cm.RdBu_r
axB.scatter(ex, ey, c=tvals, cmap=cmap, norm=norm, s=26,
            edgecolors="white", linewidths=0.3, zorder=4)

# threshold + cluster: the channels crossing |t| > 2 (the single anterior blob)
in_cluster = tvals < -2.0
axB.scatter(ex[in_cluster], ey[in_cluster], s=66, facecolors="none",
            edgecolors=C_GREEN, linewidths=1.0, zorder=5)
axB.text(0, 1.40, r"$|t| > 2.0$  cluster-forming", ha="center", va="center",
         fontsize=FONT_SMALL, color="#333333")
axB.text(0, -1.32,
         r"cluster mass $=\sum |t|$" "\n" "(Delaunay adjacency)",
         ha="center", va="center", fontsize=FONT_SMALL, color=C_BLUE,
         fontweight="bold")

# tiny colorbar for t
cax = axB.inset_axes([0.86, 0.30, 0.05, 0.42])
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, cax=cax)
cb.set_ticks([-3, 0, 3]); cb.ax.tick_params(length=1.2, labelsize=FONT_TINY)
cb.outline.set_linewidth(0.5)
cax.set_title(r"$t$", fontsize=FONT_SMALL, pad=2)

# ======================================================================
# Panel C — Freedman-Lane residual permutation loop
# ======================================================================
panel_label(axC, "c", "Freedman–Lane permutation")
axC.set_xlim(0, 1); axC.set_ylim(0, 1)

steps = [
    ("Fit reduced model\n(Age + Sex)", 0.86),
    ("Permute residuals\nacross participants", 0.635),
    ("Add back → refit full;\nrecompute $t$-map", 0.41),
    ("Store max\ncluster mass", 0.185),
]
box_w, box_h = 0.78, 0.135
for txt, yc in steps:
    fb = FancyBboxPatch((0.5 - box_w / 2, yc - box_h / 2), box_w, box_h,
                        boxstyle="round,pad=0.012,rounding_size=0.02",
                        linewidth=0.7, edgecolor=C_BLUE, facecolor=C_BOX, zorder=2)
    axC.add_patch(fb)
    axC.text(0.5, yc, txt, ha="center", va="center", fontsize=FONT_SMALL, zorder=3)

for y0, y1 in [(0.86 - box_h / 2, 0.635 + box_h / 2),
               (0.635 - box_h / 2, 0.41 + box_h / 2),
               (0.41 - box_h / 2, 0.185 + box_h / 2)]:
    axC.add_patch(FancyArrowPatch((0.5, y0), (0.5, y1), arrowstyle="-|>",
                  mutation_scale=7, lw=0.9, color=C_GREY, zorder=1))

# loop-back arrow (x5,000): one connected polyline + a single arrowhead.
box_left = 0.5 - box_w / 2
x_v = 0.045                      # vertical run sits left of the boxes
y_bot, y_top = 0.185, 0.86
axC.plot([box_left, x_v, x_v, box_left - 0.005],
         [y_bot, y_bot, y_top, y_top],
         color=C_ORANGE, lw=1.0, solid_capstyle="round",
         solid_joinstyle="round", zorder=1)
axC.annotate("", xy=(box_left, y_top), xytext=(x_v + 0.02, y_top),
             arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, lw=1.0,
                             shrinkA=0, shrinkB=0, mutation_scale=8), zorder=1)
axC.text(x_v - 0.022, (y_bot + y_top) / 2, "× 5,000", rotation=90,
         ha="center", va="center", fontsize=FONT_SMALL, color=C_ORANGE,
         fontweight="bold")

# ======================================================================
# Panel D — null distribution & corrected p
# ======================================================================
panel_label(axD, "d", "Null distribution → corrected $p$")
axD.set_xlim(0, 1); axD.set_ylim(0, 1)
inD = axD.inset_axes([0.16, 0.20, 0.80, 0.62])

# null max-cluster-mass distribution (gamma-like), observed in tail
null = rng.gamma(shape=2.4, scale=6.0, size=20000)
obs = np.percentile(null, 98.3)   # observed cluster mass in upper tail
bins = np.linspace(0, null.max() * 0.9, 46)
inD.hist(null, bins=bins, color=C_LBLUE, edgecolor="white", linewidth=0.2)
# shade tail >= observed
tail = bins[bins >= obs]
counts, _ = np.histogram(null, bins=bins)
for i in range(len(bins) - 1):
    if bins[i] >= obs:
        inD.bar(bins[i], counts[i], width=bins[i + 1] - bins[i], align="edge",
                color=C_ORANGE, alpha=0.55, edgecolor="white", linewidth=0.2)
inD.axvline(obs, color=C_BLUE, lw=1.1)
ymax = inD.get_ylim()[1]
inD.annotate("observed\ncluster mass", xy=(obs, ymax * 0.62),
             xytext=(obs + 6, ymax * 0.82), fontsize=FONT_TINY, color=C_BLUE,
             ha="left", va="center",
             arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=0.7))
inD.set_xlabel("max cluster mass (permuted)", fontsize=FONT_SMALL, labelpad=1)
inD.set_ylabel("frequency", fontsize=FONT_SMALL, labelpad=1)
inD.set_yticks([])
inD.tick_params(length=1.5, labelsize=0)
inD.set_xticks([])
for s in inD.spines.values():
    s.set_linewidth(0.6); s.set_color(C_GREY)
inD.spines["top"].set_visible(False); inD.spines["right"].set_visible(False)

axD.text(0.5, 0.085,
         r"$p_{\mathrm{corr}} = \dfrac{\#\{\mathrm{perm\ max} \geq \mathrm{obs}\}}{5{,}000}$"
         "   ; sig. if $p_{\\mathrm{corr}} < 0.05$",
         ha="center", va="center", fontsize=FONT_SMALL, color=C_BLUE)

# ----------------------------------------------------------------------
out_png = PROJECT / "poster" / "Figure_methods_statistics.png"
out_pdf = PROJECT / "poster" / "Figure_methods_statistics.pdf"
out_svg = PROJECT / "poster" / "Figure_methods_statistics.svg"
fig.savefig(out_png, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
# SVG keeps text + lines as vector → stays sharp at any scale when inserted
# into PowerPoint (Insert > Pictures), and survives PDF export crisply.
plt.rcParams["svg.fonttype"] = "none"   # keep text as selectable text, not paths
fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.05)
print(f"[OK] wrote {out_png}")
print(f"[OK] wrote {out_pdf}")
print(f"[OK] wrote {out_svg}")
