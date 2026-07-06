#!/usr/bin/env python3
"""Supplementary Figure S1: layout of the 172-channel high-density montage.

Reproduces the inside172 montage (electrode positions + channel-identifier
labels; no data plotted) as a head-outline scalp plot, matching the caption in
the supplement. Saved as output/Fig_S1.png and output/Fig_S1.pdf at 600 dpi.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spindle_common import CHAN_PATH, PROJECT, load_channel_info

FIG_S1 = PROJECT / "output/Fig_S1"  # stem; saved as .png and .pdf


def _head_outline(ax, head_r: float) -> None:
    """Draw the head circle, nose, and ears in the standard scalp-plot style."""
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(head_r * np.cos(t), head_r * np.sin(t), color="black", lw=1.5, zorder=1)
    # nose (triangle at the top / anterior)
    nose = np.array(
        [[-0.09 * head_r, 0.99 * head_r], [0.0, 1.15 * head_r], [0.09 * head_r, 0.99 * head_r]]
    )
    ax.plot(nose[:, 0], nose[:, 1], color="black", lw=1.5, zorder=1)
    # ears (simple lobes on the left and right)
    ear = np.array(
        [[0.00, 0.10], [0.05, 0.09], [0.07, 0.02], [0.07, -0.02], [0.05, -0.09], [0.00, -0.10]]
    )
    for sgn in (-1.0, 1.0):
        ex = sgn * (head_r + np.abs(ear[:, 1]) * 0.0 + ear[:, 0] * head_r * 0.9) + sgn * head_r * 0.0
        # place the lobe just outside the circle at the 3 o'clock / 9 o'clock point
        lobe_x = sgn * head_r + sgn * (ear[:, 0] * head_r * 0.9)
        lobe_y = ear[:, 1] * head_r * 0.9
        ax.plot(lobe_x, lobe_y, color="black", lw=1.5, zorder=1)


def main() -> None:
    ch = load_channel_info(CHAN_PATH)
    x, y, labels = np.asarray(ch.x), np.asarray(ch.y), ch.labels
    r_max = float(np.max(np.hypot(x, y)))
    head_r = r_max * 1.08
    off = r_max * 0.03  # label offset to the right of each marker

    fig, ax = plt.subplots(figsize=(10, 10))
    _head_outline(ax, head_r)
    ax.scatter(x, y, s=70, color="black", zorder=3)
    for xi, yi, lab in zip(x, y, labels):
        ax.text(xi + off, yi, str(lab), fontsize=7, ha="left", va="center", zorder=4)

    ax.set_title(f"inside172 montage — {len(labels)} channels", fontsize=15, pad=12)
    ax.set_aspect("equal")
    ax.axis("off")
    lim = head_r * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG_S1}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_S1}.png / .pdf ({len(labels)} channels)")


if __name__ == "__main__":
    main()
