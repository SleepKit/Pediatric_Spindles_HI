#!/usr/bin/env python3
"""Supplementary figure: HI distribution and fast-duration cluster robustness."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from spindle_common import PROJECT, TABLE_DIR, load_covariates

FIG_S4 = PROJECT / "output/Fig_S4"  # stem; saved as .png and .pdf at 600 dpi
LOO_CSV = TABLE_DIR / "loo_fast_duration_sensitivity.csv"
BOOT_DRAWS = TABLE_DIR / "bootstrap_fast_duration_draws.npy"
BOOT_CSV = TABLE_DIR / "bootstrap_fast_duration_effect.csv"
FULL_P = 0.0172


def main() -> None:
    cov = load_covariates()
    hi = cov["overall_hi"].to_numpy(dtype=float)
    log_hi = np.log10(hi + 1)
    raw_skew, log_skew = float(stats.skew(hi)), float(stats.skew(log_hi))

    loo = pd.read_csv(LOO_CSV)
    n = len(loo)
    boot = np.load(BOOT_DRAWS)
    bs = pd.read_csv(BOOT_CSV).iloc[0]
    lo, hi_ci, p_dir = bs["ci_lo"], bs["ci_hi"], bs["prop_negative"]

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2))
    a, b, c, d = axes.ravel()

    a.hist(hi, bins=18, color="#4C72B0", edgecolor="white")
    a.set(xlabel="Hypopnea index (events/hour)", ylabel="Number of children",
          title=f"(a) Raw HI  (skew = {raw_skew:.2f})")

    b.hist(log_hi, bins=18, color="#55A868", edgecolor="white")
    b.set(xlabel=r"$\log_{10}(\mathrm{HI} + 1)$", ylabel="Number of children",
          title=f"(b) Log-transformed HI  (skew = {log_skew:.2f})")

    c.scatter(loo["dropped_hi"], loo["top_cluster_p"], s=28, color="#4C72B0",
              edgecolor="white", zorder=3)
    c.axhline(0.05, color="#C44E52", ls="--", lw=1, label="p = 0.05")
    c.axhline(FULL_P, color="0.4", ls=":", lw=1, label=f"full sample (p = {FULL_P:.3f})")
    c.set(xlabel="Dropped subject's HI (events/hour)", ylabel="Leave-one-out cluster p",
          title="(c) Leave-one-out sensitivity")
    c.legend(fontsize=8, loc="upper right")

    d.hist(boot, bins=40, color="#8172B3", edgecolor="white")
    d.axvline(0, color="0.3", lw=1)
    d.axvline(lo, color="#C44E52", ls="--", lw=1)
    d.axvline(hi_ci, color="#C44E52", ls="--", lw=1, label=f"95% CI [{lo:.2f}, {hi_ci:.2f}]")
    d.set(xlabel="Standardized HI effect on fast duration", ylabel="Bootstrap resamples",
          title=f"(d) Subject bootstrap: {p_dir*100:.1f}% in direction")
    d.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Hypopnea index distribution and robustness of the fast-duration "
                 f"cluster (N = {n})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG_S4}.{ext}", dpi=600)
    print(f"wrote {FIG_S4}.png / .pdf")


if __name__ == "__main__":
    main()
