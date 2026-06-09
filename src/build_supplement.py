#!/usr/bin/env python3
"""Produce the Supplementary Information document.

Emits ``output/supplement_spec.json`` from existing analysis artifacts plus
recomputed ROI channel membership, then builds the SI ``.docx`` via the global
``docx-tools`` CLI. The SI is a separate document from the main paper.

Reads ``output/tables/cluster_permutation_results.csv`` and
``output/tables/behavioral_model_results.csv``, so the ``cluster`` and
``behavior`` stages must run first.

Contents:
  Table S1   ROI channel membership (recomputed, mapped to channel labels)
  Figure S1  inside172 montage (assets/inside172_montage.png)
  Table S2   Full cluster-permutation results
  Table S3   Full behavioral models (ROI x TOVA outcome)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import mne
import pandas as pd

from cluster_permutation import largest_cluster_channels
from compute_rois_corrected import ADJACENCY_DIST, find_roi_channels
from spindle_common import (
    CHAN_PATH,
    COV_PATH,
    PREDICTORS,
    SPINDLE_DIR,
    TABLE_DIR,
    data_matrix_for_metric,
    load_channel_info,
    load_covariates,
    load_spindle_matrices,
)

PROJECT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT / "output/supplement_spec.json"
OUTPUT = PROJECT / "output/supplement.docx"
MONTAGE = PROJECT / "assets/inside172_montage.png"
AUTHORS = PROJECT / "authors.json"
REFS = PROJECT / "references.bib"


def fmt_p(p: float) -> str:
    """Format a p-value to 3 decimals; bold when significant (< 0.05)."""
    s = f"{p:.3f}"
    return f"**{s}**" if p < 0.05 else s


def roi_channel_memberships() -> list[dict]:
    """Recompute the four ROI channel sets and map indices to channel labels."""
    mne.set_log_level("ERROR")
    channels = load_channel_info(CHAN_PATH)
    cov = load_covariates(COV_PATH)
    matrices = load_spindle_matrices(
        cov, len(channels.labels), SPINDLE_DIR, include_frequency=True
    )
    adjacency = (channels.distance_matrix < ADJACENCY_DIST) & (channels.distance_matrix > 0)
    labels = channels.labels

    rows = []
    # Three exploratory ROIs from uncorrected map via distance-based contiguity.
    for roi, band, data_dict, metric in [
        ("Anterior fast duration", "Fast", matrices.fast, "Duration"),
        ("Anterior slow duration", "Slow", matrices.slow, "Duration"),
        ("Posterior slow amplitude", "Slow", matrices.slow, "Amplitude"),
    ]:
        ch = find_roi_channels(
            data_dict, metric, cov, matrices.sleep_mins, adjacency, channels.x
        )
        rows.append((roi, band, metric, ch))

    # Fourth ROI: cluster-corrected anterior slow peak frequency.
    adj_mne = mne.channels.find_ch_adjacency(channels.info, ch_type="eeg")[0].toarray().astype(bool)
    freq_ch = largest_cluster_channels(
        data_matrix_for_metric(matrices.slow, "Frequency", matrices.sleep_mins),
        cov,
        adj_mne,
        PREDICTORS,
    )
    rows.append(("Anterior slow peak frequency", "Slow", "Peak frequency", freq_ch))

    order = [
        "Anterior fast duration",
        "Anterior slow peak frequency",
        "Posterior slow amplitude",
        "Anterior slow duration",
    ]
    by_roi = {r[0]: r for r in rows}
    out = []
    for name in order:
        roi, band, metric, ch = by_roi[name]
        out.append(
            {
                "roi": roi,
                "band": band,
                "metric": metric,
                "n": len(ch),
                "labels": ", ".join(labels[i] for i in ch),
            }
        )
    return out


def table_s1(memberships: list[dict]) -> dict:
    return {
        "type": "table",
        "headers": ["Region of interest", "Band", "Metric", "*n*", "Channel labels"],
        "rows": [
            [m["roi"], m["band"], m["metric"], str(m["n"]), m["labels"]]
            for m in memberships
        ],
        "title": "Channel membership of the four spindle regions of interest.",
        "note": "Note. Anterior fast duration and anterior slow peak frequency are the "
        "cluster-corrected effects; posterior slow amplitude and anterior slow "
        "duration are exploratory regions from the uncorrected topographic maps. "
        "The channels listed are the behavioral averaging ROIs: for anterior slow "
        "peak frequency this ROI is exactly the 24-channel surviving cluster, whereas "
        "for anterior fast duration it spans the full anterior extent of the effect "
        "(47 channels) and is broader than the 29-channel surviving cluster (Table S2). "
        "Channel labels follow the 172-channel high-density montage (Figure S1).",
        "number": "S1",
        "font_size": 8,
    }


def table_s2() -> dict:
    df = pd.read_csv(TABLE_DIR / "cluster_permutation_results.csv")
    rows = []
    for _, r in df.iterrows():
        mass = "—" if pd.isna(r["cluster_mass"]) else f"{r['cluster_mass']:.1f}"
        peak = "—" if pd.isna(r["peak_t"]) else f"{r['peak_t']:+.2f}"
        pval = "—" if pd.isna(r["p_value"]) else fmt_p(r["p_value"])
        rows.append([r["metric_label"], str(int(r["n_channels"])), mass, peak, pval])
    return {
        "type": "table",
        "headers": [
            "Spindle metric",
            "Cluster size (ch)",
            "Cluster mass (Σ|*t*|)",
            "Peak *t*",
            "Corrected *p*",
        ],
        "rows": rows,
        "title": "Cluster-based permutation results for hypopnea-related spindle effects.",
        "note": "Note. Association between the hypopnea index and each spindle metric "
        "(channel-wise model adjusted for age and sex; cluster-forming |t| > 2.0; "
        "5,000 Freedman–Lane residual permutations). Corrected p is the proportion "
        "of permutations whose maximal cluster mass met or exceeded the observed "
        "value; significant values (< 0.05) are shown in bold. Em dashes denote "
        "metrics with no suprathreshold cluster.",
        "number": "S2",
        "font_size": 9,
    }


def table_s3() -> dict:
    df = pd.read_csv(TABLE_DIR / "behavioral_model_results.csv")
    rows = []
    for _, r in df.iterrows():
        ci = f"[{r['CI_lower']:.2f}, {r['CI_upper']:.2f}]"
        rows.append(
            [
                r["TOVA_outcome"],
                r["ROI_predictor"],
                f"{r['beta']:+.3f}",
                f"{r['SE']:.3f}",
                f"{r['t']:+.2f}",
                fmt_p(r["p"]),
                ci,
                f"{r['R2_adj']:.3f}",
            ]
        )
    return {
        "type": "table",
        "headers": [
            "TOVA outcome",
            "ROI predictor",
            "*β*",
            "SE",
            "*t*",
            "*p*",
            "95% CI",
            "Adj. *R*^2",
        ],
        "rows": rows,
        "title": "ROI predictors of attentional performance.",
        "note": "Note. Standardized linear-regression coefficients for each ROI "
        "predicting each TOVA attention outcome, adjusted for age, sex, and the "
        "(log) hypopnea index (N = 56). Coefficients are in standard-deviation "
        "units; the adjusted R² is for the full model. Significant p-values "
        "(< 0.05) are shown in bold.",
        "number": "S3",
        "font_size": 8,
    }


def figure_s1() -> dict:
    return {
        "type": "figure",
        "image": "figureS1.png",
        "caption": "Layout of the 172-channel high-density EEG montage (inside172) "
        "used throughout the analysis. Each marker denotes a recording channel; "
        "labels are the channel identifiers referenced in Table S1. No data are "
        "plotted.",
        "number": "S1",
        "id": "figS1",
    }


def build_spec() -> dict:
    memberships = roi_channel_memberships()
    return {
        "content": [
            {
                "type": "title",
                "text": "Supplementary Information",
                "subtitle": "Sleep-Related Respiratory Disruption is Associated "
                "with Shorter Fast Spindles and Poorer Attention in Children",
                "id": "si-title",
            },
            {"type": "authors", "data": "authors.json"},
            {"type": "pagebreak"},
            table_s1(memberships),
            figure_s1(),
            {"type": "pagebreak"},
            table_s2(),
            table_s3(),
        ]
    }


def main() -> None:
    spec = build_spec()
    SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPEC_PATH.write_text(json.dumps(spec, indent=2))
    print(f"[SAVED] {SPEC_PATH}")

    if not MONTAGE.exists():
        raise FileNotFoundError(f"montage image not found: {MONTAGE}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.copy(MONTAGE, tmp_path / "figureS1.png")
        shutil.copy(AUTHORS, tmp_path / "authors.json")
        shutil.copy(REFS, tmp_path / "refs.bib")
        subprocess.run(
            [
                "docx-tools", "build", str(SPEC_PATH),
                "-o", str(OUTPUT),
                "--base-dir", str(tmp_path),
            ],
            check=True,
        )
    print(f"[BUILT] {OUTPUT}")


if __name__ == "__main__":
    main()
