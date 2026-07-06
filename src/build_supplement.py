#!/usr/bin/env python3
"""Produce the Supplementary Information document.

Emits ``output/supplement_spec.json`` from existing analysis artifacts plus
recomputed ROI channel membership, then builds a versioned SI ``.docx`` via the
global ``docx-tools`` CLI. The SI is a separate document from the main paper.

Contents:
  Supp. Table S1  ROI channel membership (recomputed, mapped to channel labels)
  Supp. Figure S1 inside172 montage (output/Fig_S1.png)
  Supp. Table S2  Full cluster-permutation results
  Supp. Table S3  Full behavioral models (ROI x TOVA outcome)
  Supp. Figure S2 Age x HI slow-amplitude interaction (output/Fig_S2.png)
  Supp. Table S4  ROI-level Age x HI interaction
  Supp. Figure S3 ROI-level Age x HI duration slopes (output/Fig_S3.png)
  Supp. Figure S4 HI distribution + fast-duration robustness (output/Fig_S4.png)
  Supp. Table S5  47- vs 29-channel fast-duration ROI sensitivity
  Supp. Figure S5 Slow + fast spindle density HI trends, topomaps (output/Fig_S5.png)
"""

from __future__ import annotations

import json
import re
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
OUTPUT_DIR = PROJECT / "output"
MONTAGE = PROJECT / "output/Fig_S1.png"       # Supplementary Figure S1
AGE_HI_SLOPES = PROJECT / "output/Fig_S2.png"  # Supplementary Figure S2
AGE_HI_ROI = PROJECT / "output/Fig_S3.png"     # Supplementary Figure S3
HI_ROBUSTNESS = PROJECT / "output/Fig_S4.png"  # Supplementary Figure S4
SLOW_DENSITY = PROJECT / "output/Fig_S5.png"   # Supplementary Figure S5
AUTHORS = PROJECT / "authors.json"
REFS = PROJECT / "references.bib"


def main_text_version() -> str:
    """Highest draft_vX.Y in output so the SI tracks the main text."""
    vers = []
    for f in OUTPUT_DIR.glob("draft_v*.docx"):
        m = re.match(r"draft_v(\d+)\.(\d+)", f.name)
        if m:
            vers.append((int(m.group(1)), int(m.group(2))))
    major, minor = max(vers) if vers else (0, 1)
    return f"{major}.{minor}"


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
        "image": "Fig_S1.png",
        "caption": "Layout of the 172-channel high-density EEG montage (inside172) "
        "used throughout the analysis. Each marker denotes a recording channel; "
        "labels are the channel identifiers referenced in Table S1. No data are "
        "plotted.",
        "number": "S1",
        "id": "figS1",
    }


def table_s4() -> dict:
    """ROI-level Age x HI interaction stats from age_hi_roi_interaction.py."""
    df = pd.read_csv(TABLE_DIR / "age_hi_roi_interaction.csv")
    age_mean = float(df["age_mean"].iloc[0])
    age_sd = float(df["age_sd"].iloc[0])
    # Derive the ±1 SD probe ages from the *rounded* mean/SD so the printed
    # bounds reconcile with the printed mean and SD (7.7 − 1.8 = 5.9, not 5.8).
    age_mean_r = round(age_mean, 1)
    age_sd_r = round(age_sd, 1)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                r["label"],
                f"{r['interaction_beta']:+.4f}",
                f"{r['interaction_t']:+.2f}",
                fmt_p(r["interaction_p"]),
                f"{r['slope_lo']:+.3f}",
                f"{r['slope_mid']:+.3f}",
                f"{r['slope_hi']:+.3f}",
            ]
        )
    return {
        "type": "table",
        "headers": [
            "Region of interest",
            "Age × HI *β*",
            "*t*",
            "*p*",
            "HI slope (−1 SD)",
            "HI slope (Mean)",
            "HI slope (+1 SD)",
        ],
        "rows": rows,
        "title": "ROI-level Age × hypopnea-index interaction for the spindle "
        "regions of interest.",
        "note": "Note. Each predefined ROI series (Table S1) was fit with "
        "roi ~ age_c + logHI_c + age_c:logHI_c + sex (N = 62, mean-centered "
        "predictors). The interaction coefficient (β, t, p) tests whether age "
        "moderates the hypopnea-index association; simple slopes give the "
        "model-implied association between log(HI + 1) and the ROI metric "
        f"probed at the mean age and ±1 SD (mean = {age_mean_r:.1f}, SD = "
        f"{age_sd_r:.1f}; i.e. {age_mean_r - age_sd_r:.1f}, {age_mean_r:.1f}, and "
        f"{age_mean_r + age_sd_r:.1f} years), following Aiken and West. Units per "
        "ROI: duration in s, amplitude in µV, peak frequency in Hz. Neither "
        "anterior duration ROI shows a significant Age × HI interaction, "
        "indicating a developmentally stable hypopnea effect across the "
        "3–11-year range. Significant p-values (< 0.05) are shown in bold.",
        "number": "S4",
        "font_size": 8,
    }


def age_hi_roi_subsection() -> list[dict]:
    """ROI-level Age x HI interaction text, Table S4, and duration-slope figure."""
    return [
        {
            "type": "body",
            "text": "To test the same question within the predefined regions of "
            "interest—rather than across the whole channel array—each ROI series "
            "(Table S1) was refit with an Age × log(HI + 1) interaction term "
            "(adjusting for age, sex, and the HI main effect; N = 62). Because "
            "each ROI is a single averaged series, this is a single ordinary "
            "least-squares interaction model per ROI and does not depend on the "
            "cluster-null assumptions of the channel-wise test. Simple slopes "
            "were probed at the mean age and ±1 SD (Aiken and West) rather than "
            "at arbitrary ages, so that every probed age falls within "
            "well-sampled data. Neither the "
            "anterior fast-duration ROI (interaction p = 0.77) nor the anterior "
            "slow-duration ROI (p = 0.44) showed a significant Age × HI "
            "interaction (Table S4, Figure S3): the negative association between "
            "hypopnea index and anterior spindle duration was broadly stable "
            "across the sampled age range. This stability is consistent with—but "
            "does not by itself establish—a developmentally invariant respiratory "
            "effect on duration; as noted in the main text, the narrow "
            "prepubertal age window and modest sample size limit power to detect "
            "age-by-exposure interactions, so this null is not evidence against "
            "developmental modulation.",
        },
        table_s4(),
        {
            "type": "figure",
            "image": "Fig_S3.png",
            "caption": "ROI-level Age × hypopnea-index interaction for the two "
            "anterior spindle-duration regions (N = 62). Model-estimated spindle "
            "duration as a function of hypopnea index at the mean age and ±1 SD "
            "(−1 SD, Mean, +1 SD; adjusted for sex), for (a) the anterior "
            "fast-duration ROI "
            "and (b) the anterior slow-duration ROI; shaded bands are 95% "
            "confidence intervals. The HI–duration slopes are negative and "
            "broadly parallel across ages, and the Age × HI interaction is "
            "non-significant for both ROIs (Table S4), indicating that the "
            "hypopnea-related shortening of anterior spindle duration does not "
            "vary detectably with age within this prepubertal sample.",
            "number": "S3",
            "id": "figS3",
        },
    ]


def age_hi_section() -> list[dict]:
    """Age x HI interaction text and the slow-amplitude simple-slopes figure (S2)."""
    return [
        {"type": "pagebreak"},
        {
            "type": "heading",
            "text": "Age × Hypopnea Index Interaction",
            "level": 1,
        },
        {
            "type": "body",
            "text": "To test whether the whole-sample hypopnea-index (HI) "
            "associations masked developmentally distinct effects, each spindle "
            "metric was refit channel-wise with an Age × log(HI + 1) "
            "interaction term, adjusting for age, sex, and the HI main effect "
            "(mean-centered predictors) in the full analytic sample (N = 62). "
            "The interaction term was evaluated with a Freedman–Lane residual "
            "permutation test (5,000 permutations; channel-forming threshold of "
            "|t| > 2.0—the two-sided p < .05 critical value, applied identically "
            "to the observed and permutation-null t-maps and matching the main HI "
            "cluster test) with a spatially valid cluster correction over "
            "the channel adjacency, in which a single shared subject reordering "
            "is applied across all channels per permutation (permuting channels "
            "independently makes the null spatially white and inflates "
            "significance). Under this corrected procedure, among the primary "
            "metrics—density, count, duration, and amplitude—only slow spindle "
            "amplitude showed an age-moderated HI association that reached "
            "corrected significance, and then only narrowly: it formed a "
            "25-channel anterior cluster (cluster p = 0.047, mean t = −2.32). "
            "Because this p-value sits just below the 0.05 threshold and the "
            "underlying slow-amplitude HI effect is itself exploratory (it did "
            "not survive correction in the whole-sample analysis), we treat this "
            "developmental pattern cautiously rather than as a robust "
            "interaction (Figure S2): the association between HI and slow spindle "
            "amplitude was positive in younger children and flattened to "
            "negative in older children. Density, count, and both duration "
            "metrics showed no age-moderated HI association surviving correction. "
            "Among secondary metrics, fast spindle "
            "peak frequency showed a cluster-corrected interaction (33-channel "
            "cluster, p = 0.012); this metric is not part of the primary "
            "duration-focused analysis and is documented for completeness. "
            "Crucially, the primary anterior fast-duration effect was not "
            "moderated by age and remained stable in the whole-sample model.",
        },
        {
            "type": "figure",
            "image": "Fig_S2.png",
            "caption": "Exploratory Age × hypopnea-index interaction for slow "
            "spindle amplitude (N = 62). (a) Channel-wise Age × log(HI + 1) "
            "interaction t-map; black markers denote the largest (25-channel) "
            "anterior cluster, which narrowly reached corrected significance "
            "under the spatially valid cluster-based permutation correction "
            "(cluster p = 0.047). (b) Model-estimated slow spindle amplitude as a function of "
            "hypopnea index at the mean age and ±1 SD (−1 SD, Mean, +1 SD; "
            "adjusted for sex), averaged over the cluster region; shaded bands "
            "are 95% "
            "confidence intervals. The association between hypopnea index and "
            "slow spindle amplitude is positive in younger children and "
            "flat-to-negative in older children; because this developmental "
            "pattern sits just past the 0.05 threshold and the underlying "
            "slow-amplitude HI effect is itself exploratory, it should be "
            "interpreted cautiously and regarded as hypothesis-generating.",
            "number": "S2",
            "id": "figS2",
        },
        *age_hi_roi_subsection(),
    ]


def hi_robustness_section() -> list[dict]:
    return [
        {
            "type": "heading",
            "text": "Hypopnea index distribution and robustness of the primary "
            "effect",
            "level": 1,
        },
        {
            "type": "body",
            "text": "The hypopnea index was strongly right-skewed (skewness = "
            "4.49; range 0.1–34.1 events/hour), so all regression models used "
            "log10(HI + 1), which substantially reduced the skew (1.10) "
            "(Figure S4a–b). To confirm that the primary cluster-corrected "
            "effect (shorter anterior fast spindle duration with rising HI; "
            "cluster p = 0.017) was not driven by any single participant, the "
            "cluster-based permutation test was refit 62 times, each omitting "
            "one child (leave-one-subject-out). The corrected cluster p "
            "remained below .05 in 47 of 62 folds (leave-one-out p range "
            "0.005–0.131; Figure S4c–d). The effect was most sensitive to the "
            "single child with the highest hypopnea index (HI = 34.1), whose "
            "omission attenuated the cluster (p = 0.131); the remaining folds "
            "that exceeded .05 reached at most p = 0.109, consistent with the "
            "modest power of a cluster-corrected effect at this sample size. "
            "Because counting folds against a fixed significance threshold is "
            "itself unstable at this sample size, effect stability was "
            "quantified directly with a subject-level bootstrap (10,000 "
            "resamples with replacement) of the cluster-averaged, "
            "covariate-adjusted standardized association between log(HI + 1) "
            "and fast spindle duration. The association was negative in 99.9% "
            "of resamples, with a standardized coefficient of −0.31 (95% "
            "bootstrap CI [−0.49, −0.12]) that excluded zero (Figure S4d), "
            "indicating that the shortening of fast spindle duration with "
            "rising hypopnea index is robust to subject composition rather "
            "than contingent on individual participants.",
        },
        {
            "type": "figure",
            "image": "Fig_S4.png",
            "caption": "Hypopnea index distribution and robustness of the "
            "fast-duration cluster (N = 62). (a) Raw hypopnea index, showing "
            "strong right skew. (b) log10(HI + 1), the mean-centered predictor "
            "used in all models. (c) Leave-one-subject-out corrected p for the "
            "primary fast-duration cluster plotted against the dropped child's "
            "hypopnea index; the dashed line marks p = .05 and the dotted line "
            "the full-sample value (p = 0.017). (d) Subject-level bootstrap "
            "(10,000 resamples) of the cluster-averaged standardized "
            "association between log(HI + 1) and fast spindle duration "
            "(adjusted for age and sex); dashed lines mark the 95% percentile "
            "confidence interval, which excludes zero, and the effect is "
            "negative in 99.9% of resamples.",
            "number": "S4",
            "id": "figS4",
        },
    ]


def table_s5() -> dict:
    """47-channel ROI vs 29-channel corrected-cluster behavioral sensitivity."""
    df = pd.read_csv(TABLE_DIR / "roi_29ch_sensitivity.csv")
    outcomes = ["d-prime", "Omission errors", "Commission errors", "Reaction time"]
    rows = []
    for oc in outcomes:
        sub = df[df["outcome"] == oc]
        r47 = sub[sub["roi"] == "47-ch"].iloc[0]
        r29 = sub[sub["roi"] == "29-ch"].iloc[0]
        rows.append(
            [
                oc,
                f"{r47['beta']:+.3f}",
                fmt_p(r47["p"]),
                f"{r29['beta']:+.3f}",
                fmt_p(r29["p"]),
            ]
        )
    return {
        "type": "table",
        "headers": [
            "TOVA outcome",
            "*β* (47-ch ROI)",
            "*p* (47-ch)",
            "*β* (29-ch cluster)",
            "*p* (29-ch)",
        ],
        "rows": rows,
        "title": "Sensitivity of the anterior fast-duration behavioral results to ROI "
        "definition.",
        "note": "Note. Standardized anterior fast spindle duration predicting each TOVA "
        "outcome (linear models adjusted for age, sex, and the log hypopnea index; "
        "N = 56), computed with the primary 47-channel anterior ROI (the full spatial "
        "extent of the effect) versus the 29-channel permutation-corrected cluster "
        "(Table S2). The two definitions yield the same pattern of results: d-prime, "
        "omission errors, and commission errors are significant with unchanged sign, and "
        "reaction time is non-significant, under both definitions. Significant p-values "
        "(< 0.05) are shown in bold.",
        "number": "S5",
        "font_size": 8,
    }


def roi_sensitivity_section() -> list[dict]:
    """Rationale for the broader fast-duration ROI plus the S5 sensitivity table."""
    return [
        {"type": "pagebreak"},
        {
            "type": "heading",
            "text": "Sensitivity of the anterior fast-duration behavioral ROI",
            "level": 1,
        },
        {
            "type": "body",
            "text": "The behavioral models averaged anterior fast spindle duration over "
            "the full 47-channel anterior extent of the hypopnea-index effect (channels "
            "significant at p < 0.05 together with contiguous trending channels at "
            "p < 0.08; Table S1), which is broader than the 29-channel subset that "
            "survived cluster-based permutation correction (Table S2). The broader region "
            "was used because averaging a per-participant metric over the full spatially "
            "coherent effect improves the reliability of each participant's ROI value; "
            "cluster correction is a family-wise-error threshold on the topographic test, "
            "not a boundary on where the effect is measurable, and the trending channels "
            "immediately adjacent to the surviving cluster carry the same negative "
            "association. Because the 47-channel ROI is defined from the same "
            "hypopnea-index t-map, it is a superset of the corrected cluster and shares "
            "its sign. To confirm that this choice does not drive the cognitive findings, "
            "every attention model was refit using fast spindle duration averaged over "
            "only the 29-channel corrected cluster. The pattern was unchanged (Table S5): "
            "higher fast spindle duration predicted higher d-prime and fewer omission and "
            "commission errors (all p < 0.05), with no association for reaction time, and "
            "if anything the effects were marginally stronger for the 29-channel cluster.",
        },
        table_s5(),
    ]


def figure_s5() -> dict:
    return {
        "type": "figure",
        "image": "Fig_S5.png",
        "caption": "Channel-wise association between the hypopnea index and spindle "
        "density. Beta coefficients (left) and t-statistics (right) across the "
        "172-channel montage for (a, b) slow spindle density and (c, d) fast "
        "spindle density.",
        "number": "S5",
        "id": "figS5",
    }


def density_section() -> list[dict]:
    """Topographic maps of the non-significant slow- and fast-density HI trends."""
    return [
        {"type": "pagebreak"},
        {
            "type": "heading",
            "text": "Spindle density topography",
            "level": 1,
        },
        {
            "type": "body",
            "text": "Neither slow nor fast spindle density showed a robust "
            "topographic association with the hypopnea index. In the whole-sample "
            "channel-wise model (adjusted for age and sex; N = 62), slow spindle "
            "density was weakly positive over posterior channels but reached "
            "significance at no individual channel (peak |t| = 1.97; all p > 0.05, "
            "uncorrected), and no suprathreshold cluster formed under permutation "
            "correction (cluster p = n/a). Fast spindle density showed a weak, "
            "predominantly negative pattern, with a single channel reaching "
            "p < 0.05 and a largest cluster that did not survive correction "
            "(cluster p = 0.363).",
        },
        figure_s5(),
    ]


def build_spec() -> dict:
    memberships = roi_channel_memberships()
    return {
        "content": [
            {
                "type": "title",
                "text": "Supplementary Information",
                "subtitle": "Sleep-Related Respiratory Disruption is Associated "
                "with Altered Spindle Morphology and Poorer Attention in Children",
                "id": "si-title",
            },
            {"type": "authors", "data": "authors.json"},
            {"type": "pagebreak"},
            table_s1(memberships),
            figure_s1(),
            {"type": "pagebreak"},
            table_s2(),
            table_s3(),
            *age_hi_section(),
            *hi_robustness_section(),
            *roi_sensitivity_section(),
            *density_section(),
        ]
    }


def main() -> None:
    spec = build_spec()
    SPEC_PATH.write_text(json.dumps(spec, indent=2))
    print(f"[SAVED] {SPEC_PATH}")

    if not MONTAGE.exists():
        raise FileNotFoundError(f"montage image not found: {MONTAGE}")
    if not AGE_HI_SLOPES.exists():
        raise FileNotFoundError(f"age x HI slopes image not found: {AGE_HI_SLOPES}")
    if not AGE_HI_ROI.exists():
        raise FileNotFoundError(f"age x HI ROI image not found: {AGE_HI_ROI}")
    if not HI_ROBUSTNESS.exists():
        raise FileNotFoundError(f"HI robustness image not found: {HI_ROBUSTNESS}")
    if not SLOW_DENSITY.exists():
        raise FileNotFoundError(f"slow-density image not found: {SLOW_DENSITY}")

    version = main_text_version()
    output = OUTPUT_DIR / f"draft_supplement_v{version}.docx"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.copy(MONTAGE, tmp_path / "Fig_S1.png")
        shutil.copy(AGE_HI_SLOPES, tmp_path / "Fig_S2.png")
        shutil.copy(AGE_HI_ROI, tmp_path / "Fig_S3.png")
        shutil.copy(HI_ROBUSTNESS, tmp_path / "Fig_S4.png")
        shutil.copy(SLOW_DENSITY, tmp_path / "Fig_S5.png")
        shutil.copy(AUTHORS, tmp_path / "authors.json")
        shutil.copy(REFS, tmp_path / "refs.bib")
        subprocess.run(
            [
                "docx-tools", "build", str(SPEC_PATH),
                "-o", str(output),
                "--base-dir", str(tmp_path),
            ],
            check=True,
        )
    print(f"[BUILT] {output}")


if __name__ == "__main__":
    main()
