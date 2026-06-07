#!/usr/bin/env python3
"""Run behavioral regression models for spindle ROIs and TOVA outcomes.

Model:
  TOVA_z = beta0 + beta1 * ROI_z + beta2 * age_c + beta3 * gender
           + beta4 * logHI_c + error

Output:
  output/tables/behavioral_model_results.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from spindle_common import COV_PATH, ROI_PATH, TABLE_DIR, ensure_dir


TOVA_OUTCOMES = [
    ("d-prime", "DPRIMEQ1_z"),
    ("Omission errors", "OM_sqrt_z"),
    ("Commission errors", "COM_sqrt_z"),
    ("Reaction time", "RT_log_z"),
]

# ant_slow_peakfreq is the second cluster-corrected ROI; it is modeled here (not
# only in the figure) so the "peak frequency predicts no TOVA outcome"
# dissociation is reproducible from behavioral_model_results.csv.
ROI_PREDICTORS = [
    ("Anterior fast duration", "ant_fast_dur"),
    ("Anterior slow peak frequency", "ant_slow_peakfreq"),
    ("Anterior slow duration", "ant_slow_dur"),
    ("Posterior slow amplitude", "pos_slow_amp"),
]


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)


def load_behavioral_dataset() -> pd.DataFrame:
    cov = pd.read_csv(COV_PATH)
    assert len(cov) == 62, f"expected 62 subjects, got {len(cov)}"
    roi = pd.read_csv(ROI_PATH)
    data = cov.merge(roi, on="id", how="left")

    data = data.dropna(subset=["DPRIMEQ1"]).copy()
    print(f"After TOVA filter: N = {len(data)}")

    data["OM_sqrt"] = np.sqrt(data["OMPERQ1"])
    data["COM_sqrt"] = np.sqrt(data["COMPERQ1"])
    data["RT_log"] = np.log(data["RTMEANQ1"])

    data["OM_sqrt_z_temp"] = np.abs(stats.zscore(data["OM_sqrt"]))
    data["RT_log_z_temp"] = np.abs(stats.zscore(data["RT_log"]))
    n_before = len(data)
    data = data[(data["OM_sqrt_z_temp"] < 3) & (data["RT_log_z_temp"] < 3)].copy()
    data.drop(columns=["OM_sqrt_z_temp", "RT_log_z_temp"], inplace=True)
    print(f"After outlier removal: N = {len(data)} (removed {n_before - len(data)})")

    for col in ["DPRIMEQ1", "OM_sqrt", "COM_sqrt", "RT_log"]:
        data[f"{col}_z"] = zscore(data[col])

    data["logHI_c"] = np.log10(data["overall_hi"] + 1)
    data["logHI_c"] = data["logHI_c"] - data["logHI_c"].mean()
    data["age_c"] = data["age_years"] - data["age_years"].mean()

    return data


def fit_roi_model(data: pd.DataFrame, outcome_col: str, roi_col: str):
    sub = data.dropna(subset=[outcome_col, roi_col, "age_c", "gender", "logHI_c"]).copy()
    roi_z_col = f"{roi_col}_z"
    sub[roi_z_col] = zscore(sub[roi_col])
    fit = sm.OLS.from_formula(f"{outcome_col} ~ {roi_z_col} + age_c + gender + logHI_c", sub).fit()
    return fit, roi_z_col, len(sub)


def model_row(tova_label: str, roi_label: str, fit, roi_z_col: str, n: int) -> dict:
    ci_low, ci_high = fit.conf_int().loc[roi_z_col]
    return {
        "TOVA_outcome": tova_label,
        "ROI_predictor": roi_label,
        "beta": round(fit.params[roi_z_col], 4),
        "SE": round(fit.bse[roi_z_col], 4),
        "t": round(fit.tvalues[roi_z_col], 3),
        "p": round(fit.pvalues[roi_z_col], 4),
        "CI_lower": round(ci_low, 4),
        "CI_upper": round(ci_high, 4),
        "R2": round(fit.rsquared, 4),
        "R2_adj": round(fit.rsquared_adj, 4),
        "F": round(fit.fvalue, 2),
        "model_p": round(fit.f_pvalue, 6),
        "N": n,
    }


def print_model_summary(roi_label: str, tova_label: str, fit, roi_z_col: str) -> None:
    beta = fit.params[roi_z_col]
    se = fit.bse[roi_z_col]
    t_val = fit.tvalues[roi_z_col]
    p_val = fit.pvalues[roi_z_col]
    ci_low, ci_high = fit.conf_int().loc[roi_z_col]
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

    print(f"\n  {roi_label} -> {tova_label}:")
    print(f"    beta = {beta:.3f}, SE = {se:.3f}, t = {t_val:.3f}, p = {p_val:.4f} {sig}")
    print(f"    95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print(
        f"    R2 = {fit.rsquared:.3f}, adj R2 = {fit.rsquared_adj:.3f}, "
        f"F = {fit.fvalue:.2f}, model p = {fit.f_pvalue:.6f}"
    )
    for cov in ["age_c", "gender", "logHI_c"]:
        print(f"    {cov}: beta = {fit.params[cov]:.3f}, p = {fit.pvalues[cov]:.4f}")


def main() -> None:
    data = load_behavioral_dataset()
    available_rois = [col for _, col in ROI_PREDICTORS if col in data.columns]
    print(f"ROI predictors: {available_rois}")

    results_rows = []

    print("\n" + "=" * 70)
    print("BEHAVIORAL MODEL RESULTS")
    print("Model: TOVA_z ~ ROI_z + age_c + gender + logHI_c")
    print("=" * 70)

    for tova_label, tova_col in TOVA_OUTCOMES:
        print(f"\n{'-' * 70}")
        print(f"  {tova_label.upper()}")
        print(f"{'-' * 70}")

        for roi_label, roi_col in ROI_PREDICTORS:
            if roi_col not in data.columns:
                continue

            fit, roi_z_col, n = fit_roi_model(data, tova_col, roi_col)
            print_model_summary(roi_label, tova_label, fit, roi_z_col)
            results_rows.append(model_row(tova_label, roi_label, fit, roi_z_col, n))

    results_df = pd.DataFrame(results_rows)
    ensure_dir(TABLE_DIR)
    out_path = TABLE_DIR / "behavioral_model_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n[SAVED] {out_path}")

    print("\n" + "=" * 70)
    print("MANUSCRIPT-READY TEXT (copy-paste)")
    print("=" * 70)
    for tova_label, _ in TOVA_OUTCOMES:
        print(f"\n{tova_label}:")
        for roi_label, _ in ROI_PREDICTORS:
            match = results_df[
                (results_df["TOVA_outcome"] == tova_label)
                & (results_df["ROI_predictor"] == roi_label)
            ]
            if match.empty:
                continue
            row = match.iloc[0]
            sig_str = f"p = {row['p']:.3f}" if row["p"] >= 0.001 else "p < 0.001"
            print(
                f"  {roi_label}: beta = {row['beta']:.2f}, SE = {row['SE']:.2f}, "
                f"{sig_str}; R2 = {row['R2']:.2f}"
            )


if __name__ == "__main__":
    main()
