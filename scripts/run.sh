#!/usr/bin/env bash
# Command-line entrypoints for the Pediatric_Spindles_HI pipeline.
#
# All modules live in src/ and cross-import by name, so each stage is run
# with src/ as the working directory. Stages read clinical source data from
# datasets/ (not tracked) and write to output/ (not tracked); curated copies
# of the published figures and tables live in assets/.
#
# Usage:
#   ./scripts/run.sh rois         # compute_rois_corrected.py   -> ROI values
#   ./scripts/run.sh cluster      # cluster_permutation.py      -> corrected p-values
#   ./scripts/run.sh behavior     # behavioral_models.py        -> ROI/TOVA models
#   ./scripts/run.sh sensitivity  # roi_29ch_sensitivity.py     -> 47- vs 29-ch ROI (Table S5)
#   ./scripts/run.sh robustness   # LOO + bootstrap + Figure S4 (needs cluster)
#   ./scripts/run.sh agehi        # Age x HI channel + ROI interaction (Figures S2/S3, Table S4)
#   ./scripts/run.sh figuredata   # compute_figure_data.py      -> cached quantities for Figures 2-5, S5
#   ./scripts/run.sh montage      # generate_montage_figure.py  -> Figure S1
#   ./scripts/run.sh figures      # figuredata -> generate_manuscript_figures.py (Figures 1-5, S5)
#   ./scripts/run.sh supplement   # build_supplement.py         -> SI document
#   ./scripts/run.sh main         # build_main.py               -> main manuscript (needs output/spec.json)
#   ./scripts/run.sh all          # rois -> cluster -> behavior -> sensitivity -> robustness ->
#                                 #   agehi -> montage -> figures -> supplement
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO/output"
cd "$REPO/src"
PY="${PYTHON:-python3}"

run_rois()        { "$PY" compute_rois_corrected.py; }
run_cluster()     { "$PY" cluster_permutation.py; }
run_behavior()    { "$PY" behavioral_models.py; }
run_sensitivity() { "$PY" roi_29ch_sensitivity.py; }
run_robustness()  { "$PY" loo_sensitivity.py; "$PY" bootstrap_cluster_effect.py; "$PY" hi_robustness_figure.py; }
run_agehi()       { "$PY" age_hi_interaction.py; "$PY" age_hi_roi_interaction.py; }
run_figuredata()  { "$PY" compute_figure_data.py; }
run_montage()     { "$PY" generate_montage_figure.py; }
run_figures()     { run_figuredata; "$PY" generate_manuscript_figures.py; }
run_supplement()  { "$PY" build_supplement.py; }
run_main()        { "$PY" build_main.py; }

case "${1:-help}" in
  rois)        run_rois ;;
  cluster)     run_cluster ;;
  behavior)    run_behavior ;;
  sensitivity) run_sensitivity ;;
  robustness)  run_robustness ;;
  agehi)       run_agehi ;;
  figuredata)  run_figuredata ;;
  montage)     run_montage ;;
  figures)     run_figures ;;
  supplement)  run_supplement ;;
  main)        run_main ;;
  all)         run_rois; run_cluster; run_behavior; run_sensitivity; run_robustness; \
               run_agehi; run_montage; run_figures; run_supplement ;;
  *)
    grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;
esac
