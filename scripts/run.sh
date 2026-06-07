#!/usr/bin/env bash
# Command-line entrypoints for the Pediatric_Spindles_HI pipeline.
#
# All modules live in src/ and cross-import by name, so each stage is run
# with src/ as the working directory. Stages read clinical source data from
# datasets/ (not tracked) and write to output/ (not tracked); curated copies
# of the published figures and tables live in assets/.
#
# Usage:
#   ./scripts/run.sh rois        # compute_rois_corrected.py  -> ROI values
#   ./scripts/run.sh cluster     # cluster_permutation.py     -> corrected p-values
#   ./scripts/run.sh behavior    # behavioral_models.py       -> ROI/TOVA models
#   ./scripts/run.sh figures     # generate_manuscript_figures.py
#   ./scripts/run.sh all         # rois -> cluster -> behavior -> figures
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO/src"
PY="${PYTHON:-python3}"

run_rois()     { "$PY" compute_rois_corrected.py; }
run_cluster()  { "$PY" cluster_permutation.py; }
run_behavior() { "$PY" behavioral_models.py; }
run_figures()  { "$PY" generate_manuscript_figures.py; }

case "${1:-help}" in
  rois)     run_rois ;;
  cluster)  run_cluster ;;
  behavior) run_behavior ;;
  figures)  run_figures ;;
  all)      run_rois; run_cluster; run_behavior; run_figures ;;
  *)
    grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;
esac
