## Pediatric sleep spindle topography, hypopnea burden, and attention

This repository contains code for the analysis, statistics, and figure
generation of the pediatric high-density EEG sleep-spindle study (Haber et al.).

### Codebase

- `src/`: source for configuration/data loading, cluster-based permutation testing, ROI computation, behavioral models, and figure generation.
- `scripts/`: command-line entrypoints for the analysis stages.
- `assets/`: 172-channel EEG montage geometry (`inside172.mat`).
- `authors.json` and `references.bib`: author metadata and bibliography.

See `src/README.md` for the detailed module map and workflow.

### Dependencies

- Python 3.10+ with numpy, pandas, scipy, statsmodels, matplotlib, and mne.
- Install with `uv sync` or `pip install -r requirements.txt`.
