# Pediatric_Spindles_HI — analysis source

Python pipeline for the pediatric sleep-spindle / hypopnea-index (HI) paper.
Per-subject YASA spindle detections are summarized per channel, regressed
against HI (adjusting for age and sex), corrected for multiple comparisons
with cluster-based permutation, distilled into regions of interest (ROIs), and
tested as predictors of attention (TOVA).

## Layout

```
src/
├── spindle_common.py             # Single source of truth: paths, constants,
│                                 # channel geometry, covariate + spindle
│                                 # loaders, and the channel-wise OLS regression.
├── cluster_permutation.py        # Cluster-based permutation test
│                                 # (Maris & Oostenveld 2007): covariate-adjusted
│                                 # Freedman-Lane residual permutation, |t| > 2.0
│                                 # cluster forming over MNE Delaunay adjacency,
│                                 # shared single relabeling across channels.
│                                 # Exposes largest_cluster_channels() for ROI
│                                 # membership. -> cluster_permutation_results.csv
├── compute_rois_corrected.py     # Derives the four analysis ROIs and writes
│                                 # per-subject averages. -> roi_values_corrected.csv
├── behavioral_models.py          # TOVA_z ~ ROI_z + age_c + gender + logHI_c
│                                 # for each ROI x TOVA outcome.
│                                 # -> behavioral_model_results.csv
├── compute_figure_data.py        # Precomputes every quantity Figures 2-5 and
│                                 # Figure S5 need (normative topographies,
│                                 # channel-wise HI maps incl. slow/fast density,
│                                 # ROI/cluster membership, scatter series).
│                                 # -> output/figure_data/main_figure_data.pkl
├── generate_manuscript_figures.py# Renders Figures 1-5 and Figure S5 (methods,
│                                 # normative topography, corrected/uncorrected
│                                 # HI effects, cognition forest plot, density).
├── generate_montage_figure.py    # Renders Figure S1 (172-channel montage).
├── age_hi_interaction.py         # Channel-wise Age x HI interaction -> Figure S2.
├── age_hi_roi_interaction.py     # ROI-level Age x HI interaction -> Figure S3,
│                                 # Table S4.
├── roi_29ch_sensitivity.py       # 47- vs 29-channel fast-duration ROI behavioral
│                                 # sensitivity -> Table S5.
├── loo_sensitivity.py            # Leave-one-subject-out cluster robustness.
├── bootstrap_cluster_effect.py   # Subject-level bootstrap of the fast-duration
│                                 # cluster effect.
├── hi_robustness_figure.py       # HI distribution + robustness -> Figure S4.
├── build_supplement.py           # Builds the Supplementary Information document
│                                 # (Tables S1-S5, Figures S1-S5) via docx-tools.
│                                 # -> output/supplement.docx
└── build_main.py                 # Builds the main manuscript .docx from
                                  # output/spec.json (needs the manuscript spec).
```

Nothing in `src/` hardcodes an absolute path: every input/output location is
derived from the repository root in `spindle_common.py` (`PROJECT`).

## Data contract

The only data shipped here is the EEG montage geometry, in `assets/`:

| Path | Contents |
|------|----------|
| `assets/inside172.mat` | 172-channel hdEEG montage geometry. |

The pediatric clinical inputs are **not** distributed (privacy). The pipeline
expects them under `datasets/` at the repo root:

| Path | Contents |
|------|----------|
| `datasets/clean/analysis_sample_complete_N62.csv` | Covariate master (id, age, sex, HI, sleep minutes, TOVA). |
| `datasets/clean/sample_ids_N62.csv` | The canonical N = 62 analytic sample. |
| `datasets/original/spindles_individual_data/*_all_spindles_detailed.csv` | Per-subject YASA spindle detections. |

Outputs are written to `output/figures/` and `output/tables/` (not tracked).

## Workflow

Run stages via `scripts/run.sh` (sets `src/` as cwd so the flat cross-imports
resolve), or invoke the modules directly:

```bash
./scripts/run.sh rois         # ROI values         -> roi_values_corrected.csv
./scripts/run.sh cluster      # corrected p-values  -> cluster_permutation_results.csv
./scripts/run.sh behavior     # ROI/TOVA models     -> behavioral_model_results.csv
./scripts/run.sh sensitivity  # 47- vs 29-ch ROI    -> Table S5
./scripts/run.sh robustness   # LOO + bootstrap + Figure S4
./scripts/run.sh agehi        # Age x HI            -> Figures S2/S3, Table S4
./scripts/run.sh montage      # Figure S1
./scripts/run.sh figures      # figuredata -> Figures 1-5, S5
./scripts/run.sh supplement   # SI document         -> output/supplement.docx
./scripts/run.sh all          # all of the above, in order
```

`compute_rois_corrected.py` must run before `behavioral_models.py` and
`generate_manuscript_figures.py`, both of which read
`roi_values_corrected.csv`; `compute_figure_data.py` (the `figuredata`
step, run automatically by `figures`) precomputes the cached quantities the
figures draw from. `build_supplement.py` reads the cluster, behavioral, and
interaction result tables plus the generated Figures S1-S5, so it runs after
those stages (it needs the `docx-tools` CLI).

## Canonical analysis parameters (see `spindle_common.py` / `cluster_permutation.py`)

| Parameter | Value |
|-----------|-------|
| Analytic sample | N = 62 (pinned via `sample_ids_N62.csv`) |
| Cognitive sample | N = 56 (TOVA-valid, 1 outlier removed) |
| Channels | 172 (hdEEG) |
| Slow / fast spindle bands | 10-12 Hz / 12-16 Hz |
| HI transform | `log10(HI + 1)`, mean-centered |
| Channel model | `metric ~ logHI_c + age_c + gender_bin` (OLS); effect of interest = `logHI_c` |
| Minimum N per channel | 30 |
| Cluster-forming threshold | \|t\| > 2.0 |
| Adjacency | MNE Delaunay (`mne.channels.find_ch_adjacency`) |
| Cluster mass | Σ\|t\| over connected suprathreshold channels |
| Permutation | Freedman-Lane residual, single shared relabeling across channels, 5000 permutations |
| Corrected p | (#null_max ≥ observed + 1) / (N_perm + 1) |
