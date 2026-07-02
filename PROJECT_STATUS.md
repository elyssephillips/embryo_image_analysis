# Dataset001_implantation — Nucleus Tracking Project Status

**Last updated:** 2026-05-13  
**Dataset:** 63 nuclei × 101 timepoints, 61×1400×1400 voxels (Z=2.0 µm, XY=0.208 µm/px)

---

## Directory layout

```
── Code (git repo) ──────────────────────────────────────────────────────────────
/mnt/md1/elysse/code/embryo_image_analysis/
    pipelines/tracking/
        00_preprocess_segmentation.py   nnUNet binary → instance-classified label TIFs
        01_extract_features.py          labels + raw → features.csv
        02_register_centroids.py        features.csv → features_registered.csv
        03_link_tracks.py               custom Hungarian linker (current best method)
        03_run_btrack.py                btrack Kalman filter (baseline)
        04_curate_tracks.ipynb          napari curation GUI
        05_extract_motion.py … 11_*     motion analysis scripts
    configs/tracking/
        dataset001_implantation.yaml    ← all paths and parameters live here
        cell_config.json                btrack motion model config
    scripts/
        log_tracking_result.py          interactive prompt → appends to TRACKING_LOG.md
    src/
        tracking.py, motion.py, …       shared library code

── Data (not in git) ─────────────────────────────────────────────────────────────
/mnt/md1/elysse/dataset001_implantation/
    250913_stack2/             ← one folder per stack/embryo
      data/
        raw/
          cam long/         Cam_long_*.tif              nuclear channel (H2B), used for segmentation + tracking
          cam short/        Cam_short_*_cropped.tif     ERK biosensor channel; 91 z-planes, bottom-aligned with cam_long (z_short = z_long + 30); cam_short z=0..29 blank
        labels/             *_instances_reclassified.tif  instance label TIFs
        features/
            features.csv              per-nucleus features (centroid, vol, intensity, shape)
            features_registered.csv   + registered coords (z_um_reg, y_um_reg, x_um_reg)
            transforms.npz            cumulative affine transforms (101 × 4 × 4)
    results/
        tracks/
            btrack/         tracks_v1_raw.*, tracks_v2_visual.*, tracks_v3_visual_shape.*
            linked/         tracks_linked*.csv, track_labels_linked_c*.tif
        analysis/           plots, motion CSVs, MSD outputs (from pipelines 05–11)
```

---

## Pipeline order (run from embryo_image_analysis/ root)

```bash
conda run -n btrack2 python3 pipelines/tracking/00_preprocess_segmentation.py
conda run -n btrack2 python3 pipelines/tracking/01_extract_features.py
conda run -n btrack2 python3 pipelines/tracking/02_register_centroids.py
conda run -n btrack2 python3 pipelines/tracking/03_link_tracks.py   # custom Hungarian (recommended)
# then open 04_curate_tracks.ipynb in napari_env
```

All paths are set in `configs/tracking/dataset001_implantation.yaml` — no hard-coded paths in scripts.

VS Code: `Ctrl+Shift+P → Tasks: Run Task` for one-click script execution.

---

## Results summary (as of 2026-05-13)

| Version | Tracker | Features | Tracks | Full (101) | ≥50 frames | Median |
|---------|---------|----------|--------|------------|------------|--------|
| v1 | btrack MOTION | position only | 287 | 4 | 42 | 5 |
| v2 | btrack MOTION+VISUAL | +vol, intensity | 361 | 2 | 30 | 7 |
| v3 | btrack MOTION+VISUAL | +elongation, sphericity | 370 | 2 | 31 | 7 |
| linked (best) | custom Hungarian | position+vol+int+neighborhood | 254 | 6 | 35 | 11 |

**Canonical result (250913_stack2):** `tracks_linked_c23.csv` — 23 rounds of manual curation in napari (Apr 6–7), full analysis pipeline run Apr 13–16  
**Pre-crash best:** v4sc — also manually curated, lost with drive crash

---

## Known hard transitions

Lumen collapse oscillations at t=2,3,4,8,9,10,12,15,29–30  
Residual displacement ~7–9 µm vs inter-nuclear spacing ~17.7 µm → ~40% of cells ambiguous

---

## Open questions / next steps

- [ ] Continue analysis on `tracks_linked_c23` (already curated, 23 rounds); re-run analysis pipeline (05–11) if further corrections are made
- [ ] For next embryo: try btrack VISUAL mode, shape-consistency gate before linking, and systematic parameter recording via `log_tracking_result.py`
- [ ] Why does btrack VISUAL mode not benefit from elongation/sphericity? Worth inspecting btrack source to see how VISUAL properties enter the Bayesian update
- [ ] Shape-consistency gate before linking at collapse frames: reject candidate links where sphericity or intensity changes by >2σ of within-track distribution

---

## Key parameter constraints

- `btrack accuracy` must be ≤ 10 — higher values cause Eigen crash in the C++ library
- `cell_config.json`: accuracy=7.5, theta_dist=20, dist_thresh=40, max_lost=5, prob_not_assign=0.1
- Voxel sizes: Z=2.0 µm, XY=0.208 µm/px (all distances throughout are in µm)
