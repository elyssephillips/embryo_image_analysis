# Tracking Log

## Dataset Index

| Dataset | Description | N / Conditions | Data path | Last updated | Status |
|---|---|---|---|---|---|
| dataset001_implantation | 4D nuclei tracking, implantation timelapse (H2B + ERK) | 113 tracks, 101t, 1 embryo (250913_stack2) | `/mnt/md1/elysse/dataset001_implantation/` | 2026-05-26 | active analysis — linked_c62 | <!-- row:dataset001_implantation -->

<!-- index-end -->

---

<!-- ds:dataset001_implantation -->
## dataset001_implantation

**Description:** 4D nuclei tracking implantation timelapse — H2B nuclear channel for segmentation/tracking, ERK biosensor (cam short) for C/N ratio analysis. Single objective (250913_stack2).  
**N / Conditions:** 1 embryo, 101 timepoints (t0–t100), 61×1400×1400 voxels  
**Data path:** /mnt/md1/elysse/dataset001_implantation/250913_stack2/

### ~2025-09-13 | run_btrack.py
**Done:** v0 — btrack MOTION+volume (pre-crash, recovered from napari session save)
**Params/findings:** MOTION only + area_vox/area_um3 properties; no intensity or shape; params not recoverable from h5. 187 tracks | 0 full | 53 ≥50 | median unknown. 53 ≥50 is better than current best — may be different timepoint range or aggressive fragmentation.
**Next:** Recovered from /mnt/md1/elysse/nuclei segmentation/. Worth reloading coordinates to compare against current results.

### ~2026-03-01 | run_btrack.py
**Output:** `results/tracks/btrack/tracks_v1_raw.csv`
**Done:** v1 — btrack MOTION only. cell_config.json (accuracy=10, prob_not_assign=0.25, max_lost=2, theta_dist=12). 287 tracks | 4 full | 42 ≥50 | median 5
**Params/findings:** Baseline. Many short fragments. Hard transitions completely unresolved.
**Next:** Add VISUAL features (volume + intensity)

### ~2026-03-15 | run_tracking_visual.py
**Output:** `results/tracks/btrack/tracks_v2_visual.csv`
**Done:** v2 — btrack MOTION+VISUAL (vol+intensity). tracking_updates=["MOTION","VISUAL"]. 361 tracks | 2 full | 30 ≥50 | median 7
**Params/findings:** VISUAL update increases total tracks (more fragments) but full-timecourse count drops — VISUAL features may be creating false breaks. Hard transitions still unresolved.
**Next:** Add shape features (elongation, sphericity)

### ~2026-04-15 | run_tracking_visual.py
**Output:** `results/tracks/btrack/tracks_v3_visual_shape.csv`
**Done:** v3 — btrack MOTION+VISUAL + shape (elongation, sphericity). Same as v2; added elongation+sphericity to btrack objects. 370 tracks | 2 full | 31 ≥50 | median 7
**Params/findings:** Essentially identical to v2. Shape features passed to btrack objects but btrack's VISUAL update may not use them meaningfully in the Bayesian step.
**Next:** Try custom Hungarian linker instead of btrack

### ~2026-04-20 | 03_link_tracks.py
**Output:** `results/tracks/linked/tracks_linked.csv`
**Done:** v — custom Hungarian linker (replaced btrack). K_NEIGHBORS=5, MAX_SPATIAL_DIST=16, MAX_COST=2.5, W_SPATIAL=1.0, W_VOL=0.5, W_INT=0.5, W_NEIGHBOR=1.0, NEIGHBOR_SCALE=20, STITCH_MAX_DIST=22. 254 tracks | 6 full | 33 ≥50 | median 11
**Params/findings:** Fewer total tracks (good — less fragmentation), more full-timecourse tracks, better median. Neighborhood geometry fingerprint helps at non-collapse frames. Hard transitions still fail.
**Next:** Manual curation in napari

### 2026-04-06 to 2026-04-07 | 04_curate_tracks.ipynb
**Output:** `results/tracks/linked/tracks_linked_c.csv` … `tracks_linked_c23.csv`; label TIFs at c15, c20, c23
**Done:** linked_c through linked_c23 — 23 rounds of manual curation in napari over two days. c23 is the final curated result.
**Params/findings:** "c" = curated. Each version is a saved checkpoint of manual corrections.
**Next:** Run full analysis pipeline (05–11) on linked_c23

### 2026-04-13 to 2026-04-16 | 05_extract_motion.py … 11_volume_motion_analysis.py ★ CANONICAL RESULT
**Output:** `results/analysis/motion_kinematics_linked_c23.csv`, `motion_track_stats_linked_c23.csv`, MSD outputs, plots
**Done:** Full analysis pipeline on linked_c23 (manually curated, 23 rounds). 265 tracks | 5 full | 32 ≥50 | median 9.
**Params/findings:** High pre-implantation ERK C/N cells are closest to ICM at t=30 (r=-0.5, p<0.0001). ICM-proximal cells move faster post-implantation (r=-0.53***). Pre-ERK C/N does NOT independently predict motion once controlling for ICM distance.
**Next:** Re-curate to add ICM-proximal cells missed in c23 → linked_c62

### 2026-05-26 | 04_curate_tracks.ipynb + 05–11 pipeline
**Done:** Re-curation to add ICM-proximal cells → linked_c62. Full analysis pipeline re-run.
**Params/findings:** 113 tracks total. ICM centroid (t=30): Z=38, Y=178, X=140 µm. Implantation onset t=30 (450 min). Key result: ERK adds nothing to motion prediction after controlling for ICM distance (pure position effect). Within-cell ERK/speed cross-correlogram flat across ±150 min lags.
**Next:** Continue analysis; Z motion excluded from motion analysis during implantation (embryo flattening artifact).

<!-- ds-end:dataset001_implantation -->
