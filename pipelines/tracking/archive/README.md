# Archived tracking analyses

Exploratory analyses retired in the Oct 2026 cleanup: motion/kinematics, MSD,
flow field, speed vs ERK, quadrant/tertile groupings, multivariate regression,
nuclear-only ERK, etc.

They won't run as-is from here: `REPO_ROOT = parents[2]` now points one level
too shallow, and most read `motion_kinematics_*.csv` (old 05) and
`volume_track_stats_*.csv` (old 11). To revive one, move it (and the scripts
whose outputs it reads) back up into `pipelines/tracking/`.

Current replacements: 06_build_track_table → 07_track_plots / 08_compare_tracks
(covers 17, 25, 30, 34_single, 36).
