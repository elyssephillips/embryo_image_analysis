"""
Layer 1 motion feature extraction.

For each nucleus at each timepoint, computes:
  - Frame-to-frame displacement components (dz, dy, dx) in µm
  - Scalar step size and speed in µm/min
  - Cumulative path length from track origin
  - Net displacement from track origin

Usage
-----
  python 05_extract_motion.py                                    # default config
  python 05_extract_motion.py configs/tracking/control_2.yaml   # specific dataset

Outputs  (written to output_dir from config)
-------
  motion_kinematics_{version}.csv   — one row per (track_id, t)
  motion_track_stats_{version}.csv  — one row per track_id
"""

import sys
from pathlib import Path
import yaml
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.motion import compute_kinematics, summarize_tracks
from src.log import log_run

# ── config ────────────────────────────────────────────────────────────────────
from src.tracking_config import load_tracking_config, tracks_csv as _tracks_csv
cfg, CONFIG_PATH = load_tracking_config()

OUTPUT_DIR         = Path(cfg['paths']['output_dir'])
VERSION            = cfg['tracking']['input_version']
T_START            = cfg['tracking']['t_start']
T_END              = cfg['tracking']['t_end']
FRAME_INTERVAL_MIN = cfg['tracking']['frame_interval_min']
MIN_TRACK_LENGTH   = cfg['tracking']['min_track_length']
DATASET_ID         = cfg['project']['dataset_id']

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'Dataset:     {DATASET_ID}')
print(f'Config:      {CONFIG_PATH}')
print(f'Output dir:  {OUTPUT_DIR}')

# ── load tracks ───────────────────────────────────────────────────────────────
tracks_csv = _tracks_csv(cfg)
print(f'\nLoading {tracks_csv}')
df = pd.read_csv(tracks_csv)

# Trackastra-curated CSVs carry only ids + coordinates; pull the per-nucleus
# segmentation features (area_um3, intensity, shape - used by 11+) back in
# from features_csv on (t, label_id), as 03_link_tracks.py output already has.
feat = pd.read_csv(cfg['paths']['features_csv'])
missing = [c for c in feat.columns if c not in df.columns and c not in ('z_um', 'y_um', 'x_um')]
if missing:
    df = df.merge(feat[['t', 'label_id'] + missing], on=['t', 'label_id'], how='left')
    print(f'Joined from features_csv: {", ".join(missing)}  '
          f'({df[missing[0]].isna().sum()} rows unmatched)')

has_reg = 'z_um_reg' in df.columns
z_col, y_col, x_col = ('z_um_reg', 'y_um_reg', 'x_um_reg') if has_reg \
                       else ('z_um', 'y_um', 'x_um')

if has_reg:
    print('Using drift-corrected registered coordinates (z_um_reg, y_um_reg, x_um_reg)')
else:
    print('WARNING: registered coordinates not found — using raw image coordinates')

# Filter to curated timepoint range
df = df[(df['t'] >= T_START) & (df['t'] <= T_END)].copy()
print(f'Tracks: {df["track_id"].nunique()}  |  Timepoints: {df["t"].nunique()}  '
      f'(t{T_START}–t{T_END})')

# Drop tracks shorter than the minimum length
track_lengths = df.groupby('track_id')['t'].count()
valid_ids = track_lengths[track_lengths >= MIN_TRACK_LENGTH].index
n_dropped = df['track_id'].nunique() - len(valid_ids)
df = df[df['track_id'].isin(valid_ids)].copy()
print(f'Dropped {n_dropped} tracks with < {MIN_TRACK_LENGTH} frames  '
      f'→  {df["track_id"].nunique()} tracks remaining')

# ── compute kinematics ────────────────────────────────────────────────────────
print(f'\nFrame interval: {FRAME_INTERVAL_MIN} min')
kin_df = compute_kinematics(df, FRAME_INTERVAL_MIN, z_col, y_col, x_col)

# ── per-track summary ─────────────────────────────────────────────────────────
stats_df = summarize_tracks(kin_df)

# ── save ──────────────────────────────────────────────────────────────────────
kin_out   = OUTPUT_DIR / f'motion_kinematics_{VERSION}.csv'
stats_out = OUTPUT_DIR / f'motion_track_stats_{VERSION}.csv'

kin_df.to_csv(kin_out, index=False)
stats_df.to_csv(stats_out, index=False)

print(f'\nSaved kinematics:  {kin_out}')
print(f'Saved track stats: {stats_out}')

# ── quick summary ─────────────────────────────────────────────────────────────
print(f'\n--- Track stats summary ---')
print(stats_df[['total_path_um', 'net_disp_um', 'straightness',
                 'mean_speed_um_per_min']].describe().round(3).to_string())

log_run("tracking", CONFIG_PATH.stem, "05_extract_motion.py",
        output_path=str(OUTPUT_DIR), detail="light",
        data_path=str(OUTPUT_DIR))