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
import argparse
from pathlib import Path
import yaml
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.motion import compute_kinematics, summarize_tracks

# ── config ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    'config',
    nargs='?',
    default='configs/tracking/dataset001_implantation.yaml',
    help='path to dataset config yaml (relative to repo root)',
)
args = parser.parse_args()

CONFIG_PATH = REPO_ROOT / args.config
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

TRACKS_DIR         = Path(cfg['paths']['tracks_dir'])
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
tracks_csv = TRACKS_DIR / f'tracks_{VERSION}.csv'
print(f'\nLoading {tracks_csv}')
df = pd.read_csv(tracks_csv)

has_orig = 'z_um_orig' in df.columns
z_col, y_col, x_col = ('z_um_orig', 'y_um_orig', 'x_um_orig') if has_orig \
                       else ('z_um', 'y_um', 'x_um')

if not has_orig:
    print('WARNING: original image coordinates not found — using registered coords')

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