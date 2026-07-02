"""
Dual-axis timecourse for individual tracks: speed (left y) and ERK C/N (right y).

Usage
-----
  # plot a random sample of 12 tracks
  conda run -n napari_env python3 pipelines/tracking/17_individual_track_plots.py

  # plot specific track IDs
  conda run -n napari_env python3 pipelines/tracking/17_individual_track_plots.py --tracks 4 7 23 55

  # change grid size
  conda run -n napari_env python3 pipelines/tracking/17_individual_track_plots.py --n 20
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

version  = cfg['tracking']['input_version']
out_dir  = Path(cfg['paths']['output_dir'])
interval = cfg['tracking']['frame_interval_min']

# ── Edit these directly in VS Code ───────────────────────────────────────────
TRACK_IDS = [168, 107, 103, 43, 115]   # set to [] for a random sample
N_SAMPLE  = 12                          # how many to sample if TRACK_IDS is []
T_START   = 30                          # first timepoint to show
SEED      = 42                          # random seed for sampling
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--tracks', nargs='+', type=int, default=None)
parser.add_argument('--n',       type=int, default=None)
parser.add_argument('--t_start', type=int, default=None)
parser.add_argument('--seed',    type=int, default=None)
args = parser.parse_args()

# Command-line args override the in-file settings above
if args.tracks  is not None: TRACK_IDS = args.tracks
if args.n       is not None: N_SAMPLE  = args.n
if args.t_start is not None: T_START   = args.t_start
if args.seed    is not None: SEED      = args.seed

# ── Load data ─────────────────────────────────────────────────────────────────

erk_df  = pd.read_csv(out_dir / f'erk_cn_ratio_{version}.csv')
kine_df = pd.read_csv(out_dir / f'motion_kinematics_{version}.csv')
vstats  = pd.read_csv(out_dir / f'volume_track_stats_{version}.csv')[['track_id', 'icm_dist_um']]

kine_df = kine_df.sort_values(['track_id', 't'])

df = (erk_df
      .merge(kine_df[['track_id', 't', 'speed_um_per_min']], on=['track_id', 't'], how='inner')
      .merge(vstats, on='track_id', how='left'))

df = df[df['t'] >= T_START].copy()
df['time_min'] = df['t'] * interval

# ── Select tracks ─────────────────────────────────────────────────────────────

all_tracks = sorted(df['track_id'].unique())

if TRACK_IDS:
    track_ids = [tid for tid in TRACK_IDS if tid in all_tracks]
    missing   = [tid for tid in TRACK_IDS if tid not in all_tracks]
    if missing:
        print(f'Warning: track IDs not found: {missing}')
else:
    rng = np.random.default_rng(SEED)
    n   = min(N_SAMPLE, len(all_tracks))
    track_ids = sorted(rng.choice(all_tracks, size=n, replace=False).tolist())

print(f'Plotting {len(track_ids)} tracks: {track_ids}')

# ── Colormap for ICM distance (same as script 16) ─────────────────────────────

dist_min = vstats['icm_dist_um'].min()
dist_max = vstats['icm_dist_um'].max()
norm     = mcolors.Normalize(vmin=dist_min, vmax=dist_max)
cmap     = plt.cm.coolwarm_r
track_icm = vstats.set_index('track_id')['icm_dist_um']

# ── Plot ──────────────────────────────────────────────────────────────────────

n_tracks = len(track_ids)
n_cols   = min(4, n_tracks)
n_rows   = int(np.ceil(n_tracks / n_cols))

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(5 * n_cols, 3.5 * n_rows),
                         constrained_layout=True)
axes_flat = np.array(axes).flatten() if n_tracks > 1 else [axes]

COLOR_SPD = '#2166ac'   # blue  — speed
COLOR_ERK = '#d6604d'   # red   — ERK C/N

for ax_left, tid in zip(axes_flat, track_ids):
    grp = df[df['track_id'] == tid].sort_values('t')
    if grp.empty:
        ax_left.set_visible(False)
        continue

    ax_right = ax_left.twinx()

    ax_left.plot(grp['time_min'], grp['speed_um_per_min'],
                 color=COLOR_SPD, linewidth=1.5, label='speed')
    ax_right.plot(grp['time_min'], grp['erk_cn_ratio'],
                  color=COLOR_ERK, linewidth=1.5, label='ERK C/N')

    # Title: track ID + ICM distance, colored by position
    icm = track_icm.get(tid, np.nan)
    icm_str = f'{icm:.0f} µm from ICM' if not np.isnan(icm) else 'ICM dist: n/a'
    face_col = cmap(norm(icm)) if not np.isnan(icm) else 'white'
    ax_left.set_title(f'track {tid}  |  {icm_str}',
                      fontsize=9, backgroundcolor=face_col, pad=4)

    ax_left.set_xlabel('Time (min)', fontsize=8)
    ax_left.set_ylabel('Speed (µm/min)', color=COLOR_SPD, fontsize=8)
    ax_right.set_ylabel('ERK C/N ratio', color=COLOR_ERK, fontsize=8)
    ax_left.tick_params(axis='y',  labelcolor=COLOR_SPD, labelsize=7)
    ax_right.tick_params(axis='y', labelcolor=COLOR_ERK, labelsize=7)
    ax_left.tick_params(axis='x',  labelsize=7)

# Hide unused axes
for ax in axes_flat[n_tracks:]:
    ax.set_visible(False)

fig.suptitle(
    f'Individual track timecourses — speed (blue) vs ERK C/N (red)  '
    f'|  t ≥ {T_START} ({T_START * interval} min)  '
    f'|  speed = instantaneous (last frame)',
    fontsize=10
)

suffix = 'sample' if not TRACK_IDS else 'selected'
out_path = out_dir / f'individual_tracks_{suffix}_{version}.png'
fig.savefig(out_path, dpi=150)
plt.close()
print(f'Saved: {out_path}')

# ── Scatter: Δspeed vs ΔERK (frame-to-frame changes) ─────────────────────────
# Plotting changes rather than levels removes the shared time trend.
# A negative slope means: frames where speed increases → ERK decreases (and vice versa).

fig2, axes2 = plt.subplots(n_rows, n_cols,
                            figsize=(4.5 * n_cols, 4 * n_rows),
                            constrained_layout=True)
axes2_flat = np.array(axes2).flatten() if n_tracks > 1 else [axes2]

for ax, tid in zip(axes2_flat, track_ids):
    grp = df[df['track_id'] == tid].sort_values('t').dropna(
        subset=['speed_um_per_min', 'erk_cn_ratio'])

    d_spd = grp['speed_um_per_min'].diff().dropna().values
    d_erk = grp['erk_cn_ratio'].diff().dropna().values
    n     = min(len(d_spd), len(d_erk))
    d_spd, d_erk = d_spd[:n], d_erk[:n]

    r = np.corrcoef(d_spd, d_erk)[0, 1]

    ax.scatter(d_spd, d_erk, s=18, alpha=0.7, color=COLOR_SPD, edgecolors='none')
    ax.axhline(0, color='k', linewidth=0.6, alpha=0.4)
    ax.axvline(0, color='k', linewidth=0.6, alpha=0.4)

    # Regression line
    m, b = np.polyfit(d_spd, d_erk, 1)
    xs = np.linspace(d_spd.min(), d_spd.max(), 100)
    ax.plot(xs, m * xs + b, color=COLOR_ERK, linewidth=1.5)

    icm = track_icm.get(tid, np.nan)
    icm_str  = f'{icm:.0f} µm' if not np.isnan(icm) else 'n/a'
    face_col = cmap(norm(icm)) if not np.isnan(icm) else 'white'
    ax.set_title(f'track {tid}  |  {icm_str} from ICM  |  r = {r:.2f}',
                 fontsize=8, backgroundcolor=face_col, pad=4)
    ax.set_xlabel('Δ speed (µm/min)', fontsize=8)
    ax.set_ylabel('Δ ERK C/N ratio', fontsize=8)
    ax.tick_params(labelsize=7)

for ax in axes2_flat[n_tracks:]:
    ax.set_visible(False)

fig2.suptitle(
    'Frame-to-frame changes: Δspeed vs Δ ERK C/N\n'
    'Negative slope = speed and ERK move in opposite directions (time trend removed)',
    fontsize=10
)

xcorr_path = out_dir / f'individual_tracks_diffs_{suffix}_{version}.png'
fig2.savefig(xcorr_path, dpi=150)
plt.close()
print(f'Saved: {xcorr_path}')
