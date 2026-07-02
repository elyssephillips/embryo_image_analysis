"""
Layer 1 motion diagnostics — saves four plots to output_dir.

Usage
-----
  python 06_motion_plots.py                                    # default config
  python 06_motion_plots.py configs/tracking/control_2.yaml   # specific dataset
"""

import sys
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

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

OUTPUT_DIR = Path(cfg['paths']['output_dir'])
VERSION    = cfg['tracking']['input_version']
T_START    = cfg['tracking']['t_start']
T_END      = cfg['tracking']['t_end']
FRAME_MIN  = cfg['tracking']['frame_interval_min']
DATASET_ID = cfg['project']['dataset_id']

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────────────
kin   = pd.read_csv(OUTPUT_DIR / f'motion_kinematics_{VERSION}.csv')
stats = pd.read_csv(OUTPUT_DIR / f'motion_track_stats_{VERSION}.csv')

has_orig = 'z_um_orig' in kin.columns
y_col = 'y_um_orig' if has_orig else 'y_um'
x_col = 'x_um_orig' if has_orig else 'x_um'

print(f'Dataset:    {DATASET_ID}')
print(f'Output dir: {OUTPUT_DIR}')
print(f'{stats["track_id"].nunique()} tracks  |  '
      f'{kin["t"].nunique()} timepoints  |  '
      f'frame interval: {FRAME_MIN} min')

# ── plot 1: speed over time ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))

for tid, grp in kin.groupby('track_id'):
    grp = grp.sort_values('t')
    ax.plot(grp['t'], grp['speed_um_per_min'],
            color='steelblue', alpha=0.15, linewidth=0.8)

median_speed = kin.groupby('t')['speed_um_per_min'].median()
ax.plot(median_speed.index, median_speed.values,
        color='navy', linewidth=2, label='median')

ax.set_xlabel('Timepoint')
ax.set_ylabel('Speed (µm/min)')
ax.set_title(f'Instantaneous speed over time — {DATASET_ID}')
ax.set_xlim(T_START, T_END)
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'speed_over_time.png', dpi=150)
plt.close()
print('Saved: speed_over_time.png')

# ── plot 2: straightness vs track length ─────────────────────────────────────
min_len = cfg['tracking']['min_track_length']

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(
    stats['n_frames'], stats['straightness'],
    c=stats['mean_speed_um_per_min'],
    cmap='plasma', s=30, alpha=0.8, edgecolors='none',
)
plt.colorbar(sc, ax=ax, label='Mean speed (µm/min)')
ax.axvline(min_len, color='red', linestyle='--', linewidth=1,
           label=f'min_track_length = {min_len}')
ax.set_xlabel('Track length (frames)')
ax.set_ylabel('Straightness  (net displacement / total path)')
ax.set_title(f'Straightness vs track length — {DATASET_ID}')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'straightness_vs_length.png', dpi=150)
plt.close()
print('Saved: straightness_vs_length.png')

# ── plot 3: spatial speed map ─────────────────────────────────────────────────
all_t      = sorted(kin['t'].unique())
idx        = np.round(np.linspace(0, len(all_t) - 1, 6)).astype(int)
timepoints = [all_t[i] for i in idx]

vmax = kin['speed_um_per_min'].quantile(0.95)
norm = mcolors.Normalize(vmin=0, vmax=vmax)
cmap = plt.cm.inferno

fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharex=True, sharey=True)
for ax, t in zip(axes.flat, timepoints):
    frame = kin[kin['t'] == t].dropna(subset=['speed_um_per_min'])
    ax.scatter(
        frame[x_col], frame[y_col],
        c=frame['speed_um_per_min'],
        cmap=cmap, norm=norm,
        s=20, edgecolors='none', alpha=0.9,
    )
    ax.set_title(f't = {int(t)}  ({int(t * FRAME_MIN)} min)', fontsize=10)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')

fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
             ax=axes, label='Speed (µm/min)', shrink=0.6)
fig.suptitle(f'Spatial speed map — {DATASET_ID}', y=1.01)
plt.savefig(OUTPUT_DIR / 'spatial_speed_map.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: spatial_speed_map.png')

# ── plot 4: speed distribution ────────────────────────────────────────────────
speeds = kin['speed_um_per_min'].dropna()

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(speeds, bins=60, color='steelblue', edgecolor='none', alpha=0.8)
ax.axvline(speeds.median(), color='navy',      linestyle='--', label=f'median {speeds.median():.2f}')
ax.axvline(speeds.mean(),   color='firebrick', linestyle='--', label=f'mean   {speeds.mean():.2f}')
ax.set_xlabel('Speed (µm/min)')
ax.set_ylabel('Count (nucleus-steps)')
ax.set_title(f'Distribution of instantaneous speeds — {DATASET_ID}')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'speed_distribution.png', dpi=150)
plt.close()
print('Saved: speed_distribution.png')