"""
Script 38: Smoothed pre-implantation ERK C/N comparison across tracks.

Single-panel plot overlaying the Savitzky-Golay-smoothed ERK C/N ratio timecourse
(pre-implantation window, t_start <= t < T_SPLIT) for a small set of tracks.
Smoothed lines only — no raw points.

Usage
-----
  conda run -n napari_env python3 pipelines/tracking/38_compare_tracks_erk_preimplant.py --tracks 48 71

  Or set TRACK_IDS below and just hit Run in VS Code — no CLI args needed.
"""

TRACK_IDS = [48, 71]   # <-- set track ids here to run directly (e.g. VS Code Run button)

import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter

REPO_ROOT   = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

version  = cfg['tracking']['input_version']
out_dir  = Path(cfg['paths']['output_dir'])
interval = cfg['tracking']['frame_interval_min']

T_START = cfg['tracking']['t_start']
T_SPLIT = 30   # implantation onset

SG_WINDOW  = 5   # points (must be odd, <= number of points in window)
SG_POLYORD = 2

parser = argparse.ArgumentParser()
parser.add_argument('--tracks', type=int, nargs='+', default=None,
                     help='track_ids to compare (overrides TRACK_IDS above)')
args, _unknown = parser.parse_known_args()
if args.tracks is not None:
    TRACK_IDS = args.tracks

# ── Load data (same as scripts 34/37) ──────────────────────────────────────────

kine = pd.read_csv(out_dir / f'motion_kinematics_{version}.csv').sort_values(['track_id', 't'])
erk  = pd.read_csv(out_dir / f'erk_cn_ratio_{version}.csv').sort_values(['track_id', 't'])
kine['time_min'] = kine['t'] * interval

merged = (kine[['track_id', 't', 'time_min']]
          .merge(erk[['track_id', 't', 'erk_cn_ratio']], on=['track_id', 't']))
merged = merged[merged['t'] < T_SPLIT]

colors = ['#d6604d', '#4393c3', '#4daf4a', '#984ea3', '#ff7f00']

fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

for i, track_id in enumerate(TRACK_IDS):
    cell = merged[merged['track_id'] == track_id].sort_values('t')
    if cell.empty:
        print(f'WARNING: track_id {track_id} has no points in the pre-implantation window — skipping')
        continue

    n_pts = len(cell)
    sg_window = min(SG_WINDOW, n_pts if n_pts % 2 == 1 else n_pts - 1)
    color = colors[i % len(colors)]

    ax.plot(cell['time_min'], cell['erk_cn_ratio'], color=color, linewidth=0.8, alpha=0.35,
             marker='o', markersize=3, zorder=1)

    if sg_window >= SG_POLYORD + 2:
        smoothed = savgol_filter(cell['erk_cn_ratio'].values, window_length=sg_window, polyorder=SG_POLYORD)
        ax.plot(cell['time_min'], smoothed, color=color, linewidth=2.2, zorder=2,
                 label=f'track {track_id} (n={n_pts})')
    else:
        print(f'WARNING: track_id {track_id} has too few points (n={n_pts}) to smooth — skipping')

ax.set_xlabel('Time (min)', fontsize=10)
ax.set_ylabel('ERK C/N ratio', fontsize=10)
ax.set_title(f'Pre-implantation ERK C/N (smoothed)  |  t < {T_SPLIT}', fontsize=11)
ax.legend(fontsize=9, loc='best')
ax.tick_params(labelsize=8)

tracks_str = '_'.join(str(t) for t in TRACK_IDS)
out_path = out_dir / f'erk_preimplant_compare_{tracks_str}_{version}.png'
fig.savefig(out_path, dpi=150)
plt.close()
print(f'Saved: {out_path}')
