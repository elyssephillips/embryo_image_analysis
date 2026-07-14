"""
Script 39: Raw (unsmoothed) pre-implantation ERK C/N comparison across tracks.

Single-panel plot of raw ERK C/N ratio timecourse (pre-implantation window,
t_start <= t < T_SPLIT) for a small set of tracks. No smoothing. Legend labels
each track by its ICM distance (t=30 snapshot) rather than track_id/n.

Usage
-----
  conda run -n napari_env python3 pipelines/tracking/39_compare_tracks_erk_preimplant_raw.py --tracks 48 71

  Or set TRACK_IDS below and just hit Run in VS Code — no CLI args needed.
"""

TRACK_IDS = [48, 71]   # <-- set track ids here to run directly (e.g. VS Code Run button)

import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

version  = cfg['tracking']['input_version']
out_dir  = Path(cfg['paths']['output_dir'])
interval = cfg['tracking']['frame_interval_min']

T_SPLIT = 30   # implantation onset

parser = argparse.ArgumentParser()
parser.add_argument('--tracks', type=int, nargs='+', default=None,
                     help='track_ids to compare (overrides TRACK_IDS above)')
args, _unknown = parser.parse_known_args()
if args.tracks is not None:
    TRACK_IDS = args.tracks

# ── Load data (same as scripts 34/37/38) ───────────────────────────────────────

kine = pd.read_csv(out_dir / f'motion_kinematics_{version}.csv').sort_values(['track_id', 't'])
erk  = pd.read_csv(out_dir / f'erk_cn_ratio_{version}.csv').sort_values(['track_id', 't'])
vstats = pd.read_csv(out_dir / f'volume_track_stats_{version}.csv')[['track_id', 'icm_dist_um']]
icm_lookup = vstats.set_index('track_id')['icm_dist_um']

kine['time_min'] = kine['t'] * interval

merged = (kine[['track_id', 't', 'time_min']]
          .merge(erk[['track_id', 't', 'erk_cn_ratio']], on=['track_id', 't']))
merged = merged[merged['t'] < T_SPLIT]

# Matches the highlight colors in 40_highlight_tracks_cn_overlay.py (kept off
# blue/red so they don't get confused with an ERK C/N colormap elsewhere)
colors = ['#984ea3', '#2ca02c', '#ff7f00', '#33a02c']

# Small physical figure, oversized fonts/lines — stays legible shrunk way down
# (e.g. embedded small in a slide or figure panel).
fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

for i, track_id in enumerate(TRACK_IDS):
    cell = merged[merged['track_id'] == track_id].sort_values('t')
    if cell.empty:
        print(f'WARNING: track_id {track_id} has no points in the pre-implantation window — skipping')
        continue

    color = colors[i % len(colors)]
    icm = icm_lookup.get(track_id, np.nan)
    icm_label = f'{icm:.0f} µm from ICM' if not pd.isna(icm) else 'ICM dist: n/a'

    ax.plot(cell['time_min'], cell['erk_cn_ratio'], color=color, linewidth=3,
             marker='o', markersize=7, label=icm_label)

ax.set_xlabel('Time (min)', fontsize=18)
ax.set_ylabel('ERK C/N ratio', fontsize=18)
ax.set_title('Pre-implantation ERK C/N', fontsize=17)
ax.legend(fontsize=13, loc='best')
ax.tick_params(labelsize=14, width=1.5, length=6)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

tracks_str = '_'.join(str(t) for t in TRACK_IDS)
out_path = out_dir / f'erk_preimplant_compare_raw_{tracks_str}_{version}.png'
fig.savefig(out_path, dpi=150)
plt.close()
print(f'Saved: {out_path}')
