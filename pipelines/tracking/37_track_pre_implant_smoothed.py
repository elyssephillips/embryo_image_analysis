"""
Script 37: Smoothed pre-implantation timecourse for a single track — ERK C/N, Z position,
radial distance.

Same three panels as script 34, but restricted to the pre-implantation window
(t_start <= t < T_SPLIT) and smoothed with a Savitzky-Golay filter. Raw points
are shown light/faint underneath the smoothed line for reference.

Usage
-----
  conda run -n napari_env python3 pipelines/tracking/37_track_pre_implant_smoothed.py --track 48

  Or set TRACK_ID below and just hit Run in VS Code — no CLI args needed.
"""

TRACK_ID = 48   # <-- set track id here to run directly (e.g. VS Code Run button)

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
parser.add_argument('--track', type=int, default=None, help='track_id to inspect (overrides TRACK_ID above)')
args, _unknown = parser.parse_known_args()
if args.track is not None:
    TRACK_ID = args.track

# ── Load data + dynamic radial distance (same as script 34) ──────────────────

kine = pd.read_csv(out_dir / f'motion_kinematics_{version}.csv').sort_values(['track_id', 't'])
erk  = pd.read_csv(out_dir / f'erk_cn_ratio_{version}.csv').sort_values(['track_id', 't'])
vstats = pd.read_csv(out_dir / f'volume_track_stats_{version}.csv')[['track_id', 'icm_dist_um']]
icm_lookup = vstats.set_index('track_id')['icm_dist_um']

centroid = (kine.groupby('t')[['x_um_reg', 'y_um_reg']]
                .mean()
                .rename(columns={'x_um_reg': 'cx', 'y_um_reg': 'cy'}))
kine = kine.join(centroid, on='t')
kine['radial_dist_dyn'] = np.sqrt(
    (kine['x_um_reg'] - kine['cx'])**2 +
    (kine['y_um_reg'] - kine['cy'])**2
)
kine['time_min'] = kine['t'] * interval

merged = (kine[['track_id', 't', 'time_min', 'z_um_reg', 'radial_dist_dyn']]
          .merge(erk[['track_id', 't', 'erk_cn_ratio']], on=['track_id', 't']))

if TRACK_ID not in merged['track_id'].unique():
    raise SystemExit(f'track_id {TRACK_ID} not found in {version}')

cell = merged[(merged['track_id'] == TRACK_ID) & (merged['t'] < T_SPLIT)].sort_values('t')
if cell.empty:
    raise SystemExit(f'track_id {TRACK_ID} has no points in the pre-implantation window (t<{T_SPLIT})')

icm  = icm_lookup.get(TRACK_ID, np.nan)
icm_str = f'{icm:.0f} µm from ICM' if not pd.isna(icm) else 'ICM dist: n/a'

n_pts = len(cell)
sg_window = min(SG_WINDOW, n_pts if n_pts % 2 == 1 else n_pts - 1)
can_smooth = sg_window >= SG_POLYORD + 2

# ── Plot ────────────────────────────────────────────────────────────────────────

vars3 = [
    ('erk_cn_ratio',    'ERK C/N ratio',        '#d6604d'),
    ('z_um_reg',        'Z position (µm)',      '#4393c3'),
    ('radial_dist_dyn', 'Radial distance (µm)', '#4daf4a'),
]

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)

for ax, (vcol, ylabel, color) in zip(axes, vars3):
    ax.plot(cell['time_min'], cell[vcol], color=color, linewidth=0.8, alpha=0.35,
             marker='o', markersize=3, zorder=1, label='raw')
    if can_smooth:
        smoothed = savgol_filter(cell[vcol].values, window_length=sg_window, polyorder=SG_POLYORD)
        ax.plot(cell['time_min'], smoothed, color=color, linewidth=2.2, zorder=2, label='smoothed')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=8)

axes[0].legend(fontsize=7, loc='best')
axes[-1].set_xlabel('Time (min)', fontsize=10)
fig.suptitle(
    f'track {TRACK_ID}  |  {icm_str}  |  pre-implantation window\n'
    f't = {cell["t"].min():.0f}–{cell["t"].max():.0f}  (n={n_pts})'
    + ('' if can_smooth else '  |  too few points to smooth — raw only'),
    fontsize=11,
)

out_path = out_dir / f'track_{TRACK_ID}_preimplant_smoothed_{version}.png'
fig.savefig(out_path, dpi=150)
plt.close()
print(f'Saved: {out_path}')
