"""
Script 34: Close-up timecourse for a single track — ERK C/N, Z position, radial distance.

For one track_id, plots three stacked panels (ERK C/N ratio, Z position, dynamic
XY radial distance) over its full observed timespan, raw values, no smoothing.
Same data/derivation as scripts 27/30 (radial = distance from per-timepoint
centroid of all tracked cells), just zoomed to one cell for close inspection.

Usage
-----
  conda run -n napari_env python3 pipelines/tracking/34_single_track_inspect.py --track 48
"""

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
parser.add_argument('--track', type=int, required=True, help='track_id to inspect')
args = parser.parse_args()
TRACK_ID = args.track

# ── Load data + dynamic radial distance (same as scripts 27/30) ───────────────

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

cell = merged[merged['track_id'] == TRACK_ID].sort_values('t')
icm  = icm_lookup.get(TRACK_ID, np.nan)
icm_str = f'{icm:.0f} µm from ICM' if not pd.isna(icm) else 'ICM dist: n/a'

# ── Plot ────────────────────────────────────────────────────────────────────────

vars3 = [
    ('erk_cn_ratio',    'ERK C/N ratio',        '#d6604d'),
    ('z_um_reg',        'Z position (µm)',      '#4393c3'),
    ('radial_dist_dyn', 'Radial distance (µm)', '#4daf4a'),
]

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)

for ax, (vcol, ylabel, color) in zip(axes, vars3):
    ax.axvline(T_SPLIT * interval, color='0.4', linewidth=0.8, linestyle=':', zorder=1)
    ax.axvspan(cell['time_min'].min(), T_SPLIT * interval, color='0.93', zorder=0)
    ax.plot(cell['time_min'], cell[vcol], color=color, linewidth=1.8,
             marker='o', markersize=3, zorder=2)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=8)

axes[-1].set_xlabel('Time (min)', fontsize=10)
fig.suptitle(
    f'track {TRACK_ID}  |  {icm_str}\n'
    f't = {cell["t"].min()}–{cell["t"].max()}  |  grey = pre-implantation (t<{T_SPLIT})',
    fontsize=11,
)

out_path = out_dir / f'track_{TRACK_ID}_inspect_{version}.png'
fig.savefig(out_path, dpi=150)
plt.close()
print(f'Saved: {out_path}')
