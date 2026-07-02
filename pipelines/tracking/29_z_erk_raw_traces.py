"""
Script 29: Raw Z and ERK traces per cell — no detection, no thresholds.

The onset-timing analysis (script 28) reduces each cell's trajectory to a single
number (a threshold-crossing time), which hides exactly the ambiguity -- noisy Z,
partial flattening, unclear transitions -- that's best judged by eye. This script
just plots the raw Z position and ERK C/N ratio over time, dual-axis, one panel per
cell, so the actual data can be inspected directly instead of trusting a detector.

t < T_MIN_PLOT (20) is excluded -- early timepoints are noisier / less reliable.

Cells are split into ICM-distance tertiles (near / mid / far) and plotted as
separate pages within each group, since cells with different ICM proximity may
follow different Z/ERK dynamics (e.g. the FGF4-driven ICM-proximal population vs
the distal population) and lumping them together obscures both stories.

Outputs
-------
  z_erk_raw_traces_{version}_{icm_group}_page{N}.png — grid of individual cell panels

Run with:
  conda run -n napari_env python3 pipelines/tracking/29_z_erk_raw_traces.py
"""

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

T_SPLIT        = 30   # implantation onset
T_MIN_PLOT     = 20   # exclude timepoints before this
MIN_PTS        = 15   # minimum timepoints (after T_MIN_PLOT cut) to bother plotting
CELLS_PER_PAGE = 25
N_COLS         = 5
N_ICM_GROUPS   = 3
ICM_GROUP_LABELS = ['near', 'mid', 'far']

# ── Load data ──────────────────────────────────────────────────────────────────

kine = pd.read_csv(out_dir / f'motion_kinematics_{version}.csv').sort_values(['track_id', 't'])
erk  = pd.read_csv(out_dir / f'erk_cn_ratio_{version}.csv').sort_values(['track_id', 't'])
vstats = pd.read_csv(out_dir / f'volume_track_stats_{version}.csv')[['track_id', 'icm_dist_um']]

kine['time_min'] = kine['t'] * interval

merged = (kine[['track_id', 't', 'time_min', 'z_um_reg']]
          .merge(erk[['track_id', 't', 'erk_cn_ratio']], on=['track_id', 't'])
          .merge(vstats, on='track_id', how='left'))

merged = merged[merged['t'] >= T_MIN_PLOT].copy()

track_counts = merged.groupby('track_id')['t'].count()
valid_tracks = track_counts[track_counts >= MIN_PTS].index
merged = merged[merged['track_id'].isin(valid_tracks)]

# Only tracks with a valid ICM distance can be grouped
icm_lookup = vstats.set_index('track_id')['icm_dist_um'].dropna()
valid_tracks = [t for t in valid_tracks if t in icm_lookup.index]
merged = merged[merged['track_id'].isin(valid_tracks)]

icm_series = icm_lookup.loc[valid_tracks]
icm_groups = pd.qcut(icm_series, q=N_ICM_GROUPS, labels=ICM_GROUP_LABELS)

print(f'Tracks with valid ICM distance and enough timepoints (t>={T_MIN_PLOT}): {len(valid_tracks)}')
for lbl in ICM_GROUP_LABELS:
    ids = icm_groups[icm_groups == lbl].index
    rng = icm_series.loc[ids]
    print(f'  {lbl}: n={len(ids)}  ICM range [{rng.min():.0f}, {rng.max():.0f}] µm')

# shared y-limits across all panels for honest visual comparison
z_lo, z_hi     = merged['z_um_reg'].quantile([0.01, 0.99])
erk_lo, erk_hi = merged['erk_cn_ratio'].quantile([0.01, 0.99])
xlim = (T_MIN_PLOT * interval, merged['time_min'].max())

# ── Plot per ICM group, paginated ──────────────────────────────────────────────

for lbl in ICM_GROUP_LABELS:
    group_tracks = icm_groups[icm_groups == lbl].index.tolist()
    sorted_tracks = sorted(group_tracks, key=lambda t: icm_lookup.get(t, np.inf))

    n_pages = int(np.ceil(len(sorted_tracks) / CELLS_PER_PAGE))

    for page in range(n_pages):
        page_tracks = sorted_tracks[page * CELLS_PER_PAGE:(page + 1) * CELLS_PER_PAGE]
        n_cells = len(page_tracks)
        n_rows  = int(np.ceil(n_cells / N_COLS))

        fig, axes = plt.subplots(n_rows, N_COLS, figsize=(4.5 * N_COLS, 2.8 * n_rows),
                                  constrained_layout=True)
        axes_flat = np.array(axes).flatten() if n_cells > 1 else [axes]

        for ax, tid in zip(axes_flat, page_tracks):
            cell = merged[merged['track_id'] == tid].sort_values('t')
            icm  = icm_lookup.get(tid, np.nan)

            ax.axvline(T_SPLIT * interval, color='0.4', linewidth=0.8, linestyle=':', zorder=1)
            ax.axvspan(xlim[0], T_SPLIT * interval, color='0.93', zorder=0)

            ax.plot(cell['time_min'], cell['z_um_reg'], color='#4393c3', linewidth=1.4, zorder=2)
            ax.set_ylim(z_lo, z_hi)
            ax.set_xlim(xlim)
            ax.tick_params(axis='y', labelcolor='#4393c3', labelsize=7)
            ax.tick_params(axis='x', labelsize=7)

            ax2 = ax.twinx()
            ax2.plot(cell['time_min'], cell['erk_cn_ratio'], color='#d6604d', linewidth=1.4, zorder=2)
            ax2.set_ylim(erk_lo, erk_hi)
            ax2.tick_params(axis='y', labelcolor='#d6604d', labelsize=7)

            ax.set_title(f'track {tid}  |  ICM {icm:.0f} µm', fontsize=8)

        for ax in axes_flat[n_cells:]:
            ax.set_visible(False)

        fig.suptitle(
            f'Z position (blue) and ERK C/N (red) over time — ICM-{lbl} group, page {page+1}/{n_pages}\n'
            f'Shaded = pre-implantation  |  t≥{T_MIN_PLOT}  |  sorted by ICM distance within group',
            fontsize=10,
        )
        out_path = out_dir / f'z_erk_raw_traces_{version}_icm-{lbl}_page{page+1}.png'
        fig.savefig(out_path, dpi=140)
        plt.close()
        print(f'Saved: {out_path.name}  ({n_cells} cells)')
