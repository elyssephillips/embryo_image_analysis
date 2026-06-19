"""
Script 30: Individual ERK / Z / radial timecourses for the most-confident tracks.

Restricts to the 42 tracks that span t=20 through t=80 (the best-covered subset --
see script 28/29 discussion: requiring zero gaps drops this to 12 tracks, almost
all ICM-far, so this uses the looser "spans the window" criterion instead).

For each cell, plots three lines side by side: ERK C/N ratio, Z position, and
dynamic XY radial distance, all on the same time axis -- raw data, no smoothing,
no detection -- for direct visual inspection.

Outputs
-------
  confident_tracks_individual_{version}_page{N}.png

Run with:
  conda run -n napari_env python3 pipelines/tracking/30_confident_tracks_individual.py
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

T_SPLIT      = 30
T_LO, T_HI   = 20, 80   # span criterion
T_MIN_PLOT   = 20       # don't plot before this
CELLS_PER_PAGE = 14
N_ICM_GROUPS = 3

# ── Load data + dynamic radial distance ───────────────────────────────────────

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

# ── Identify the 42 tracks spanning t=20-80 ────────────────────────────────────

tmin = kine.groupby('track_id')['t'].min()
tmax = kine.groupby('track_id')['t'].max()
confident_tracks = tmin[(tmin <= T_LO) & (tmax >= T_HI)].index.tolist()
print(f'Confident tracks (span t={T_LO}-{T_HI}): {len(confident_tracks)}')

merged = merged[(merged['track_id'].isin(confident_tracks)) & (merged['t'] >= T_MIN_PLOT)].copy()

sorted_tracks = sorted(confident_tracks, key=lambda t: icm_lookup.get(t, np.inf))

# ── tertile labels for context ─────────────────────────────────────────────────

all_icm = icm_lookup.dropna()
near_hi = all_icm.quantile(1/3)
far_lo  = all_icm.quantile(2/3)

def icm_bucket(icm):
    if pd.isna(icm):
        return 'n/a'
    if icm <= near_hi:
        return 'near'
    if icm >= far_lo:
        return 'far'
    return 'mid'

# shared y-limits across all panels
z_lo, z_hi     = merged['z_um_reg'].quantile([0.01, 0.99])
erk_lo, erk_hi = merged['erk_cn_ratio'].quantile([0.01, 0.99])
rad_lo, rad_hi = merged['radial_dist_dyn'].quantile([0.01, 0.99])
xlim = (T_MIN_PLOT * interval, merged['time_min'].max())

vars3 = [
    ('erk_cn_ratio',    'ERK C/N ratio',         (erk_lo, erk_hi), '#d6604d'),
    ('z_um_reg',        'Z position (µm)',       (z_lo, z_hi),     '#4393c3'),
    ('radial_dist_dyn', 'Radial distance (µm)',  (rad_lo, rad_hi), '#4daf4a'),
]

# ── Paginate and plot ───────────────────────────────────────────────────────────

n_pages = int(np.ceil(len(sorted_tracks) / CELLS_PER_PAGE))

for page in range(n_pages):
    page_tracks = sorted_tracks[page * CELLS_PER_PAGE:(page + 1) * CELLS_PER_PAGE]
    n_cells = len(page_tracks)

    fig, axes = plt.subplots(n_cells, 3, figsize=(13, 2.3 * n_cells),
                              squeeze=False, constrained_layout=True)

    for row_i, tid in enumerate(page_tracks):
        cell = merged[merged['track_id'] == tid].sort_values('t')
        icm  = icm_lookup.get(tid, np.nan)
        bucket = icm_bucket(icm)
        icm_str = f'{icm:.0f} µm' if not pd.isna(icm) else 'n/a'

        for col_i, (vcol, ylabel, ylim, color) in enumerate(vars3):
            ax = axes[row_i, col_i]
            ax.axvline(T_SPLIT * interval, color='0.4', linewidth=0.8, linestyle=':', zorder=1)
            ax.axvspan(xlim[0], T_SPLIT * interval, color='0.93', zorder=0)
            ax.axvspan(T_LO * interval, T_HI * interval, color='#fffbcc', alpha=0.4, zorder=0)
            ax.plot(cell['time_min'], cell[vcol], color=color, linewidth=1.4, zorder=2)
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.tick_params(labelsize=7)
            if row_i == 0:
                ax.set_title(ylabel, fontsize=9)
            if col_i == 0:
                ax.set_ylabel(f'track {tid}\nICM {icm_str} ({bucket})', fontsize=7.5)
            if row_i == n_cells - 1:
                ax.set_xlabel('Time (min)', fontsize=8)

    fig.suptitle(
        f'Confident tracks (span t={T_LO}-{T_HI}) — page {page+1}/{n_pages}\n'
        f'Grey shading = pre-implantation  |  yellow shading = t={T_LO}-{T_HI} confident window  |  '
        f'sorted by ICM distance',
        fontsize=10,
    )
    out_path = out_dir / f'confident_tracks_individual_{version}_page{page+1}.png'
    fig.savefig(out_path, dpi=140)
    plt.close()
    print(f'Saved: {out_path.name}  ({n_cells} cells)')
