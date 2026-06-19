"""
Script 33: ERK / Z / radial timecourses, all confident tracks, coloured continuously by ICM distance.

Instead of binning into discrete ICM groups (script 32), plot every one of the 42
confident tracks (span t=20-80, see script 30) as an individual line, coloured on a
continuous scale by its pre-implantation ICM distance. Avoids artifacts from bin
boundary choice and shows the full gradient + any non-monotonic structure directly.

Outputs
-------
  icm_continuous_color_confident_{version}.png — 3 panels (ERK, Z, radial), individual lines

Run with:
  conda run -n napari_env python3 pipelines/tracking/33_icm_continuous_color_confident.py
"""

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

T_SPLIT    = 30
T_LO, T_HI = 20, 80

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

# ── Restrict to confident tracks with valid ICM distance ──────────────────────

tmin = kine.groupby('track_id')['t'].min()
tmax = kine.groupby('track_id')['t'].max()
confident_tracks = tmin[(tmin <= T_LO) & (tmax >= T_HI)].index.tolist()
confident_tracks = [t for t in confident_tracks if t in icm_lookup.dropna().index]
print(f'Confident tracks with valid ICM distance: {len(confident_tracks)}')

merged = (kine[['track_id', 't', 'time_min', 'z_um_reg', 'radial_dist_dyn']]
          .merge(erk[['track_id', 't', 'erk_cn_ratio']], on=['track_id', 't']))
merged = merged[merged['track_id'].isin(confident_tracks)].copy()

# ── Continuous ICM colormap ────────────────────────────────────────────────────

icm_vals = icm_lookup.loc[confident_tracks]
norm = mcolors.Normalize(vmin=icm_vals.min(), vmax=icm_vals.max())
cmap = plt.cm.coolwarm_r  # red = near ICM, blue = far from ICM (consistent with other scripts)

vars3 = [
    ('erk_cn_ratio',    'ERK C/N ratio'),
    ('z_um_reg',        'Z position (µm)\n← top of embryo    bottom / dish →'),
    ('radial_dist_dyn', 'Radial distance (µm)'),
]

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# sort tracks far-to-near so near (drawn last) sits on top, easier to see
plot_order = sorted(confident_tracks, key=lambda t: -icm_lookup.get(t))

for ax, (vcol, ylabel) in zip(axes, vars3):
    ax.axvline(T_SPLIT * interval, color='0.4', linewidth=0.8, linestyle=':', zorder=1)
    ax.axvspan(0, T_SPLIT * interval, color='0.92', zorder=0)
    for tid in plot_order:
        cell = merged[merged['track_id'] == tid].sort_values('t')
        color = cmap(norm(icm_lookup.get(tid)))
        ax.plot(cell['time_min'], cell[vcol], color=color, linewidth=1.1, alpha=0.8, zorder=2)
    ax.set_ylabel(ylabel, fontsize=9)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=axes, label='ICM distance (µm)\n← near        far →', shrink=0.6, pad=0.02)

axes[2].set_xlabel('Time (min)')
fig.suptitle(
    f'ERK / Z / radial distance, all confident tracks (span t={T_LO}-{T_HI}, n={len(confident_tracks)})\n'
    f'Coloured continuously by pre-implantation ICM distance (fixed t=30 snapshot)',
    fontsize=10
)
out_path = out_dir / f'icm_continuous_color_confident_{version}.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out_path.name}')
