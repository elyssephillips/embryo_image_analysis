"""
Script 32: ERK / Z / radial timecourses by ICM-distance group — confident tracks only.

Same idea as script 31, but grouping by pre-implantation ICM distance (near/mid/far
tertiles, fixed t=30 snapshot) instead of by late-timepoint ERK/radial quadrant.
Restricted to the 42 tracks spanning t=20-80 (see script 30).

Outputs
-------
  icm_group_timecourses_confident_{version}.png

Run with:
  conda run -n napari_env python3 pipelines/tracking/32_icm_group_timecourse_confident.py
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

T_SPLIT    = 30
T_LO, T_HI = 20, 80   # "confident" span criterion
ICM_GROUP_LABELS = ['near half', 'far half']

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

# ── Restrict to confident tracks (span t=20-80) with valid ICM distance ───────

tmin = kine.groupby('track_id')['t'].min()
tmax = kine.groupby('track_id')['t'].max()
confident_tracks = tmin[(tmin <= T_LO) & (tmax >= T_HI)].index.tolist()
confident_tracks = [t for t in confident_tracks if t in icm_lookup.dropna().index]
print(f'Confident tracks with valid ICM distance: {len(confident_tracks)}')

merged = (kine[['track_id', 't', 'time_min', 'z_um_reg', 'radial_dist_dyn']]
          .merge(erk[['track_id', 't', 'erk_cn_ratio']], on=['track_id', 't']))
merged = merged[merged['track_id'].isin(confident_tracks)].copy()

# ── ICM median-split groups (boundary from the full dataset, as established earlier) ─

all_icm = icm_lookup.dropna()
median_icm = all_icm.quantile(0.5)

def icm_bucket(icm):
    if pd.isna(icm):
        return None
    return 'near half' if icm <= median_icm else 'far half'

icm_group = {tid: icm_bucket(icm_lookup.get(tid)) for tid in confident_tracks}
merged['icm_group'] = merged['track_id'].map(icm_group)

print(f'Median boundary: {median_icm:.0f} µm')
print('ICM group counts (confident tracks):')
group_counts = pd.Series(icm_group).value_counts()
print(group_counts.to_dict())

# ── Group mean timecourses (ERK, Z, radial) ────────────────────────────────────

group_colors = {
    'near half': '#d6604d',
    'far half':  '#4393c3',
}
vars3 = [
    ('erk_cn_ratio',    'ERK C/N ratio'),
    ('z_um_reg',        'Z position (µm)\n← top of embryo    bottom / dish →'),
    ('radial_dist_dyn', 'Radial distance (µm)'),
]

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for ax, (vcol, ylabel) in zip(axes, vars3):
    ax.axvline(T_SPLIT * interval, color='0.4', linewidth=0.8, linestyle=':', zorder=1)
    ax.axvspan(0, T_SPLIT * interval, color='0.92', zorder=0)
    for lbl in ICM_GROUP_LABELS:
        grp = merged[merged['icm_group'] == lbl]
        if grp.empty:
            continue
        ts = grp.groupby('time_min')[vcol].agg(['mean', 'sem'])
        n  = grp['track_id'].nunique()
        ax.plot(ts.index, ts['mean'],
                color=group_colors[lbl], linewidth=2.2, label=f'ICM-{lbl} (n={n})', zorder=3)
        ax.fill_between(ts.index,
                        ts['mean'] - ts['sem'],
                        ts['mean'] + ts['sem'],
                        color=group_colors[lbl], alpha=0.2, zorder=2)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, loc='upper left')

axes[2].set_xlabel('Time (min)')
fig.suptitle(
    f'Timecourses by pre-implantation ICM-distance group — confident tracks only '
    f'(span t={T_LO}-{T_HI}, n={len(confident_tracks)})\n'
    f'near half ≤{median_icm:.0f}µm  |  far half >{median_icm:.0f}µm  (fixed t=30 snapshot)',
    fontsize=10
)
plt.tight_layout()
out_path = out_dir / f'icm_halfsplit_timecourses_confident_{version}.png'
fig.savefig(out_path, dpi=150)
plt.close()
print(f'Saved: {out_path.name}')
