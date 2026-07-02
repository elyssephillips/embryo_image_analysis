"""
Script 31: Peripheral/central ERK quadrant group timecourses — confident tracks only.

Same as script 27's group timecourse figure, but restricted to the 42 tracks that
span t=20-80 (the best-covered subset, identified in script 30 -- requiring zero
gaps drops this to 12 tracks, almost all ICM-far, so the looser "spans the window"
criterion is used here instead).

Within this subset, late-window (t>=80) quadrant counts are:
  peripheral ERK-high: 9, central ERK-low: 6, peripheral ERK-low: 3, central ERK-high: 0

Outputs
-------
  peripheral_erk_group_timecourses_confident_{version}.png

Run with:
  conda run -n napari_env python3 pipelines/tracking/31_peripheral_erk_group_timecourse_confident.py
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
T_LO, T_HI   = 20, 80    # "confident" span criterion
T_LATE       = 80        # start of late classification window
MIN_LATE_PTS = 5
TERTILE_LO   = 0.33
TERTILE_HI   = 0.67

# ── Load data + dynamic radial distance ───────────────────────────────────────

kine = pd.read_csv(out_dir / f'motion_kinematics_{version}.csv').sort_values(['track_id', 't'])
erk  = pd.read_csv(out_dir / f'erk_cn_ratio_{version}.csv').sort_values(['track_id', 't'])

centroid = (kine.groupby('t')[['x_um_reg', 'y_um_reg']]
                .mean()
                .rename(columns={'x_um_reg': 'cx', 'y_um_reg': 'cy'}))
kine = kine.join(centroid, on='t')
kine['radial_dist_dyn'] = np.sqrt(
    (kine['x_um_reg'] - kine['cx'])**2 +
    (kine['y_um_reg'] - kine['cy'])**2
)
kine['time_min'] = kine['t'] * interval

# ── Restrict to confident tracks (span t=20-80) ────────────────────────────────

tmin = kine.groupby('track_id')['t'].min()
tmax = kine.groupby('track_id')['t'].max()
confident_tracks = tmin[(tmin <= T_LO) & (tmax >= T_HI)].index.tolist()
print(f'Confident tracks (span t={T_LO}-{T_HI}): {len(confident_tracks)}')

merged = (kine[['track_id', 't', 'time_min', 'z_um_reg', 'radial_dist_dyn']]
          .merge(erk[['track_id', 't', 'erk_cn_ratio']], on=['track_id', 't']))
merged = merged[merged['track_id'].isin(confident_tracks)].copy()

# ── Classify by late-window (radial, ERK) quadrant ────────────────────────────

late = merged[merged['t'] >= T_LATE].copy()
late_counts = late.groupby('track_id')['erk_cn_ratio'].count()
valid_late  = late_counts[late_counts >= MIN_LATE_PTS].index

late_summary = (late[late['track_id'].isin(valid_late)]
                .groupby('track_id')[['radial_dist_dyn', 'erk_cn_ratio']]
                .mean()
                .rename(columns={'radial_dist_dyn': 'late_radial', 'erk_cn_ratio': 'late_erk'}))

rad_lo = late_summary['late_radial'].quantile(TERTILE_LO)
rad_hi = late_summary['late_radial'].quantile(TERTILE_HI)
erk_lo = late_summary['late_erk'].quantile(TERTILE_LO)
erk_hi = late_summary['late_erk'].quantile(TERTILE_HI)

def classify(row):
    rad = 'periph' if row['late_radial'] >= rad_hi else ('central' if row['late_radial'] <= rad_lo else None)
    erkc = 'high'  if row['late_erk']    >= erk_hi else ('low'     if row['late_erk']    <= erk_lo else None)
    if rad is None or erkc is None:
        return None
    return f'{rad}_{erkc}'

late_summary['quadrant'] = late_summary.apply(classify, axis=1)
late_summary = late_summary.dropna(subset=['quadrant'])

quadrant_colors = {
    'periph_high':  '#d6604d',
    'periph_low':   '#f4a582',
    'central_high': '#4393c3',
    'central_low':  '#92c5de',
}
quadrant_labels = {
    'periph_high':  'Peripheral ERK-high',
    'periph_low':   'Peripheral ERK-low',
    'central_high': 'Central ERK-high',
    'central_low':  'Central ERK-low',
}

print('Quadrant counts (confident tracks only):')
for q, n in late_summary['quadrant'].value_counts().items():
    print(f'  {quadrant_labels[q]}: n={n}')

# ── Group mean timecourses (ERK, Z, radial) ────────────────────────────────────

quad_order = [q for q in ['periph_high', 'central_high', 'periph_low', 'central_low']
              if q in late_summary['quadrant'].values]
vars3 = [
    ('erk_cn_ratio',    'ERK C/N ratio'),
    ('z_um_reg',        'Z position (µm)\n← top of embryo    bottom / dish →'),
    ('radial_dist_dyn', 'Radial distance (µm)'),
]

merged['quadrant'] = merged['track_id'].map(late_summary['quadrant'])
quad_data = merged.dropna(subset=['quadrant'])

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for ax, (vcol, ylabel) in zip(axes, vars3):
    ax.axvline(T_SPLIT * interval, color='0.4', linewidth=0.8, linestyle=':', zorder=1)
    ax.axvspan(0, T_SPLIT * interval, color='0.92', zorder=0)
    for q in quad_order:
        grp = quad_data[quad_data['quadrant'] == q]
        ts  = grp.groupby('time_min')[vcol].agg(['mean', 'sem'])
        n   = grp['track_id'].nunique()
        lw  = 2.5 if 'high' in q else 1.2
        ls  = '-' if 'periph' in q else '--'
        ax.plot(ts.index, ts['mean'],
                color=quadrant_colors[q], linewidth=lw, linestyle=ls,
                label=f'{quadrant_labels[q]} (n={n})', zorder=3)
        ax.fill_between(ts.index,
                        ts['mean'] - ts['sem'],
                        ts['mean'] + ts['sem'],
                        color=quadrant_colors[q], alpha=0.15, zorder=2)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, loc='upper left')

axes[2].set_xlabel('Time (min)')
fig.suptitle(
    f'Timecourses by late-timepoint quadrant — confident tracks only (span t={T_LO}-{T_HI}, n={len(confident_tracks)})\n'
    f'Solid = peripheral, dashed = central  |  bright = ERK-high, pale = ERK-low',
    fontsize=10
)
plt.tight_layout()
out_path = out_dir / f'peripheral_erk_group_timecourses_confident_{version}.png'
fig.savefig(out_path, dpi=150)
plt.close()
print(f'Saved: {out_path.name}')
