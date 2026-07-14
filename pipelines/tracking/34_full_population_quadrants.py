"""
Script 34: Full-population quadrant classification (median split) vs extreme-tertile
classification (script 27/31) — and ICM distance comparison between the two.

Script 27 classified late-timepoint (radial, ERK) into quadrants using top/bottom
TERTILES, which excluded ~54% of eligible tracks (those landing in the middle
tertile on either axis -- see prior discussion). This script instead uses a MEDIAN
split on each axis, so every eligible track gets assigned to one of the four
quadrants -- nothing excluded.

Then compares the pre-implantation ICM distance distribution per quadrant group
between the two classification schemes, to see whether the "extreme" comparison
(script 27) generalizes to the full population or is specific to the most
extreme cells.

Outputs
-------
  full_population_quadrant_scatter_{version}.png   — (radial, ERK) scatter, both
                                                       classification schemes shown
  icm_by_quadrant_comparison_{version}.png          — ICM distance by group,
                                                       extreme-tertile vs median-split
  full_population_group_timecourses_{version}.png   — ERK/Z/radial timecourses,
                                                       median-split groups (all tracks)

Run with:
  conda run -n napari_env python3 pipelines/tracking/34_full_population_quadrants.py
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
T_LATE       = 80
MIN_LATE_PTS = 5
TERTILE_LO   = 0.33
TERTILE_HI   = 0.67

# ── Load data + dynamic radial distance ───────────────────────────────────────

kine = pd.read_csv(out_dir / f'motion_kinematics_{version}.csv').sort_values(['track_id', 't'])
erk  = pd.read_csv(out_dir / f'erk_cn_ratio_{version}.csv').sort_values(['track_id', 't'])
vstats = pd.read_csv(out_dir / f'volume_track_stats_{version}.csv')[['track_id', 'icm_dist_um']]

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

# ── Eligible tracks: >=MIN_LATE_PTS timepoints in the late window ─────────────

late = merged[merged['t'] >= T_LATE].copy()
late_counts = late.groupby('track_id')['erk_cn_ratio'].count()
valid_late  = late_counts[late_counts >= MIN_LATE_PTS].index

late_summary = (late[late['track_id'].isin(valid_late)]
                .groupby('track_id')[['radial_dist_dyn', 'erk_cn_ratio']]
                .mean()
                .rename(columns={'radial_dist_dyn': 'late_radial', 'erk_cn_ratio': 'late_erk'}))
late_summary = late_summary.merge(vstats, on='track_id', how='left')

print(f'Eligible tracks (>= {MIN_LATE_PTS} timepoints in t>={T_LATE}): {len(late_summary)}')

# ── Scheme 1: extreme tertile classification (script 27/31) ──────────────────

rad_lo = late_summary['late_radial'].quantile(TERTILE_LO)
rad_hi = late_summary['late_radial'].quantile(TERTILE_HI)
erk_lo = late_summary['late_erk'].quantile(TERTILE_LO)
erk_hi = late_summary['late_erk'].quantile(TERTILE_HI)

def classify_extreme(row):
    rad = 'periph' if row['late_radial'] >= rad_hi else ('central' if row['late_radial'] <= rad_lo else None)
    erkc = 'high'  if row['late_erk']    >= erk_hi else ('low'     if row['late_erk']    <= erk_lo else None)
    if rad is None or erkc is None:
        return None
    return f'{rad}_{erkc}'

late_summary['quadrant_extreme'] = late_summary.apply(classify_extreme, axis=1)

# ── Scheme 2: median split, all tracks classified ──────────────────────────────

rad_med = late_summary['late_radial'].median()
erk_med = late_summary['late_erk'].median()

def classify_median(row):
    rad = 'periph' if row['late_radial'] >= rad_med else 'central'
    erkc = 'high'  if row['late_erk']    >= erk_med else 'low'
    return f'{rad}_{erkc}'

late_summary['quadrant_median'] = late_summary.apply(classify_median, axis=1)

print('\nExtreme-tertile quadrant counts:')
print(late_summary['quadrant_extreme'].value_counts(dropna=False).to_dict())
print('\nMedian-split quadrant counts (all tracks classified):')
print(late_summary['quadrant_median'].value_counts().to_dict())

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
quad_order = ['periph_high', 'central_high', 'periph_low', 'central_low']

# ── Figure 1: scatter showing both schemes ─────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

for ax, qcol, title in [
    (axes[0], 'quadrant_extreme', f'Extreme-tertile classification\n(n={late_summary["quadrant_extreme"].notna().sum()} of {len(late_summary)} classified)'),
    (axes[1], 'quadrant_median',  f'Median-split classification\n(n={len(late_summary)} of {len(late_summary)} classified)'),
]:
    for q in quad_order:
        sub = late_summary[late_summary[qcol] == q]
        if sub.empty:
            continue
        ax.scatter(sub['late_radial'], sub['late_erk'],
                   color=quadrant_colors[q], label=f'{quadrant_labels[q]} (n={len(sub)})',
                   s=45, edgecolors='0.3', linewidths=0.4, zorder=3)
    unclassified = late_summary[late_summary[qcol].isna()] if qcol == 'quadrant_extreme' else late_summary.iloc[0:0]
    if not unclassified.empty:
        ax.scatter(unclassified['late_radial'], unclassified['late_erk'],
                   color='0.8', label=f'excluded (n={len(unclassified)})', s=30, zorder=2)
    ax.axvline(rad_med, color='0.4', linewidth=0.8, linestyle='-', alpha=0.4)
    ax.axhline(erk_med, color='0.4', linewidth=0.8, linestyle='-', alpha=0.4)
    if qcol == 'quadrant_extreme':
        ax.axvline(rad_lo, color='0.5', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.axvline(rad_hi, color='0.5', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.axhline(erk_lo, color='0.5', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.axhline(erk_hi, color='0.5', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel(f'Mean radial distance (t≥{T_LATE}, µm)')
    ax.set_ylabel(f'Mean ERK C/N (t≥{T_LATE})')
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc='best')

plt.tight_layout()
fig.savefig(out_dir / f'full_population_quadrant_scatter_{version}.png', dpi=150)
plt.close()
print(f'\nSaved: full_population_quadrant_scatter_{version}.png')

# ── Figure 2: ICM distance by group, both schemes side by side ────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, qcol, title in [
    (axes[0], 'quadrant_extreme', 'Extreme-tertile groups'),
    (axes[1], 'quadrant_median',  'Median-split groups (all tracks)'),
]:
    box_data, box_labels, box_colors = [], [], []
    for q in quad_order:
        sub = late_summary[late_summary[qcol] == q]['icm_dist_um'].dropna()
        if len(sub) == 0:
            continue
        box_data.append(sub.values)
        box_labels.append(f'{quadrant_labels[q]}\n(n={len(sub)})')
        box_colors.append(quadrant_colors[q])

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for i, sub in enumerate(box_data):
        jitter = np.random.normal(0, 0.05, size=len(sub))
        ax.scatter(np.full(len(sub), i + 1) + jitter, sub, color='0.2', s=15, zorder=3, alpha=0.7)

    ax.set_ylabel('ICM distance at t=30 (µm)')
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis='x', labelsize=8)

plt.tight_layout()
fig.savefig(out_dir / f'icm_by_quadrant_comparison_{version}.png', dpi=150)
plt.close()
print(f'Saved: icm_by_quadrant_comparison_{version}.png')

# print numeric summary
print('\n--- ICM distance by group (median [IQR], n) ---')
for scheme, qcol in [('Extreme-tertile', 'quadrant_extreme'), ('Median-split', 'quadrant_median')]:
    print(f'\n{scheme}:')
    for q in quad_order:
        sub = late_summary[late_summary[qcol] == q]['icm_dist_um'].dropna()
        if len(sub) == 0:
            continue
        print(f'  {quadrant_labels[q]}: median={sub.median():.0f}  IQR=[{sub.quantile(0.25):.0f}, {sub.quantile(0.75):.0f}]  n={len(sub)}')

# ── Figure 3: group timecourses, median-split (all tracks) ────────────────────

merged['quadrant_median'] = merged['track_id'].map(late_summary.set_index('track_id')['quadrant_median'])
quad_data = merged.dropna(subset=['quadrant_median'])

vars3 = [
    ('erk_cn_ratio',    'ERK C/N ratio'),
    ('z_um_reg',        'Z position (µm)\n← top of embryo    bottom / dish →'),
    ('radial_dist_dyn', 'Radial distance (µm)'),
]

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for ax, (vcol, ylabel) in zip(axes, vars3):
    ax.axvline(T_SPLIT * interval, color='0.4', linewidth=0.8, linestyle=':', zorder=1)
    ax.axvspan(0, T_SPLIT * interval, color='0.92', zorder=0)
    for q in quad_order:
        grp = quad_data[quad_data['quadrant_median'] == q]
        if grp.empty:
            continue
        ts = grp.groupby('time_min')[vcol].agg(['mean', 'sem'])
        n  = grp['track_id'].nunique()
        lw = 2.5 if 'high' in q else 1.2
        ls = '-' if 'periph' in q else '--'
        ax.plot(ts.index, ts['mean'], color=quadrant_colors[q], linewidth=lw, linestyle=ls,
                label=f'{quadrant_labels[q]} (n={n})', zorder=3)
        ax.fill_between(ts.index, ts['mean'] - ts['sem'], ts['mean'] + ts['sem'],
                        color=quadrant_colors[q], alpha=0.15, zorder=2)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, loc='upper left')

axes[2].set_xlabel('Time (min)')
fig.suptitle(
    f'Timecourses by median-split quadrant — ALL eligible tracks (n={len(late_summary)})\n'
    f'Solid = peripheral, dashed = central  |  bright = ERK-high, pale = ERK-low',
    fontsize=10
)
plt.tight_layout()
fig.savefig(out_dir / f'full_population_group_timecourses_{version}.png', dpi=150)
plt.close()
print(f'Saved: full_population_group_timecourses_{version}.png')
