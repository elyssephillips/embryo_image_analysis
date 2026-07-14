"""
Script 35: Final nuclear volume by late-timepoint quadrant — both classification schemes.

Same quadrant classification as script 34 (extreme-tertile vs median-split, based on
late radial distance and ERK C/N), now compared against final nuclear volume
(mean area_um3, t>=95) instead of ICM distance.

Outputs
-------
  volume_by_quadrant_comparison_{version}.png — boxplots, extreme-tertile vs median-split

Run with:
  conda run -n napari_env python3 pipelines/tracking/35_volume_by_quadrant.py
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import kruskal, mannwhitneyu
from itertools import combinations

REPO_ROOT   = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

version  = cfg['tracking']['input_version']
out_dir  = Path(cfg['paths']['output_dir'])

T_LATE       = 80
MIN_LATE_PTS = 5
TERTILE_LO   = 0.33
TERTILE_HI   = 0.67
VOL_LATE_T0  = 95   # mean volume window for "final volume"

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

merged = (kine[['track_id', 't', 'radial_dist_dyn']]
          .merge(erk[['track_id', 't', 'erk_cn_ratio']], on=['track_id', 't']))

# ── Eligible tracks + classification (same as script 34) ──────────────────────

late = merged[merged['t'] >= T_LATE].copy()
late_counts = late.groupby('track_id')['erk_cn_ratio'].count()
valid_late  = late_counts[late_counts >= MIN_LATE_PTS].index

late_summary = (late[late['track_id'].isin(valid_late)]
                .groupby('track_id')[['radial_dist_dyn', 'erk_cn_ratio']]
                .mean()
                .rename(columns={'radial_dist_dyn': 'late_radial', 'erk_cn_ratio': 'late_erk'}))
late_summary = late_summary.merge(vstats, on='track_id', how='left')

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

rad_med = late_summary['late_radial'].median()
erk_med = late_summary['late_erk'].median()

def classify_median(row):
    rad = 'periph' if row['late_radial'] >= rad_med else 'central'
    erkc = 'high'  if row['late_erk']    >= erk_med else 'low'
    return f'{rad}_{erkc}'

late_summary['quadrant_median'] = late_summary.apply(classify_median, axis=1)

# ── Final volume (mean t>=VOL_LATE_T0) ─────────────────────────────────────────

vol_late = kine[kine['t'] >= VOL_LATE_T0].groupby('track_id')['area_um3'].mean().rename('vol_late')
late_summary = late_summary.merge(vol_late, on='track_id', how='left')

quadrant_colors = {
    'periph_high':  '#d6604d',
    'periph_low':   '#f4a582',
    'central_high': '#4393c3',
    'central_low':  '#92c5de',
}
quadrant_labels = {
    'periph_high':  'Peripheral\nERK-high',
    'periph_low':   'Peripheral\nERK-low',
    'central_high': 'Central\nERK-high',
    'central_low':  'Central\nERK-low',
}
quad_order = ['periph_high', 'central_high', 'periph_low', 'central_low']

# ── Plot: boxplots, both schemes side by side ──────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

stats_summary = {}
for ax, qcol, title in [
    (axes[0], 'quadrant_extreme', 'Extreme-tertile groups'),
    (axes[1], 'quadrant_median',  'Median-split groups (all tracks)'),
]:
    box_data, box_labels, box_colors = [], [], []
    groups = {}
    for q in quad_order:
        sub = late_summary[late_summary[qcol] == q]['vol_late'].dropna()
        if len(sub) == 0:
            continue
        box_data.append(sub.values)
        box_labels.append(f'{quadrant_labels[q]}\n(n={len(sub)})')
        box_colors.append(quadrant_colors[q])
        if len(sub) >= 2:
            groups[q] = sub.values

    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for i, sub in enumerate(box_data):
        jitter = np.random.normal(0, 0.05, size=len(sub))
        ax.scatter(np.full(len(sub), i + 1) + jitter, sub, color='0.2', s=15, zorder=3, alpha=0.7)

    ax.set_ylabel('Final nuclear volume (µm³, mean t≥95)')
    ax.tick_params(axis='x', labelsize=8)

    if len(groups) >= 2:
        stat, p = kruskal(*groups.values())
        p_str = f'{p:.2e}' if p < 0.001 else f'{p:.3f}'
        ax.set_title(f'{title}\nKruskal-Wallis p={p_str}', fontsize=9)
    else:
        ax.set_title(title, fontsize=9)

    stats_summary[qcol] = groups

plt.tight_layout()
out_path = out_dir / f'volume_by_quadrant_comparison_{version}.png'
fig.savefig(out_path, dpi=150)
plt.close()
print(f'Saved: {out_path.name}')

# ── Print pairwise stats ───────────────────────────────────────────────────────

for scheme, qcol in [('Extreme-tertile', 'quadrant_extreme'), ('Median-split', 'quadrant_median')]:
    groups = stats_summary[qcol]
    print(f'\n{scheme} pairwise (Mann-Whitney):')
    for a, b in combinations(groups.keys(), 2):
        u, p = mannwhitneyu(groups[a], groups[b])
        print(f'  {a} (n={len(groups[a])}) vs {b} (n={len(groups[b])}): p={p:.4f}')
