"""
Script 27: Individual trajectories of late peripheral ERK-high cells.

Classifies cells by their late-timepoint (t>=T_LATE) radial distance and ERK C/N
into four quadrants (peripheral/central × ERK-high/low), then plots full timecourses
of ERK, Z position, and XY radial distance for each cell individually.

The key question: for cells that end up peripheral AND ERK-high —
  did Z drop early (surface contact) → then radial increase (outward migration)?
  or were they already peripheral with Z dropping later (new contacts)?

Comparison groups (central/peripheral × ERK-low/high) are shown as group means
for context alongside the individual peripheral ERK-high traces.

Outputs
-------
  peripheral_erk_quadrant_scatter_{version}.png  — (radial, ERK) scatter at late timepoints
  peripheral_erk_group_timecourses_{version}.png — group mean ± SEM: ERK, Z, radial
  peripheral_erkhigh_individual_{version}.png    — per-cell panels for peripheral ERK-high

Run with:
  conda run -n napari_env python3 pipelines/tracking/27_peripheral_erk_trajectories.py
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

T_SPLIT      = 30    # implantation onset
T_LATE       = 80    # start of late classification window
MIN_LATE_PTS = 5     # min timepoints in late window to classify a track
TERTILE_HI   = 0.67  # top third = "high"
TERTILE_LO   = 0.33  # bottom third = "low"

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
          .merge(erk[['track_id', 't', 'erk_cn_ratio']], on=['track_id', 't'])
          .merge(vstats, on='track_id', how='left'))

# ── Classify tracks by late-window (radial, ERK) ──────────────────────────────

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
    erk = 'high'   if row['late_erk']    >= erk_hi else ('low'     if row['late_erk']    <= erk_lo else None)
    if rad is None or erk is None:
        return None
    return f'{rad}_{erk}'

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

for q, n in late_summary['quadrant'].value_counts().items():
    print(f'  {quadrant_labels[q]}: n={n}')

# ── Figure 1: quadrant scatter ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 5))
for q, grp in late_summary.groupby('quadrant'):
    ax.scatter(grp['late_radial'], grp['late_erk'],
               color=quadrant_colors[q], label=f'{quadrant_labels[q]} (n={len(grp)})',
               s=55, edgecolors='0.3', linewidths=0.4, zorder=3)
ax.axvline(rad_lo, color='0.5', linewidth=0.8, linestyle='--', alpha=0.5)
ax.axvline(rad_hi, color='0.5', linewidth=0.8, linestyle='--', alpha=0.5)
ax.axhline(erk_lo, color='0.5', linewidth=0.8, linestyle='--', alpha=0.5)
ax.axhline(erk_hi, color='0.5', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xlabel(f'Mean radial distance  (t≥{T_LATE}, µm)')
ax.set_ylabel(f'Mean ERK C/N  (t≥{T_LATE})')
ax.set_title(f'Late-timepoint quadrant classification\n(top/bottom tertiles of radial and ERK)')
ax.legend(fontsize=8, loc='upper left')
plt.tight_layout()
fig.savefig(out_dir / f'peripheral_erk_quadrant_scatter_{version}.png', dpi=150)
plt.close()
print(f'Saved: peripheral_erk_quadrant_scatter_{version}.png')

# ── Figure 2: group mean timecourses (ERK, Z, radial) ─────────────────────────

quad_order = ['periph_high', 'central_high', 'periph_low', 'central_low']
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
fig.suptitle('Timecourses by late-timepoint quadrant\nSolid = peripheral, dashed = central  |  '
             'bright = ERK-high, pale = ERK-low', fontsize=10)
plt.tight_layout()
fig.savefig(out_dir / f'peripheral_erk_group_timecourses_{version}.png', dpi=150)
plt.close()
print(f'Saved: peripheral_erk_group_timecourses_{version}.png')

# ── Figure 3: individual cell panels for peripheral ERK-high ──────────────────
# Each cell: 3 rows (ERK, Z, radial) × 1 column, stacked vertically.
# Cells sorted by ICM distance (close → far).

focus_ids = late_summary[late_summary['quadrant'] == 'periph_high'].index.tolist()
focus_data = merged[merged['track_id'].isin(focus_ids)].copy()
icm_lookup = vstats.set_index('track_id')['icm_dist_um']
focus_ids_sorted = sorted(focus_ids, key=lambda t: icm_lookup.get(t, np.inf))

n_cells = len(focus_ids_sorted)
n_cols  = min(6, n_cells)
n_rows  = int(np.ceil(n_cells / n_cols))

# shared axis limits
xlim = (0, merged['time_min'].max())
ylims = {
    'erk_cn_ratio':    (merged['erk_cn_ratio'].quantile(0.01),    merged['erk_cn_ratio'].quantile(0.99)),
    'z_um_reg':        (merged['z_um_reg'].quantile(0.01),         merged['z_um_reg'].quantile(0.99)),
    'radial_dist_dyn': (merged['radial_dist_dyn'].quantile(0.01),  merged['radial_dist_dyn'].quantile(0.99)),
}

# ICM distance colormap
icm_vals = icm_lookup[focus_ids_sorted].dropna()
icm_norm = mcolors.Normalize(vmin=icm_vals.min(), vmax=icm_vals.max())
icm_cmap = plt.cm.coolwarm_r

fig, axes = plt.subplots(
    n_cells, 3,
    figsize=(14, 2.8 * n_cells),
    squeeze=False,
    constrained_layout=True,
)

for row_i, tid in enumerate(focus_ids_sorted):
    cell = focus_data[focus_data['track_id'] == tid].sort_values('t')
    icm  = icm_lookup.get(tid, np.nan)
    col  = icm_cmap(icm_norm(icm)) if not np.isnan(icm) else '0.5'

    for col_i, (vcol, ylabel) in enumerate(vars3):
        ax = axes[row_i, col_i]
        ax.axvline(T_SPLIT * interval, color='0.4', linewidth=0.8, linestyle=':', zorder=1)
        ax.axvspan(0, T_SPLIT * interval, color='0.95', zorder=0)
        ax.plot(cell['time_min'], cell[vcol], color=col, linewidth=1.4, zorder=2)
        ax.set_xlim(xlim)
        ax.set_ylim(ylims[vcol])
        ax.tick_params(labelsize=7)

        if row_i == 0:
            ax.set_title(ylabel, fontsize=8)
        if col_i == 0:
            icm_str = f'{icm:.0f} µm' if not np.isnan(icm) else 'n/a'
            ax.set_ylabel(f'track {tid}\n{icm_str} ICM', fontsize=7)
        if row_i == n_cells - 1:
            ax.set_xlabel('Time (min)', fontsize=7)

sm = plt.cm.ScalarMappable(cmap=icm_cmap, norm=icm_norm)
sm.set_array([])
fig.colorbar(sm, ax=axes[:, -1].tolist(),
             label='ICM distance (µm)\n← near    far →', shrink=0.5)

fig.suptitle(
    f'Peripheral ERK-high cells — individual timecourses  (n={n_cells})\n'
    f'Columns: ERK C/N  |  Z position  |  XY radial distance  |  '
    f'colour = ICM distance  |  shaded = pre-implantation',
    fontsize=9,
)
fig.savefig(out_dir / f'peripheral_erkhigh_individual_{version}.png', dpi=150)
plt.close()
print(f'Saved: peripheral_erkhigh_individual_{version}.png')
