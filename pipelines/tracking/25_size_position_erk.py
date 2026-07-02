"""
Dynamic size, position, and ERK analysis.

At each timepoint, recomputes the embryo centroid (mean XY of all tracked nuclei)
and measures each nucleus's radial distance from it.  Volume is also tracked
per-timepoint.  Per-track summaries are then used to ask three pairwise questions:

  1. Are pre-ERK-low cells bigger?           (pre_erk_mean vs vol_pre / vol_post)
  2. Are bigger cells more peripheral?        (vol_post vs radial_post)
  3. Are bigger / peripheral cells ERK-high?  (vol_post vs late_erk, radial_post vs late_erk)

Definitions
-----------
  pre_erk_mean  — mean ERK C/N for t < T_SPLIT (30)
  late_erk_mean — mean ERK C/N for t >= T_LATE  (80)
  vol_pre       — mean volume (µm³) for t < T_SPLIT
  vol_post      — mean volume (µm³) for t >= T_SPLIT
  radial_pre    — mean dynamic radial dist (µm) for t < T_SPLIT
  radial_post   — mean dynamic radial dist (µm) for t >= T_SPLIT

Outputs
-------
  size_position_erk_{version}.csv           — per-track summary table
  volume_timecourse_by_preerk_{version}.png — mean volume over time by pre-ERK group
  radial_timecourse_by_preerk_{version}.png — mean radial distance over time by pre-ERK group
  size_position_scatter_{version}.png       — 2x2 scatter grid, Spearman ρ for each pair

Run with:
  conda run -n napari_env python3 pipelines/tracking/25_size_position_erk.py
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

version  = cfg['tracking']['input_version']
out_dir  = Path(cfg['paths']['output_dir'])
interval = cfg['tracking']['frame_interval_min']

T_SPLIT  = 30    # implantation onset (timepoints)
T_LATE   = 80    # start of "late" ERK window
N_GROUPS = 3     # tertiles for pre-ERK grouping
MIN_PRE  = 3     # minimum pre-implantation timepoints to include a track

# ── Load data ──────────────────────────────────────────────────────────────────

kine = pd.read_csv(out_dir / f'motion_kinematics_{version}.csv')
erk  = pd.read_csv(out_dir / f'erk_cn_ratio_{version}.csv')

kine = kine.sort_values(['track_id', 't'])
erk  = erk.sort_values(['track_id', 't'])

# ── Dynamic radial distance ────────────────────────────────────────────────────
# At each timepoint, embryo centroid = mean XY of all tracks present at that t.
# Radial distance is then 2D (XY), consistent with excluding Z from motion analysis.

centroid = (kine.groupby('t')[['x_um_reg', 'y_um_reg']]
                .mean()
                .rename(columns={'x_um_reg': 'cx', 'y_um_reg': 'cy'}))

kine = kine.join(centroid, on='t')
kine['radial_dist_dyn'] = np.sqrt(
    (kine['x_um_reg'] - kine['cx'])**2 +
    (kine['y_um_reg'] - kine['cy'])**2
)

# ── Per-track summaries ────────────────────────────────────────────────────────

def window_mean(df, t_col, val_col, t_lo, t_hi):
    """Mean of val_col for rows where t_lo <= t_col < t_hi (t_hi=None means no upper bound)."""
    mask = df[t_col] >= t_lo
    if t_hi is not None:
        mask &= df[t_col] < t_hi
    return df.loc[mask].groupby('track_id')[val_col].mean()

vol_pre    = window_mean(kine, 't', 'area_um3',        0,       T_SPLIT)
vol_post   = window_mean(kine, 't', 'area_um3',        T_SPLIT, None)
rad_pre    = window_mean(kine, 't', 'radial_dist_dyn', 0,       T_SPLIT)
rad_post   = window_mean(kine, 't', 'radial_dist_dyn', T_SPLIT, None)

pre_erk    = window_mean(erk,  't', 'erk_cn_ratio',    0,       T_SPLIT)
late_erk   = window_mean(erk,  't', 'erk_cn_ratio',    T_LATE,  None)

summary = (pd.DataFrame({
    'pre_erk_mean':  pre_erk,
    'late_erk_mean': late_erk,
    'vol_pre':       vol_pre,
    'vol_post':      vol_post,
    'radial_pre':    rad_pre,
    'radial_post':   rad_post,
}).reset_index().rename(columns={'index': 'track_id'}))

# Drop tracks without enough pre-implantation coverage
pre_counts = erk[erk['t'] < T_SPLIT].groupby('track_id')['erk_cn_ratio'].count()
valid = pre_counts[pre_counts >= MIN_PRE].index
summary = summary[summary['track_id'].isin(valid)].copy()

print(f'Tracks in summary: {len(summary)}')

# ── Pre-ERK group assignment ───────────────────────────────────────────────────

group_labels = ['low', 'mid', 'high']
summary['preerk_group'] = pd.qcut(
    summary['pre_erk_mean'], q=N_GROUPS, labels=group_labels
)
summary.to_csv(out_dir / f'size_position_erk_{version}.csv', index=False)
print(f'Saved: size_position_erk_{version}.csv')

# ── Timecourse data: volume and radial distance by pre-ERK group ──────────────

group_lookup  = summary.set_index('track_id')['preerk_group']
kine['preerk_group'] = kine['track_id'].map(group_lookup)
kine['time_min']     = kine['t'] * interval

group_colors = {'low': '#4393c3', 'mid': '#f7f7f7', 'high': '#d6604d'}
group_edge   = {'low': '#2166ac', 'mid': '#999999', 'high': '#b2182b'}

def timecourse_plot(df, val_col, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axvline(T_SPLIT * interval, color='0.4', linewidth=0.8, linestyle=':', zorder=1)
    ax.axvspan(0, T_SPLIT * interval, color='0.92', zorder=0)

    for lbl in group_labels:
        grp = df[df['preerk_group'] == lbl]
        ts  = grp.groupby('time_min')[val_col].agg(['mean', 'sem'])
        n   = grp['track_id'].nunique()
        ax.plot(ts.index, ts['mean'],
                color=group_edge[lbl], linewidth=2.2, label=f'{lbl}  (n={n})', zorder=3)
        ax.fill_between(ts.index,
                        ts['mean'] - ts['sem'],
                        ts['mean'] + ts['sem'],
                        color=group_colors[lbl], alpha=0.35, zorder=2)

    ax.set_xlabel('Time (min)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title='Pre-ERK group', fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved: {out_path.name}')

timecourse_plot(
    kine.dropna(subset=['preerk_group']),
    'area_um3',
    'Nuclear volume (µm³)',
    f'Nuclear volume over time by pre-ERK group  (shaded = pre-implantation)',
    out_dir / f'volume_timecourse_by_preerk_{version}.png',
)

timecourse_plot(
    kine.dropna(subset=['preerk_group']),
    'radial_dist_dyn',
    'Radial distance from embryo centroid (µm)',
    f'Radial distance over time by pre-ERK group  (dynamic centroid)',
    out_dir / f'radial_timecourse_by_preerk_{version}.png',
)

# ── Scatter grid: pairwise questions ──────────────────────────────────────────
#
#  Panel layout (2 rows x 2 cols):
#   [pre_erk vs vol_post]   [pre_erk vs radial_post]
#   [vol_post vs late_erk]  [radial_post vs late_erk]

pairs = [
    ('pre_erk_mean',  'vol_post',      'Pre-ERK C/N mean', 'Post-implant volume (µm³)',
     'Q1: Are pre-ERK-low cells bigger?'),
    ('pre_erk_mean',  'radial_post',   'Pre-ERK C/N mean', 'Post-implant radial dist (µm)',
     'Q2: Are pre-ERK-low cells more peripheral?'),
    ('vol_post',      'late_erk_mean', 'Post-implant volume (µm³)', f'Late ERK C/N (t≥{T_LATE})',
     'Q3: Are bigger cells ERK-high?'),
    ('radial_post',   'late_erk_mean', 'Post-implant radial dist (µm)', f'Late ERK C/N (t≥{T_LATE})',
     'Q4: Are peripheral cells ERK-high?'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes_flat = axes.flatten()

# colour points by pre-ERK group
group_pt_color = {'low': '#4393c3', 'mid': '#aaaaaa', 'high': '#d6604d'}
point_colors = summary['preerk_group'].map(group_pt_color).values

for ax, (xcol, ycol, xlabel, ylabel, title) in zip(axes_flat, pairs):
    x = summary[xcol].values
    y = summary[ycol].values
    mask = np.isfinite(x) & np.isfinite(y)

    ax.scatter(x[mask], y[mask],
               c=point_colors[mask], s=45, alpha=0.85,
               edgecolors='0.3', linewidths=0.4, zorder=3)

    # regression line
    m, b = np.polyfit(x[mask], y[mask], 1)
    xs = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xs, m * xs + b, 'k--', linewidth=1.2, alpha=0.6, zorder=4)

    rho, p = spearmanr(x[mask], y[mask])
    p_str = f'{p:.3f}' if p >= 0.001 else f'{p:.2e}'
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f'{title}\nρ = {rho:.2f},  p = {p_str},  n = {mask.sum()}', fontsize=9)

# shared legend
from matplotlib.patches import Patch
legend_handles = [Patch(color=group_pt_color[lbl], label=f'pre-ERK {lbl}') for lbl in group_labels]
fig.legend(handles=legend_handles, title='Pre-ERK group',
           loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    f'Size, position, and ERK — pairwise relationships  (linked_c62)\n'
    f'Colour = pre-implantation ERK group  |  dashed = linear fit',
    fontsize=10, y=1.01
)
plt.tight_layout()
fig.savefig(out_dir / f'size_position_scatter_{version}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: size_position_scatter_{version}.png')

# ── Per-timepoint radial vs ERK correlation timecourse ────────────────────────
# The per-track summary (Q4 above) uses radial means over broad windows which
# dilutes the signal.  Here we compute Spearman(radial_dist_dyn, erk_cn_ratio)
# separately at each timepoint to show when the relationship emerges.

merged_tp = (kine[['track_id', 't', 'time_min', 'radial_dist_dyn']]
             .merge(erk[['track_id', 't', 'erk_cn_ratio']], on=['track_id', 't']))

tp_corrs = []
for t, grp in merged_tp.groupby('t'):
    sub = grp.dropna(subset=['radial_dist_dyn', 'erk_cn_ratio'])
    if len(sub) < 5:
        continue
    rho, p = spearmanr(sub['radial_dist_dyn'], sub['erk_cn_ratio'])
    tp_corrs.append({'t': t, 'time_min': t * interval, 'rho': rho, 'p': p, 'n': len(sub)})

tp_corrs = pd.DataFrame(tp_corrs)

fig, ax = plt.subplots(figsize=(12, 4))
ax.axvline(T_SPLIT * interval, color='0.4', linewidth=0.8, linestyle=':', zorder=1)
ax.axvspan(0, T_SPLIT * interval, color='0.92', zorder=0)
ax.axhline(0, color='0.4', linewidth=0.8, linestyle='--', alpha=0.4)

sig  = tp_corrs['p'] < 0.05
nsig = ~sig
ax.scatter(tp_corrs.loc[sig,  'time_min'], tp_corrs.loc[sig,  'rho'],
           color='#2166ac', s=40, zorder=3, label='p < 0.05')
ax.scatter(tp_corrs.loc[nsig, 'time_min'], tp_corrs.loc[nsig, 'rho'],
           color='0.6',     s=40, zorder=3, label='p ≥ 0.05')
ax.plot(tp_corrs['time_min'], tp_corrs['rho'], color='0.5', linewidth=0.8, zorder=2)

ax.set_xlabel('Time (min)')
ax.set_ylabel("Spearman ρ\n(radial dist vs ERK C/N)")
ax.set_title('Per-timepoint correlation: radial distance vs ERK C/N\n'
             'Positive ρ = peripheral cells are ERK-high at that timepoint')
ax.set_ylim(-0.8, 0.8)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(out_dir / f'radial_erk_corr_timecourse_{version}.png', dpi=150)
plt.close()
print(f'Saved: radial_erk_corr_timecourse_{version}.png')

# Pooled scatter: t=80-100
late_merged = merged_tp[merged_tp['t'] >= T_LATE].dropna(subset=['radial_dist_dyn', 'erk_cn_ratio'])
rho_pool, p_pool = spearmanr(late_merged['radial_dist_dyn'], late_merged['erk_cn_ratio'])

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(late_merged['radial_dist_dyn'], late_merged['erk_cn_ratio'],
           s=12, alpha=0.4, color='#2166ac', edgecolors='none', zorder=2)
m, b = np.polyfit(late_merged['radial_dist_dyn'], late_merged['erk_cn_ratio'], 1)
xs = np.linspace(late_merged['radial_dist_dyn'].min(), late_merged['radial_dist_dyn'].max(), 100)
ax.plot(xs, m * xs + b, 'k--', linewidth=1.2, alpha=0.7, zorder=3)
p_str = f'{p_pool:.2e}' if p_pool < 0.001 else f'{p_pool:.3f}'
ax.set_xlabel('Radial distance from embryo centroid (µm)')
ax.set_ylabel('ERK C/N ratio')
ax.set_title(f'Radial distance vs ERK C/N  (pooled t={T_LATE}–100)\n'
             f'ρ = {rho_pool:.3f},  p = {p_str},  n = {len(late_merged)} observations')
plt.tight_layout()
fig.savefig(out_dir / f'radial_vs_erk_late_scatter_{version}.png', dpi=150)
plt.close()
print(f'Saved: radial_vs_erk_late_scatter_{version}.png')

# ── Print summary stats ────────────────────────────────────────────────────────

print(f'\n--- Pairwise Spearman correlations (n={len(summary)}) ---')
for xcol, ycol, xlabel, *_ in pairs:
    x = summary[xcol].dropna()
    y = summary[ycol].dropna()
    idx = x.index.intersection(y.index)
    rho, p = spearmanr(x[idx], y[idx])
    print(f'  {xcol} vs {ycol}: ρ={rho:.3f}  p={p:.4f}  n={len(idx)}')

print('\n--- Volume by pre-ERK group (post-implantation mean) ---')
for lbl in group_labels:
    v = summary.loc[summary['preerk_group'] == lbl, 'vol_post']
    print(f'  {lbl}: median={v.median():.0f} µm³  IQR=[{v.quantile(0.25):.0f}, {v.quantile(0.75):.0f}]  n={len(v)}')
