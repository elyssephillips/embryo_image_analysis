"""
ERK C/N ratio vs speed — two analyses:

  1. Speed vs C/N scatter, colored by ICM-distance group (near vs far),
     with separate trend lines per group
  2. Spearman correlation between speed and C/N computed at each timepoint
     (split by ICM group) — shows whether the relationship is stage-specific

Requires: run 12_erk_cn_ratio.py first.
Run with:  conda run -n napari_env python3 pipelines/tracking/13_erk_motion_analysis.py
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from scipy.stats import spearmanr

REPO_ROOT   = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

version  = cfg['tracking']['input_version']
out_dir  = Path(cfg['paths']['output_dir'])
interval = cfg['tracking']['frame_interval_min']

SPEED_WINDOW  = 3    # frames to rolling-average speed over
MIN_N_CORR    = 8   # minimum nuclei per timepoint to compute correlation
N_DIST_BINS   = 4   # number of ICM-distance quantile bins for timecourse

# ── Load and merge ────────────────────────────────────────────────────────────

erk_df  = pd.read_csv(out_dir / f'erk_cn_ratio_{version}.csv')
kine_df = pd.read_csv(out_dir / f'motion_kinematics_{version}.csv')

kine_df = kine_df.sort_values(['track_id', 't'])
kine_df['speed_rolling'] = (kine_df.groupby('track_id')['speed_um_per_min']
                             .transform(lambda s: s.rolling(SPEED_WINDOW,
                                                            min_periods=1).mean()))

df = erk_df.merge(
    kine_df[['track_id', 't', 'speed_rolling']],
    on=['track_id', 't'], how='inner'
).dropna(subset=['erk_cn_ratio', 'speed_rolling'])

# ICM distance — continuous, per track
vstats_path = out_dir / f'volume_track_stats_{version}.csv'
has_icm = vstats_path.exists()

if has_icm:
    vstats = pd.read_csv(vstats_path)[['track_id', 'icm_dist_um']].dropna()
    df     = df.merge(vstats, on='track_id', how='left')
    print(f'ICM distance range: {df["icm_dist_um"].min():.1f} – {df["icm_dist_um"].max():.1f} µm')
else:
    df['icm_dist_um'] = np.nan
    print('No volume_track_stats found — ICM distance unavailable')

# Quantile bins for timecourse grouping
df['dist_bin'] = pd.qcut(df['icm_dist_um'], q=N_DIST_BINS,
                          labels=[f'Q{i+1}' for i in range(N_DIST_BINS)])
print(f'Total (track, t) rows: {len(df)}')

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Speed vs C/N — scatter + binned trend, colored by ICM group
# ═══════════════════════════════════════════════════════════════════════════════

print('\n[1] Speed vs C/N colored by ICM distance...')

valid = df.dropna(subset=['erk_cn_ratio', 'speed_rolling', 'icm_dist_um'])
r_all, p_all = spearmanr(valid['speed_rolling'], valid['erk_cn_ratio'])
print(f'    Overall: ρ={r_all:.3f}, p={p_all:.2e}, n={len(valid)}')

fig, ax = plt.subplots(figsize=(8, 6))

sc = ax.scatter(valid['speed_rolling'], valid['erk_cn_ratio'],
                c=valid['icm_dist_um'], cmap='coolwarm_r',
                alpha=0.25, s=6, rasterized=True)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Distance from ICM at t=30 (µm)\n← near ICM       far from ICM →')

# Binned mean ± SEM for near and far halves (shown as lines for readability)
median_dist = valid['icm_dist_um'].median()
for label, mask, col in [('near ICM', valid['icm_dist_um'] <= median_dist, '#c0392b'),
                          ('far ICM',  valid['icm_dist_um'] >  median_dist, '#2980b9')]:
    sub = valid[mask]
    bins = np.percentile(sub['speed_rolling'], np.linspace(0, 100, 9))
    bins = np.unique(bins)
    if len(bins) < 3:
        continue
    sub = sub.copy()
    sub['speed_bin'] = pd.cut(sub['speed_rolling'], bins=bins)
    binned = (sub.groupby('speed_bin', observed=True)['erk_cn_ratio']
               .agg(mean='mean', sem=lambda x: x.sem()).reset_index())
    centers = [iv.mid for iv in binned['speed_bin']]
    r_g, p_g = spearmanr(sub['speed_rolling'], sub['erk_cn_ratio'])
    ax.errorbar(centers, binned['mean'], yerr=binned['sem'],
                fmt='o-', color=col, lw=2, zorder=5,
                label=f'{label}  ρ={r_g:.2f} p={p_g:.2e}')
    print(f'    {label}: ρ={r_g:.3f}, p={p_g:.2e}')

ax.set_xlabel(f'Speed — {SPEED_WINDOW}-frame rolling mean (µm/min)')
ax.set_ylabel('ERK C/N ratio')
ax.set_title(f'Speed vs ERK C/N  (overall ρ={r_all:.3f}, p={p_all:.2e})\n'
             'Lines = binned mean ± SEM for near/far halves')
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(out_dir / f'erk_vs_speed_icm_dist_{version}.png', dpi=150)
plt.close()
print(f'    saved erk_vs_speed_icm_dist_{version}.png')

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Correlation timecourse — ρ(speed, C/N) at each timepoint, by ICM group
# ═══════════════════════════════════════════════════════════════════════════════

print('\n[2] Correlation timecourse by ICM distance quartile...')

timepoints = sorted(df['t'].unique())
bins_ordered = [f'Q{i+1}' for i in range(N_DIST_BINS)]
cmap_dist = plt.get_cmap('coolwarm_r', N_DIST_BINS)
bin_colors = {b: cmap_dist(i) for i, b in enumerate(bins_ordered)}

records = []
for grp in bins_ordered:
    sub = df[df['dist_bin'] == grp]
    for t in timepoints:
        t_sub = sub[sub['t'] == t].dropna(subset=['speed_rolling', 'erk_cn_ratio'])
        if len(t_sub) < MIN_N_CORR:
            continue
        r, p = spearmanr(t_sub['speed_rolling'], t_sub['erk_cn_ratio'])
        records.append({'t': t, 'dist_bin': grp, 'rho': r, 'p': p, 'n': len(t_sub)})

# Also overall (all cells)
for t in timepoints:
    t_sub = df[df['t'] == t].dropna(subset=['speed_rolling', 'erk_cn_ratio'])
    if len(t_sub) < MIN_N_CORR:
        continue
    r, p = spearmanr(t_sub['speed_rolling'], t_sub['erk_cn_ratio'])
    records.append({'t': t, 'dist_bin': 'all', 'rho': r, 'p': p, 'n': len(t_sub)})

corr_df = pd.DataFrame(records)
corr_df.to_csv(out_dir / f'erk_speed_corr_timecourse_{version}.csv', index=False)

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1]})

ax = axes[0]
ax.axhline(0, color='k', lw=0.8, linestyle='--', alpha=0.5)

# Overall in grey behind the quartiles
all_sub = corr_df[corr_df['dist_bin'] == 'all'].sort_values('t')
ax.plot(all_sub['t'] * interval, all_sub['rho'],
        color='0.6', lw=1.5, linestyle='--', label='all cells', zorder=2)

# Per-quartile lines (Q1 = nearest ICM, Q4 = furthest)
bin_labels = {'Q1': 'Q1 nearest ICM', 'Q2': 'Q2', 'Q3': 'Q3', 'Q4': 'Q4 furthest'}
for grp in bins_ordered:
    sub = corr_df[corr_df['dist_bin'] == grp].sort_values('t')
    if sub.empty:
        continue
    col = bin_colors[grp]
    ax.plot(sub['t'] * interval, sub['rho'], color=col, lw=2,
            label=bin_labels.get(grp, grp), zorder=3)
    sig = sub[sub['p'] < 0.05]
    ax.scatter(sig['t'] * interval, sig['rho'], color=col, s=35, zorder=4)

ax.set_ylabel('Spearman ρ  (speed vs ERK C/N)')
ax.set_title('Speed–ERK C/N correlation over time, by ICM distance quartile\n'
             f'filled dots = p < 0.05   |   min n={MIN_N_CORR} per timepoint')
ax.legend(fontsize=9, ncol=2)
ax.set_ylim(-1, 1)

ax2 = axes[1]
ax2.plot(all_sub['t'] * interval, all_sub['n'], color='0.5', lw=1.5)
ax2.set_xlabel('Time (min)')
ax2.set_ylabel('n nuclei (all)')
ax2.set_title('Nuclei per timepoint')

plt.tight_layout()
fig.savefig(out_dir / f'erk_speed_corr_timecourse_{version}.png', dpi=150)
plt.close()
print(f'    saved erk_speed_corr_timecourse_{version}.png')

print('\nDone.')
