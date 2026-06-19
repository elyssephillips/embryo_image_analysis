"""
Per-track slope analysis — fit a linear trend to ERK C/N and motion parameters
over time for each track, then scatter slopes against ICM distance.

Slopes are normalized by time so short and long tracks are comparable.
Requires a minimum number of timepoints per track for a reliable fit.

Outputs
-------
  track_slopes_{version}.csv          — per-track slopes + ICM distance
  slope_vs_icm_{version}.png          — each slope scattered vs ICM distance
  slope_scatter_{version}.png         — ERK slope vs displacement slope, coloured by ICM distance

Run with:
  conda run -n napari_env python3 pipelines/tracking/20_slope_analysis.py
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress, spearmanr
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

version  = cfg['tracking']['input_version']
out_dir  = Path(cfg['paths']['output_dir'])
interval = cfg['tracking']['frame_interval_min']

T_START   = 30
MIN_PTS   = 8    # minimum timepoints per track to fit a slope

# ── Load data ─────────────────────────────────────────────────────────────────

erk_df  = pd.read_csv(out_dir / f'erk_cn_ratio_{version}.csv')
kine_df = pd.read_csv(out_dir / f'motion_kinematics_{version}.csv')
vstats  = pd.read_csv(out_dir / f'volume_track_stats_{version}.csv')[['track_id', 'icm_dist_um']]

rad_path = out_dir / f'radial_cumdisp_{version}.csv'
has_rad  = rad_path.exists()
if has_rad:
    rad_df = pd.read_csv(rad_path)[['track_id', 't', 'cumul_radial_disp']]

kine_df = kine_df.sort_values(['track_id', 't'])

df = (erk_df[['track_id', 't', 'erk_cn_ratio']]
      .merge(kine_df[['track_id', 't', 'net_disp_um']], on=['track_id', 't'], how='inner')
      .merge(vstats, on='track_id', how='left'))

if has_rad:
    df = df.merge(rad_df, on=['track_id', 't'], how='left')

df = df[df['t'] >= T_START].copy()
df['time_min'] = df['t'] * interval

# ── Fit slopes per track ──────────────────────────────────────────────────────

SLOPE_VARS = {
    'erk_cn_ratio':      'ERK C/N slope (per min)',
    'net_disp_um':       'Net displacement slope (µm/min)',
}
if has_rad:
    SLOPE_VARS['cumul_radial_disp'] = 'Radial displacement slope (µm/min)'

records = []
for tid, grp in df.groupby('track_id'):
    grp = grp.sort_values('t').dropna(subset=list(SLOPE_VARS.keys()))
    if len(grp) < MIN_PTS:
        continue
    row = {'track_id': tid}
    t   = grp['time_min'].values
    for col in SLOPE_VARS:
        slope, intercept, r, p, se = linregress(t, grp[col].values)
        row[f'{col}_slope'] = slope
        row[f'{col}_r2']    = r**2
        row[f'{col}_p']     = p
    records.append(row)

slopes = pd.DataFrame(records).merge(vstats, on='track_id', how='left')
slopes = slopes.dropna(subset=['icm_dist_um'])

out_csv = out_dir / f'track_slopes_{version}.csv'
slopes.to_csv(out_csv, index=False)
print(f'Tracks with slopes: {len(slopes)}  (min {MIN_PTS} timepoints required)')
print(f'Saved: {out_csv.name}')

# ── Colormap ──────────────────────────────────────────────────────────────────

dist_min = slopes['icm_dist_um'].min()
dist_max = slopes['icm_dist_um'].max()
norm     = mcolors.Normalize(vmin=dist_min, vmax=dist_max)
cmap     = plt.cm.coolwarm_r
colors   = cmap(norm(slopes['icm_dist_um'].values))

# ── Plot 1: each slope vs ICM distance ───────────────────────────────────────

slope_cols = [f'{v}_slope' for v in SLOPE_VARS]
n_vars = len(slope_cols)

fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 5))
if n_vars == 1:
    axes = [axes]

for ax, col, ylabel in zip(axes, slope_cols, SLOPE_VARS.values()):
    x = slopes['icm_dist_um'].values
    y = slopes[col].values
    mask = np.isfinite(x) & np.isfinite(y)

    ax.scatter(x[mask], y[mask], c=colors[mask], s=40, alpha=0.8,
               edgecolors='none', zorder=3)
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.4)

    # Regression line + Spearman ρ
    m, b = np.polyfit(x[mask], y[mask], 1)
    xs = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xs, m * xs + b, 'k--', linewidth=1.2, alpha=0.6)

    rho, p = spearmanr(x[mask], y[mask])
    ax.set_xlabel('ICM distance at t=30 (µm)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'ρ = {rho:.2f},  p = {p:.3f},  n = {mask.sum()}')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=axes, label='ICM distance (µm)\n← near        far →',
             shrink=0.6, pad=0.02)
fig.suptitle(f'Per-track slopes vs ICM distance  (t ≥ {T_START}, min {MIN_PTS} pts)',
             y=1.02)
plt.tight_layout()
fig.savefig(out_dir / f'slope_vs_icm_{version}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: slope_vs_icm_{version}.png')

# ── Plot 2: ERK slope vs radial displacement slope, coloured by ICM distance ──

fig, ax = plt.subplots(figsize=(7, 6))

x = slopes['erk_cn_ratio_slope'].values
y = slopes['cumul_radial_disp_slope'].values
mask = np.isfinite(x) & np.isfinite(y)

sc = ax.scatter(x[mask], y[mask], c=colors[mask], s=50, alpha=0.85,
                edgecolors='0.3', linewidths=0.4, zorder=3)

ax.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.3)
ax.axvline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.3)

rho, p = spearmanr(x[mask], y[mask])
ax.set_xlabel('ERK C/N slope (per min)')
ax.set_ylabel('Radial displacement slope (µm/min)\n← inward      outward →')
ax.set_title(f'ERK slope vs radial displacement slope\nρ = {rho:.2f},  p = {p:.3f},  n = {mask.sum()}')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='ICM distance (µm)\n← near        far →', shrink=0.8)

plt.tight_layout()
fig.savefig(out_dir / f'slope_scatter_{version}.png', dpi=150)
plt.close()
print(f'Saved: slope_scatter_{version}.png')

# ── Print summary ─────────────────────────────────────────────────────────────

print(f'\n--- Slope summaries (n={len(slopes)}) ---')
for col, label in SLOPE_VARS.items():
    s = slopes[f'{col}_slope']
    rho, p = spearmanr(slopes['icm_dist_um'], slopes[f'{col}_slope'])
    print(f'{label}:')
    print(f'  median={s.median():.4f}  range=[{s.min():.4f}, {s.max():.4f}]  '
          f'ρ vs ICM dist={rho:.2f}  p={p:.3f}')
