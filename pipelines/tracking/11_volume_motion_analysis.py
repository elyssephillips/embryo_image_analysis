"""
Layer 4 — Nuclear volume vs motion analysis.

Correlates nuclear volume (from segmentation) with motion characteristics:
  - speed            : do larger nuclei move faster or slower?
  - flow alignment   : do larger nuclei move more independently from the collective?
  - persistence (α)  : do larger nuclei move more directionally?
  - leading edge     : are larger/smaller nuclei at the embryo periphery?
  - volume stability : volume CV over time (proxy for cell-cycle activity)

Volume is already in the flow CSV (area_um3 per nucleus per timepoint).

Usage
-----
  python 11_volume_motion_analysis.py
  python 11_volume_motion_analysis.py configs/tracking/dataset001_implantation.yaml

Outputs  (written to output_dir)
-------
  volume_track_stats_{version}.csv   — per-track volume + motion summary
  volume_vs_speed.png
  volume_vs_alignment.png
  volume_vs_alpha.png
  volume_vs_leading_edge.png
  volume_correlation_matrix.png
"""

import sys
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# ── config ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('config', nargs='?',
                    default='configs/tracking/dataset001_implantation.yaml')
args = parser.parse_args()

CONFIG_PATH = REPO_ROOT / args.config
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

OUTPUT_DIR = Path(cfg['paths']['output_dir'])
VERSION    = cfg['tracking']['input_version']
DATASET_ID = cfg['project']['dataset_id']
FRAME_MIN  = cfg['tracking']['frame_interval_min']
ICM_ZYX    = (cfg.get('biology') or {}).get('icm_centroid_t30_zyx')  # None until set in config

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'Dataset: {DATASET_ID}')

# ── load data ─────────────────────────────────────────────────────────────────
flow  = pd.read_csv(OUTPUT_DIR / f'motion_flow_{VERSION}.csv')
stats = pd.read_csv(OUTPUT_DIR / f'motion_track_stats_{VERSION}.csv')
fits  = pd.read_csv(OUTPUT_DIR / f'msd_track_fits_{VERSION}.csv')

has_orig = 'z_um_orig' in flow.columns
z_col = 'z_um_orig' if has_orig else 'z_um'
y_col = 'y_um_orig' if has_orig else 'y_um'
x_col = 'x_um_orig' if has_orig else 'x_um'

print(f'{flow["track_id"].nunique()} tracks, {len(flow)} timepoint observations')

# ── per-track volume summary ───────────────────────────────────────────────────
vol = (
    flow.groupby('track_id')['area_um3']
    .agg(
        vol_mean   = 'mean',
        vol_median = 'median',
        vol_std    = 'std',
        vol_min    = 'min',
        vol_max    = 'max',
    )
    .reset_index()
)
# Coefficient of variation — how much does volume fluctuate over the track?
# High CV may indicate cell-cycle-related swelling/shrinkage
vol['vol_cv'] = vol['vol_std'] / vol['vol_mean']

# ── per-track flow alignment summary ──────────────────────────────────────────
align = (
    flow.dropna(subset=['flow_alignment'])
    .groupby('track_id')['flow_alignment']
    .agg(align_mean='mean', align_std='std')
    .reset_index()
)

# ── positions at t=30 ────────────────────────────────────────────────────────
# Z position and radial distance are taken at t=30 (±T_WINDOW) for consistency
# with ICM distance — all three classify "where was this nucleus at outgrowth".
# Tracks not observed in the window get NaN for all three.
T_REF    = 30
T_WINDOW = 2   # ±frames — tracks outside this window get NaN

within_window = flow[flow['t'].between(T_REF - T_WINDOW, T_REF + T_WINDOW)]
ref_pos = (
    within_window
    .assign(_dt=lambda d: (d['t'] - T_REF).abs())
    .sort_values(['track_id', '_dt', 't'])
    .groupby('track_id')
    .first()
    .reset_index()
    [['track_id', z_col, y_col, x_col]]
    .rename(columns={z_col: 'z_t30', y_col: 'y_t30', x_col: 'x_t30'})
)

# Radial distance from the centroid of all t=30 nuclei
cy = ref_pos['y_t30'].mean()
cx = ref_pos['x_t30'].mean()
ref_pos['radial_dist_um'] = np.sqrt(
    (ref_pos['y_t30'] - cy)**2 + (ref_pos['x_t30'] - cx)**2
)

n_total  = flow['track_id'].nunique()
n_scored = len(ref_pos)
print(f't=30 ±{T_WINDOW} window: {n_scored}/{n_total} tracks scored, {n_total - n_scored} → NaN')

# ── ICM distance at t=30 ─────────────────────────────────────────────────────
if ICM_ZYX is not None:
    iz, iy, ix = ICM_ZYX
    ref_pos['icm_dist_um'] = np.sqrt(
        (ref_pos['z_t30'] - iz)**2 +
        (ref_pos['y_t30'] - iy)**2 +
        (ref_pos['x_t30'] - ix)**2
    )
    print(f'ICM centroid: Z={iz} Y={iy} X={ix} µm')
else:
    ref_pos['icm_dist_um'] = np.nan
    print('ICM centroid not set — icm_dist_um will be NaN (set biology.icm_centroid_t30_zyx in config)')

# ── assemble per-track summary ────────────────────────────────────────────────
summary = (
    stats
    .merge(vol,      on='track_id', how='inner')
    .merge(align,    on='track_id', how='left')
    .merge(fits[['track_id', 'alpha', 'D_eff_um2_per_min']], on='track_id', how='left')
    .merge(ref_pos[['track_id', 'x_t30', 'y_t30', 'z_t30', 'radial_dist_um', 'icm_dist_um']], on='track_id', how='left')
)

summary.to_csv(OUTPUT_DIR / f'volume_track_stats_{VERSION}.csv', index=False)
print(f'Saved volume_track_stats_{VERSION}.csv  ({len(summary)} tracks)')

print('\n--- Volume summary (µm³) ---')
print(summary['vol_mean'].describe().round(1).to_string())

# ── helper: scatter with regression line and stats ────────────────────────────
def _scatter(ax, x, y, xlabel, ylabel, title, cmap_col=None, cmap='viridis'):
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    if cmap_col is not None:
        sc = ax.scatter(xm, ym, c=cmap_col[mask], cmap=cmap,
                        s=25, alpha=0.7, linewidths=0)
    else:
        sc = ax.scatter(xm, ym, s=25, alpha=0.6, linewidths=0, color='steelblue')
    # regression line
    m, b = np.polyfit(xm, ym, 1)
    xs = np.linspace(xm.min(), xm.max(), 100)
    ax.plot(xs, m * xs + b, 'k--', linewidth=1, alpha=0.6)
    rho, p = spearmanr(xm, ym)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title}\nρ={rho:.2f}  p={p:.3f}')
    return sc


# ── plot 1: volume vs speed ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
_scatter(axes[0],
         summary['vol_mean'].values, summary['mean_speed_um_per_min'].values,
         'Mean nuclear volume (µm³)', 'Mean speed (µm/min)',
         'Volume vs speed', cmap_col=summary['straightness'].values)
_scatter(axes[1],
         summary['vol_mean'].values, summary['straightness'].values,
         'Mean nuclear volume (µm³)', 'Straightness (net/total path)',
         'Volume vs persistence')
fig.suptitle(f'Nuclear volume vs motility — {DATASET_ID}')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'volume_vs_speed.png', dpi=150)
plt.close()
print('Saved: volume_vs_speed.png')

# ── plot 2: volume vs flow alignment ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
_scatter(axes[0],
         summary['vol_mean'].values, summary['align_mean'].values,
         'Mean nuclear volume (µm³)', 'Mean flow alignment (cos θ)',
         'Volume vs collective alignment')
_scatter(axes[1],
         summary['vol_cv'].values, summary['align_mean'].values,
         'Volume CV (variability over time)', 'Mean flow alignment (cos θ)',
         'Volume stability vs alignment')
fig.suptitle(f'Nuclear volume vs collective motion — {DATASET_ID}')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'volume_vs_alignment.png', dpi=150)
plt.close()
print('Saved: volume_vs_alignment.png')

# ── plot 3: volume vs α (MSD exponent) ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
_scatter(axes[0],
         summary['vol_mean'].values, summary['alpha'].values,
         'Mean nuclear volume (µm³)', 'α (anomalous diffusion exponent)',
         'Volume vs motion mode',
         cmap_col=summary['mean_speed_um_per_min'].values, cmap='plasma')
_scatter(axes[1],
         summary['vol_mean'].values, summary['D_eff_um2_per_min'].values,
         'Mean nuclear volume (µm³)', 'D_eff (µm²/min)',
         'Volume vs diffusivity')
for ax in axes:
    ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
fig.suptitle(f'Nuclear volume vs MSD parameters — {DATASET_ID}')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'volume_vs_alpha.png', dpi=150)
plt.close()
print('Saved: volume_vs_alpha.png')

# ── plot 4: volume vs spatial position (leading edge) ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: spatial map coloured by volume
norm_vol = mcolors.Normalize(
    vmin=summary['vol_mean'].quantile(0.05),
    vmax=summary['vol_mean'].quantile(0.95),
)
sc = axes[0].scatter(
    summary['x_t30'], summary['y_t30'],
    c=summary['vol_mean'], cmap='RdYlBu_r', norm=norm_vol,
    s=40, linewidths=0,
)
plt.colorbar(sc, ax=axes[0], label='Mean volume (µm³)', shrink=0.8)
axes[0].set_aspect('equal')
axes[0].invert_yaxis()
axes[0].set_xlabel('X (µm)')
axes[0].set_ylabel('Y (µm)')
axes[0].set_title('Spatial map — coloured by volume')

# Right: radial distance vs volume
_scatter(axes[1],
         summary['radial_dist_um'].values, summary['vol_mean'].values,
         'Radial distance from centroid (µm)', 'Mean nuclear volume (µm³)',
         'Leading edge position vs volume')

fig.suptitle(f'Nuclear volume — spatial distribution — {DATASET_ID}')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'volume_vs_leading_edge.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: volume_vs_leading_edge.png')

# ── plot 5: correlation matrix ────────────────────────────────────────────────
corr_cols = {
    'vol_mean':             'Mean volume',
    'vol_cv':               'Volume CV',
    'mean_speed_um_per_min':'Mean speed',
    'straightness':         'Straightness',
    'align_mean':           'Flow alignment',
    'alpha':                'α (MSD)',
    'D_eff_um2_per_min':    'D_eff',
    'radial_dist_um':       'Radial distance',
    'z_t30':                'Z position (t30)',
    'icm_dist_um':          'ICM distance (t30)',
    'n_frames':             'Track length',
}
corr_data = summary[list(corr_cols.keys())].dropna()
corr_mat  = corr_data.corr(method='spearman')
corr_mat.index   = list(corr_cols.values())
corr_mat.columns = list(corr_cols.values())

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr_mat.values, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label='Spearman ρ', shrink=0.8)
ticks = range(len(corr_cols))
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(list(corr_cols.values()), rotation=45, ha='right')
ax.set_yticklabels(list(corr_cols.values()))
# Annotate cells
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        val = corr_mat.values[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=7, color='white' if abs(val) > 0.5 else 'black')
ax.set_title(f'Spearman correlation matrix — {DATASET_ID}')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'volume_correlation_matrix.png', dpi=150)
plt.close()
print('Saved: volume_correlation_matrix.png')
