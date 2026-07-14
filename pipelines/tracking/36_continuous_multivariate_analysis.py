"""
Script 36: Continuous multivariate analysis — replaces the quadrant/quartile patchwork
(scripts 21, 27, 29, 32, 34, 35) with one coherent test on un-binned variables.

Builds a single per-track table of five continuous metrics:
  pre_erk_mean   — mean ERK C/N, t<30 (pre-implantation)
  late_erk_mean  — mean ERK C/N, t>=80 (late post-implantation)
  icm_dist_um    — distance from ICM centroid, fixed t=30 snapshot
  late_radial    — mean dynamic XY radial distance, t>=80
  vol_late       — mean nuclear volume (um3), t>=95 (final size)

Then:
  1. Full pairwise Spearman correlation matrix (no binning) -- heatmap with rho/p.
  2. Multiple regression (OLS, standardized predictors) for two outcomes:
       vol_late      ~ icm_dist_um + late_radial
       late_erk_mean ~ icm_dist_um + late_radial + vol_late + pre_erk_mean
     Standardized coefficients are directly comparable effect sizes, and each
     coefficient's p-value tests whether that predictor has an independent
     relationship with the outcome once the others are controlled for.

No external regression library required -- OLS is implemented directly via
numpy linear algebra with the standard formula for coefficient SEs / p-values,
since statsmodels is not installed in this environment.

Outputs
-------
  continuous_corr_matrix_{version}.png      — Spearman correlation heatmap
  continuous_per_track_table_{version}.csv  — the underlying per-track table
  (regression results printed to console)

Run with:
  conda run -n napari_env python3 pipelines/tracking/36_continuous_multivariate_analysis.py
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, t as t_dist

REPO_ROOT   = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

version  = cfg['tracking']['input_version']
out_dir  = Path(cfg['paths']['output_dir'])

T_START  = cfg['tracking']['t_start']
T_SPLIT  = 30
T_LATE   = 80
VOL_T0   = 95

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

# ── Build per-track table of continuous variables ─────────────────────────────

pre_erk  = erk[(erk['t'] >= T_START) & (erk['t'] < T_SPLIT)].groupby('track_id')['erk_cn_ratio'].mean().rename('pre_erk_mean')
late_erk = erk[erk['t'] >= T_LATE].groupby('track_id')['erk_cn_ratio'].mean().rename('late_erk_mean')
late_radial = kine[kine['t'] >= T_LATE].groupby('track_id')['radial_dist_dyn'].mean().rename('late_radial')
vol_late = kine[kine['t'] >= VOL_T0].groupby('track_id')['area_um3'].mean().rename('vol_late')

table = (pre_erk.to_frame()
         .join(late_erk, how='outer')
         .join(late_radial, how='outer')
         .join(vol_late, how='outer')
         .merge(vstats.set_index('track_id'), left_index=True, right_index=True, how='left'))
table.index.name = 'track_id'
table = table.reset_index()

table.to_csv(out_dir / f'continuous_per_track_table_{version}.csv', index=False)
print(f'Per-track table: {len(table)} tracks total')
for col in ['pre_erk_mean', 'late_erk_mean', 'icm_dist_um', 'late_radial', 'vol_late']:
    print(f'  {col}: n={table[col].notna().sum()}')

# ── Pairwise Spearman correlation matrix ───────────────────────────────────────

cols = ['pre_erk_mean', 'icm_dist_um', 'late_radial', 'vol_late', 'late_erk_mean']
col_labels = ['Pre-ERK\n(t<30)', 'ICM dist\n(t=30)', 'Late radial\n(t≥80)', 'Final volume\n(t≥95)', 'Late ERK\n(t≥80)']

n = len(cols)
rho_mat = np.full((n, n), np.nan)
p_mat   = np.full((n, n), np.nan)
n_mat   = np.zeros((n, n), dtype=int)

for i in range(n):
    for j in range(n):
        if i == j:
            rho_mat[i, j] = 1.0
            p_mat[i, j] = 0.0
            n_mat[i, j] = table[cols[i]].notna().sum()
            continue
        sub = table[[cols[i], cols[j]]].dropna()
        if len(sub) < 3:
            continue
        rho, p = spearmanr(sub.iloc[:, 0].values, sub.iloc[:, 1].values)
        rho_mat[i, j] = rho
        p_mat[i, j] = p
        n_mat[i, j] = len(sub)

fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(rho_mat, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(n)); ax.set_xticklabels(col_labels, fontsize=8, rotation=30, ha='right')
ax.set_yticks(range(n)); ax.set_yticklabels(col_labels, fontsize=8)

for i in range(n):
    for j in range(n):
        if i == j:
            continue
        rho, p, cnt = rho_mat[i, j], p_mat[i, j], n_mat[i, j]
        if np.isnan(rho):
            continue
        sig = '*' if p < 0.05 else ''
        ax.text(j, i, f'{rho:.2f}{sig}\n(n={cnt})', ha='center', va='center', fontsize=7,
                color='white' if abs(rho) > 0.5 else 'black')

fig.colorbar(im, ax=ax, label='Spearman ρ', shrink=0.8)
ax.set_title('Pairwise Spearman correlations\n(* = p<0.05, no binning)', fontsize=10)
plt.tight_layout()
out_path = out_dir / f'continuous_corr_matrix_{version}.png'
fig.savefig(out_path, dpi=150)
plt.close()
print(f'\nSaved: {out_path.name}')

# ── OLS with standard errors / p-values (no external dependency) ──────────────

def ols_fit(df, y_col, x_cols, standardize=True):
    sub = df[[y_col] + x_cols].dropna()
    n_obs = len(sub)
    y = sub[y_col].values.astype(float)
    X = sub[x_cols].values.astype(float)

    if standardize:
        y = (y - y.mean()) / y.std()
        X = (X - X.mean(axis=0)) / X.std(axis=0)

    X_design = np.column_stack([np.ones(n_obs), X])
    p = X_design.shape[1]

    beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    y_pred = X_design @ beta
    resid = y - y_pred
    rss = np.sum(resid**2)
    tss = np.sum((y - y.mean())**2)
    r2 = 1 - rss / tss
    df_resid = n_obs - p
    sigma2 = rss / df_resid
    cov_beta = sigma2 * np.linalg.inv(X_design.T @ X_design)
    se = np.sqrt(np.diag(cov_beta))
    t_stats = beta / se
    p_vals = 2 * t_dist.sf(np.abs(t_stats), df_resid)

    names = ['intercept'] + x_cols
    results = pd.DataFrame({'coef': beta, 'se': se, 't': t_stats, 'p': p_vals}, index=names)
    return results, r2, n_obs

print('\n' + '='*70)
print('Model A: final volume ~ ICM distance + late radial distance')
print('='*70)
res_a, r2_a, n_a = ols_fit(table, 'vol_late', ['icm_dist_um', 'late_radial'])
print(f'n={n_a}, R²={r2_a:.3f}')
print(res_a.round(4))

print('\n' + '='*70)
print('Model B: late ERK ~ ICM distance + late radial + final volume + pre-ERK')
print('='*70)
res_b, r2_b, n_b = ols_fit(table, 'late_erk_mean', ['icm_dist_um', 'late_radial', 'vol_late', 'pre_erk_mean'])
print(f'n={n_b}, R²={r2_b:.3f}')
print(res_b.round(4))

print('\n' + '='*70)
print('Model C (reduced): late ERK ~ late radial + final volume   (drop weakest predictors)')
print('='*70)
res_c, r2_c, n_c = ols_fit(table, 'late_erk_mean', ['late_radial', 'vol_late'])
print(f'n={n_c}, R²={r2_c:.3f}')
print(res_c.round(4))
