"""
Pairwise comparisons of the per-track summary measures - size, position
(pre-implantation and end), and ERK (pre and during implantation).

Variables (from track_summary_{version}.csv, see 06_build_track_table.py for
the window definitions):
  vol_end        nuclear volume, last analysis.end_frames frames of the movie
  radial_pre     XY distance from embryo centre, pre-implantation
  radial_end     ... end of movie
  icm_dist_pre   distance to ICM hull edge, pre-implantation
  icm_dist_end   ... end of movie
  erk_pre        ERK C/N, pre-implantation
  erk_during     ERK C/N, during implantation

Each pair uses every track that has both values (n shown per pair). Spearman
correlation throughout - no binning, no regression.

Requires 06_build_track_table.py.

Outputs (output_dir)
-------
  compare_corr_matrix_{version}.png   Spearman ρ heatmap, * p < 0.05
  compare_scatter_{version}.png       lower-triangle scatter grid
  compare_stats_{version}.csv         ρ, p, n for every pair

Run with:
  conda run -n napari_env python3 pipelines/tracking/08_compare_tracks.py
"""

import sys
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.tracking_config import load_tracking_config
cfg, CONFIG_PATH = load_tracking_config()

version = cfg['tracking']['input_version']
out_dir = Path(cfg['paths']['output_dir'])

VARS = {
    'vol_end':      'Volume\n(end)',
    'radial_pre':   'Radial\n(pre)',
    'radial_end':   'Radial\n(end)',
    'icm_dist_pre': 'ICM dist\n(pre)',
    'icm_dist_end': 'ICM dist\n(end)',
    'erk_pre':      'ERK C/N\n(pre)',
    'erk_during':   'ERK C/N\n(during)',
}

summary = pd.read_csv(out_dir / f'track_summary_{version}.csv')
cols = [c for c in VARS if summary[c].notna().sum() >= 3]
dropped = [c for c in VARS if c not in cols]
if dropped:
    print(f'Skipping (fewer than 3 tracks with values): {dropped}')

# ── Pairwise Spearman ─────────────────────────────────────────────────────────

rows = []
for a, b in combinations(cols, 2):
    sub = summary[[a, b]].dropna()
    rho, p = spearmanr(sub[a], sub[b]) if len(sub) >= 3 else (np.nan, np.nan)
    rows.append({'var_a': a, 'var_b': b, 'rho': rho, 'p': p, 'n': len(sub)})
stats = pd.DataFrame(rows)
stats.to_csv(out_dir / f'compare_stats_{version}.csv', index=False)
print(stats.round(3).to_string(index=False))
lookup = {(r.var_a, r.var_b): r for r in stats.itertuples()}
lookup.update({(r.var_b, r.var_a): r for r in stats.itertuples()})

# ── Correlation heatmap ───────────────────────────────────────────────────────

n = len(cols)
rho_mat = np.eye(n)
for i, j in combinations(range(n), 2):
    rho_mat[i, j] = rho_mat[j, i] = lookup[(cols[i], cols[j])].rho

fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.1 * n + 1))
im = ax.imshow(rho_mat, cmap='RdBu_r', vmin=-1, vmax=1)
labels = [VARS[c] for c in cols]
ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=8)
ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
for i, j in combinations(range(n), 2):
    r = lookup[(cols[i], cols[j])]
    if np.isnan(r.rho):
        continue
    txt = f'{r.rho:.2f}{"*" if r.p < 0.05 else ""}\nn={r.n}'
    for y, x in ((i, j), (j, i)):
        ax.text(x, y, txt, ha='center', va='center', fontsize=7,
                color='white' if abs(r.rho) > 0.5 else 'black')
fig.colorbar(im, ax=ax, label='Spearman ρ', shrink=0.8)
ax.set_title(f'Per-track Spearman correlations  (* p < 0.05)\n{version}', fontsize=10)
fig.tight_layout()
fig.savefig(out_dir / f'compare_corr_matrix_{version}.png', dpi=150)
plt.close(fig)
print(f'\nSaved: compare_corr_matrix_{version}.png')

# ── Scatter grid (lower triangle) ─────────────────────────────────────────────

fig, axes = plt.subplots(n - 1, n - 1, figsize=(2.3 * (n - 1), 2.3 * (n - 1)),
                         squeeze=False, constrained_layout=True)
for i in range(1, n):          # row variable (y)
    for j in range(n - 1):     # column variable (x)
        ax = axes[i - 1, j]
        if j >= i:
            ax.set_visible(False)
            continue
        x, y = cols[j], cols[i]
        sub = summary[[x, y]].dropna()
        r = lookup[(x, y)]
        ax.scatter(sub[x], sub[y], s=10, alpha=0.7, color='#4393c3', edgecolors='none')
        if not np.isnan(r.rho):
            ax.set_title(f'ρ={r.rho:.2f}{"*" if r.p < 0.05 else ""}  n={r.n}', fontsize=7.5,
                         color='#b2182b' if r.p < 0.05 else '0.3')
        ax.tick_params(labelsize=6.5)
        if j == 0:
            ax.set_ylabel(VARS[y].replace('\n', ' '), fontsize=8)
        if i == n - 1:
            ax.set_xlabel(VARS[x].replace('\n', ' '), fontsize=8)
fig.suptitle(f'Per-track comparisons  |  {version}', fontsize=10)
fig.savefig(out_dir / f'compare_scatter_{version}.png', dpi=150)
plt.close(fig)
print(f'Saved: compare_scatter_{version}.png')
