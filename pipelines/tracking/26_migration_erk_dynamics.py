"""
Script 26: Migration dynamics and ERK.

Asks whether the late-timepoint peripheral = ERK-high relationship arises from:
  (a) cells migrating outward and staying/becoming ERK-high, or
  (b) cells already at the periphery turning ERK on (e.g. substrate contact)

Three analyses:
  1. Z vs ERK timecourse   — does proximity to the dish surface (high Z) predict ERK?
  2. Rank stability        — do cells keep their radial rank from early to late post-implantation?
                             Stable = cells staying put; shuffled = migration contributing.
  3. Cross-lag correlogram — does radial distance lead ERK (contact → ERK on) or lag it
                             (ERK-high cells migrate out)?

Geometry note: high Z ≈ dish surface (Z=60 is bottom). The dish is not perfectly flat
so Z alone is an imperfect proxy, but it's the best available.

Outputs
-------
  z_erk_corr_timecourse_{version}.png      — per-timepoint Spearman(Z, ERK C/N)
  radial_rank_stability_{version}.png      — early vs late radial rank, coloured by ΔERK
  erk_rank_stability_{version}.png         — early vs late ERK rank, coloured by Δradial
  radial_erk_crosscorr_{version}.png       — mean within-track cross-correlogram

Run with:
  conda run -n napari_env python3 pipelines/tracking/26_migration_erk_dynamics.py
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr, pearsonr
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

version  = cfg['tracking']['input_version']
out_dir  = Path(cfg['paths']['output_dir'])
interval = cfg['tracking']['frame_interval_min']

T_SPLIT   = 30    # implantation onset
T_EARLY_END = 55  # end of "early post-implantation" window
T_LATE    = 80    # start of "late" window
MAX_LAG   = 20    # timepoints (= 300 min) for cross-lag
MIN_TRACK_PTS = 15  # minimum overlapping timepoints for cross-lag

# ── Load data + compute dynamic radial distance ────────────────────────────────

kine = pd.read_csv(out_dir / f'motion_kinematics_{version}.csv')
erk  = pd.read_csv(out_dir / f'erk_cn_ratio_{version}.csv')

kine = kine.sort_values(['track_id', 't'])
erk  = erk.sort_values(['track_id', 't'])

centroid = (kine.groupby('t')[['x_um_reg', 'y_um_reg']]
                .mean()
                .rename(columns={'x_um_reg': 'cx', 'y_um_reg': 'cy'}))
kine = kine.join(centroid, on='t')
kine['radial_dist_dyn'] = np.sqrt(
    (kine['x_um_reg'] - kine['cx'])**2 +
    (kine['y_um_reg'] - kine['cy'])**2
)

merged = (kine[['track_id', 't', 'z_um_reg', 'radial_dist_dyn']]
          .merge(erk[['track_id', 't', 'erk_cn_ratio']], on=['track_id', 't']))
merged['time_min'] = merged['t'] * interval

# ── 1. Z vs ERK timecourse ─────────────────────────────────────────────────────
# High Z = dish surface.  Positive ρ = bottom cells are ERK-high.

def corr_timecourse(df, xcol, ycol, t_col='t'):
    rows = []
    for t, grp in df.groupby(t_col):
        sub = grp[[xcol, ycol]].dropna()
        if len(sub) < 5:
            continue
        rho, p = spearmanr(sub[xcol], sub[ycol])
        rows.append({'t': t, 'time_min': t * interval, 'rho': rho, 'p': p, 'n': len(sub)})
    return pd.DataFrame(rows)

z_corr  = corr_timecourse(merged, 'z_um_reg',       'erk_cn_ratio')
rad_corr = corr_timecourse(merged, 'radial_dist_dyn', 'erk_cn_ratio')

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

for ax, df, ylabel, title in [
    (axes[0], z_corr,
     'ρ  (Z vs ERK C/N)',
     'Z position vs ERK C/N\nPositive ρ = cells closer to dish surface are ERK-high'),
    (axes[1], rad_corr,
     'ρ  (radial dist vs ERK C/N)',
     'Radial distance (XY) vs ERK C/N\nPositive ρ = peripheral cells are ERK-high'),
]:
    ax.axvline(T_SPLIT * interval, color='0.4', linewidth=0.8, linestyle=':', zorder=1)
    ax.axvspan(0, T_SPLIT * interval, color='0.92', zorder=0)
    ax.axhline(0, color='0.4', linewidth=0.8, linestyle='--', alpha=0.4)
    sig  = df['p'] < 0.05
    ax.scatter(df.loc[ sig, 'time_min'], df.loc[ sig, 'rho'],
               color='#2166ac', s=40, zorder=3, label='p < 0.05')
    ax.scatter(df.loc[~sig, 'time_min'], df.loc[~sig, 'rho'],
               color='0.6',     s=40, zorder=3, label='p ≥ 0.05')
    ax.plot(df['time_min'], df['rho'], color='0.5', linewidth=0.8, zorder=2)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(-0.8, 0.8)
    ax.legend(fontsize=8, loc='upper left')

axes[1].set_xlabel('Time (min)')
fig.suptitle('Spatial correlates of ERK C/N over time', fontsize=10)
plt.tight_layout()
fig.savefig(out_dir / f'z_erk_corr_timecourse_{version}.png', dpi=150)
plt.close()
print(f'Saved: z_erk_corr_timecourse_{version}.png')

# ── 2. Rank stability ──────────────────────────────────────────────────────────
# For each track, compute mean radial distance and mean ERK in the early and late
# post-implantation windows, then assign ranks across tracks within each window.
# Stable ranks → cells keeping their spatial positions (not migrating).
# Colour by the *other* variable's change to see if the two are coupled.

def window_mean(df, t_lo, t_hi, val_col):
    sub = df[(df['t'] >= t_lo) & (df['t'] < t_hi)]
    return sub.groupby('track_id')[val_col].mean()

rad_early = window_mean(merged, T_SPLIT,    T_EARLY_END, 'radial_dist_dyn')
rad_late  = window_mean(merged, T_LATE,     101,         'radial_dist_dyn')
erk_early = window_mean(merged, T_SPLIT,    T_EARLY_END, 'erk_cn_ratio')
erk_late  = window_mean(merged, T_LATE,     101,         'erk_cn_ratio')

rank_df = pd.DataFrame({
    'rad_early': rad_early,
    'rad_late':  rad_late,
    'erk_early': erk_early,
    'erk_late':  erk_late,
}).dropna()

rank_df['rad_early_rank'] = rank_df['rad_early'].rank(pct=True)
rank_df['rad_late_rank']  = rank_df['rad_late'].rank(pct=True)
rank_df['erk_early_rank'] = rank_df['erk_early'].rank(pct=True)
rank_df['erk_late_rank']  = rank_df['erk_late'].rank(pct=True)

rank_df['delta_erk']    = rank_df['erk_late']  - rank_df['erk_early']
rank_df['delta_radial'] = rank_df['rad_late']  - rank_df['rad_early']

# Normalise deltas for colouring
def diverging_norm(series):
    vmax = np.abs(series).quantile(0.95)
    return mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, xrank, yrank, xlabel, ylabel, delta_col, cbar_label, title in [
    (axes[0],
     'rad_early_rank', 'rad_late_rank',
     f'Radial rank  (t={T_SPLIT}–{T_EARLY_END})',
     f'Radial rank  (t={T_LATE}–100)',
     'delta_erk',
     'ΔERK rank\n(late − early)',
     'Radial position rank stability\nColour = change in ERK rank'),
    (axes[1],
     'erk_early_rank', 'erk_late_rank',
     f'ERK rank  (t={T_SPLIT}–{T_EARLY_END})',
     f'ERK rank  (t={T_LATE}–100)',
     'delta_radial',
     'Δradial rank\n(late − early)',
     'ERK rank stability\nColour = change in radial rank'),
]:
    norm  = diverging_norm(rank_df[delta_col])
    sc = ax.scatter(rank_df[xrank], rank_df[yrank],
                    c=rank_df[delta_col], cmap='RdBu_r', norm=norm,
                    s=55, edgecolors='0.3', linewidths=0.4, zorder=3)
    # diagonal = perfect rank stability
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.9, alpha=0.5, zorder=2)
    rho, p = spearmanr(rank_df[xrank], rank_df[yrank])
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f'{title}\nρ = {rho:.2f},  p = {p:.3f},  n = {len(rank_df)}', fontsize=9)
    fig.colorbar(sc, ax=ax, label=cbar_label, shrink=0.8)

plt.tight_layout()
fig.savefig(out_dir / f'rank_stability_{version}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: rank_stability_{version}.png')

# ── 3. Cross-lag correlogram: within-track, generic for any spatial variable ──
# Within each track, z-score both signals, then compute Pearson r between
# spatial_var[t] and erk_cn_ratio[t + lag] for lags -MAX_LAG to +MAX_LAG.
# Average r across tracks.
# Convention: positive lag = spatial var leads ERK (e.g. moving/contact → ERK turns on)
#             negative lag = ERK leads spatial var  (ERK-high → cell moves/contacts)

lags = np.arange(-MAX_LAG, MAX_LAG + 1)

def crosscorr(df, spatial_col, t_lo=None, t_hi=None, min_pts=MIN_TRACK_PTS):
    sub = df
    if t_lo is not None:
        sub = sub[sub['t'] >= t_lo]
    if t_hi is not None:
        sub = sub[sub['t'] < t_hi]

    track_corrs = []
    for tid, grp in sub.groupby('track_id'):
        both = grp.sort_values('t')[[spatial_col, 'erk_cn_ratio']].dropna()
        if len(both) < min_pts:
            continue

        spat = both[spatial_col].values
        erk  = both['erk_cn_ratio'].values
        n    = len(spat)

        spat_z = (spat - spat.mean()) / (spat.std() + 1e-9)
        erk_z  = (erk  - erk.mean())  / (erk.std()  + 1e-9)

        row = {'track_id': tid}
        for lag in lags:
            if lag >= 0:
                x = spat_z[:n - lag] if lag > 0 else spat_z
                y = erk_z[lag:]      if lag > 0 else erk_z
            else:
                abs_lag = -lag
                x = spat_z[abs_lag:]
                y = erk_z[:n - abs_lag]

            if len(x) < 5 or len(x) != len(y):
                row[lag] = np.nan
                continue
            r, _ = pearsonr(x, y)
            row[lag] = r

        track_corrs.append(row)

    corr_df = pd.DataFrame(track_corrs).set_index('track_id')[lags] if track_corrs else pd.DataFrame(columns=lags)
    return corr_df

def plot_crosscorr(corr_df, xlabel_pair, title, out_name):
    mean_r = corr_df.mean(skipna=True)
    sem_r  = corr_df.sem(skipna=True)
    n_full = corr_df.notna().all(axis=1).sum()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color='0.4', linewidth=0.8, linestyle='--', alpha=0.4)
    ax.axvline(0, color='0.4', linewidth=0.8, linestyle=':',  alpha=0.6)

    lag_min = lags * interval
    ax.fill_between(lag_min, mean_r - sem_r, mean_r + sem_r, alpha=0.3, color='#2166ac')
    ax.plot(lag_min, mean_r, color='#2166ac', linewidth=2, zorder=3)

    peak_idx = mean_r.abs().idxmax()
    ax.scatter([peak_idx * interval], [mean_r[peak_idx]],
               color='#b2182b', s=80, zorder=4,
               label=f'peak at lag={peak_idx} ({peak_idx*interval:.0f} min)')

    ax.set_xlabel(f'Lag (min)\n← {xlabel_pair[0]} leads          {xlabel_pair[1]} leads →')
    ax.set_ylabel('Mean within-track Pearson r')
    ax.set_title(f'{title}\nn = {n_full} tracks with ≥{MIN_TRACK_PTS} timepoints  |  shaded = ± SEM')
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / out_name, dpi=150)
    plt.close()
    print(f'Saved: {out_name}')
    return peak_idx, mean_r[peak_idx]

# Radial vs ERK, full range (as before)
rad_corr_df = crosscorr(merged, 'radial_dist_dyn')
rad_peak_idx, rad_peak_r = plot_crosscorr(
    rad_corr_df, ('ERK', 'radial'),
    'Cross-lag correlogram: radial distance vs ERK C/N',
    f'radial_erk_crosscorr_{version}.png',
)

# Z vs ERK, full range — likely diluted once the embryo flattens (Z stops varying)
z_corr_df_full = crosscorr(merged, 'z_um_reg')
z_peak_idx_full, z_peak_r_full = plot_crosscorr(
    z_corr_df_full, ('ERK', 'Z'),
    'Cross-lag correlogram: Z position vs ERK C/N  (full t=0-100)',
    f'z_erk_crosscorr_full_{version}.png',
)

# Z vs ERK, windowed to the active surface-contact period (t=30-75) where the
# Z-ERK timecourse correlation was actually significant — see z_erk_corr_timecourse plot
z_corr_df_win = crosscorr(merged, 'z_um_reg', t_lo=T_SPLIT, t_hi=75, min_pts=10)
z_peak_idx_win, z_peak_r_win = plot_crosscorr(
    z_corr_df_win, ('ERK', 'Z'),
    f'Cross-lag correlogram: Z position vs ERK C/N  (windowed t={T_SPLIT}-75)',
    f'z_erk_crosscorr_windowed_{version}.png',
)

# ── Summary ───────────────────────────────────────────────────────────────────

print(f'\n--- Rank stability (n={len(rank_df)}) ---')
rho_rad, p_rad = spearmanr(rank_df['rad_early_rank'], rank_df['rad_late_rank'])
rho_erk, p_erk = spearmanr(rank_df['erk_early_rank'], rank_df['erk_late_rank'])
print(f'  Radial rank stability: ρ={rho_rad:.3f}  p={p_rad:.4f}')
print(f'  ERK rank stability:    ρ={rho_erk:.3f}  p={p_erk:.4f}')

print(f'\n--- Cross-lag peaks ---')
print(f'  Radial vs ERK (full):        peak lag={rad_peak_idx} ({rad_peak_idx*interval:.0f} min),  r={rad_peak_r:.3f}')
print(f'  Z vs ERK (full t=0-100):     peak lag={z_peak_idx_full} ({z_peak_idx_full*interval:.0f} min),  r={z_peak_r_full:.3f}')
print(f'  Z vs ERK (windowed t={T_SPLIT}-75): peak lag={z_peak_idx_win} ({z_peak_idx_win*interval:.0f} min),  r={z_peak_r_win:.3f}')
print(f'  (positive lag = spatial var leads ERK; negative lag = ERK leads spatial var)')
