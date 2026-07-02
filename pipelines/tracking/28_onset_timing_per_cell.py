"""
Script 28: Per-cell event timing — does Z-touchdown or ERK-onset happen first?

The cross-lag analysis in script 26 averages each cell's full trajectory at a fixed
lag, which is the wrong test when cells touch down / activate at different absolute
times: most of each track is flat before/after the transition, so averaging over the
whole track dilutes the one brief window that actually carries directional information.

This script instead finds, per cell, the TIME it crosses a threshold defined relative
to its OWN observed range — no slope detection, no smoothing, just "when did this
cell get close to where it ends up":

  - Z touchdown:  first time Z >= (own max Z during the track) - Z_TOL.
                  Z's physical ceiling (the dish surface) is roughly fixed, but the
                  dish isn't perfectly flat, so we use each cell's own observed max
                  rather than one global threshold.
  - radial onset: same idea — first time radial distance >= (own max) - a tolerance
                  fraction of its own range (no fixed physical ceiling for this one).
  - ERK onset:    first time ERK C/N reaches 80% of the way from its own
                  pre-implantation baseline to its own post-implantation peak.

A cell only gets an onset if it shows a genuine rise (own peak meaningfully above
baseline / starting point) — flat or noisy cells are excluded rather than forced.

Per cell: delta_t = t_erk_onset - t_other_onset
  positive delta_t -> the spatial transition happens BEFORE ERK onset (contact -> ERK)
  negative delta_t -> ERK onset happens BEFORE the spatial transition (ERK -> movement)

Outputs
-------
  onset_scatter_{version}.png      — per-cell onset time: Z (or radial) vs ERK, paired
  onset_delta_hist_{version}.png   — histograms of delta_t for Z-ERK and radial-ERK
  onset_aligned_traces_{version}.png — traces aligned to each cell's own onset time (t=0)

Run with:
  conda run -n napari_env python3 pipelines/tracking/28_onset_timing_per_cell.py
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

version  = cfg['tracking']['input_version']
out_dir  = Path(cfg['paths']['output_dir'])
interval = cfg['tracking']['frame_interval_min']

T_SPLIT   = 30    # implantation onset; onset search starts here
MIN_PTS   = 15    # minimum post-implantation timepoints required
Z_TOL     = 10.0  # µm tolerance band below own max Z to call "touchdown"
RAD_FRAC  = 0.8   # fraction of own radial range to call "radial onset"
ERK_FRAC  = 0.8   # fraction of own (peak - baseline) ERK rise to call "ERK onset"
MIN_RISE_Z   = 15.0  # minimum µm rise required to call a genuine Z touchdown
MIN_RISE_RAD = 15.0  # minimum µm rise required to call a genuine radial onset

# ── Load data + dynamic radial distance ───────────────────────────────────────

kine = pd.read_csv(out_dir / f'motion_kinematics_{version}.csv').sort_values(['track_id', 't'])
erk  = pd.read_csv(out_dir / f'erk_cn_ratio_{version}.csv').sort_values(['track_id', 't'])

centroid = (kine.groupby('t')[['x_um_reg', 'y_um_reg']]
                .mean()
                .rename(columns={'x_um_reg': 'cx', 'y_um_reg': 'cy'}))
kine = kine.join(centroid, on='t')
kine['radial_dist_dyn'] = np.sqrt(
    (kine['x_um_reg'] - kine['cx'])**2 +
    (kine['y_um_reg'] - kine['cy'])**2
)
kine['time_min'] = kine['t'] * interval

merged_full = (kine[['track_id', 't', 'time_min', 'z_um_reg', 'radial_dist_dyn']]
               .merge(erk[['track_id', 't', 'erk_cn_ratio']], on=['track_id', 't']))
merged = merged_full[merged_full['t'] >= T_SPLIT].copy()

# ── Per-cell onset detection ───────────────────────────────────────────────────

def detect_ceiling_onset(t, vals, min_rise):
    """First time vals reaches within a tolerance band of its own max (post-T_SPLIT)."""
    own_max = vals.max()
    own_min = vals.min()
    if (own_max - own_min) < min_rise:
        return np.nan
    thresh = own_max - Z_TOL if min_rise == MIN_RISE_Z else own_max - RAD_FRAC * (own_max - own_min)
    mask = vals >= thresh
    if not mask.any():
        return np.nan
    return t[mask].min()

def detect_erk_onset(t, vals, baseline):
    own_peak = vals.max()
    if (own_peak - baseline) <= 0:
        return np.nan
    thresh = baseline + ERK_FRAC * (own_peak - baseline)
    mask = vals >= thresh
    if not mask.any():
        return np.nan
    return t[mask].min()

# pre-implantation ERK baseline per track (t < T_SPLIT)
pre_erk = (merged_full[merged_full['t'] < T_SPLIT]
           .groupby('track_id')['erk_cn_ratio'].mean())

records = []
for tid, grp in merged.groupby('track_id'):
    grp = grp.dropna(subset=['z_um_reg', 'radial_dist_dyn', 'erk_cn_ratio']).sort_values('t')
    if len(grp) < MIN_PTS:
        continue

    t_arr   = grp['time_min'].values
    z_arr   = grp['z_um_reg'].values
    rad_arr = grp['radial_dist_dyn'].values
    erk_arr = grp['erk_cn_ratio'].values

    t_z   = detect_ceiling_onset(t_arr, z_arr,   MIN_RISE_Z)
    t_rad = detect_ceiling_onset(t_arr, rad_arr, MIN_RISE_RAD)

    baseline = pre_erk.get(tid, np.nan)
    t_erk = detect_erk_onset(t_arr, erk_arr, baseline) if not np.isnan(baseline) else np.nan

    records.append({
        'track_id': tid,
        't_z_onset': t_z,
        't_rad_onset': t_rad,
        't_erk_onset': t_erk,
    })

onsets = pd.DataFrame(records)
print(f'Tracks evaluated: {len(onsets)}')
print(f'  with Z onset:      {onsets["t_z_onset"].notna().sum()}')
print(f'  with radial onset: {onsets["t_rad_onset"].notna().sum()}')
print(f'  with ERK onset:    {onsets["t_erk_onset"].notna().sum()}')

z_pairs   = onsets.dropna(subset=['t_z_onset', 't_erk_onset'])
rad_pairs = onsets.dropna(subset=['t_rad_onset', 't_erk_onset'])

z_pairs   = z_pairs.assign(delta_t=z_pairs['t_erk_onset'] - z_pairs['t_z_onset'])
rad_pairs = rad_pairs.assign(delta_t=rad_pairs['t_erk_onset'] - rad_pairs['t_rad_onset'])

print(f'\nPaired cells (both onsets detected):')
print(f'  Z vs ERK:      n={len(z_pairs)}')
print(f'  Radial vs ERK: n={len(rad_pairs)}')

# ── Figure 1: paired onset scatter ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, pairs, xcol, xlabel, title in [
    (axes[0], z_pairs,   't_z_onset',   'Z onset time (min)',      'Z onset vs ERK onset'),
    (axes[1], rad_pairs, 't_rad_onset', 'Radial onset time (min)', 'Radial onset vs ERK onset'),
]:
    if len(pairs) == 0:
        ax.set_title(f'{title}\n(no paired cells)')
        continue
    lims = [
        min(pairs[xcol].min(), pairs['t_erk_onset'].min()) - 20,
        max(pairs[xcol].max(), pairs['t_erk_onset'].max()) + 20,
    ]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5, zorder=1, label='equal timing')
    ax.scatter(pairs[xcol], pairs['t_erk_onset'], s=50, alpha=0.8,
               color='#2166ac', edgecolors='0.3', linewidths=0.4, zorder=3)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('ERK onset time (min)')
    n_before = (pairs['delta_t'] > 0).sum()  # spatial transition before ERK
    n_after  = (pairs['delta_t'] < 0).sum()  # ERK before spatial transition
    ax.set_title(f'{title}\n{xlabel.split()[0]} first: n={n_before}  |  ERK first: n={n_after}  (n={len(pairs)})')
    ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(out_dir / f'onset_scatter_{version}.png', dpi=150)
plt.close()
print(f'Saved: onset_scatter_{version}.png')

# ── Figure 2: delta_t histograms ───────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, pairs, label in [
    (axes[0], z_pairs,   'Z onset'),
    (axes[1], rad_pairs, 'Radial onset'),
]:
    if len(pairs) == 0:
        ax.set_title(f'{label} vs ERK onset\n(no paired cells)')
        continue
    ax.axvline(0, color='k', linewidth=1, linestyle='--', alpha=0.6)
    ax.hist(pairs['delta_t'], bins=12, color='#4393c3', edgecolor='0.3', alpha=0.85)
    median_dt = pairs['delta_t'].median()
    try:
        stat, p = wilcoxon(pairs['delta_t'])
    except ValueError:
        p = np.nan
    ax.axvline(median_dt, color='#b2182b', linewidth=1.5, label=f'median={median_dt:.0f} min')
    ax.set_xlabel(f'Δt = ERK onset − {label}  (min)\n← ERK leads          {label} leads →')
    ax.set_ylabel('Count')
    ax.set_title(f'{label} vs ERK timing  (n={len(pairs)})\nWilcoxon p={p:.3f}')
    ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(out_dir / f'onset_delta_hist_{version}.png', dpi=150)
plt.close()
print(f'Saved: onset_delta_hist_{version}.png')

# ── Figure 3: traces aligned to each cell's own onset time ────────────────────

def aligned_traces(pairs, signal_col, onset_col, value_lookup):
    rows = []
    for _, row in pairs.iterrows():
        tid = row['track_id']
        onset_t = row[onset_col]
        cell = value_lookup[value_lookup['track_id'] == tid]
        for _, r in cell.iterrows():
            rows.append({'rel_t': r['time_min'] - onset_t, 'value': r[signal_col]})
    return pd.DataFrame(rows)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Z-aligned: Z and ERK traces aligned to each cell's own Z onset
z_aligned_z   = aligned_traces(z_pairs, 'z_um_reg',      't_z_onset', merged)
z_aligned_erk = aligned_traces(z_pairs, 'erk_cn_ratio',  't_z_onset', merged)

ax = axes[0]
ax.axvline(0, color='k', linewidth=1, linestyle='--', alpha=0.6, label='Z onset (t=0)')
for df, col, label, axis in [(z_aligned_z, 'value', 'Z position', ax)]:
    pass
bins = np.arange(-300, 301, interval)
z_aligned_z['bin']   = pd.cut(z_aligned_z['rel_t'], bins)
z_aligned_erk['bin'] = pd.cut(z_aligned_erk['rel_t'], bins)
zb = z_aligned_z.groupby('bin')['value'].agg(['mean', 'sem'])
eb = z_aligned_erk.groupby('bin')['value'].agg(['mean', 'sem'])
bin_centers = [iv.mid for iv in zb.index]
ax2 = ax.twinx()
ax.plot(bin_centers, zb['mean'], color='#4393c3', linewidth=2, label='Z position')
ax.fill_between(bin_centers, zb['mean']-zb['sem'], zb['mean']+zb['sem'], color='#4393c3', alpha=0.2)
ax2.plot(bin_centers, eb['mean'], color='#d6604d', linewidth=2, label='ERK C/N')
ax2.fill_between(bin_centers, eb['mean']-eb['sem'], eb['mean']+eb['sem'], color='#d6604d', alpha=0.2)
ax.set_ylabel('Z position (µm)', color='#4393c3')
ax2.set_ylabel('ERK C/N', color='#d6604d')
ax.set_title(f'Traces aligned to each cell\'s own Z onset  (n={len(z_pairs)})')
ax.legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8)

# Radial-aligned: radial and ERK traces aligned to each cell's own radial onset
rad_aligned_rad = aligned_traces(rad_pairs, 'radial_dist_dyn', 't_rad_onset', merged)
rad_aligned_erk = aligned_traces(rad_pairs, 'erk_cn_ratio',    't_rad_onset', merged)

ax = axes[1]
ax.axvline(0, color='k', linewidth=1, linestyle='--', alpha=0.6, label='Radial onset (t=0)')
rad_aligned_rad['bin'] = pd.cut(rad_aligned_rad['rel_t'], bins)
rad_aligned_erk['bin'] = pd.cut(rad_aligned_erk['rel_t'], bins)
rb = rad_aligned_rad.groupby('bin')['value'].agg(['mean', 'sem'])
erb = rad_aligned_erk.groupby('bin')['value'].agg(['mean', 'sem'])
ax2 = ax.twinx()
ax.plot(bin_centers, rb['mean'], color='#4393c3', linewidth=2, label='Radial distance')
ax.fill_between(bin_centers, rb['mean']-rb['sem'], rb['mean']+rb['sem'], color='#4393c3', alpha=0.2)
ax2.plot(bin_centers, erb['mean'], color='#d6604d', linewidth=2, label='ERK C/N')
ax2.fill_between(bin_centers, erb['mean']-erb['sem'], erb['mean']+erb['sem'], color='#d6604d', alpha=0.2)
ax.set_ylabel('Radial distance (µm)', color='#4393c3')
ax2.set_ylabel('ERK C/N', color='#d6604d')
ax.set_xlabel('Time relative to onset (min)')
ax.set_title(f'Traces aligned to each cell\'s own radial onset  (n={len(rad_pairs)})')
ax.legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8)

plt.tight_layout()
fig.savefig(out_dir / f'onset_aligned_traces_{version}.png', dpi=150)
plt.close()
print(f'Saved: onset_aligned_traces_{version}.png')

# ── Summary ───────────────────────────────────────────────────────────────────

print('\n--- Summary ---')
if len(z_pairs) > 0:
    print(f'Z vs ERK onset:      median Δt = {z_pairs["delta_t"].median():.0f} min  '
          f'(positive = Z transition before ERK)')
if len(rad_pairs) > 0:
    print(f'Radial vs ERK onset: median Δt = {rad_pairs["delta_t"].median():.0f} min  '
          f'(positive = radial transition before ERK)')
