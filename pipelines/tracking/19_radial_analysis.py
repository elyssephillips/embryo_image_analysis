"""
Radial displacement analysis — track each cell's cumulative outward movement
relative to the time-varying embryo centroid (2D, y-x plane).

The key quantity is cumulative radial displacement from t_start: how much further
from (or closer to) the embryo centroid has each cell moved since implantation began.
This is less noisy than frame-to-frame radial velocity and directly tests whether
Q4 cells drift outward relative to Q1 over time.

Positive = moving away from embryo centre (outward).
Negative = moving toward centre.

Outputs
-------
  radial_cumdisp_{version}.csv             — per (track_id, t) cumulative radial displacement
  radial_cumdisp_timecourse_{version}.png  — all tracks coloured by ICM distance
  radial_binned_Q{1..N}_{version}.png      — mean cumulative radial disp + ERK per ICM bin
  radial_separation_{version}.png          — mean radial position change per bin over time

Run with:
  conda run -n napari_env python3 pipelines/tracking/19_radial_analysis.py
  conda run -n napari_env python3 pipelines/tracking/19_radial_analysis.py --n_bins 3
"""

import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('--n_bins',  type=int, default=4)
parser.add_argument('--t_start', type=int, default=30)
args = parser.parse_args()

N_BINS  = args.n_bins
T_START = args.t_start

COLOR_ERK = '#d6604d'
COLOR_RAD = '#1a9641'

# ── Load data ─────────────────────────────────────────────────────────────────

kin    = pd.read_csv(out_dir / f'motion_kinematics_{version}.csv')
erk_df = pd.read_csv(out_dir / f'erk_cn_ratio_{version}.csv')
vstats = pd.read_csv(out_dir / f'volume_track_stats_{version}.csv')[['track_id', 'icm_dist_um']]

kin = kin.sort_values(['track_id', 't'])

# ── Time-varying embryo centroid (all cells, registered coords) ───────────────

centroid = (kin.groupby('t')[['y_um_reg', 'x_um_reg']]
              .mean()
              .rename(columns={'y_um_reg': 'cy', 'x_um_reg': 'cx'}))
kin = kin.join(centroid, on='t')

# ── Radial distance from embryo centroid at every timepoint ───────────────────

kin['radial_dist'] = np.sqrt(
    (kin['y_um_reg'] - kin['cy'])**2 +
    (kin['x_um_reg'] - kin['cx'])**2
)

# ── Cumulative radial displacement from t_start ───────────────────────────────
# For each track: subtract that track's radial distance at its closest
# observation to T_START (within ±T_WINDOW frames). Anchors all tracks to 0
# at implantation onset regardless of whether they were observed at exactly t=30.

T_WINDOW = 2

within = kin[kin['t'].between(T_START - T_WINDOW, T_START + T_WINDOW)].copy()
within['_dt'] = (within['t'] - T_START).abs()
ref_dist = (within.sort_values(['track_id', '_dt', 't'])
                  .groupby('track_id')['radial_dist']
                  .first()
                  .rename('radial_dist_t0'))

kin = kin.join(ref_dist, on='track_id')
kin['cumul_radial_disp'] = kin['radial_dist'] - kin['radial_dist_t0']

n_before = kin['track_id'].nunique()
kin = kin[kin['radial_dist_t0'].notna()]
n_after = kin['track_id'].nunique()
print(f'Reference point: t={T_START} ±{T_WINDOW} frames  '
      f'({n_after} tracks with reference, {n_before - n_after} without '
      f'— these also lack icm_dist_um and are excluded from binned plots)')

# ── Merge with ERK and ICM distance ──────────────────────────────────────────

df = (erk_df
      .merge(kin[['track_id', 't', 'cumul_radial_disp']], on=['track_id', 't'], how='inner')
      .merge(vstats, on='track_id', how='left'))

df = df[df['t'] >= T_START].dropna(subset=['icm_dist_um']).copy()
df['time_min'] = df['t'] * interval

print(f'Tracks: {df["track_id"].nunique()}  |  Timepoints: {df["t"].nunique()}')
rv = df['cumul_radial_disp'].dropna()
print(f'Cumulative radial displacement range: {rv.min():.1f} – {rv.max():.1f} µm')

# ── ICM distance bins ─────────────────────────────────────────────────────────

track_dist = vstats.dropna(subset=['icm_dist_um']).set_index('track_id')['icm_dist_um']
bin_labels = [f'Q{i+1}' for i in range(N_BINS)]
df['dist_bin'] = pd.qcut(df['icm_dist_um'], q=N_BINS, labels=bin_labels)
_, edges = pd.qcut(track_dist, q=N_BINS, retbins=True)

cmap_bins  = plt.cm.coolwarm_r
bin_colors = [cmap_bins(i / max(N_BINS - 1, 1)) for i in range(N_BINS)]

# ── Save CSV ──────────────────────────────────────────────────────────────────

out_csv = out_dir / f'radial_cumdisp_{version}.csv'
kin[['track_id', 't', 'radial_dist', 'radial_dist_t0',
     'cumul_radial_disp']].to_csv(out_csv, index=False)
print(f'Saved: {out_csv.name}')

# ── Plot 1: all tracks coloured by ICM distance ───────────────────────────────

dist_min  = vstats['icm_dist_um'].min()
dist_max  = vstats['icm_dist_um'].max()
norm      = mcolors.Normalize(vmin=dist_min, vmax=dist_max)
cmap      = plt.cm.coolwarm_r
track_icm = vstats.set_index('track_id')['icm_dist_um']

fig, (ax_rad, ax_erk) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                      gridspec_kw={'hspace': 0.06})

for tid, grp in df.groupby('track_id'):
    if tid not in track_icm.index or np.isnan(track_icm[tid]):
        continue
    col = cmap(norm(track_icm[tid]))
    grp = grp.sort_values('t')
    ax_rad.plot(grp['time_min'], grp['cumul_radial_disp'],
                color=col, alpha=0.3, linewidth=0.8)
    ax_erk.plot(grp['time_min'], grp['erk_cn_ratio'],
                color=col, alpha=0.3, linewidth=0.8)

ax_rad.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.4)
ax_rad.set_ylabel('Cumulative radial displacement from t=30 (µm)\n← inward      outward →')
ax_rad.set_title('Cumulative radial displacement and ERK C/N — coloured by ICM distance')
ax_erk.set_ylabel('ERK C/N ratio')
ax_erk.set_xlabel('Time (min)')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax_rad, ax_erk], shrink=0.85, pad=0.02)
cbar.set_label('Distance from ICM (µm)\n← near        far →')

fig.savefig(out_dir / f'radial_cumdisp_timecourse_{version}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: radial_cumdisp_timecourse_{version}.png')

# ── Plot 2: binned mean cumulative radial displacement + ERK ──────────────────

def agg(s):
    return pd.Series({'mean': s.mean(), 'sem': s.sem()})

# Pre-compute timeseries for all bins to get shared axis limits
all_ts = {}
for label in bin_labels:
    sub = df[df['dist_bin'] == label].dropna(subset=['cumul_radial_disp', 'erk_cn_ratio'])
    all_ts[label] = (sub.groupby('time_min')['cumul_radial_disp'].apply(agg).unstack(),
                     sub.groupby('time_min')['erk_cn_ratio'].apply(agg).unstack())

pad = lambda lo, hi: (lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))

rad_vals = np.concatenate([(ts[0]['mean'] + ts[0]['sem']).values for ts in all_ts.values()] +
                           [(ts[0]['mean'] - ts[0]['sem']).values for ts in all_ts.values()])
erk_vals = np.concatenate([(ts[1]['mean'] + ts[1]['sem']).values for ts in all_ts.values()] +
                           [(ts[1]['mean'] - ts[1]['sem']).values for ts in all_ts.values()])
rad_ylim = pad(rad_vals[np.isfinite(rad_vals)].min(), rad_vals[np.isfinite(rad_vals)].max())
erk_ylim = pad(erk_vals[np.isfinite(erk_vals)].min(), erk_vals[np.isfinite(erk_vals)].max())

for i, (label, color) in enumerate(zip(bin_labels, bin_colors)):
    sub      = df[df['dist_bin'] == label].dropna(subset=['cumul_radial_disp', 'erk_cn_ratio'])
    n_tracks = sub['track_id'].nunique()
    dist_lo, dist_hi = edges[i], edges[i + 1]
    title = (f'Cumulative radial displacement  |  ICM bin {label} '
             f'({dist_lo:.0f}–{dist_hi:.0f} µm)  |  n={n_tracks} tracks')
    print(f'  {label}: {n_tracks} tracks')

    rad_ts, erk_ts = all_ts[label]
    t = rad_ts.index.values

    fig, ax_left = plt.subplots(figsize=(10, 4))
    ax_right = ax_left.twinx()

    ax_left.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.4)
    ax_left.plot(t, rad_ts['mean'], color=COLOR_RAD, linewidth=2,
                 label='cumul. radial disp.')
    ax_left.fill_between(t,
                         rad_ts['mean'] - rad_ts['sem'],
                         rad_ts['mean'] + rad_ts['sem'],
                         color=COLOR_RAD, alpha=0.2)

    ax_right.plot(t, erk_ts['mean'], color=COLOR_ERK, linewidth=2, label='ERK C/N')
    ax_right.fill_between(t,
                          erk_ts['mean'] - erk_ts['sem'],
                          erk_ts['mean'] + erk_ts['sem'],
                          color=COLOR_ERK, alpha=0.2)

    ax_left.set_ylim(rad_ylim)
    ax_right.set_ylim(erk_ylim)
    ax_left.set_xlabel('Time (min)')
    ax_left.set_ylabel('Cumulative radial displacement (µm)\n← inward      outward →',
                       color=COLOR_RAD)
    ax_right.set_ylabel('ERK C/N ratio', color=COLOR_ERK)
    ax_left.tick_params(axis='y', labelcolor=COLOR_RAD)
    ax_right.tick_params(axis='y', labelcolor=COLOR_ERK)
    ax_left.set_title(title, pad=8)

    lines  = ax_left.get_lines() + ax_right.get_lines()
    labels = [l.get_label() for l in lines]
    ax_left.legend(lines, labels, loc='upper left', fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / f'radial_binned_{label}_{version}.png', dpi=150)
    plt.close()
    print(f'    Saved: radial_binned_{label}_{version}.png')

# ── Plot 3: mean cumulative radial displacement per bin over time ─────────────

fig, ax = plt.subplots(figsize=(12, 4))
ax.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.4)

for i, (label, color) in enumerate(zip(bin_labels, bin_colors)):
    sub = df[df['dist_bin'] == label]
    ts  = sub.groupby('time_min')['cumul_radial_disp'].apply(agg).unstack()
    dist_lo, dist_hi = edges[i], edges[i + 1]
    ax.plot(ts.index, ts['mean'], color=color, linewidth=2,
            label=f'{label} ({dist_lo:.0f}–{dist_hi:.0f} µm)')
    ax.fill_between(ts.index,
                    ts['mean'] - ts['sem'],
                    ts['mean'] + ts['sem'],
                    color=color, alpha=0.2)

ax.set_xlabel('Time (min)')
ax.set_ylabel('Mean cumulative radial displacement from t=30 (µm)')
ax.set_title('Outward drift by ICM-distance bin since implantation onset\n'
             'All tracks anchored to 0 at t=30')
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(out_dir / f'radial_separation_{version}.png', dpi=150)
plt.close()
print(f'Saved: radial_separation_{version}.png')

print('\nDone.')
