"""
Per-nucleus ERK C/N timecourse for the low pre-ERK group.

Nuclei are grouped by their mean ERK C/N before implantation onset (t < T_SPLIT)
into low / mid / high tertiles.  Every nucleus in the low group gets its own panel
showing ERK C/N over the full timecourse, with the pre-implantation window shaded.

Usage
-----
  # default: low tertile, t_split=30
  conda run -n napari_env python3 pipelines/tracking/21_preerk_individual_tracks.py

  # look at the high group instead
  conda run -n napari_env python3 pipelines/tracking/21_preerk_individual_tracks.py --group high

  # change the split timepoint or number of groups
  conda run -n napari_env python3 pipelines/tracking/21_preerk_individual_tracks.py --t_split 25 --n_groups 4
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

# ── Edit these directly in VS Code ───────────────────────────────────────────
GROUP    = 'high'   # 'low', 'mid', or 'high'
N_GROUPS = 3       # number of quantile groups (3 = tertiles)
T_SPLIT  = 30      # timepoint separating pre- from post-implantation
MIN_PRE  = 3       # minimum pre-implantation timepoints required per track
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--group',    default=None, choices=['low', 'mid', 'high'])
parser.add_argument('--n_groups', type=int, default=None)
parser.add_argument('--t_split',  type=int, default=None)
parser.add_argument('--min_pre',  type=int, default=None)
args = parser.parse_args()

if args.group    is not None: GROUP    = args.group
if args.n_groups is not None: N_GROUPS = args.n_groups
if args.t_split  is not None: T_SPLIT  = args.t_split
if args.min_pre  is not None: MIN_PRE  = args.min_pre

# map group name to quantile rank (0-indexed)
GROUP_IDX = {'low': 0, 'mid': N_GROUPS // 2, 'high': N_GROUPS - 1}
group_idx = GROUP_IDX[GROUP]
group_labels = ['low', 'mid', 'high'] if N_GROUPS == 3 else [f'G{i+1}' for i in range(N_GROUPS)]

# ── Load data ─────────────────────────────────────────────────────────────────

erk_df = pd.read_csv(out_dir / f'erk_cn_ratio_{version}.csv')
vstats = pd.read_csv(out_dir / f'volume_track_stats_{version}.csv')[['track_id', 'icm_dist_um']]

erk_df = erk_df.merge(vstats, on='track_id', how='left')
erk_df['time_min'] = erk_df['t'] * interval

# ── Compute pre-ERK mean per track ───────────────────────────────────────────

pre = erk_df[erk_df['t'] < T_SPLIT].copy()
pre_mean = (pre.groupby('track_id')['erk_cn_ratio']
               .agg(pre_erk_mean='mean', pre_erk_n='count')
               .reset_index())
pre_mean = pre_mean[pre_mean['pre_erk_n'] >= MIN_PRE]

# quantile-cut into N_GROUPS
pre_mean['preerk_group'] = pd.qcut(
    pre_mean['pre_erk_mean'], q=N_GROUPS,
    labels=group_labels[:N_GROUPS] if N_GROUPS == 3 else None
)

target_label = group_labels[group_idx] if N_GROUPS == 3 else pre_mean['preerk_group'].cat.categories[group_idx]

n_total = len(pre_mean)
target_tracks = pre_mean[pre_mean['preerk_group'] == target_label]['track_id'].tolist()

print(f'Pre-ERK groups (t < {T_SPLIT}, min {MIN_PRE} timepoints):')
for lbl in pre_mean['preerk_group'].cat.categories:
    n = (pre_mean['preerk_group'] == lbl).sum()
    vals = pre_mean.loc[pre_mean['preerk_group'] == lbl, 'pre_erk_mean']
    print(f'  {lbl}: n={n}  mean C/N range [{vals.min():.3f}, {vals.max():.3f}]')
print(f'\nPlotting {len(target_tracks)} nuclei in the "{GROUP}" group')

# ── Plot: one panel per nucleus ───────────────────────────────────────────────

# Colormap by ICM distance (consistent with other scripts)
dist_min = vstats['icm_dist_um'].min()
dist_max = vstats['icm_dist_um'].max()
norm_icm = mcolors.Normalize(vmin=dist_min, vmax=dist_max)
cmap_icm = plt.cm.coolwarm_r
track_icm = vstats.set_index('track_id')['icm_dist_um']

# Color each line by pre-ERK level within this group
pre_erk_lookup = pre_mean.set_index('track_id')['pre_erk_mean']
group_erk_vals = pre_erk_lookup[pre_erk_lookup.index.isin(target_tracks)]
erk_lo, erk_hi = group_erk_vals.min(), group_erk_vals.max()
norm_erk = mcolors.Normalize(vmin=erk_lo, vmax=erk_hi)
cmap_erk = plt.cm.YlOrRd

n_tracks  = len(target_tracks)
split_time = T_SPLIT * interval

# ── Shared y-limits (computed once, applied to all plots) ────────────────────

# Raw ERK: use all data across all groups so every plot is on the same scale
raw_vals = erk_df['erk_cn_ratio'].dropna()
_y_lo = raw_vals.quantile(0.01)
_y_hi = raw_vals.quantile(0.99)
_pad  = 0.05 * (_y_hi - _y_lo)
ylim_raw = (_y_lo - _pad, _y_hi + _pad)

# Normalized ERK: computed after normalization below (placeholder until Plot 3)
ylim_norm = None

# Shared x-axis: always start at 0, end at last timepoint
xlim = (0, erk_df['time_min'].max())

fig, ax = plt.subplots(figsize=(12, 5))

# Shade pre-implantation window
ax.axvspan(erk_df['time_min'].min(), split_time, color='0.92', zorder=0)
ax.axvline(split_time, color='0.4', linewidth=0.8, linestyle=':', zorder=1)

for tid in sorted(target_tracks):
    grp = erk_df[erk_df['track_id'] == tid].sort_values('t')

    pre_erk  = pre_erk_lookup.get(tid, np.nan)
    line_col = cmap_erk(norm_erk(pre_erk)) if not np.isnan(pre_erk) else '0.5'

    ax.plot(grp['time_min'], grp['erk_cn_ratio'],
            color=line_col, linewidth=0.9, alpha=0.7, zorder=2)

# Mean ± SEM across all nuclei in this group
group_data = erk_df[erk_df['track_id'].isin(target_tracks)]
mean_ts = group_data.groupby('time_min')['erk_cn_ratio'].agg(['mean', 'sem'])
ax.plot(mean_ts.index, mean_ts['mean'],
        color='k', linewidth=2.5, zorder=4, label='group mean')
ax.fill_between(mean_ts.index,
                mean_ts['mean'] - mean_ts['sem'],
                mean_ts['mean'] + mean_ts['sem'],
                color='k', alpha=0.15, zorder=3, label='± SEM')

sm = plt.cm.ScalarMappable(cmap=cmap_erk, norm=norm_erk)
sm.set_array([])
fig.colorbar(sm, ax=ax, label=f'pre-ERK C/N mean (t < {T_SPLIT})', shrink=0.85)

erk_range = f'[{erk_lo:.3f}–{erk_hi:.3f}]'
ax.set_xlabel('Time (min)')
ax.set_ylabel('ERK C/N ratio')
ax.set_title(
    f'ERK C/N timecourses — "{GROUP}" pre-ERK group  (n={n_tracks})  |  '
    f'pre-ERK range {erk_range}\n'
    f'Shaded = pre-implantation (t < {T_SPLIT})  |  '
    f'line colour = pre-ERK level  |  black = group mean ± SEM'
)
ax.set_xlim(xlim)
ax.set_ylim(ylim_raw)
ax.legend(fontsize=9)

plt.tight_layout()
out_path = out_dir / f'preerk_{GROUP}_overlay_erk_{version}.png'
fig.savefig(out_path, dpi=150)
plt.close()
print(f'Saved: {out_path}')

# ── Plot 2: one panel per nucleus, colored by ICM distance ───────────────────

n_cols   = min(5, n_tracks)
n_rows   = int(np.ceil(n_tracks / n_cols))

fig2, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(4.5 * n_cols, 3 * n_rows),
    constrained_layout=True
)
axes_flat = np.array(axes).flatten() if n_tracks > 1 else [axes]

for ax, tid in zip(axes_flat, sorted(target_tracks)):
    grp = erk_df[erk_df['track_id'] == tid].sort_values('t')

    icm      = track_icm.get(tid, np.nan)
    line_col = cmap_icm(norm_icm(icm)) if not np.isnan(icm) else '0.5'

    ax.axvspan(grp['time_min'].min(), split_time, color='0.92', zorder=0)
    ax.axvline(split_time, color='0.4', linewidth=0.8, linestyle=':', zorder=1)
    ax.plot(grp['time_min'], grp['erk_cn_ratio'],
            color=line_col, linewidth=1.4, zorder=2)

    pre_erk = pre_erk_lookup.get(tid, np.nan)
    if not np.isnan(pre_erk):
        pre_t = grp.loc[grp['t'] < T_SPLIT, 'time_min']
        if not pre_t.empty:
            ax.hlines(pre_erk, pre_t.min(), split_time,
                      color=line_col, linewidth=1, linestyle='--', alpha=0.6, zorder=3)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim_raw)

    icm_str = f'{icm:.0f} µm' if not np.isnan(icm) else 'n/a'
    pre_str = f'pre={pre_erk:.3f}' if not np.isnan(pre_erk) else ''
    ax.set_title(f'track {tid}  |  {icm_str} ICM  |  {pre_str}',
                 fontsize=8, pad=3)
    ax.set_xlabel('Time (min)', fontsize=7)
    ax.set_ylabel('ERK C/N', fontsize=7)
    ax.tick_params(labelsize=6)

for ax in axes_flat[n_tracks:]:
    ax.set_visible(False)

sm2 = plt.cm.ScalarMappable(cmap=cmap_icm, norm=norm_icm)
sm2.set_array([])
fig2.colorbar(sm2, ax=axes_flat.tolist(),
              label='ICM distance (µm)\n← near        far →', shrink=0.6)

fig2.suptitle(
    f'ERK C/N per nucleus — "{GROUP}" pre-ERK group  (n={n_tracks})\n'
    f'Line colour = ICM distance  |  dashed = pre-ERK mean  |  '
    f'shaded = pre-implantation (t < {T_SPLIT})',
    fontsize=9
)

out_path2 = out_dir / f'preerk_{GROUP}_individual_erk_{version}.png'
fig2.savefig(out_path2, dpi=150)
plt.close()
print(f'Saved: {out_path2}')

# ── Plot 3: normalized timecourse — all tracks in group ──────────────────────
# Normalize each track to its pre-implantation mean so all start at 1.0.
# All tracks with a valid pre-implantation mean are included.

full_tracks = [tid for tid in target_tracks if tid in pre_erk_lookup.index]

print(f'\nTracks in "{GROUP}" group with valid pre-implantation baseline: {len(full_tracks)}')

if full_tracks:
    fig3, ax3 = plt.subplots(figsize=(12, 5))

    ax3.axvspan(erk_df['time_min'].min(), split_time, color='0.92', zorder=0)
    ax3.axvline(split_time, color='0.4', linewidth=0.8, linestyle=':', zorder=1)
    ax3.axhline(1.0, color='0.4', linewidth=0.8, linestyle='--', alpha=0.6, zorder=1)

    norm_traces = []
    for tid in sorted(full_tracks):
        grp     = erk_df[erk_df['track_id'] == tid].sort_values('t')
        pre_erk = pre_erk_lookup.get(tid, np.nan)
        if np.isnan(pre_erk) or pre_erk == 0:
            continue

        grp = grp.copy()
        grp['erk_norm'] = grp['erk_cn_ratio'] / pre_erk

        icm      = track_icm.get(tid, np.nan)
        line_col = cmap_icm(norm_icm(icm)) if not np.isnan(icm) else '0.5'

        ax3.plot(grp['time_min'], grp['erk_norm'],
                 color=line_col, linewidth=0.9, alpha=0.75, zorder=2)

        norm_traces.append(grp[['time_min', 'erk_norm']].copy())

    # Mean ± SEM of normalized traces; also set shared ylim_norm for plot 4
    if norm_traces:
        all_norm = pd.concat(norm_traces)
        _yn_lo = all_norm['erk_norm'].quantile(0.01)
        _yn_hi = all_norm['erk_norm'].quantile(0.99)
        _yn_pad = 0.05 * (_yn_hi - _yn_lo)
        ylim_norm = (_yn_lo - _yn_pad, _yn_hi + _yn_pad)
        mean_norm = all_norm.groupby('time_min')['erk_norm'].agg(['mean', 'sem'])
        ax3.plot(mean_norm.index, mean_norm['mean'],
                 color='k', linewidth=2.5, zorder=4, label='group mean')
        ax3.fill_between(mean_norm.index,
                         mean_norm['mean'] - mean_norm['sem'],
                         mean_norm['mean'] + mean_norm['sem'],
                         color='k', alpha=0.15, zorder=3, label='± SEM')

    sm3 = plt.cm.ScalarMappable(cmap=cmap_icm, norm=norm_icm)
    sm3.set_array([])
    fig3.colorbar(sm3, ax=ax3,
                  label='ICM distance (µm)\n← near        far →', shrink=0.85)

    ax3.set_xlim(xlim)
    if ylim_norm:
        ax3.set_ylim(ylim_norm)
    ax3.set_xlabel('Time (min)')
    ax3.set_ylabel('ERK C/N  (normalized to pre-implantation mean)')
    ax3.set_title(
        f'Normalized ERK timecourse — "{GROUP}" pre-ERK group  (n={len(full_tracks)})\n'
        f'Each track divided by its mean ERK C/N for t < {T_SPLIT}  |  '
        f'1.0 = pre-implantation baseline  |  colour = ICM distance'
    )
    ax3.legend(fontsize=9)

    plt.tight_layout()
    out_path3 = out_dir / f'preerk_{GROUP}_normalized_erk_{version}.png'
    fig3.savefig(out_path3, dpi=150)
    plt.close()
    print(f'Saved: {out_path3}')
else:
    print('No full-span tracks found in this group — skipping normalized plots.')

# ── Plot 4: individual panels, normalized — full-span tracks only ─────────────

if full_tracks:
    # Build normalized data for each full-span track
    norm_data = {}
    for tid in sorted(full_tracks):
        grp     = erk_df[erk_df['track_id'] == tid].sort_values('t')
        pre_erk = pre_erk_lookup.get(tid, np.nan)
        if np.isnan(pre_erk) or pre_erk == 0:
            continue
        grp = grp.copy()
        grp['erk_norm'] = grp['erk_cn_ratio'] / pre_erk
        norm_data[tid] = grp

    n_full   = len(norm_data)
    n_cols4  = min(5, n_full)
    n_rows4  = int(np.ceil(n_full / n_cols4))

    fig4, axes4 = plt.subplots(
        n_rows4, n_cols4,
        figsize=(4.5 * n_cols4, 3 * n_rows4),
        constrained_layout=True
    )
    axes4_flat = np.array(axes4).flatten() if n_full > 1 else [axes4]

    for ax, (tid, grp) in zip(axes4_flat, norm_data.items()):
        icm      = track_icm.get(tid, np.nan)
        line_col = cmap_icm(norm_icm(icm)) if not np.isnan(icm) else '0.5'
        pre_erk  = pre_erk_lookup.get(tid, np.nan)

        ax.axvspan(grp['time_min'].min(), split_time, color='0.92', zorder=0)
        ax.axvline(split_time, color='0.4', linewidth=0.8, linestyle=':', zorder=1)
        ax.axhline(1.0, color='0.4', linewidth=0.8, linestyle='--', alpha=0.5, zorder=1)
        ax.plot(grp['time_min'], grp['erk_norm'],
                color=line_col, linewidth=1.4, zorder=2)

        ax.set_xlim(xlim)
        if ylim_norm:
            ax.set_ylim(ylim_norm)

        icm_str = f'{icm:.0f} µm' if not np.isnan(icm) else 'n/a'
        pre_str = f'pre={pre_erk:.3f}' if not np.isnan(pre_erk) else ''
        ax.set_title(f'track {tid}  |  {icm_str} ICM  |  {pre_str}',
                     fontsize=8, pad=3)
        ax.set_xlabel('Time (min)', fontsize=7)
        ax.set_ylabel('ERK C/N (norm.)', fontsize=7)
        ax.tick_params(labelsize=6)

    for ax in axes4_flat[n_full:]:
        ax.set_visible(False)

    sm4 = plt.cm.ScalarMappable(cmap=cmap_icm, norm=norm_icm)
    sm4.set_array([])
    fig4.colorbar(sm4, ax=list(axes4_flat),
                  label='ICM distance (µm)\n← near        far →', shrink=0.6)

    fig4.suptitle(
        f'Normalized ERK C/N per nucleus — "{GROUP}" pre-ERK group  (n={n_full})\n'
        f'1.0 = pre-implantation mean  |  colour = ICM distance  |  '
        f'shaded = pre-implantation (t < {T_SPLIT})',
        fontsize=9
    )

    out_path4 = out_dir / f'preerk_{GROUP}_normalized_individual_erk_{version}.png'
    fig4.savefig(out_path4, dpi=150)
    plt.close()
    print(f'Saved: {out_path4}')

# ── Plot 5: group mean ± SEM for all pre-ERK groups on one axis ──────────────

group_colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, N_GROUPS))
all_group_labels = list(pre_mean['preerk_group'].cat.categories)

fig5, ax5 = plt.subplots(figsize=(12, 5))

ax5.axvspan(erk_df['time_min'].min(), split_time, color='0.92', zorder=0)
ax5.axvline(split_time, color='0.4', linewidth=0.8, linestyle=':', zorder=1)

for lbl, col in zip(all_group_labels, group_colors):
    grp_tracks = pre_mean.loc[pre_mean['preerk_group'] == lbl, 'track_id']
    grp_data   = erk_df[erk_df['track_id'].isin(grp_tracks)]
    ts = grp_data.groupby('time_min')['erk_cn_ratio'].agg(['mean', 'sem', 'count'])
    n  = grp_tracks.nunique()
    ax5.plot(ts.index, ts['mean'], color=col, linewidth=2, label=f'{lbl}  (n={n})', zorder=3)
    ax5.fill_between(ts.index,
                     ts['mean'] - ts['sem'],
                     ts['mean'] + ts['sem'],
                     color=col, alpha=0.2, zorder=2)

ax5.set_xlim(xlim)
ax5.set_ylim(ylim_raw)
ax5.set_xlabel('Time (min)')
ax5.set_ylabel('ERK C/N ratio')
ax5.set_title(
    f'Mean ERK C/N by pre-implantation ERK group  '
    f'({N_GROUPS} quantile groups, t < {T_SPLIT})\n'
    f'Shaded = ± SEM  |  vertical line = implantation onset'
)
ax5.legend(fontsize=9)

plt.tight_layout()
out_path5 = out_dir / f'preerk_group_means_erk_{version}.png'
fig5.savefig(out_path5, dpi=150)
plt.close()
print(f'Saved: {out_path5}')

# ── Plot 6: spatial positions — full-span tracks only ────────────────────────
# Only tracks that span ≥80% of the recording have a meaningful "final position".
# For shorter tracks the last point is just where tracking ended, not where the cell is.

t_max_global  = erk_df['t'].max()
MIN_SPAN_FRAC = 0.8
track_tmax    = erk_df.groupby('track_id')['t'].max()
spatial_tracks = [
    tid for tid in target_tracks
    if tid in track_tmax.index and track_tmax[tid] >= MIN_SPAN_FRAC * t_max_global
]
print(f'\nFull-span tracks for spatial plot '
      f'(t_max >= {MIN_SPAN_FRAC * t_max_global:.0f}): {len(spatial_tracks)}')

if spatial_tracks:
    kine_df = pd.read_csv(out_dir / f'motion_kinematics_{version}.csv')

    # Final position of every track
    last_pos = (kine_df.sort_values('t')
                       .groupby('track_id')[['x_um_reg', 'y_um_reg', 'z_um_reg']]
                       .last()
                       .reset_index())

    # Final ERK for background coloring
    last_erk = (erk_df.sort_values('t')
                      .groupby('track_id')['erk_cn_ratio']
                      .last()
                      .reset_index()
                      .rename(columns={'erk_cn_ratio': 'final_erk'}))

    bg = last_pos.merge(last_erk, on='track_id', how='left')
    bg_other = bg[~bg['track_id'].isin(spatial_tracks)]
    bg_focus = bg[bg['track_id'].isin(spatial_tracks)]

    erk_norm_bg = mcolors.Normalize(
        vmin=bg['final_erk'].quantile(0.02),
        vmax=bg['final_erk'].quantile(0.98)
    )

    projections = [
        ('x_um_reg', 'y_um_reg', 'X (µm)', 'Y (µm)', 'Top-down (XY)'),
        ('x_um_reg', 'z_um_reg', 'X (µm)', 'Z (µm)', 'Side view (XZ)'),
    ]

    fig6, axes6 = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    for ax6, (xcol, ycol, xlabel, ylabel, title) in zip(axes6, projections):
        # All other nuclei
        ax6.scatter(bg_other[xcol], bg_other[ycol],
                    c=bg_other['final_erk'], cmap='RdYlBu_r', norm=erk_norm_bg,
                    s=12, alpha=0.4, linewidths=0, zorder=2, label='other nuclei')

        # Paths of full-span tracks — grey so they don't clash with the ERK colormap
        for tid in sorted(spatial_tracks):
            traj = kine_df[kine_df['track_id'] == tid].sort_values('t')
            ax6.plot(traj[xcol], traj[ycol],
                     color='0.4', linewidth=0.8, alpha=0.5, zorder=3)

        # Final positions of full-span tracks — same ERK colormap, larger with outline
        ax6.scatter(bg_focus[xcol], bg_focus[ycol],
                    c=bg_focus['final_erk'], cmap='RdYlBu_r', norm=erk_norm_bg,
                    s=100, edgecolors='k', linewidths=1.0, zorder=4,
                    label=f'{GROUP} pre-ERK, full-span (n={len(spatial_tracks)})')

        ax6.set_xlabel(xlabel)
        ax6.set_ylabel(ylabel)
        ax6.set_title(title)
        ax6.set_aspect('equal')
        if 'z_um' in ycol:
            ax6.invert_yaxis()  # z=0 at top, dish bottom at high z

    sm_erk = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=erk_norm_bg)
    sm_erk.set_array([])
    fig6.colorbar(sm_erk, ax=axes6, label='Final ERK C/N', shrink=0.8)

    axes6[0].legend(fontsize=8, loc='upper left')

    fig6.suptitle(
        f'Final positions — "{GROUP}" pre-ERK group, full-span tracks  (n={len(spatial_tracks)})\n'
        f'All nuclei coloured by final ERK C/N  |  '
        f'outlined circles = full-span tracks  |  lines = track paths',
        fontsize=9
    )

    out_path6 = out_dir / f'preerk_{GROUP}_spatial_final_{version}.png'
    fig6.savefig(out_path6, dpi=150)
    plt.close()
    print(f'Saved: {out_path6}')
