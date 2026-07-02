"""
Nuclear ERK analysis using normalized nuclear intensity instead of C/N ratio.

With an ERK translocation reporter, ERK "on" = nuclear export = nucleus appears
dark.  The minimum nuclear intensity for each track represents its most active ERK
state ("on" / empty nucleus).  We normalize each track to its own 5th-percentile
nuclear intensity (robust minimum) so that:
  1.0  = nucleus maximally empty  (ERK fully on)
  > 1  = nucleus filling up       (ERK going off)

Nuclei are then grouped by their mean normalized nuclear intensity before
implantation onset (t < T_SPLIT): low = already active pre-implantation,
high = nucleus still full / ERK off pre-implantation.

Mirrors 21_preerk_individual_tracks.py — run both to check whether trends seen
in the C/N ratio hold when cytoplasmic signal is removed.

Usage
-----
  conda run -n napari_env python3 pipelines/tracking/22_nuclear_erk_analysis.py
  conda run -n napari_env python3 pipelines/tracking/22_nuclear_erk_analysis.py --group high
  conda run -n napari_env python3 pipelines/tracking/22_nuclear_erk_analysis.py --t_split 25 --n_groups 4
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

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--group',    default='low', choices=['low', 'mid', 'high'])
parser.add_argument('--n_groups', type=int, default=3)
parser.add_argument('--t_split',  type=int, default=30)
parser.add_argument('--min_pre',  type=int, default=3)
parser.add_argument('--on_pct',   type=float, default=5.0,
                    help='percentile of nuclear intensity used as the "on" (empty) reference')
args = parser.parse_args()

GROUP    = args.group
N_GROUPS = args.n_groups
T_SPLIT  = args.t_split
MIN_PRE  = args.min_pre
ON_PCT   = args.on_pct

GROUP_IDX    = {'low': 0, 'mid': N_GROUPS // 2, 'high': N_GROUPS - 1}
group_idx    = GROUP_IDX[GROUP]
group_labels = ['low', 'mid', 'high'] if N_GROUPS == 3 else [f'G{i+1}' for i in range(N_GROUPS)]

SIGNAL = 'nuc_norm'   # column name used throughout

# ── Load & normalize ──────────────────────────────────────────────────────────

erk_df = pd.read_csv(out_dir / f'erk_cn_ratio_{version}.csv')
vstats = pd.read_csv(out_dir / f'volume_track_stats_{version}.csv')[['track_id', 'icm_dist_um']]

erk_df = erk_df.merge(vstats, on='track_id', how='left')
erk_df['time_min'] = erk_df['t'] * interval

# Per-track "on" reference = low percentile of nuclear intensity over full timecourse
on_ref = (erk_df.groupby('track_id')['erk_nuc_mean']
                .quantile(ON_PCT / 100.0)
                .rename('on_ref'))
erk_df = erk_df.join(on_ref, on='track_id')

# Exclude tracks whose "on" reference is zero or negative (bad segmentation)
valid_on = on_ref[on_ref > 0].index
erk_df = erk_df[erk_df['track_id'].isin(valid_on)].copy()
erk_df[SIGNAL] = erk_df['erk_nuc_mean'] / erk_df['on_ref']

print(f'Tracks after filtering zero on-ref: {erk_df["track_id"].nunique()}')
print(f'Normalized nuclear intensity (1.0 = on/empty):')
print(f'  range [{erk_df[SIGNAL].quantile(0.01):.2f}, {erk_df[SIGNAL].quantile(0.99):.2f}]')

# ── Pre-ERK grouping ──────────────────────────────────────────────────────────

pre = erk_df[erk_df['t'] < T_SPLIT].copy()
pre_mean = (pre.groupby('track_id')[SIGNAL]
               .agg(pre_nuc_mean='mean', pre_nuc_n='count')
               .reset_index())
pre_mean = pre_mean[pre_mean['pre_nuc_n'] >= MIN_PRE]

pre_mean['preerk_group'] = pd.qcut(
    pre_mean['pre_nuc_mean'], q=N_GROUPS,
    labels=group_labels[:N_GROUPS] if N_GROUPS == 3 else None
)

target_label  = group_labels[group_idx] if N_GROUPS == 3 else pre_mean['preerk_group'].cat.categories[group_idx]
target_tracks = pre_mean[pre_mean['preerk_group'] == target_label]['track_id'].tolist()

print(f'\nPre-implantation nuclear groups (t < {T_SPLIT}, {ON_PCT}th-pct normalised):')
for lbl in pre_mean['preerk_group'].cat.categories:
    n    = (pre_mean['preerk_group'] == lbl).sum()
    vals = pre_mean.loc[pre_mean['preerk_group'] == lbl, 'pre_nuc_mean']
    print(f'  {lbl}: n={n}  norm-nuc range [{vals.min():.3f}, {vals.max():.3f}]')
print(f'\nPlotting {len(target_tracks)} nuclei in the "{GROUP}" group')

# ── Colormaps ─────────────────────────────────────────────────────────────────

dist_min = vstats['icm_dist_um'].min()
dist_max = vstats['icm_dist_um'].max()
norm_icm = mcolors.Normalize(vmin=dist_min, vmax=dist_max)
cmap_icm = plt.cm.coolwarm_r
track_icm = vstats.set_index('track_id')['icm_dist_um']

pre_erk_lookup = pre_mean.set_index('track_id')['pre_nuc_mean']
group_erk_vals = pre_erk_lookup[pre_erk_lookup.index.isin(target_tracks)]
erk_lo, erk_hi = group_erk_vals.min(), group_erk_vals.max()
norm_erk = mcolors.Normalize(vmin=erk_lo, vmax=erk_hi)
cmap_erk = plt.cm.YlOrRd

n_tracks   = len(target_tracks)
split_time = T_SPLIT * interval

# ── Shared axis limits ────────────────────────────────────────────────────────

raw_vals = erk_df[SIGNAL].dropna()
_y_lo = raw_vals.quantile(0.01)
_y_hi = raw_vals.quantile(0.99)
_pad  = 0.05 * (_y_hi - _y_lo)
ylim_raw  = (_y_lo - _pad, _y_hi + _pad)
ylim_norm = None
xlim      = (0, erk_df['time_min'].max())

# ── Plot 1: overlay, colored by pre-nuc level ─────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))

ax.axvspan(erk_df['time_min'].min(), split_time, color='0.92', zorder=0)
ax.axvline(split_time, color='0.4', linewidth=0.8, linestyle=':', zorder=1)
ax.axhline(1.0, color='0.4', linewidth=0.8, linestyle='--', alpha=0.5, zorder=1,
           label='on / empty nucleus')

for tid in sorted(target_tracks):
    grp      = erk_df[erk_df['track_id'] == tid].sort_values('t')
    pre_val  = pre_erk_lookup.get(tid, np.nan)
    line_col = cmap_erk(norm_erk(pre_val)) if not np.isnan(pre_val) else '0.5'
    ax.plot(grp['time_min'], grp[SIGNAL],
            color=line_col, linewidth=0.9, alpha=0.7, zorder=2)

group_data = erk_df[erk_df['track_id'].isin(target_tracks)]
mean_ts = group_data.groupby('time_min')[SIGNAL].agg(['mean', 'sem'])
ax.plot(mean_ts.index, mean_ts['mean'],
        color='k', linewidth=2.5, zorder=4, label='group mean')
ax.fill_between(mean_ts.index,
                mean_ts['mean'] - mean_ts['sem'],
                mean_ts['mean'] + mean_ts['sem'],
                color='k', alpha=0.15, zorder=3, label='± SEM')

sm = plt.cm.ScalarMappable(cmap=cmap_erk, norm=norm_erk)
sm.set_array([])
fig.colorbar(sm, ax=ax, label=f'pre-implantation norm. nuclear intensity', shrink=0.85)

ax.set_xlim(xlim)
ax.set_ylim(ylim_raw)
ax.set_xlabel('Time (min)')
ax.set_ylabel(f'Nuclear intensity  (norm. to {ON_PCT}th-pct = "on")')
ax.set_title(
    f'Normalised nuclear ERK — "{GROUP}" pre-implantation group  (n={n_tracks})\n'
    f'1.0 = nucleus empty (ERK on)  |  higher = nucleus filling (ERK off)  |  '
    f'shaded = pre-implantation (t < {T_SPLIT})'
)
ax.legend(fontsize=9)

plt.tight_layout()
out_path = out_dir / f'nucnorm_{GROUP}_overlay_{version}.png'
fig.savefig(out_path, dpi=150)
plt.close()
print(f'Saved: {out_path}')

# ── Plot 2: individual panels, colored by ICM distance ───────────────────────

n_cols   = min(5, n_tracks)
n_rows   = int(np.ceil(n_tracks / n_cols))
fig2, axes = plt.subplots(n_rows, n_cols,
                           figsize=(4.5 * n_cols, 3 * n_rows),
                           constrained_layout=True)
axes_flat = np.array(axes).flatten() if n_tracks > 1 else [axes]

for ax, tid in zip(axes_flat, sorted(target_tracks)):
    grp      = erk_df[erk_df['track_id'] == tid].sort_values('t')
    icm      = track_icm.get(tid, np.nan)
    line_col = cmap_icm(norm_icm(icm)) if not np.isnan(icm) else '0.5'
    pre_val  = pre_erk_lookup.get(tid, np.nan)

    ax.axvspan(grp['time_min'].min(), split_time, color='0.92', zorder=0)
    ax.axvline(split_time, color='0.4', linewidth=0.8, linestyle=':', zorder=1)
    ax.axhline(1.0, color='0.4', linewidth=0.6, linestyle='--', alpha=0.4, zorder=1)
    ax.plot(grp['time_min'], grp[SIGNAL],
            color=line_col, linewidth=1.4, zorder=2)

    if not np.isnan(pre_val):
        pre_t = grp.loc[grp['t'] < T_SPLIT, 'time_min']
        if not pre_t.empty:
            ax.hlines(pre_val, pre_t.min(), split_time,
                      color=line_col, linewidth=1, linestyle='--', alpha=0.6, zorder=3)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim_raw)

    icm_str = f'{icm:.0f} µm' if not np.isnan(icm) else 'n/a'
    pre_str = f'pre={pre_val:.3f}' if not np.isnan(pre_val) else ''
    ax.set_title(f'track {tid}  |  {icm_str} ICM  |  {pre_str}',
                 fontsize=8, pad=3)
    ax.set_xlabel('Time (min)', fontsize=7)
    ax.set_ylabel('Norm. nuclear', fontsize=7)
    ax.tick_params(labelsize=6)

for ax in axes_flat[n_tracks:]:
    ax.set_visible(False)

sm2 = plt.cm.ScalarMappable(cmap=cmap_icm, norm=norm_icm)
sm2.set_array([])
fig2.colorbar(sm2, ax=axes_flat.tolist(),
              label='ICM distance (µm)\n← near        far →', shrink=0.6)
fig2.suptitle(
    f'Normalised nuclear ERK per nucleus — "{GROUP}" pre-implantation group  (n={n_tracks})\n'
    f'1.0 = on / empty  |  colour = ICM distance  |  dashed = pre-implantation mean',
    fontsize=9
)

out_path2 = out_dir / f'nucnorm_{GROUP}_individual_{version}.png'
fig2.savefig(out_path2, dpi=150)
plt.close()
print(f'Saved: {out_path2}')

# ── Full-span tracks ──────────────────────────────────────────────────────────

t_max_global  = erk_df['t'].max()
MIN_SPAN_FRAC = 0.8
track_tmax    = erk_df.groupby('track_id')['t'].max()
full_tracks   = [
    tid for tid in target_tracks
    if tid in track_tmax.index and track_tmax[tid] >= MIN_SPAN_FRAC * t_max_global
]
print(f'\nFull-span tracks in "{GROUP}" group: {len(full_tracks)}')

# ── Plot 3: normalized overlay — full-span tracks ─────────────────────────────
# Here "normalized" means each track divided by its own pre-implantation mean,
# so all lines start at 1.0 and deviations reflect change from that baseline.

if full_tracks:
    norm_traces = []
    fig3, ax3 = plt.subplots(figsize=(12, 5))

    ax3.axvspan(erk_df['time_min'].min(), split_time, color='0.92', zorder=0)
    ax3.axvline(split_time, color='0.4', linewidth=0.8, linestyle=':', zorder=1)
    ax3.axhline(1.0, color='0.4', linewidth=0.8, linestyle='--', alpha=0.6, zorder=1)

    for tid in sorted(full_tracks):
        grp     = erk_df[erk_df['track_id'] == tid].sort_values('t')
        pre_val = pre_erk_lookup.get(tid, np.nan)
        if np.isnan(pre_val) or pre_val == 0:
            continue
        grp = grp.copy()
        grp['sig_norm'] = grp[SIGNAL] / pre_val

        icm      = track_icm.get(tid, np.nan)
        line_col = cmap_icm(norm_icm(icm)) if not np.isnan(icm) else '0.5'
        ax3.plot(grp['time_min'], grp['sig_norm'],
                 color=line_col, linewidth=0.9, alpha=0.75, zorder=2)
        norm_traces.append(grp[['time_min', 'sig_norm']].copy())

    if norm_traces:
        all_norm = pd.concat(norm_traces)
        _yn_lo = all_norm['sig_norm'].quantile(0.01)
        _yn_hi = all_norm['sig_norm'].quantile(0.99)
        _yn_pad = 0.05 * (_yn_hi - _yn_lo)
        ylim_norm = (_yn_lo - _yn_pad, _yn_hi + _yn_pad)

        mean_norm = all_norm.groupby('time_min')['sig_norm'].agg(['mean', 'sem'])
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
    ax3.set_ylabel('Norm. nuclear  (÷ pre-implantation mean)')
    ax3.set_title(
        f'Normalised nuclear ERK — "{GROUP}" group, full-span tracks  (n={len(full_tracks)})\n'
        f'Each track ÷ own pre-implantation mean  |  1.0 = pre-implantation baseline  |  '
        f'colour = ICM distance'
    )
    ax3.legend(fontsize=9)
    plt.tight_layout()
    out_path3 = out_dir / f'nucnorm_{GROUP}_normalized_overlay_{version}.png'
    fig3.savefig(out_path3, dpi=150)
    plt.close()
    print(f'Saved: {out_path3}')

# ── Plot 4: individual panels, normalized — full-span tracks ──────────────────

if full_tracks:
    norm_data = {}
    for tid in sorted(full_tracks):
        grp     = erk_df[erk_df['track_id'] == tid].sort_values('t')
        pre_val = pre_erk_lookup.get(tid, np.nan)
        if np.isnan(pre_val) or pre_val == 0:
            continue
        grp = grp.copy()
        grp['sig_norm'] = grp[SIGNAL] / pre_val
        norm_data[tid] = grp

    n_full  = len(norm_data)
    n_cols4 = min(5, n_full)
    n_rows4 = int(np.ceil(n_full / n_cols4))

    fig4, axes4 = plt.subplots(n_rows4, n_cols4,
                                figsize=(4.5 * n_cols4, 3 * n_rows4),
                                constrained_layout=True)
    axes4_flat = np.array(axes4).flatten() if n_full > 1 else [axes4]

    for ax, (tid, grp) in zip(axes4_flat, norm_data.items()):
        icm      = track_icm.get(tid, np.nan)
        line_col = cmap_icm(norm_icm(icm)) if not np.isnan(icm) else '0.5'
        pre_val  = pre_erk_lookup.get(tid, np.nan)

        ax.axvspan(grp['time_min'].min(), split_time, color='0.92', zorder=0)
        ax.axvline(split_time, color='0.4', linewidth=0.8, linestyle=':', zorder=1)
        ax.axhline(1.0, color='0.4', linewidth=0.6, linestyle='--', alpha=0.5, zorder=1)
        ax.plot(grp['time_min'], grp['sig_norm'],
                color=line_col, linewidth=1.4, zorder=2)

        ax.set_xlim(xlim)
        if ylim_norm:
            ax.set_ylim(ylim_norm)

        icm_str = f'{icm:.0f} µm' if not np.isnan(icm) else 'n/a'
        pre_str = f'pre={pre_val:.3f}' if not np.isnan(pre_val) else ''
        ax.set_title(f'track {tid}  |  {icm_str} ICM  |  {pre_str}',
                     fontsize=8, pad=3)
        ax.set_xlabel('Time (min)', fontsize=7)
        ax.set_ylabel('Norm. nuclear (÷ pre)', fontsize=7)
        ax.tick_params(labelsize=6)

    for ax in axes4_flat[n_full:]:
        ax.set_visible(False)

    sm4 = plt.cm.ScalarMappable(cmap=cmap_icm, norm=norm_icm)
    sm4.set_array([])
    fig4.colorbar(sm4, ax=list(axes4_flat),
                  label='ICM distance (µm)\n← near        far →', shrink=0.6)
    fig4.suptitle(
        f'Normalised nuclear ERK per nucleus — "{GROUP}" group, full-span tracks  (n={n_full})\n'
        f'1.0 = pre-implantation mean  |  colour = ICM distance  |  '
        f'shaded = pre-implantation (t < {T_SPLIT})',
        fontsize=9
    )

    out_path4 = out_dir / f'nucnorm_{GROUP}_normalized_individual_{version}.png'
    fig4.savefig(out_path4, dpi=150)
    plt.close()
    print(f'Saved: {out_path4}')

# ── Plot 5: group means — all pre-implantation groups on one axis ─────────────

group_colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, N_GROUPS))
all_group_labels = list(pre_mean['preerk_group'].cat.categories)

fig5, ax5 = plt.subplots(figsize=(12, 5))
ax5.axvspan(erk_df['time_min'].min(), split_time, color='0.92', zorder=0)
ax5.axvline(split_time, color='0.4', linewidth=0.8, linestyle=':', zorder=1)
ax5.axhline(1.0, color='0.4', linewidth=0.8, linestyle='--', alpha=0.5, zorder=1,
            label='on / empty nucleus')

for lbl, col in zip(all_group_labels, group_colors):
    grp_tracks = pre_mean.loc[pre_mean['preerk_group'] == lbl, 'track_id']
    grp_data   = erk_df[erk_df['track_id'].isin(grp_tracks)]
    ts = grp_data.groupby('time_min')[SIGNAL].agg(['mean', 'sem'])
    n  = grp_tracks.nunique()
    ax5.plot(ts.index, ts['mean'], color=col, linewidth=2,
             label=f'{lbl}  (n={n})', zorder=3)
    ax5.fill_between(ts.index,
                     ts['mean'] - ts['sem'],
                     ts['mean'] + ts['sem'],
                     color=col, alpha=0.2, zorder=2)

ax5.set_xlim(xlim)
ax5.set_ylim(ylim_raw)
ax5.set_xlabel('Time (min)')
ax5.set_ylabel(f'Nuclear intensity  (norm. to {ON_PCT}th-pct = "on")')
ax5.set_title(
    f'Mean normalised nuclear ERK by pre-implantation group  '
    f'({N_GROUPS} quantile groups, t < {T_SPLIT})\n'
    f'Shaded = ± SEM  |  1.0 = nucleus empty (ERK on)  |  '
    f'vertical line = implantation onset'
)
ax5.legend(fontsize=9)
plt.tight_layout()
out_path5 = out_dir / f'nucnorm_group_means_{version}.png'
fig5.savefig(out_path5, dpi=150)
plt.close()
print(f'Saved: {out_path5}')

# ── Plot 6: spatial — full-span tracks vs all nuclei, final ERK coloring ──────

if full_tracks:
    kine_df = pd.read_csv(out_dir / f'motion_kinematics_{version}.csv')

    last_pos = (kine_df.sort_values('t')
                       .groupby('track_id')[['x_um_reg', 'y_um_reg', 'z_um_reg']]
                       .last().reset_index())
    last_nuc = (erk_df.sort_values('t')
                      .groupby('track_id')[SIGNAL]
                      .last().reset_index()
                      .rename(columns={SIGNAL: 'final_nuc'}))

    bg       = last_pos.merge(last_nuc, on='track_id', how='left')
    bg_other = bg[~bg['track_id'].isin(full_tracks)]
    bg_focus = bg[bg['track_id'].isin(full_tracks)]

    nuc_norm_bg = mcolors.Normalize(
        vmin=bg['final_nuc'].quantile(0.02),
        vmax=bg['final_nuc'].quantile(0.98)
    )

    projections = [
        ('x_um_reg', 'y_um_reg', 'X (µm)', 'Y (µm)', 'Top-down (XY)'),
        ('x_um_reg', 'z_um_reg', 'X (µm)', 'Z (µm)', 'Side view (XZ)'),
    ]

    fig6, axes6 = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    for ax6, (xcol, ycol, xlabel, ylabel, title) in zip(axes6, projections):
        ax6.scatter(bg_other[xcol], bg_other[ycol],
                    c=bg_other['final_nuc'], cmap='RdYlBu_r', norm=nuc_norm_bg,
                    s=12, alpha=0.4, linewidths=0, zorder=2, label='other nuclei')

        for tid in sorted(full_tracks):
            traj = kine_df[kine_df['track_id'] == tid].sort_values('t')
            ax6.plot(traj[xcol], traj[ycol],
                     color='0.4', linewidth=0.8, alpha=0.5, zorder=3)

        ax6.scatter(bg_focus[xcol], bg_focus[ycol],
                    c=bg_focus['final_nuc'], cmap='RdYlBu_r', norm=nuc_norm_bg,
                    s=100, edgecolors='k', linewidths=1.0, zorder=4,
                    label=f'low pre-nuc, full-span (n={len(full_tracks)})')

        ax6.set_xlabel(xlabel)
        ax6.set_ylabel(ylabel)
        ax6.set_title(title)
        ax6.set_aspect('equal')
        if 'z_um' in ycol:
            ax6.invert_yaxis()

    sm6 = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=nuc_norm_bg)
    sm6.set_array([])
    fig6.colorbar(sm6, ax=axes6, label='Final normalised nuclear intensity', shrink=0.8)
    axes6[0].legend(fontsize=8, loc='upper left')
    fig6.suptitle(
        f'Final positions — "{GROUP}" pre-implantation group, full-span tracks  '
        f'(n={len(full_tracks)})\n'
        f'All nuclei coloured by final normalised nuclear intensity  |  '
        f'outlined circles = full-span tracks  |  lines = track paths',
        fontsize=9
    )

    out_path6 = out_dir / f'nucnorm_{GROUP}_spatial_final_{version}.png'
    fig6.savefig(out_path6, dpi=150)
    plt.close()
    print(f'Saved: {out_path6}')
