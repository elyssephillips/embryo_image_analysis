"""
Bin tracks by ICM distance, then plot mean motion parameter and ERK C/N over time.
One dual-axis figure per bin per parameter, with shaded SEM.

Available motion parameters (from kinematics and flow CSVs):
  speed_um_per_min      instantaneous frame-to-frame speed
  rel_speed_um_per_min  speed relative to local collective flow
  flow_alignment        alignment with local flow direction (cos θ, -1 to 1)
  turn_cos              cosine of turn angle between consecutive steps (directionality/persistence)
  net_disp_um           cumulative net displacement from track origin

Run with:
  conda run -n napari_env python3 pipelines/tracking/18_binned_track_plots.py

  # specific parameters
  conda run -n napari_env python3 pipelines/tracking/18_binned_track_plots.py \\
      --params speed_um_per_min turn_cos

  # change number of bins
  conda run -n napari_env python3 pipelines/tracking/18_binned_track_plots.py --n_bins 3
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

version  = cfg['tracking']['input_version']
out_dir  = Path(cfg['paths']['output_dir'])
interval = cfg['tracking']['frame_interval_min']

# Human-readable label and y-axis unit for each motion parameter
PARAM_META = {
    'speed_um_per_min':     ('Speed',                        'µm/min'),
    'rel_speed_um_per_min': ('Speed relative to flow',       'µm/min'),
    'flow_alignment':       ('Flow alignment (cos θ)',        '−1 to 1'),
    'turn_cos':             ('Directionality (cos turn angle)','−1 to 1'),
    'net_disp_um':          ('Net displacement from origin', 'µm'),
}
DEFAULT_PARAMS = list(PARAM_META.keys())

parser = argparse.ArgumentParser()
parser.add_argument('--params', nargs='+', default=DEFAULT_PARAMS,
                    choices=list(PARAM_META.keys()),
                    help='motion parameters to plot')
parser.add_argument('--n_bins',  type=int, default=4)
parser.add_argument('--t_start', type=int, default=30)
args = parser.parse_args()

N_BINS     = args.n_bins
T_START    = args.t_start
PARAMS     = args.params

COLOR_ERK = '#d6604d'
COLOR_MOT = '#2166ac'

# ── Load data ─────────────────────────────────────────────────────────────────

erk_df   = pd.read_csv(out_dir / f'erk_cn_ratio_{version}.csv')
kine_df  = pd.read_csv(out_dir / f'motion_kinematics_{version}.csv')
vstats   = pd.read_csv(out_dir / f'volume_track_stats_{version}.csv')[['track_id', 'icm_dist_um']]

# Compute turn angle cosine: cos of angle between consecutive displacement vectors
# 1 = same direction, 0 = right-angle turn, -1 = reversal
kine_df = kine_df.sort_values(['track_id', 't'])
for col in ['dz_um', 'dy_um', 'dx_um']:
    kine_df[f'{col}_prev'] = kine_df.groupby('track_id')[col].shift(1)
dot      = (kine_df['dz_um'] * kine_df['dz_um_prev'] +
            kine_df['dy_um'] * kine_df['dy_um_prev'] +
            kine_df['dx_um'] * kine_df['dx_um_prev'])
norm_cur = np.sqrt(kine_df['dz_um']**2 + kine_df['dy_um']**2 + kine_df['dx_um']**2)
norm_prv = np.sqrt(kine_df['dz_um_prev']**2 + kine_df['dy_um_prev']**2 + kine_df['dx_um_prev']**2)
kine_df['turn_cos'] = np.where((norm_cur > 0) & (norm_prv > 0), dot / (norm_cur * norm_prv), np.nan)
kine_df.drop(columns=['dz_um_prev', 'dy_um_prev', 'dx_um_prev'], inplace=True)

flow_path = out_dir / f'motion_flow_{version}.csv'
flow_cols = ['rel_speed_um_per_min', 'flow_alignment']
if flow_path.exists() and any(p in flow_cols for p in PARAMS):
    flow_df = pd.read_csv(flow_path)[['track_id', 't'] + flow_cols]
    kine_df = kine_df.merge(flow_df, on=['track_id', 't'], how='left')

motion_cols = [p for p in PARAMS if p in kine_df.columns]
missing = [p for p in PARAMS if p not in kine_df.columns]
if missing:
    print(f'Warning: columns not found, skipping: {missing}')

kine_df = kine_df.sort_values(['track_id', 't'])

df = (erk_df
      .merge(kine_df[['track_id', 't'] + motion_cols], on=['track_id', 't'], how='inner')
      .merge(vstats, on='track_id', how='left'))

df = df[df['t'] >= T_START].dropna(subset=['icm_dist_um']).copy()
df['time_min'] = df['t'] * interval

# ── Bin tracks by ICM distance ────────────────────────────────────────────────

track_dist = vstats.dropna(subset=['icm_dist_um']).set_index('track_id')['icm_dist_um']
bin_labels = [f'Q{i+1}' for i in range(N_BINS)]

df['dist_bin'] = pd.qcut(df['icm_dist_um'], q=N_BINS, labels=bin_labels)
_, edges = pd.qcut(track_dist, q=N_BINS, retbins=True)

cmap_bins  = plt.cm.coolwarm_r
bin_colors = [cmap_bins(i / max(N_BINS - 1, 1)) for i in range(N_BINS)]

# ── One figure per (bin, param) ───────────────────────────────────────────────

def agg(series):
    return pd.Series({'mean': series.mean(), 'sem': series.sem()})

for param in motion_cols:
    param_label, param_unit = PARAM_META.get(param, (param, ''))
    ylabel_mot = f'{param_label} ({param_unit})' if param_unit else param_label
    print(f'\n[{param}]')

    # Pre-compute timeseries for all bins so we can set shared axis limits
    all_ts = {}
    for bin_label in bin_labels:
        sub = df[df['dist_bin'] == bin_label].dropna(subset=[param, 'erk_cn_ratio'])
        mot_ts = sub.groupby('time_min')[param].apply(agg).unstack()
        erk_ts = sub.groupby('time_min')['erk_cn_ratio'].apply(agg).unstack()
        all_ts[bin_label] = (mot_ts, erk_ts)

    # Shared y-limits: include mean ± sem across all bins
    mot_vals = np.concatenate([(ts[0]['mean'] + ts[0]['sem']).values for ts in all_ts.values()] +
                               [(ts[0]['mean'] - ts[0]['sem']).values for ts in all_ts.values()])
    erk_vals = np.concatenate([(ts[1]['mean'] + ts[1]['sem']).values for ts in all_ts.values()] +
                               [(ts[1]['mean'] - ts[1]['sem']).values for ts in all_ts.values()])
    mot_vals = mot_vals[np.isfinite(mot_vals)]
    erk_vals = erk_vals[np.isfinite(erk_vals)]
    pad = lambda lo, hi: (lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    mot_ylim = pad(mot_vals.min(), mot_vals.max())
    erk_ylim = pad(erk_vals.min(), erk_vals.max())

    for i, (bin_label, color) in enumerate(zip(bin_labels, bin_colors)):
        sub      = df[df['dist_bin'] == bin_label].dropna(subset=[param, 'erk_cn_ratio'])
        n_tracks = sub['track_id'].nunique()
        dist_lo  = edges[i]
        dist_hi  = edges[i + 1]
        title    = (f'{param_label}  |  ICM bin {bin_label} '
                    f'({dist_lo:.0f}–{dist_hi:.0f} µm)  |  n={n_tracks} tracks')
        print(f'  {bin_label}: {n_tracks} tracks, {len(sub)} rows')

        mot_ts, erk_ts = all_ts[bin_label]
        t = mot_ts.index.values

        fig, ax_left  = plt.subplots(figsize=(10, 4))
        ax_right = ax_left.twinx()

        ax_left.plot(t, mot_ts['mean'], color=COLOR_MOT, linewidth=2, label=param_label)
        ax_left.fill_between(t,
                             mot_ts['mean'] - mot_ts['sem'],
                             mot_ts['mean'] + mot_ts['sem'],
                             color=COLOR_MOT, alpha=0.2)

        ax_right.plot(t, erk_ts['mean'], color=COLOR_ERK, linewidth=2, label='ERK C/N')
        ax_right.fill_between(t,
                              erk_ts['mean'] - erk_ts['sem'],
                              erk_ts['mean'] + erk_ts['sem'],
                              color=COLOR_ERK, alpha=0.2)

        ax_left.set_ylim(mot_ylim)
        ax_right.set_ylim(erk_ylim)

        ax_left.set_xlabel('Time (min)')
        ax_left.set_ylabel(ylabel_mot, color=COLOR_MOT)
        ax_right.set_ylabel('ERK C/N ratio', color=COLOR_ERK)
        ax_left.tick_params(axis='y',  labelcolor=COLOR_MOT)
        ax_right.tick_params(axis='y', labelcolor=COLOR_ERK)
        ax_left.set_title(title, pad=8)

        lines  = ax_left.get_lines() + ax_right.get_lines()
        labels = [l.get_label() for l in lines]
        ax_left.legend(lines, labels, loc='upper left', fontsize=9)

        plt.tight_layout()
        out_name = f'binned_timecourse_{param}_{bin_label}_{version}.png'
        fig.savefig(out_dir / out_name, dpi=150)
        plt.close()
        print(f'  Saved: {out_name}')

print('\nDone.')
