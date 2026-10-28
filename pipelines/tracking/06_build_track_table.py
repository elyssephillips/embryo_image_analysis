"""
Build the per-timepoint and per-track tables that 07 (track plots) and
08 (comparisons) read.

Joins, on (t, label_id):
  tracks_{version}.csv        track_id + registered centroid (z/y/x_um_reg)
  features_registered_csv     nuclear volume (area_um3) for every nucleus
  erk_cn_ratio_{version}.csv  ERK C/N (05_erk_cn_ratio.py)
  icm_distance_csv            signed distance to the ICM hull edge (01_extract_icm_region.py;
                              optional - columns are NaN if the config has none)

and adds radial_dist_um: XY distance from that timepoint's embryo centroid,
the mean registered XY of *all* segmented nuclei at t (not just tracked
ones, so incomplete tracking doesn't shift the centre).

Per-track windows (config `biology` / `analysis`):
  pre     t_start <= t < implantation_onset_t
  during  implantation_onset_t <= t <= analysis.during_end_t (default t_end)
  end     the last analysis.end_frames frames of the movie (t_end - end_frames < t <= t_end),
          the same window for every track - tracks that stop earlier get NaN
A window value is NaN when the track has fewer than analysis.min_window_frames
frames in it.

Outputs (output_dir)
-------
  track_table_{version}.csv    one row per (track_id, t)
  track_summary_{version}.csv  one row per track

Run with:
  conda run -n napari_env python3 pipelines/tracking/06_build_track_table.py [--config X.yaml]
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.tracking_config import (load_tracking_config, tracks_csv, implantation_onset,
                                 analysis_windows)
cfg, CONFIG_PATH = load_tracking_config()

version  = cfg['tracking']['input_version']
out_dir  = Path(cfg['paths']['output_dir'])
interval = cfg['tracking']['frame_interval_min']
T_START  = cfg['tracking']['t_start']
T_END    = cfg['tracking']['t_end']
MIN_LEN  = cfg['tracking']['min_track_length']
win      = analysis_windows(cfg)

# ── Load ───────────────────────────────────────────────────────────────────────

tracks = pd.read_csv(tracks_csv(cfg))[['track_id', 't', 'label_id',
                                        'z_um_reg', 'y_um_reg', 'x_um_reg']]
feat = pd.read_csv(cfg['paths']['features_registered_csv'])
erk  = pd.read_csv(out_dir / f'erk_cn_ratio_{version}.csv')[['t', 'label_id', 'erk_cn_ratio',
                                                             'erk_nuc_mean']]

icm_path = cfg['paths'].get('icm_distance_csv')
if icm_path and Path(icm_path).exists():
    icm = pd.read_csv(icm_path)[['t', 'label_id', 'icm_edge_dist_um']]
    icm = icm.rename(columns={'icm_edge_dist_um': 'icm_dist_um'})
else:
    print(f'No icm_distance_csv ({icm_path}) - icm_dist_um will be NaN. '
          f'Run 01_extract_icm_region.py first.')
    icm = pd.DataFrame(columns=['t', 'label_id', 'icm_dist_um'])

for df in (tracks, feat, erk, icm):
    df['label_id'] = df['label_id'].astype('Int64')

# ── Per-timepoint table ────────────────────────────────────────────────────────

centre = (feat.groupby('t')[['x_um_reg', 'y_um_reg']].mean()
              .rename(columns={'x_um_reg': 'cx', 'y_um_reg': 'cy'}))

table = (tracks
         .merge(feat[['t', 'label_id', 'area_um3']], on=['t', 'label_id'], how='left')
         .merge(erk, on=['t', 'label_id'], how='left')
         .merge(icm, on=['t', 'label_id'], how='left')
         .join(centre, on='t'))
table['radial_dist_um'] = np.hypot(table['x_um_reg'] - table['cx'],
                                   table['y_um_reg'] - table['cy'])
table['time_min'] = table['t'] * interval
table = table.drop(columns=['cx', 'cy'])

table = table[(table['t'] >= T_START) & (table['t'] <= T_END)]
lengths = table.groupby('track_id')['t'].count()
table = table[table['track_id'].isin(lengths[lengths >= MIN_LEN].index)]
table = table.sort_values(['track_id', 't']).reset_index(drop=True)

for col in ['area_um3', 'erk_cn_ratio', 'icm_dist_um']:
    n_miss = table[col].isna().sum()
    if n_miss:
        print(f'  {col}: {n_miss}/{len(table)} rows unmatched')

table.to_csv(out_dir / f'track_table_{version}.csv', index=False)
print(f'Saved track_table_{version}.csv  ({table["track_id"].nunique()} tracks '
      f'≥ {MIN_LEN} frames, {len(table)} rows)')

# ── Per-track summary ──────────────────────────────────────────────────────────

def window_mean(df, col, name):
    """Per-track mean of col over df, NaN where fewer than min_window_frames values."""
    g = df.groupby('track_id')[col]
    return g.mean().where(g.count() >= win['min_frames']).rename(name)

pre    = table[(table['t'] >= T_START) & (table['t'] < win['onset'])]
during = table[(table['t'] >= win['onset']) & (table['t'] <= win['during_end'])]
end    = table[table['t'] > T_END - win['end_frames']]

span = table.groupby('track_id')['t'].agg(n_frames='count', t_first='min', t_last='max')

summary = pd.concat([
    span,
    window_mean(end,    'area_um3',       'vol_end'),
    window_mean(pre,    'radial_dist_um', 'radial_pre'),
    window_mean(end,    'radial_dist_um', 'radial_end'),
    window_mean(pre,    'icm_dist_um',    'icm_dist_pre'),
    window_mean(end,    'icm_dist_um',    'icm_dist_end'),
    window_mean(pre,    'z_um_reg',       'z_pre'),
    window_mean(end,    'z_um_reg',       'z_end'),
    window_mean(pre,    'erk_cn_ratio',   'erk_pre'),
    window_mean(during, 'erk_cn_ratio',   'erk_during'),
], axis=1).reset_index()

summary.to_csv(out_dir / f'track_summary_{version}.csv', index=False)
print(f'Saved track_summary_{version}.csv  ({len(summary)} tracks)')
print(f'  windows: pre t{T_START}-{win["onset"] - 1}, '
      f'during t{win["onset"]}-{win["during_end"]}, end t{T_END - win["end_frames"] + 1}-{T_END}, '
      f'≥ {win["min_frames"]} frames per window')
print(summary.drop(columns='track_id').notna().sum().rename('n tracks').to_string())
