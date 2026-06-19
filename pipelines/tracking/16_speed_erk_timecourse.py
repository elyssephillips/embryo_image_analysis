"""
Per-track speed and ERK C/N ratio timecourse, colored by ICM distance.

Two-panel figure: speed (top) and ERK C/N ratio (bottom), shared time axis.
Each track line is colored by its distance from the ICM (from volume_track_stats).

Run with:
  conda run -n napari_env python3 pipelines/tracking/16_speed_erk_timecourse.py
"""

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

SPEED_WINDOW = 3     # frames for rolling speed average (set to 1 to disable)
LINE_ALPHA   = 0.35
LINE_WIDTH   = 0.8
T_START      = 30    # first timepoint to show (implantation onset)

# ── Load data ─────────────────────────────────────────────────────────────────

erk_df  = pd.read_csv(out_dir / f'erk_cn_ratio_{version}.csv')
kine_df = pd.read_csv(out_dir / f'motion_kinematics_{version}.csv')
vstats  = pd.read_csv(out_dir / f'volume_track_stats_{version}.csv')[['track_id', 'icm_dist_um']]

kine_df = kine_df.sort_values(['track_id', 't'])
kine_df['speed_rolling'] = (
    kine_df.groupby('track_id')['speed_um_per_min']
    .transform(lambda s: s.rolling(SPEED_WINDOW, min_periods=1).mean())
)

df = (erk_df
      .merge(kine_df[['track_id', 't', 'speed_rolling']], on=['track_id', 't'], how='inner')
      .merge(vstats, on='track_id', how='left'))

df = df[df['t'] >= T_START].copy()
df['time_min'] = df['t'] * interval

print(f'Tracks:        {df["track_id"].nunique()}')
print(f'Timepoints:    {df["t"].nunique()}')
print(f'ICM dist range: {df["icm_dist_um"].min():.1f} – {df["icm_dist_um"].max():.1f} µm')

# ── Colormap by ICM distance ──────────────────────────────────────────────────

dist_min = vstats['icm_dist_um'].min()
dist_max = vstats['icm_dist_um'].max()
norm     = mcolors.Normalize(vmin=dist_min, vmax=dist_max)
cmap     = plt.cm.coolwarm_r  # red = near ICM, blue = far

track_icm = vstats.set_index('track_id')['icm_dist_um']

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, (ax_spd, ax_erk) = plt.subplots(
    2, 1, figsize=(14, 8), sharex=True,
    gridspec_kw={'hspace': 0.06}
)

for tid, grp in df.groupby('track_id'):
    if tid not in track_icm.index or np.isnan(track_icm[tid]):
        continue
    col = cmap(norm(track_icm[tid]))
    grp = grp.sort_values('t')
    ax_spd.plot(grp['time_min'], grp['speed_rolling'],
                color=col, alpha=LINE_ALPHA, linewidth=LINE_WIDTH)
    ax_erk.plot(grp['time_min'], grp['erk_cn_ratio'],
                color=col, alpha=LINE_ALPHA, linewidth=LINE_WIDTH)

ax_spd.set_ylabel(f'Speed  ({SPEED_WINDOW}-frame rolling mean, µm/min)')
ax_spd.set_title('Per-track speed and ERK C/N ratio over time — colored by distance from ICM')

ax_erk.set_ylabel('ERK C/N ratio')
ax_erk.set_xlabel('Time (min)')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax_spd, ax_erk], shrink=0.85, pad=0.02)
cbar.set_label('Distance from ICM (µm)\n← near        far →')

out_path = out_dir / f'speed_erk_timecourse_{version}.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out_path}')
