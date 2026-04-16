"""
Layer 3 — Local flow field analysis.

For each nucleus at each timepoint, computes:
  - local flow  : mean velocity of k nearest spatial neighbors (collective tissue motion)
  - autonomous motion : nucleus velocity minus local flow (cell-intrinsic motion)

Usage
-----
  python 09_flow_field.py
  python 09_flow_field.py configs/tracking/control_2.yaml

Outputs  (written to output_dir)
-------
  motion_flow_{version}.csv          — kinematics + local/autonomous flow columns
  flow_field_mean.png                — mean flow vectors over full timecourse
  flow_field_animation_{version}.mp4 — time-varying quiver animation
"""

import sys
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, FFMpegWriter
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.motion import compute_local_flow

# ── config ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('config', nargs='?',
                    default='configs/tracking/dataset001_implantation.yaml')
args = parser.parse_args()

CONFIG_PATH = REPO_ROOT / args.config
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

OUTPUT_DIR  = Path(cfg['paths']['output_dir'])
VERSION     = cfg['tracking']['input_version']
FRAME_MIN   = cfg['tracking']['frame_interval_min']
DATASET_ID  = cfg['project']['dataset_id']
K           = cfg['motion']['flow_field']['k_neighbors']

FPS         = 8    # animation playback speed

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'Dataset:     {DATASET_ID}')
print(f'k_neighbors: {K}')

# ── load kinematics ───────────────────────────────────────────────────────────
kin = pd.read_csv(OUTPUT_DIR / f'motion_kinematics_{VERSION}.csv')
has_orig = 'z_um_orig' in kin.columns
z_col = 'z_um_orig' if has_orig else 'z_um'
y_col = 'y_um_orig' if has_orig else 'y_um'
x_col = 'x_um_orig' if has_orig else 'x_um'

# ── compute local flow ────────────────────────────────────────────────────────
print('Computing local flow field...')
flow_df = compute_local_flow(kin, K, FRAME_MIN, z_col=z_col, y_col=y_col, x_col=x_col)

flow_df.to_csv(OUTPUT_DIR / f'motion_flow_{VERSION}.csv', index=False)
print(f'Saved motion_flow_{VERSION}.csv')

# ── plot 1: mean flow field over full timecourse ──────────────────────────────
# Average local and autonomous vectors per track (across all timepoints)
mean_per_track = (
    flow_df
    .dropna(subset=['local_vy_um', 'local_vx_um'])
    .groupby('track_id')
    .agg(
        y=   (y_col, 'mean'),
        x=   (x_col, 'mean'),
        local_vy    = ('local_vy_um',            'mean'),
        local_vx    = ('local_vx_um',            'mean'),
        local_speed = ('local_speed_um_per_min', 'mean'),
        rel_vy      = ('rel_vy_um',              'mean'),
        rel_vx      = ('rel_vx_um',              'mean'),
        rel_speed   = ('rel_speed_um_per_min',   'mean'),
    )
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
cmap  = plt.cm.plasma
scale = 10   # quiver arrow scale — increase to shorten arrows

for ax, vy_col, vx_col, speed_col, label, title in [
    (axes[0], 'local_vy', 'local_vx', 'local_speed',
     'Mean collective speed (µm/min)', 'Collective (local) flow'),
    (axes[1], 'rel_vy',   'rel_vx',   'rel_speed',
     'Mean autonomous speed (µm/min)', 'Autonomous motion'),
]:
    vmax = mean_per_track[speed_col].quantile(0.95)
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    sc = ax.scatter(
        mean_per_track['x'], mean_per_track['y'],
        c=mean_per_track[speed_col], cmap=cmap, norm=norm,
        s=20, zorder=3, edgecolors='none',
    )
    ax.quiver(
        mean_per_track['x'], mean_per_track['y'],
        mean_per_track[vx_col], -mean_per_track[vy_col],
        angles='xy', scale_units='xy', scale=scale,
        color='white', alpha=0.7, width=0.003,
    )
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_title(title)
    ax.set_facecolor('#1a1a2e')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=ax, label=label, shrink=0.8)

fig.suptitle(f'Mean flow field — {DATASET_ID}', y=1.01)
plt.savefig(OUTPUT_DIR / 'flow_field_mean.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.close()
print('Saved: flow_field_mean.png')

# ── animation: time-varying flow field ───────────────────────────────────────
print('Rendering flow field animation...')

timepoints = sorted(flow_df['t'].unique())

x_min, x_max = flow_df[x_col].min(), flow_df[x_col].max()
y_min, y_max = flow_df[y_col].min(), flow_df[y_col].max()
pad = 10

# Separate colour scales for collective vs autonomous speed
norm_local = mcolors.Normalize(
    vmin=0, vmax=flow_df['local_speed_um_per_min'].quantile(0.95))
norm_rel   = mcolors.Normalize(
    vmin=0, vmax=flow_df['rel_speed_um_per_min'].quantile(0.95))

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='black')
time_text = fig.text(0.5, 0.96, '', ha='center', color='white', fontsize=11)


def _style_ax(ax, title):
    ax.set_facecolor('black')
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_max + pad, y_min - pad)
    ax.set_aspect('equal')
    ax.set_xlabel('X (µm)', color='white')
    ax.set_ylabel('Y (µm)', color='white')
    ax.set_title(title, color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')


def update(frame_idx):
    for ax in axes:
        ax.cla()
    _style_ax(axes[0], 'Collective flow')
    _style_ax(axes[1], 'Autonomous motion')

    t = timepoints[frame_idx]
    frame = flow_df[flow_df['t'] == t].dropna(
        subset=['local_vy_um', 'local_vx_um', 'rel_vy_um', 'rel_vx_um'])

    xs = frame[x_col].values
    ys = frame[y_col].values

    for ax, vy_col, vx_col, speed_col, norm_i in [
        (axes[0], 'local_vy_um', 'local_vx_um', 'local_speed_um_per_min', norm_local),
        (axes[1], 'rel_vy_um',   'rel_vx_um',   'rel_speed_um_per_min',   norm_rel),
    ]:
        ax.scatter(xs, ys, c=frame[speed_col].values, cmap=cmap, norm=norm_i,
                   s=15, linewidths=0, zorder=3)
        ax.quiver(xs, ys, frame[vx_col].values, -frame[vy_col].values,
                  angles='xy', scale_units='xy', scale=scale,
                  color='white', alpha=0.7, width=0.003)

    time_text.set_text(f't = {int(t)}   ({int(t * FRAME_MIN)} min)')
    return []


anim = FuncAnimation(fig, update, frames=len(timepoints),
                     interval=1000 / FPS, blit=False)

mp4_path = OUTPUT_DIR / f'flow_field_animation_{VERSION}.mp4'
writer = FFMpegWriter(fps=FPS, bitrate=2000)
anim.save(mp4_path, writer=writer, dpi=120,
          savefig_kwargs={'facecolor': 'black'})
print(f'Saved MP4:  {mp4_path.name}')

# TIFF stack for napari
print('Rendering TIFF stack for napari...')
frames_rgb = []
for i in range(len(timepoints)):
    update(i)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    frames_rgb.append(buf.reshape(h, w, 4)[:, :, 1:])

tiff_path = OUTPUT_DIR / f'flow_field_animation_{VERSION}.tif'
tifffile.imwrite(tiff_path, np.stack(frames_rgb, axis=0), imagej=True)
print(f'Saved TIFF: {tiff_path.name}  {np.stack(frames_rgb).shape}')
plt.close()