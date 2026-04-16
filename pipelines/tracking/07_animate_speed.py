"""
Animate nucleus positions coloured by instantaneous speed, with short track tails.
Saves an MP4 to output_dir.

Usage
-----
  python 07_animate_speed.py                                    # default config
  python 07_animate_speed.py configs/tracking/control_2.yaml   # specific dataset
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

# ── config ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    'config',
    nargs='?',
    default='configs/tracking/dataset001_implantation.yaml',
    help='path to dataset config yaml (relative to repo root)',
)
args = parser.parse_args()

CONFIG_PATH = REPO_ROOT / args.config
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

OUTPUT_DIR = Path(cfg['paths']['output_dir'])
VERSION    = cfg['tracking']['input_version']
T_START    = cfg['tracking']['t_start']
T_END      = cfg['tracking']['t_end']
FRAME_MIN  = cfg['tracking']['frame_interval_min']
DATASET_ID = cfg['project']['dataset_id']

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── animation settings ────────────────────────────────────────────────────────
TAIL_LENGTH = 8     # number of past frames to draw as a fading trail
FPS         = 8     # frames per second in the output video
POINT_SIZE  = 40    # nucleus marker size

# ── load ──────────────────────────────────────────────────────────────────────
kin = pd.read_csv(OUTPUT_DIR / f'motion_kinematics_{VERSION}.csv')

has_orig = 'z_um_orig' in kin.columns
y_col = 'y_um_orig' if has_orig else 'y_um'
x_col = 'x_um_orig' if has_orig else 'x_um'

timepoints = sorted(kin['t'].unique())

# Pre-group by track for fast tail lookup: {track_id -> DataFrame sorted by t}
track_groups = {tid: grp.sort_values('t') for tid, grp in kin.groupby('track_id')}

# Shared colour scale: cap at 95th percentile so outliers don't wash out colour
vmax = kin['speed_um_per_min'].quantile(0.95)
norm = mcolors.Normalize(vmin=0, vmax=vmax)
cmap = plt.cm.inferno

# Axis limits from all positions
x_min, x_max = kin[x_col].min(), kin[x_col].max()
y_min, y_max = kin[y_col].min(), kin[y_col].max()
pad = 10  # µm padding

print(f'Dataset:    {DATASET_ID}')
print(f'Frames:     {len(timepoints)}  ({timepoints[0]:.0f}–{timepoints[-1]:.0f})')
print(f'Tail:       {TAIL_LENGTH} frames  ({TAIL_LENGTH * FRAME_MIN} min)')
print(f'Output dir: {OUTPUT_DIR}')

# ── build animation ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7), facecolor='black')
ax.set_facecolor('black')
ax.set_xlim(x_min - pad, x_max + pad)
ax.set_ylim(y_max + pad, y_min - pad)   # inverted: y=0 at top
ax.set_aspect('equal')
ax.set_xlabel('X (µm)', color='white')
ax.set_ylabel('Y (µm)', color='white')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('white')

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=ax, label='Speed (µm/min)', shrink=0.7)
cbar.ax.yaxis.set_tick_params(color='white')
cbar.ax.yaxis.label.set_color('white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

time_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                    color='white', fontsize=10, va='top')

# Placeholder artists — updated each frame
scatter = ax.scatter([], [], c=[], cmap=cmap, norm=norm,
                     s=POINT_SIZE, linewidths=0, zorder=3)
tail_lines = []   # created on first frame, reused after


def update(frame_idx):
    global tail_lines
    t = timepoints[frame_idx]

    # ── tail lines ────────────────────────────────────────────────────────────
    # Remove previous tails
    for line in tail_lines:
        line.remove()
    tail_lines = []

    t_window = [tp for tp in timepoints if t - TAIL_LENGTH * 1 <= tp <= t]

    for tid, grp in track_groups.items():
        seg = grp[grp['t'].isin(t_window)]
        if len(seg) < 2:
            continue
        xs = seg[x_col].values
        ys = seg[y_col].values
        # Draw segments with alpha proportional to recency
        for i in range(len(xs) - 1):
            alpha = (i + 1) / len(xs) * 0.6   # older = more transparent
            line, = ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                            color='white', alpha=alpha, linewidth=0.8, zorder=2)
            tail_lines.append(line)

    # ── current positions coloured by speed ───────────────────────────────────
    frame = kin[kin['t'] == t].dropna(subset=['speed_um_per_min'])
    scatter.set_offsets(np.column_stack([frame[x_col], frame[y_col]]))
    scatter.set_array(frame['speed_um_per_min'].values)

    # ── timestamp ─────────────────────────────────────────────────────────────
    elapsed_min = int(t * FRAME_MIN)
    time_text.set_text(f't = {int(t)}   ({elapsed_min} min)')

    return [scatter, time_text] + tail_lines


anim = FuncAnimation(fig, update, frames=len(timepoints), interval=1000 / FPS, blit=False)

print(f'\nRendering {len(timepoints)} frames...')

# ── MP4 ───────────────────────────────────────────────────────────────────────
mp4_path = OUTPUT_DIR / f'speed_animation_{VERSION}.mp4'
writer = FFMpegWriter(fps=FPS, bitrate=2000)
anim.save(mp4_path, writer=writer, dpi=150,
          savefig_kwargs={'facecolor': 'black'})
print(f'Saved MP4:  {mp4_path}')

# ── TIFF stack (T, Y, X, 3) for napari ───────────────────────────────────────
# Re-render each frame to a numpy RGB array and stack into a single TIFF.
print('Rendering TIFF stack for napari...')
frames_rgb = []
for frame_idx in range(len(timepoints)):
    update(frame_idx)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    argb = buf.reshape(h, w, 4)
    frames_rgb.append(argb[:, :, 1:])   # drop alpha, keep R G B

tiff_stack = np.stack(frames_rgb, axis=0)   # (T, Y, X, 3)
tiff_path = OUTPUT_DIR / f'speed_animation_{VERSION}.tif'
tifffile.imwrite(tiff_path, tiff_stack, imagej=True)
print(f'Saved TIFF: {tiff_path}  {tiff_stack.shape}')

plt.close()