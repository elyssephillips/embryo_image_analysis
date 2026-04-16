"""
Layer 2 — Mean Squared Displacement (MSD) analysis.

For each track, fits MSD(τ) = 6D·τ^α to classify motion mode:
  α ≈ 1  : normal diffusion (Brownian / random walk)
  α < 1  : sub-diffusion / confined
  α > 1  : super-diffusion / directed  (α ≈ 2 = purely ballistic)

Usage
-----
  python 08_msd_analysis.py
  python 08_msd_analysis.py configs/tracking/control_2.yaml

Outputs  (written to output_dir)
-------
  msd_curves_{version}.csv     — MSD vs lag per track
  msd_track_fits_{version}.csv — alpha, D_eff per track
  msd_curves.png
  msd_alpha_distribution.png
  msd_alpha_vs_D.png
  msd_spatial.png
  alpha_animation_{version}.mp4
  alpha_animation_{version}.tif
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

import trackpy as tp

# ── config ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('config', nargs='?',
                    default='configs/tracking/dataset001_implantation.yaml')
args = parser.parse_args()

CONFIG_PATH = REPO_ROOT / args.config
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

OUTPUT_DIR         = Path(cfg['paths']['output_dir'])
VERSION            = cfg['tracking']['input_version']
FRAME_MIN          = cfg['tracking']['frame_interval_min']
DATASET_ID         = cfg['project']['dataset_id']
MAX_LAG_FRAC       = cfg['motion']['msd']['max_lag_fraction']
MIN_LAGS           = cfg['motion']['msd']['min_lags_for_fit']

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'Dataset:    {DATASET_ID}')
print(f'Config:     {CONFIG_PATH}')

# ── load kinematics ───────────────────────────────────────────────────────────
kin = pd.read_csv(OUTPUT_DIR / f'motion_kinematics_{VERSION}.csv')
has_orig = 'z_um_orig' in kin.columns
z_col = 'z_um_orig' if has_orig else 'z_um'
y_col = 'y_um_orig' if has_orig else 'y_um'
x_col = 'x_um_orig' if has_orig else 'x_um'

print(f'{kin["track_id"].nunique()} tracks loaded')

# ── compute MSD via trackpy ───────────────────────────────────────────────────
print('Computing MSD curves (trackpy)...')

# trackpy expects: particle, frame, x, y (+ z for 3D)
traj = kin.rename(columns={
    'track_id': 'particle',
    't':        'frame',
    x_col:      'x',
    y_col:      'y',
    z_col:      'z',
})[['particle', 'frame', 'x', 'y', 'z']]

max_lagtime = int(kin.groupby('track_id')['t'].count().max() * MAX_LAG_FRAC)

# fps=1/FRAME_MIN → imsd index is lag time in minutes; mpp=1 → coords already in µm
imsd = tp.imsd(traj, mpp=1.0, fps=1.0 / FRAME_MIN,
               max_lagtime=max_lagtime, pos_columns=['x', 'y', 'z'])
# imsd: rows = lag time (min), columns = track_id, values = MSD (µm²)

# Convert to long format matching downstream plot code: track_id, lag (frames), msd_um2
msd_df = (
    imsd
    .reset_index()
    .rename(columns={imsd.index.name: 'lag_min'})
    .melt(id_vars='lag_min', var_name='track_id', value_name='msd_um2')
    .dropna(subset=['msd_um2'])
    .assign(lag=lambda d: (d['lag_min'] / FRAME_MIN).round().astype(int))
    [['track_id', 'lag', 'msd_um2']]
    .sort_values(['track_id', 'lag'])
    .reset_index(drop=True)
)

# Fit α per track — only tracks with enough lag points
print('Fitting MSD α (trackpy fit_powerlaw)...')
valid_tracks = imsd.columns[imsd.count() >= MIN_LAGS]
fits_raw = tp.utils.fit_powerlaw(imsd[valid_tracks], plot=False)
# fits_raw: index=track_id, columns=['n' (=α), 'A' (prefactor)]
# 3D: MSD = 6·D·τ^α  →  D_eff = A / 6
n_lags_series = imsd[valid_tracks].count()   # Series: track_id → n valid lags
fits_df = (
    fits_raw
    .rename(columns={'n': 'alpha'})
    .assign(D_eff_um2_per_min=lambda d: d['A'] / 6.0)
    .reset_index()
    .rename(columns={'index': 'track_id'})
)
fits_df['n_lags'] = fits_df['track_id'].map(n_lags_series)
fits_df = (
    fits_df[['track_id', 'alpha', 'D_eff_um2_per_min', 'n_lags']]
    .sort_values('track_id')
    .reset_index(drop=True)
)

# ── save CSVs ─────────────────────────────────────────────────────────────────
msd_df.to_csv(OUTPUT_DIR / f'msd_curves_{VERSION}.csv', index=False)
fits_df.to_csv(OUTPUT_DIR / f'msd_track_fits_{VERSION}.csv', index=False)
print(f'Saved msd_curves_{VERSION}.csv  ({len(msd_df)} rows)')
print(f'Saved msd_track_fits_{VERSION}.csv  ({len(fits_df)} tracks fitted)')

print(f'\n--- Alpha summary ---')
print(fits_df['alpha'].describe().round(3).to_string())

# ── plot 1: MSD curves (log-log) ──────────────────────────────────────────────
# Colour lines by alpha so directed tracks stand out
alpha_min, alpha_max = fits_df['alpha'].min(), fits_df['alpha'].max()
norm  = mcolors.Normalize(vmin=0.5, vmax=1.5)   # centre on diffusive (1.0)
cmap  = plt.cm.coolwarm

fig, ax = plt.subplots(figsize=(9, 6))

for tid, grp in msd_df.groupby('track_id'):
    grp = grp.sort_values('lag')
    tau = grp['lag'].values * FRAME_MIN
    fit_row = fits_df[fits_df['track_id'] == tid]
    color = cmap(norm(fit_row['alpha'].values[0])) if len(fit_row) else 'grey'
    ax.plot(tau, grp['msd_um2'].values, color=color, alpha=0.4, linewidth=0.8)

# Reference slopes — anchored at the median lag and median MSD at that lag,
# so the lines pass through the middle of the data cloud rather than a corner.
tau_ref  = np.array([FRAME_MIN, msd_df['lag'].max() * FRAME_MIN])
mid_lag  = int(msd_df['lag'].median())
mid_tau  = mid_lag * FRAME_MIN
mid_msd  = msd_df[msd_df['lag'] == mid_lag]['msd_um2'].median()

for slope, label, ls in [(1.0, 'α=1 (diffusive)', '--'), (2.0, 'α=2 (directed)', ':')]:
    y_ref = (tau_ref / mid_tau) ** slope * mid_msd
    ax.plot(tau_ref, y_ref, 'k', linestyle=ls, linewidth=1.2, label=label)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Lag time (min)')
ax.set_ylabel('MSD (µm²)')
ax.set_title(f'MSD curves — {DATASET_ID}  (colour = α)')
ax.legend()
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
             ax=ax, label='α', shrink=0.7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'msd_curves.png', dpi=150)
plt.close()
print('Saved: msd_curves.png')

# ── plot 2: alpha distribution ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(fits_df['alpha'], bins=25, color='steelblue', edgecolor='none', alpha=0.85)
ax.axvline(1.0, color='black',    linestyle='--', linewidth=1.2, label='α=1 diffusive')
ax.axvline(2.0, color='firebrick',linestyle=':',  linewidth=1.2, label='α=2 directed')
ax.axvline(fits_df['alpha'].median(), color='navy', linewidth=1.5,
           label=f'median = {fits_df["alpha"].median():.2f}')
ax.set_xlabel('α  (anomalous diffusion exponent)')
ax.set_ylabel('Number of tracks')
ax.set_title(f'Distribution of α — {DATASET_ID}')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'msd_alpha_distribution.png', dpi=150)
plt.close()
print('Saved: msd_alpha_distribution.png')

# ── plot 3: D_eff vs alpha, coloured by track length ─────────────────────────
stats = pd.read_csv(OUTPUT_DIR / f'motion_track_stats_{VERSION}.csv')
fits_merged = fits_df.merge(stats[['track_id', 'n_frames']], on='track_id', how='left')

fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(
    fits_merged['alpha'], fits_merged['D_eff_um2_per_min'],
    c=fits_merged['n_frames'], cmap='viridis',
    s=30, alpha=0.8, edgecolors='none',
)
plt.colorbar(sc, ax=ax, label='Track length (frames)')
ax.axvline(1.0, color='black', linestyle='--', linewidth=1, label='α=1 diffusive')
ax.set_xlabel('α')
ax.set_ylabel('D_eff  (µm² / min)')
ax.set_title(f'D_eff vs α — {DATASET_ID}')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'msd_alpha_vs_D.png', dpi=150)
plt.close()
print('Saved: msd_alpha_vs_D.png')

# ── plot 4: spatial maps of alpha and D_eff ───────────────────────────────────
# Mean Y-X position per track across all timepoints
mean_pos = (
    kin.groupby('track_id')[[y_col, x_col]]
    .mean()
    .reset_index()
    .rename(columns={y_col: 'y_mean', x_col: 'x_mean'})
)
spatial = fits_df.merge(mean_pos, on='track_id', how='inner')

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

# Left: alpha
sc0 = axes[0].scatter(
    spatial['x_mean'], spatial['y_mean'],
    c=spatial['alpha'], cmap='coolwarm',
    vmin=0.5, vmax=1.5,
    s=60, edgecolors='k', linewidths=0.3,
)
plt.colorbar(sc0, ax=axes[0], label='α')
axes[0].set_title('Motion mode (α)')

# Right: D_eff — log scale colormap since D spans two orders of magnitude
d_norm = mcolors.LogNorm(
    vmin=spatial['D_eff_um2_per_min'].min(),
    vmax=spatial['D_eff_um2_per_min'].max(),
)
sc1 = axes[1].scatter(
    spatial['x_mean'], spatial['y_mean'],
    c=spatial['D_eff_um2_per_min'], cmap='plasma', norm=d_norm,
    s=60, edgecolors='k', linewidths=0.3,
)
plt.colorbar(sc1, ax=axes[1], label='D_eff  (µm²/min)')
axes[1].set_title('Mobility (D_eff)')

for ax in axes:
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')

fig.suptitle(f'Spatial distribution of MSD parameters — {DATASET_ID}')
plt.savefig(OUTPUT_DIR / 'msd_spatial.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: msd_spatial.png')

# ── animation: tracks coloured by alpha ───────────────────────────────────────
# α is a per-track scalar so each nucleus keeps its colour throughout.
# Colour scale: coolwarm centred on 1.0 (blue=confined, red=directed).
TAIL_LENGTH = 8
FPS         = 8

alpha_norm = mcolors.Normalize(vmin=0.5, vmax=1.5)
alpha_cmap = plt.cm.coolwarm

# Build per-track lookup: track_id -> alpha colour
alpha_lookup = dict(zip(fits_df['track_id'], fits_df['alpha']))

timepoints = sorted(kin['t'].unique())
x_min, x_max = kin[x_col].min(), kin[x_col].max()
y_min, y_max = kin[y_col].min(), kin[y_col].max()
pad = 10

# Pre-group tracks for fast tail lookup
track_groups = {tid: grp.sort_values('t') for tid, grp in kin.groupby('track_id')}

fig, ax = plt.subplots(figsize=(7, 7), facecolor='black')
ax.set_facecolor('black')
ax.set_xlim(x_min - pad, x_max + pad)
ax.set_ylim(y_max + pad, y_min - pad)
ax.set_aspect('equal')
ax.set_xlabel('X (µm)', color='white')
ax.set_ylabel('Y (µm)', color='white')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('white')

fig.colorbar(plt.cm.ScalarMappable(norm=alpha_norm, cmap=alpha_cmap),
             ax=ax, label='α  (blue=confined, red=directed)', shrink=0.7)
time_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                    color='white', fontsize=10, va='top')

tail_lines = []


def update_alpha(frame_idx):
    global tail_lines
    t = timepoints[frame_idx]

    for line in tail_lines:
        line.remove()
    tail_lines = []

    t_window = [tp for tp in timepoints if t - TAIL_LENGTH <= tp <= t]

    for tid, grp in track_groups.items():
        seg = grp[grp['t'].isin(t_window)]
        if len(seg) < 2:
            continue
        colour = alpha_cmap(alpha_norm(alpha_lookup.get(tid, 1.0)))
        xs, ys = seg[x_col].values, seg[y_col].values
        for i in range(len(xs) - 1):
            alpha_val = (i + 1) / len(xs) * 0.5
            line, = ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                            color=colour, alpha=alpha_val, linewidth=0.8)
            tail_lines.append(line)

    # Current positions
    frame = kin[kin['t'] == t].copy()
    frame['alpha_val'] = frame['track_id'].map(alpha_lookup)
    frame = frame.dropna(subset=['alpha_val'])

    sc = ax.scatter(
        frame[x_col], frame[y_col],
        c=frame['alpha_val'], cmap=alpha_cmap, norm=alpha_norm,
        s=40, linewidths=0, zorder=3,
    )
    tail_lines.append(sc)

    time_text.set_text(f't = {int(t)}   ({int(t * FRAME_MIN)} min)')
    return tail_lines + [time_text]


print('Rendering α animation...')
anim = FuncAnimation(fig, update_alpha, frames=len(timepoints),
                     interval=1000 / FPS, blit=False)

mp4_path = OUTPUT_DIR / f'alpha_animation_{VERSION}.mp4'
anim.save(mp4_path, writer=FFMpegWriter(fps=FPS, bitrate=2000), dpi=150,
          savefig_kwargs={'facecolor': 'black'})
print(f'Saved MP4:  {mp4_path.name}')

# TIFF stack for napari
print('Rendering TIFF stack for napari...')
frames_rgb = []
for i in range(len(timepoints)):
    update_alpha(i)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    frames_rgb.append(buf.reshape(h, w, 4)[:, :, 1:])

tiff_path = OUTPUT_DIR / f'alpha_animation_{VERSION}.tif'
tifffile.imwrite(tiff_path, np.stack(frames_rgb, axis=0), imagej=True)
print(f'Saved TIFF: {tiff_path.name}  {np.stack(frames_rgb).shape}')
plt.close()