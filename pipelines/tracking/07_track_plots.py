"""
Per-track timecourses: one row per track, one column each for ERK C/N, Z,
radial distance, ICM distance and nuclear volume. Raw values, no smoothing.

Grey shading = pre-implantation (t < implantation_onset_t). Rows are sorted by
pre-implantation ICM distance and the title shows the track's summary values.
Y limits are shared down each column (1st-99th percentile) so tracks compare
directly.

Requires 06_build_track_table.py.

Usage
-----
  # every track with ≥ MIN_FRAMES frames, paginated
  conda run -n napari_env python3 pipelines/tracking/07_track_plots.py

  # specific tracks (one figure)
  conda run -n napari_env python3 pipelines/tracking/07_track_plots.py --tracks 48 71

  Or set TRACK_IDS below and hit Run in VS Code.

Outputs (output_dir/track_plots/)
-------
  tracks_{version}_page{N}.png   or   tracks_{version}_selected.png
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.tracking_config import load_tracking_config, implantation_onset
cfg, CONFIG_PATH = load_tracking_config()

# ── Edit these directly in VS Code ───────────────────────────────────────────
TRACK_IDS      = []    # [] = all tracks with ≥ MIN_FRAMES frames
MIN_FRAMES     = 30
TRACKS_PER_PAGE = 12
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--tracks', nargs='+', type=int, default=None)
parser.add_argument('--min-frames', type=int, default=None)
args = parser.parse_args()
if args.tracks is not None:     TRACK_IDS  = args.tracks
if args.min_frames is not None: MIN_FRAMES = args.min_frames

version  = cfg['tracking']['input_version']
out_dir  = Path(cfg['paths']['output_dir'])
interval = cfg['tracking']['frame_interval_min']
T_SPLIT  = implantation_onset(cfg)

table   = pd.read_csv(out_dir / f'track_table_{version}.csv')
summary = pd.read_csv(out_dir / f'track_summary_{version}.csv').set_index('track_id')

PANELS = [
    ('erk_cn_ratio',   'ERK C/N',             '#d6604d'),
    ('z_um_reg',       'Z (µm)',              '#4393c3'),
    ('radial_dist_um', 'Radial dist (µm)',    '#4daf4a'),
    ('icm_dist_um',    'ICM edge dist (µm)',  '#984ea3'),
    ('area_um3',       'Volume (µm³)',        '#7f7f7f'),
]

# ── Select tracks ─────────────────────────────────────────────────────────────

if TRACK_IDS:
    missing = [t for t in TRACK_IDS if t not in summary.index]
    if missing:
        print(f'Warning: track IDs not found: {missing}')
    track_ids = [t for t in TRACK_IDS if t in summary.index]
    pages = [track_ids]
    names = [f'tracks_{version}_selected.png']
else:
    keep = summary[summary['n_frames'] >= MIN_FRAMES]
    track_ids = keep.sort_values('icm_dist_pre', na_position='last').index.tolist()
    pages = [track_ids[i:i + TRACKS_PER_PAGE] for i in range(0, len(track_ids), TRACKS_PER_PAGE)]
    names = [f'tracks_{version}_page{i + 1}.png' for i in range(len(pages))]
    print(f'{len(track_ids)} tracks with ≥ {MIN_FRAMES} frames → {len(pages)} pages')

if not track_ids:
    sys.exit('No tracks to plot.')

# ── Plot ──────────────────────────────────────────────────────────────────────

ylims = {col: tuple(table[col].quantile([0.01, 0.99])) for col, _, _ in PANELS}
xlim  = (table['time_min'].min(), table['time_min'].max())
plot_dir = out_dir / 'track_plots'
plot_dir.mkdir(parents=True, exist_ok=True)


def fmt(v, spec='.0f'):
    return 'n/a' if pd.isna(v) else format(v, spec)


for page_tracks, name in zip(pages, names):
    fig, axes = plt.subplots(len(page_tracks), len(PANELS),
                             figsize=(3.2 * len(PANELS), 2.1 * len(page_tracks)),
                             squeeze=False, constrained_layout=True)
    for row, tid in enumerate(page_tracks):
        cell = table[table['track_id'] == tid]
        s = summary.loc[tid]
        for c, (col, label, color) in enumerate(PANELS):
            ax = axes[row, c]
            ax.axvspan(xlim[0], T_SPLIT * interval, color='0.93', zorder=0)
            ax.plot(cell['time_min'], cell[col], color=color, lw=1.3, marker='.', ms=2)
            if all(np.isfinite(ylims[col])):
                ax.set_ylim(ylims[col])
            ax.set_xlim(xlim)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(label, fontsize=9)
            if row == len(page_tracks) - 1:
                ax.set_xlabel('Time (min)', fontsize=8)
        axes[row, 0].set_ylabel(
            f'track {tid}  (t{int(s.t_first)}-{int(s.t_last)})\n'
            f'ERK pre {fmt(s.erk_pre, ".2f")} / during {fmt(s.erk_during, ".2f")}\n'
            f'ICM pre {fmt(s.icm_dist_pre)} µm, vol end {fmt(s.vol_end)}',
            fontsize=7)

    fig.suptitle(f'{version}  |  grey = pre-implantation (t < {T_SPLIT})', fontsize=10)
    fig.savefig(plot_dir / name, dpi=130)
    plt.close(fig)
    print(f'Saved: track_plots/{name}  ({len(page_tracks)} tracks)')
