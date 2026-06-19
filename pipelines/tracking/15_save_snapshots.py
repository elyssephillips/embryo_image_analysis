"""
Save max-Z projection TIFFs for specified timepoints, for opening in Fiji.

Saves three float32 TIFFs per timepoint into {output_dir}/snapshots/:
  snapshot_t{t:03d}_erk_cn.tif       — ERK C/N ratio (float32, NaN=background)
  snapshot_t{t:03d}_icm_dist.tif     — ICM distance in µm (float32, NaN=background)
  snapshot_t{t:03d}_h2b_raw.tif      — raw H2B nuclear channel (original dtype)

Usage
-----
  python 15_save_snapshots.py 10 25 50
  python 15_save_snapshots.py --config configs/tracking/dataset001_implantation.yaml 10 25 50
"""

import sys
import argparse
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
DEFAULT_TIMEPOINTS = [20, 40, 60, 80, 100]  # edit this and run with F5 when no CLI args given

parser.add_argument('timepoints', nargs='*', type=int,
                    help='Timepoint indices to snapshot (e.g. 10 25 50)')
parser.add_argument('--config', default='configs/tracking/dataset001_implantation.yaml',
                    help='Path to YAML config (relative to repo root or absolute)')
args = parser.parse_args()

timepoints_input = args.timepoints if args.timepoints else DEFAULT_TIMEPOINTS

CONFIG_PATH = Path(args.config) if Path(args.config).is_absolute() else REPO_ROOT / args.config
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

RAW_DIR      = Path(cfg['paths']['raw_dir'])
RAW_GLOB     = cfg['paths']['raw_glob']
ERK_DIR      = Path(cfg['paths']['erk_dir'])
ERK_GLOB     = cfg['paths']['erk_glob']
LABEL_DIR    = Path(cfg['paths']['label_dir'])
LABEL_GLOB   = cfg['paths']['label_glob']
OUTPUT_DIR   = Path(cfg['paths']['output_dir'])
VERSION      = cfg['tracking']['input_version']
N_T          = cfg['microscopy']['n_timepoints']
XY_UM_PER_PX = cfg['microscopy']['voxel_size_zyx'][2]   # µm per XY pixel
SCALE_BAR_UM = 50                                        # µm to draw on each panel

SNAP_DIR = OUTPUT_DIR / 'snapshots'
SNAP_DIR.mkdir(parents=True, exist_ok=True)

timepoints = timepoints_input
for t in timepoints:
    if not (0 <= t < N_T):
        sys.exit(f'Timepoint {t} out of range [0, {N_T - 1}]')

# ── load ERK C/N ratio CSV ────────────────────────────────────────────────────
cn_csv = OUTPUT_DIR / f'erk_cn_ratio_{VERSION}.csv'
if not cn_csv.exists():
    sys.exit(f'\nERK C/N CSV not found: {cn_csv}\nRun 12_erk_cn_ratio.py first.')

print(f'Loading: {cn_csv}')
cn_df = pd.read_csv(cn_csv)
cn_df['label_id'] = cn_df['label_id'].astype('Int64')

print('Building per-timepoint label→C/N lookup tables...')
luts = {}
for t, grp in cn_df.groupby('t'):
    valid = grp['label_id'].notna() & grp['erk_cn_ratio'].notna()
    grp_v = grp[valid]
    if grp_v.empty:
        luts[t] = np.array([], dtype=np.float32)
        continue
    max_lid = int(grp_v['label_id'].max())
    arr = np.full(max_lid + 1, np.nan, dtype=np.float32)
    arr[grp_v['label_id'].astype(int).values] = grp_v['erk_cn_ratio'].astype(np.float32).values
    luts[t] = arr

# ── ICM distance ──────────────────────────────────────────────────────────────
icm_luts = {}
icm_dist_csv = OUTPUT_DIR / f'volume_track_stats_{VERSION}.csv'
if icm_dist_csv.exists():
    print(f'Loading: {icm_dist_csv}')
    vol_stats = pd.read_csv(icm_dist_csv)[['track_id', 'icm_dist_um']].dropna()
    dist_map  = dict(zip(vol_stats['track_id'], vol_stats['icm_dist_um'].astype(np.float32)))
    cn_df['icm_dist_um'] = cn_df['track_id'].map(dist_map)
    print('Building per-timepoint label→ICM-distance lookup tables...')
    for t, grp in cn_df.groupby('t'):
        valid = grp['label_id'].notna() & grp['icm_dist_um'].notna()
        grp_v = grp[valid]
        if grp_v.empty:
            icm_luts[t] = np.array([], dtype=np.float32)
            continue
        max_lid = int(grp_v['label_id'].max())
        arr = np.full(max_lid + 1, np.nan, dtype=np.float32)
        arr[grp_v['label_id'].astype(int).values] = grp_v['icm_dist_um'].astype(np.float32).values
        icm_luts[t] = arr
else:
    print(f'ICM distance CSV not found ({icm_dist_csv.name}) — skipping ICM distance layer')

# ── track identity (label_id -> track_id, for identity-colored overlays) ──────
print('Building per-timepoint label→track-ID lookup tables...')
track_luts = {}
for t, grp in cn_df.groupby('t'):
    valid = grp['label_id'].notna() & grp['track_id'].notna()
    grp_v = grp[valid]
    if grp_v.empty:
        track_luts[t] = np.array([], dtype=np.float32)
        continue
    max_lid = int(grp_v['label_id'].max())
    arr = np.full(max_lid + 1, np.nan, dtype=np.float32)
    arr[grp_v['label_id'].astype(int).values] = grp_v['track_id'].astype(np.float32).values
    track_luts[t] = arr


def _add_scalebar(ax, data_shape, um_per_px, scale_um):
    h, w = data_shape
    bar_px = scale_um / um_per_px
    pad_x  = w * 0.04
    pad_y  = h * 0.05
    bar_h  = max(h * 0.018, 4)
    x0     = pad_x
    y0     = h - pad_y - bar_h
    ax.add_patch(plt.Rectangle((x0, y0), bar_px, bar_h, color='black', clip_on=False))
    ax.text(x0 + bar_px / 2, y0 - bar_h,
            f'{scale_um} µm', ha='center', va='bottom',
            color='black', fontsize=22, fontweight='bold')


def _render_panel(ax, data, cmap_name, percentile=(2, 98), colorbar=False, label=''):
    cmap = plt.cm.get_cmap(cmap_name).copy()
    cmap.set_bad('white')
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return
    vmin, vmax = np.percentile(finite, percentile)
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.axis('off')
    if label:
        ax.set_title(label, fontsize=22, pad=8)
    if colorbar:
        cb = plt.colorbar(im, ax=ax, orientation='horizontal',
                          fraction=0.018, pad=0.02, shrink=0.5)
        cb.ax.tick_params(labelsize=14)
    return im


def _save_png(data, path, cmap_name, percentile=(2, 98)):
    h, w = data.shape
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    _render_panel(ax, data, cmap_name, percentile)
    _add_scalebar(ax, data.shape, XY_UM_PER_PX, SCALE_BAR_UM)
    plt.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close(fig)


def _save_montage(h2b, cn, icm, path, t):
    h, w = h2b.shape
    fig, axes = plt.subplots(1, 3, figsize=(w * 3 / 100, h / 100), dpi=100)
    _render_panel(axes[0], h2b, 'gray_r',  label='H2B (raw)')
    _render_panel(axes[1], cn,  'inferno', label='ERK C/N',           colorbar=True)
    _render_panel(axes[2], icm, 'viridis', label='ICM distance (µm)', colorbar=True)
    _add_scalebar(axes[0], h2b.shape, XY_UM_PER_PX, SCALE_BAR_UM)

    # Time label above the scale bar
    bar_px = SCALE_BAR_UM / XY_UM_PER_PX
    pad_x  = w * 0.04
    pad_y  = h * 0.05
    bar_h  = max(h * 0.018, 4)
    y0     = h - pad_y - bar_h
    axes[0].text(pad_x + bar_px / 2, y0 - bar_h * 4,
                 f't={t}  ({t * 15} min)',
                 ha='center', va='bottom',
                 color='black', fontsize=22, fontweight='bold')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


_GOLDEN_RATIO = 0.6180339887498949


def _track_id_to_rgb(track_id):
    """Deterministic, well-spaced hue per track_id so identity stays visually
    distinct and stable across timepoints without needing a fixed palette."""
    hue = (track_id * _GOLDEN_RATIO) % 1.0
    return mcolors.hsv_to_rgb([hue, 0.65, 0.95])


def _save_track_snapshot(h2b, track_proj, path, t, alpha=0.55):
    """H2B raw (grayscale) as background, nuclei tinted by track_id (identity)."""
    h, w = h2b.shape
    finite = h2b[np.isfinite(h2b)]
    vmin, vmax = np.percentile(finite, (1, 99.5))
    bg = np.clip((h2b - vmin) / (vmax - vmin + 1e-9), 0, 1)
    rgb = np.stack([bg] * 3, axis=-1).astype(np.float32)

    mask = np.isfinite(track_proj) & (track_proj > 0)
    comp = rgb
    if mask.any():
        ids = np.unique(track_proj[mask]).astype(int)
        colors = np.array([_track_id_to_rgb(i) for i in ids], dtype=np.float32)
        idx = np.searchsorted(ids, track_proj[mask].astype(int))
        a = np.zeros((h, w), dtype=np.float32)
        a[mask] = alpha
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        overlay[mask] = colors[idx]
        comp = rgb * (1 - a[..., None]) + overlay * a[..., None]

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(comp, interpolation='nearest')
    ax.axis('off')
    ax.set_title(f'Track identity  (n={mask.any() and len(ids) or 0})', fontsize=22, pad=8)
    _add_scalebar(ax, h2b.shape, XY_UM_PER_PX, SCALE_BAR_UM)

    bar_px = SCALE_BAR_UM / XY_UM_PER_PX
    pad_x  = w * 0.04
    pad_y  = h * 0.05
    bar_h  = max(h * 0.018, 4)
    y0     = h - pad_y - bar_h
    ax.text(pad_x + bar_px / 2, y0 - bar_h * 4,
            f't={t}  ({t * 15} min)',
            ha='center', va='bottom',
            color='black', fontsize=22, fontweight='bold')

    plt.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close(fig)


def _map_lut_to_frame(label_path, lut):
    labels = tifffile.imread(str(label_path))
    out = np.full(labels.shape, np.nan, dtype=np.float32)
    if lut is not None and len(lut) > 0:
        flat_lbl = labels.ravel()
        flat_out = out.ravel()
        nonbg = flat_lbl > 0
        if nonbg.any():
            ids = flat_lbl[nonbg]
            in_range = ids < len(lut)
            flat_out[np.where(nonbg)[0][in_range]] = lut[ids[in_range]]
    return out


# ── file lists ────────────────────────────────────────────────────────────────
label_files = sorted(LABEL_DIR.glob(LABEL_GLOB))[:N_T]
raw_files   = sorted(RAW_DIR.glob(RAW_GLOB))[:N_T]

assert len(label_files) == N_T, f'Expected {N_T} label files, got {len(label_files)}'

# ── save snapshots ────────────────────────────────────────────────────────────
for t in timepoints:
    print(f'\nt={t}')

    # ERK C/N — max-Z projection ignoring NaN background
    cn_vol  = _map_lut_to_frame(label_files[t], luts.get(t))
    cn_proj = np.nanmax(cn_vol, axis=0)
    out_cn  = SNAP_DIR / f'snapshot_t{t:03d}_erk_cn.tif'
    tifffile.imwrite(str(out_cn), cn_proj)
    print(f'  saved {out_cn.name}')
    out_cn_png = SNAP_DIR / f'snapshot_t{t:03d}_erk_cn.png'
    _save_png(cn_proj, out_cn_png, 'inferno')
    print(f'  saved {out_cn_png.name}')

    # ICM distance — max-Z projection ignoring NaN background
    if icm_luts:
        icm_vol  = _map_lut_to_frame(label_files[t], icm_luts.get(t))
        icm_proj = np.nanmax(icm_vol, axis=0)
        out_icm  = SNAP_DIR / f'snapshot_t{t:03d}_icm_dist.tif'
        tifffile.imwrite(str(out_icm), icm_proj)
        print(f'  saved {out_icm.name}')
        out_icm_png = SNAP_DIR / f'snapshot_t{t:03d}_icm_dist.png'
        _save_png(icm_proj, out_icm_png, 'viridis')
        print(f'  saved {out_icm_png.name}')

    # H2B raw — max-Z projection
    if t < len(raw_files):
        h2b_vol  = tifffile.imread(str(raw_files[t]))
        h2b_proj = h2b_vol.max(axis=0)
        out_h2b  = SNAP_DIR / f'snapshot_t{t:03d}_h2b_raw.tif'
        tifffile.imwrite(str(out_h2b), h2b_proj)
        print(f'  saved {out_h2b.name}')
        out_h2b_png = SNAP_DIR / f'snapshot_t{t:03d}_h2b_raw.png'
        _save_png(h2b_proj.astype(np.float32), out_h2b_png, 'gray_r')
        print(f'  saved {out_h2b_png.name}')
    else:
        print(f'  H2B file not found for t={t}, skipping')
        h2b_proj = None

    # 3-panel montage
    if icm_luts and t < len(raw_files) and h2b_proj is not None:
        out_montage = SNAP_DIR / f'snapshot_t{t:03d}_montage.png'
        _save_montage(h2b_proj.astype(np.float32), cn_proj, icm_proj, out_montage, t)
        print(f'  saved {out_montage.name}')

    # Track identity overlay — nuclei colored by persistent track_id over H2B
    if h2b_proj is not None:
        track_vol  = _map_lut_to_frame(label_files[t], track_luts.get(t))
        track_proj = np.nanmax(track_vol, axis=0)
        out_track_png = SNAP_DIR / f'snapshot_t{t:03d}_track_identity.png'
        _save_track_snapshot(h2b_proj.astype(np.float32), track_proj, out_track_png, t)
        print(f'  saved {out_track_png.name}')

print(f'\nDone. Files saved to: {SNAP_DIR}')
