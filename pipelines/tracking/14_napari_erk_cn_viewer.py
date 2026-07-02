"""
Pseudocolor segmentation stack in napari colored by ERK C/N ratio and ICM distance.

Each nucleus voxel is assigned a float value (C/N ratio or ICM distance) at that
timepoint. Background and untracked nuclei are NaN (transparent).

Layers
------
  ERK C/N ratio    — nuclei colored by cytoplasm/nucleus ERK ratio (RdBu_r)
  ICM distance     — nuclei colored by distance from ICM centroid at t=30 (magma)
  labels           — instance segmentation boundaries (semi-transparent)
  erk_raw          — raw ERK biosensor channel (hidden by default)
  h2b_raw          — raw nuclear H2B channel (hidden by default)

Usage
-----
  python 14_napari_erk_cn_viewer.py
  python 14_napari_erk_cn_viewer.py --no-images   # skip raw channel loading
  python 14_napari_erk_cn_viewer.py configs/tracking/dataset001_implantation.yaml
"""

import sys
import argparse
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import tifffile
import dask
import dask.array as da
import napari

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# ── config ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('config', nargs='?',
                    default='configs/tracking/dataset001_implantation.yaml')
parser.add_argument('--no-images', action='store_true',
                    help='Skip loading raw ERK and H2B channels (faster startup)')
args = parser.parse_args()

CONFIG_PATH = REPO_ROOT / args.config
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

RAW_DIR    = Path(cfg['paths']['raw_dir'])
RAW_GLOB   = cfg['paths']['raw_glob']
ERK_DIR    = Path(cfg['paths']['erk_dir'])
ERK_GLOB   = cfg['paths']['erk_glob']
LABEL_DIR  = Path(cfg['paths']['label_dir'])
LABEL_GLOB = cfg['paths']['label_glob']
OUTPUT_DIR = Path(cfg['paths']['output_dir'])
VERSION    = cfg['tracking']['input_version']
N_T        = cfg['microscopy']['n_timepoints']
VX_Z, VX_Y, VX_X = cfg['microscopy']['voxel_size_zyx']
SCALE = (1, VX_Z, VX_Y, VX_X)   # T, Z, Y, X in µm

# ── load ERK C/N ratio CSV ────────────────────────────────────────────────────
cn_csv = OUTPUT_DIR / f'erk_cn_ratio_{VERSION}.csv'
if not cn_csv.exists():
    sys.exit(
        f'\nERK C/N CSV not found: {cn_csv}\n'
        f'Run 12_erk_cn_ratio.py first, then relaunch this viewer.'
    )

print(f'Loading: {cn_csv}')
cn_df = pd.read_csv(cn_csv)
cn_df['label_id'] = cn_df['label_id'].astype('Int64')

# Per-timepoint LUT: ndarray indexed by label_id → C/N float32, NaN where absent
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

cn_vals = cn_df['erk_cn_ratio'].dropna()
cn_median = float(cn_vals.median())
cn_lo = float(cn_vals.quantile(0.02))
cn_hi = float(cn_vals.quantile(0.98))
# Make contrast limits symmetric around C/N=1 so the colormap center aligns
cn_sym = max(abs(cn_lo - 1.0), abs(cn_hi - 1.0))
contrast_lo = max(0.0, 1.0 - cn_sym)
contrast_hi = 1.0 + cn_sym
print(f'C/N  median={cn_median:.3f}  2–98pct=[{cn_lo:.3f}, {cn_hi:.3f}]')
print(f'Contrast limits (symmetric around 1.0): [{contrast_lo:.3f}, {contrast_hi:.3f}]')

# ── ICM distance per (t, label_id) ───────────────────────────────────────────
# icm_dist_um is per-track (fixed at t=30); join via track_id from cn_df.
icm_dist_csv = OUTPUT_DIR / f'volume_track_stats_{VERSION}.csv'
icm_luts = {}
icm_dist_hi = None

if icm_dist_csv.exists():
    print(f'Loading: {icm_dist_csv}')
    vol_stats = pd.read_csv(icm_dist_csv)[['track_id', 'icm_dist_um']].dropna()
    dist_map  = dict(zip(vol_stats['track_id'], vol_stats['icm_dist_um'].astype(np.float32)))

    # Join distance to cn_df rows, then build per-timepoint LUTs
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

    dist_vals   = cn_df['icm_dist_um'].dropna()
    icm_dist_hi = float(dist_vals.quantile(0.98))
    print(f'ICM distance  median={float(dist_vals.median()):.1f} µm  '
          f'98pct={icm_dist_hi:.1f} µm')
else:
    print(f'ICM distance CSV not found ({icm_dist_csv.name}) — skipping ICM distance layer')

# ── lazy per-frame C/N image builder ─────────────────────────────────────────
label_files = sorted(LABEL_DIR.glob(LABEL_GLOB))[:N_T]
assert len(label_files) == N_T, (
    f'Expected {N_T} label files, got {len(label_files)} in {LABEL_DIR}'
)

_probe = tifffile.imread(str(label_files[0]))
label_shape = _probe.shape  # (Z, Y, X)
print(f'Label frame shape: {label_shape}  dtype: {_probe.dtype}')


def _map_lut_to_frame(label_path, lut):
    """Replace each nonzero label ID with its value from lut. Background/missing → NaN."""
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


def _cn_frame(t):
    return _map_lut_to_frame(label_files[t], luts.get(t))


def _icm_frame(t):
    return _map_lut_to_frame(label_files[t], icm_luts.get(t))


cn_frames = [
    da.from_delayed(
        dask.delayed(_cn_frame)(t),
        shape=label_shape,
        dtype=np.float32,
    )
    for t in range(N_T)
]
cn_stack = da.stack(cn_frames, axis=0)   # (T, Z, Y, X), lazy
print(f'C/N stack: {cn_stack.shape}  (lazy dask — loads per frame on demand)')

if icm_luts:
    icm_frames = [
        da.from_delayed(
            dask.delayed(_icm_frame)(t),
            shape=label_shape,
            dtype=np.float32,
        )
        for t in range(N_T)
    ]
    icm_stack = da.stack(icm_frames, axis=0)
    print(f'ICM dist stack: {icm_stack.shape}  (lazy dask)')
else:
    icm_stack = None

# ── open napari ───────────────────────────────────────────────────────────────
viewer = napari.Viewer()
viewer.dims.axis_labels = ['t', 'z', 'y', 'x']

# Raw channels (lazy, hidden by default)
if not args.no_images:
    erk_files = sorted(ERK_DIR.glob(ERK_GLOB))[:N_T]
    if erk_files:
        _erk_probe = tifffile.imread(str(erk_files[0]))
        erk_shape  = _erk_probe.shape
        print(f'ERK frame shape: {erk_shape}  ({len(erk_files)} files, lazy)')
        erk_frames = [
            da.from_delayed(
                dask.delayed(tifffile.imread)(str(p)),
                shape=erk_shape,
                dtype=_erk_probe.dtype,
            )
            for p in erk_files
        ]
        viewer.add_image(
            da.stack(erk_frames, axis=0), name='erk_raw', scale=SCALE,
            colormap='green', blending='additive', opacity=0.5,
            visible=False,
        )

    raw_files = sorted(RAW_DIR.glob(RAW_GLOB))[:N_T]
    if raw_files:
        _raw_probe = tifffile.imread(str(raw_files[0]))
        raw_shape  = _raw_probe.shape
        print(f'H2B frame shape: {raw_shape}  ({len(raw_files)} files, lazy)')
        raw_frames = [
            da.from_delayed(
                dask.delayed(tifffile.imread)(str(p)),
                shape=raw_shape,
                dtype=_raw_probe.dtype,
            )
            for p in raw_files
        ]
        viewer.add_image(
            da.stack(raw_frames, axis=0), name='h2b_raw', scale=SCALE,
            colormap='gray', blending='additive', opacity=0.6,
            visible=False,
        )

# Instance label boundaries (semi-transparent overlay for nucleus outlines)
lbl_frames = [
    da.from_delayed(
        dask.delayed(tifffile.imread)(str(p)),
        shape=label_shape,
        dtype=_probe.dtype,
    )
    for p in label_files
]
viewer.add_labels(
    da.stack(lbl_frames, axis=0), name='labels', scale=SCALE,
    opacity=0.15,
)

# ERK C/N ratio pseudocolor — main layer
viewer.add_image(
    cn_stack, name='ERK C/N ratio',
    scale=SCALE,
    colormap='RdBu_r',           # blue=nuclear, white=equal, red=cytoplasmic
    contrast_limits=(contrast_lo, contrast_hi),
    blending='additive',
    opacity=0.85,
)

# ICM distance pseudocolor — hidden by default
if icm_stack is not None:
    viewer.add_image(
        icm_stack, name='ICM distance (µm)',
        scale=SCALE,
        colormap='magma',        # dark=near ICM, bright=far from ICM
        contrast_limits=(0.0, icm_dist_hi),
        blending='additive',
        opacity=0.85,
        visible=False,
    )

print('\nReady.')
print('  ERK C/N ratio:    blue=nuclear-enriched  white=C/N≈1  red=cytoplasmic-enriched')
print(f'    contrast limits: [{contrast_lo:.3f}, {contrast_hi:.3f}] (symmetric around 1.0)')
if icm_stack is not None:
    print(f'  ICM distance:     dark (magma)=near ICM  bright=far  '
          f'max={icm_dist_hi:.1f} µm  (hidden by default)')
print('  Toggle "labels" for nucleus outlines, "erk_raw"/"h2b_raw" for raw channels')
napari.run()
