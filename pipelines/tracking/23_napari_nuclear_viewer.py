"""
Pseudocolor segmentation stack in napari colored by normalized nuclear ERK intensity.

Each nucleus is colored by its nuclear intensity normalized to its own "on" state
(the Nth-percentile nuclear intensity over the full timecourse, representing the
most empty / ERK-active state for that nucleus):

  1.0  = nucleus maximally empty  (ERK fully on / active)
  > 1  = nucleus filling up       (ERK going off / inactive)

Colormap:  low (ERK on) → warm   |   high (ERK off) → cool
This lets you visually compare with the C/N ratio viewer (14_napari_erk_cn_viewer.py)
to see whether cytoplasmic signal is adding noise or skewing results.

Layers
------
  norm. nuclear ERK    — normalized nuclear intensity (RdBu, warm=on, cool=off)
  ERK C/N ratio        — original C/N ratio for side-by-side comparison (hidden)
  ICM distance         — per-track ICM distance (hidden)
  labels               — instance segmentation boundaries (semi-transparent)
  erk_raw              — raw ERK biosensor channel (hidden)
  h2b_raw              — raw H2B channel (hidden)

Usage
-----
  conda run -n napari_env python3 pipelines/tracking/23_napari_nuclear_viewer.py
  conda run -n napari_env python3 pipelines/tracking/23_napari_nuclear_viewer.py --no-images
  conda run -n napari_env python3 pipelines/tracking/23_napari_nuclear_viewer.py --on-pct 10
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

# ── Config ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('config', nargs='?',
                    default='configs/tracking/dataset001_implantation.yaml')
parser.add_argument('--no-images', action='store_true',
                    help='Skip loading raw ERK and H2B channels (faster startup)')
parser.add_argument('--on-pct', type=float, default=5.0,
                    help='Percentile of nuclear intensity used as the "on" reference (default 5)')
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
SCALE = (1, VX_Z, VX_Y, VX_X)
ON_PCT = args.on_pct

# ── Load ERK CSV and compute normalized nuclear intensity ─────────────────────

cn_csv = OUTPUT_DIR / f'erk_cn_ratio_{VERSION}.csv'
if not cn_csv.exists():
    sys.exit(f'\nERK CSV not found: {cn_csv}\nRun 12_erk_cn_ratio.py first.')

print(f'Loading: {cn_csv}')
cn_df = pd.read_csv(cn_csv)
cn_df['label_id'] = cn_df['label_id'].astype('Int64')

# Per-track "on" reference = low percentile of nuclear intensity (most empty state)
print(f'Computing per-track {ON_PCT}th-percentile nuclear intensity as "on" reference...')
on_ref = (cn_df.groupby('track_id')['erk_nuc_mean']
               .quantile(ON_PCT / 100.0)
               .rename('on_ref'))
cn_df = cn_df.join(on_ref, on='track_id')

# Drop tracks with zero/negative on-ref (bad segmentation)
valid_on = on_ref[on_ref > 0].index
cn_df = cn_df[cn_df['track_id'].isin(valid_on)].copy()
cn_df['nuc_norm'] = (cn_df['erk_nuc_mean'] / cn_df['on_ref']).astype(np.float32)

nuc_vals   = cn_df['nuc_norm'].dropna()
nuc_median = float(nuc_vals.median())
nuc_lo     = float(nuc_vals.quantile(0.02))
nuc_hi     = float(nuc_vals.quantile(0.98))
print(f'Norm. nuclear  median={nuc_median:.3f}  2–98pct=[{nuc_lo:.3f}, {nuc_hi:.3f}]')
print(f'  1.0 = nucleus empty (ERK on)  |  >{nuc_hi:.2f} = nucleus full (ERK off)')

# Also keep C/N for comparison layer
cn_vals    = cn_df['erk_cn_ratio'].dropna()
cn_lo      = float(cn_vals.quantile(0.02))
cn_hi      = float(cn_vals.quantile(0.98))
cn_sym     = max(abs(cn_lo - 1.0), abs(cn_hi - 1.0))
cn_clim_lo = max(0.0, 1.0 - cn_sym)
cn_clim_hi = 1.0 + cn_sym

# ── Build per-timepoint LUTs ──────────────────────────────────────────────────

def _build_luts(df, col):
    luts = {}
    for t, grp in df.groupby('t'):
        valid = grp['label_id'].notna() & grp[col].notna()
        grp_v = grp[valid]
        if grp_v.empty:
            luts[t] = np.array([], dtype=np.float32)
            continue
        max_lid = int(grp_v['label_id'].max())
        arr = np.full(max_lid + 1, np.nan, dtype=np.float32)
        arr[grp_v['label_id'].astype(int).values] = grp_v[col].astype(np.float32).values
        luts[t] = arr
    return luts

print('Building LUTs for normalized nuclear intensity...')
nuc_luts = _build_luts(cn_df, 'nuc_norm')

print('Building LUTs for C/N ratio (comparison layer)...')
cn_luts = _build_luts(cn_df, 'erk_cn_ratio')

# ICM distance
icm_luts    = {}
icm_dist_hi = None
icm_csv     = OUTPUT_DIR / f'volume_track_stats_{VERSION}.csv'
if icm_csv.exists():
    print(f'Loading ICM distances: {icm_csv}')
    vol_stats = pd.read_csv(icm_csv)[['track_id', 'icm_dist_um']].dropna()
    dist_map  = dict(zip(vol_stats['track_id'], vol_stats['icm_dist_um'].astype(np.float32)))
    cn_df['icm_dist_um'] = cn_df['track_id'].map(dist_map)
    print('Building LUTs for ICM distance...')
    icm_luts    = _build_luts(cn_df, 'icm_dist_um')
    dist_vals   = cn_df['icm_dist_um'].dropna()
    icm_dist_hi = float(dist_vals.quantile(0.98))
else:
    print(f'ICM distance CSV not found — skipping')

# ── Lazy per-frame image builder ──────────────────────────────────────────────

label_files = sorted(LABEL_DIR.glob(LABEL_GLOB))[:N_T]
assert len(label_files) == N_T, (
    f'Expected {N_T} label files, got {len(label_files)}'
)
_probe      = tifffile.imread(str(label_files[0]))
label_shape = _probe.shape
print(f'Label frame shape: {label_shape}')


def _map_lut_to_frame(label_path, lut):
    labels   = tifffile.imread(str(label_path))
    out      = np.full(labels.shape, np.nan, dtype=np.float32)
    if lut is not None and len(lut) > 0:
        flat_lbl = labels.ravel()
        flat_out = out.ravel()
        nonbg    = flat_lbl > 0
        if nonbg.any():
            ids      = flat_lbl[nonbg]
            in_range = ids < len(lut)
            flat_out[np.where(nonbg)[0][in_range]] = lut[ids[in_range]]
    return out


def _make_stack(luts_dict):
    frames = [
        da.from_delayed(
            dask.delayed(_map_lut_to_frame)(label_files[t], luts_dict.get(t)),
            shape=label_shape, dtype=np.float32,
        )
        for t in range(N_T)
    ]
    return da.stack(frames, axis=0)


nuc_stack = _make_stack(nuc_luts)
cn_stack  = _make_stack(cn_luts)
print(f'Norm. nuclear stack:  {nuc_stack.shape}  (lazy)')
print(f'C/N stack:            {cn_stack.shape}   (lazy)')

icm_stack = _make_stack(icm_luts) if icm_luts else None

# ── Open napari ───────────────────────────────────────────────────────────────

viewer = napari.Viewer()
viewer.dims.axis_labels = ['t', 'z', 'y', 'x']

# Raw channels (hidden by default)
if not args.no_images:
    erk_files = sorted(ERK_DIR.glob(ERK_GLOB))[:N_T]
    if erk_files:
        _ep = tifffile.imread(str(erk_files[0]))
        viewer.add_image(
            da.stack([
                da.from_delayed(dask.delayed(tifffile.imread)(str(p)),
                                shape=_ep.shape, dtype=_ep.dtype)
                for p in erk_files
            ], axis=0),
            name='erk_raw', scale=SCALE,
            colormap='green', blending='additive', opacity=0.5, visible=False,
        )

    raw_files = sorted(RAW_DIR.glob(RAW_GLOB))[:N_T]
    if raw_files:
        _rp = tifffile.imread(str(raw_files[0]))
        viewer.add_image(
            da.stack([
                da.from_delayed(dask.delayed(tifffile.imread)(str(p)),
                                shape=_rp.shape, dtype=_rp.dtype)
                for p in raw_files
            ], axis=0),
            name='h2b_raw', scale=SCALE,
            colormap='gray', blending='additive', opacity=0.6, visible=False,
        )

# Label boundaries
lbl_frames = [
    da.from_delayed(dask.delayed(tifffile.imread)(str(p)),
                    shape=label_shape, dtype=_probe.dtype)
    for p in label_files
]
viewer.add_labels(
    da.stack(lbl_frames, axis=0), name='labels', scale=SCALE, opacity=0.15,
)

# ICM distance (hidden by default)
if icm_stack is not None:
    viewer.add_image(
        icm_stack, name='ICM distance (µm)', scale=SCALE,
        colormap='magma', contrast_limits=(0.0, icm_dist_hi),
        blending='additive', opacity=0.85, visible=False,
    )

# C/N ratio comparison layer (hidden by default)
# RdBu_r: high C/N (cytoplasmic, ERK active) → red  |  low C/N (nuclear, ERK inactive) → blue
viewer.add_image(
    cn_stack, name='ERK C/N ratio', scale=SCALE,
    colormap='RdBu_r',
    contrast_limits=(cn_clim_lo, cn_clim_hi),
    blending='additive', opacity=0.85, visible=False,
)

# Normalized nuclear ERK — main layer
# RdBu: low norm_nuc (nucleus empty, ERK active) → red  |  high norm_nuc (nucleus full, ERK inactive) → blue
# Signals run opposite directions (high C/N vs low norm_nuc both = ERK active),
# so opposite colormap names produce the same color convention: red = ERK active in both layers.
nuc_sym     = max(abs(nuc_lo - 1.0), abs(nuc_hi - 1.0))
nuc_clim_lo = max(0.0, 1.0 - nuc_sym)
nuc_clim_hi = 1.0 + nuc_sym
viewer.add_image(
    nuc_stack, name='norm. nuclear ERK', scale=SCALE,
    colormap='RdBu',
    contrast_limits=(nuc_clim_lo, nuc_clim_hi),
    blending='additive', opacity=0.85,
)

print('\nReady.')
print(f'  norm. nuclear ERK:  red=low (nucleus empty, ERK active)  blue=high (nucleus full, ERK inactive)')
print(f'    consistent with C/N layer: red = ERK active in both (opposite colormap names, same convention)')
print(f'    contrast limits (symmetric around 1.0): [{nuc_clim_lo:.3f}, {nuc_clim_hi:.3f}]')
print(f'    "on" reference = {ON_PCT}th-pct nuclear intensity per track')
print(f'  ERK C/N ratio:      hidden by default — toggle on to compare spatial patterns')
print(f'  Toggle "labels" for outlines, "erk_raw"/"h2b_raw" for raw channels')
napari.run()
