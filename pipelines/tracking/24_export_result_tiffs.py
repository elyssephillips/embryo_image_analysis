"""
Export full-timecourse result TIFFs for ERK C/N ratio and ICM distance.

Saves float32 TIFFs into {output_dir} with physical voxel size metadata
(ImageJ/napari-compatible):
  erk_cn_{VERSION}_4d.tif      — (T, Z, Y, X) float32, NaN = background
  erk_cn_{VERSION}_maxz.tif    — (T, Y, X) float32, NaN = background
  icm_dist_{VERSION}_4d.tif   — (T, Z, Y, X) float32, NaN = background
  icm_dist_{VERSION}_maxz.tif — (T, Y, X) float32, NaN = background

The 4D stacks can be large (several GB); use --no-4d to skip them and only
save the max-Z projected timecourses.

Usage
-----
  python 24_export_result_tiffs.py
  python 24_export_result_tiffs.py configs/tracking/dataset001_implantation.yaml
  python 24_export_result_tiffs.py --no-4d
"""

import sys
import argparse
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('config', nargs='?',
                    default='configs/tracking/dataset001_implantation.yaml')
parser.add_argument('--no-4d', action='store_true',
                    help='Skip 4D stacks; only save max-Z projected TIFFs')
args = parser.parse_args()

CONFIG_PATH = REPO_ROOT / args.config
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

LABEL_DIR  = Path(cfg['paths']['label_dir'])
LABEL_GLOB = cfg['paths']['label_glob']
OUTPUT_DIR = Path(cfg['paths']['output_dir'])
VERSION    = cfg['tracking']['input_version']
N_T        = cfg['microscopy']['n_timepoints']
VX_Z, VX_Y, VX_X = cfg['microscopy']['voxel_size_zyx']

# ── load ERK C/N ratio CSV ────────────────────────────────────────────────────
cn_csv = OUTPUT_DIR / f'erk_cn_ratio_{VERSION}.csv'
if not cn_csv.exists():
    sys.exit(f'\nERK C/N CSV not found: {cn_csv}\nRun 12_erk_cn_ratio.py first.')

print(f'Loading: {cn_csv}')
cn_df = pd.read_csv(cn_csv)
cn_df['label_id'] = cn_df['label_id'].astype('Int64')

print('Building per-timepoint label→C/N lookup tables...')
luts: dict[int, np.ndarray] = {}
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
icm_luts: dict[int, np.ndarray] = {}
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
    print(f'ICM distance CSV not found ({icm_dist_csv.name}) — skipping ICM distance export')

# ── label files ───────────────────────────────────────────────────────────────
label_files = sorted(LABEL_DIR.glob(LABEL_GLOB))[:N_T]
assert len(label_files) == N_T, f'Expected {N_T} label files, got {len(label_files)}'

_probe = tifffile.imread(str(label_files[0]))
Z, Y, X = _probe.shape
print(f'Label frame shape: ({Z}, {Y}, {X})')
if not args.no_4d:
    gb_per_stack = N_T * Z * Y * X * 4 / 1e9
    print(f'4D stack size estimate: ~{gb_per_stack:.1f} GB per channel')


def _map_lut_to_frame(label_path: Path, lut: np.ndarray) -> np.ndarray:
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


def _save_tiff(arr: np.ndarray, path: Path, axes: str) -> None:
    res_xy = 1.0 / VX_X   # pixels per µm
    tifffile.imwrite(
        str(path),
        arr,
        imagej=True,
        resolution=(res_xy, res_xy),
        metadata={'axes': axes, 'spacing': VX_Z, 'unit': 'um', 'loop': False},
    )
    size_mb = path.stat().st_size / 1e6
    print(f'  saved {path.name}  ({size_mb:.0f} MB)')


# ── build and save stacks ─────────────────────────────────────────────────────
channels = [('erk_cn', luts, 'ERK C/N ratio')]
if icm_luts:
    channels.append(('icm_dist', icm_luts, 'ICM distance'))

for name, ch_luts, label in channels:
    print(f'\nProcessing {label} ({N_T} timepoints)...')

    stack_4d   = np.full((N_T, Z, Y, X), np.nan, dtype=np.float32) if not args.no_4d else None
    stack_maxz = np.full((N_T, Y, X), np.nan, dtype=np.float32)

    for t in range(N_T):
        if (t + 1) % 10 == 0 or t == N_T - 1:
            print(f'  frame {t + 1}/{N_T}', end='\r', flush=True)
        vol = _map_lut_to_frame(label_files[t], ch_luts.get(t))
        if stack_4d is not None:
            stack_4d[t] = vol
        stack_maxz[t] = np.nanmax(vol, axis=0)
    print()

    if stack_4d is not None:
        out_4d = OUTPUT_DIR / f'{name}_{VERSION}_4d.tif'
        print(f'  writing 4D TIFF ...')
        _save_tiff(stack_4d, out_4d, axes='TZYX')

    out_maxz = OUTPUT_DIR / f'{name}_{VERSION}_maxz.tif'
    print(f'  writing max-Z TIFF ...')
    _save_tiff(stack_maxz, out_maxz, axes='TYX')

print(f'\nDone. Files saved to: {OUTPUT_DIR}')
