"""
Compute per-nucleus ERK cytoplasm-to-nucleus (C/N) ratio from the KTR biosensor channel.

For each nucleus at each timepoint:
  1. Nuclear ERK mean  — mean ERK intensity within the instance segmentation mask
  2. Cytoplasmic ring  — voxels within RING_RADIUS_UM of the nucleus surface,
                         Voronoi-constrained (assigned to nearest nucleus only),
                         with extracellular voxels excluded via ERK threshold
  3. C/N ratio         — cyto_mean / nuc_mean

Output joined to tracks_linked_c23.csv so every row has a track_id.

Run with: conda run -n napari_env python3 pipelines/tracking/12_erk_cn_ratio.py
"""

import yaml
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from scipy.ndimage import distance_transform_edt

REPO_ROOT   = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

VX_Z, VX_Y, VX_X  = cfg['microscopy']['voxel_size_zyx']
SAMPLING           = (VX_Z, VX_Y, VX_X)
N_TIMEPOINTS       = cfg['microscopy']['n_timepoints']
RING_RADIUS_UM     = cfg['erk']['ring_radius_um']
BG_MULTIPLIER      = cfg['erk']['bg_multiplier']

erk_dir    = Path(cfg['paths']['erk_dir'])
label_dir  = Path(cfg['paths']['label_dir'])
linked_dir = Path(cfg['paths']['linked_tracks_dir'])
out_dir    = Path(cfg['paths']['output_dir'])
version    = cfg['tracking']['input_version']

erk_files   = sorted(erk_dir.glob(cfg['paths']['erk_glob']))[:N_TIMEPOINTS]
label_files = sorted(label_dir.glob(cfg['paths']['label_glob']))[:N_TIMEPOINTS]

assert len(erk_files)   == N_TIMEPOINTS, f'Expected {N_TIMEPOINTS} ERK files, found {len(erk_files)}'
assert len(label_files) == N_TIMEPOINTS, f'Expected {N_TIMEPOINTS} label files, found {len(label_files)}'

# ── ERK background threshold (estimated once from t=0) ────────────────────────

erk_t0    = tifffile.imread(erk_files[0]).astype(np.float32)
bg_level  = erk_t0[:3].mean()
ERK_THRESH = bg_level * BG_MULTIPLIER
print(f'ERK background: {bg_level:.1f}  →  threshold: {ERK_THRESH:.1f}  '
      f'(multiplier={BG_MULTIPLIER})')

# ── Load tracks CSV for joining ───────────────────────────────────────────────

tracks_path = linked_dir / f'tracks_{version}.csv'
tracks_df   = pd.read_csv(tracks_path)[['track_id', 't', 'label_id']]
tracks_df['label_id'] = tracks_df['label_id'].astype('Int64')
print(f'Loaded {tracks_df["track_id"].nunique()} tracks from {tracks_path.name}')

# ── Per-timepoint C/N computation ─────────────────────────────────────────────

records = []

for t, (erk_path, label_path) in enumerate(zip(erk_files, label_files)):
    print(f't={t:03d}', flush=True)

    erk    = tifffile.imread(erk_path).astype(np.float32)
    labels = tifffile.imread(label_path)

    label_ids = np.unique(labels)
    label_ids = label_ids[label_ids != 0]
    if len(label_ids) == 0:
        continue

    background = (labels == 0)
    dist, nearest_idx = distance_transform_edt(
        background, sampling=SAMPLING, return_indices=True
    )
    voronoi  = labels[tuple(nearest_idx)]
    in_cell  = erk >= ERK_THRESH
    ring_mask = background & (dist <= RING_RADIUS_UM) & in_cell

    for lid in label_ids:
        nuc_vox  = erk[labels == lid]
        ring_vox = erk[ring_mask & (voronoi == lid)]

        nuc_mean  = float(nuc_vox.mean())
        nuc_vox_n = int(len(nuc_vox))
        ring_mean  = float(ring_vox.mean()) if len(ring_vox) > 0 else np.nan
        ring_vox_n = int(len(ring_vox))
        cn_ratio   = ring_mean / nuc_mean if (ring_vox_n > 0 and nuc_mean > 0) else np.nan

        records.append({
            't':           t,
            'label_id':    int(lid),
            'erk_nuc_mean':  nuc_mean,
            'erk_nuc_vox':   nuc_vox_n,
            'erk_cyto_mean': ring_mean,
            'erk_cyto_vox':  ring_vox_n,
            'erk_cn_ratio':  cn_ratio,
        })

# ── Join to tracks and save ───────────────────────────────────────────────────

erk_df = pd.DataFrame(records)
erk_df['label_id'] = erk_df['label_id'].astype('Int64')

merged = erk_df.merge(tracks_df, on=['t', 'label_id'], how='left')

n_matched   = merged['track_id'].notna().sum()
n_unmatched = merged['track_id'].isna().sum()
print(f'\nMatched {n_matched} rows to track IDs, {n_unmatched} unmatched '
      f'(label present in labels but not in tracks CSV)')

out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f'erk_cn_ratio_{version}.csv'
merged.to_csv(out_path, index=False)
print(f'Saved: {out_path}')

# ── Summary ───────────────────────────────────────────────────────────────────

cn = merged['erk_cn_ratio'].dropna()
print(f'\nC/N ratio summary (all timepoints, ring={RING_RADIUS_UM} µm):')
print(f'  n rows:          {len(cn)}')
print(f'  median C/N:      {cn.median():.3f}')
print(f'  range:           {cn.min():.3f} – {cn.max():.3f}')
print(f'  nuclei with no ring voxels: {merged["erk_cyto_vox"].eq(0).sum()}')
cyto_counts = merged['erk_cyto_vox']
print(f'  ring voxels/nucleus: min={cyto_counts.min()}, '
      f'median={cyto_counts.median():.0f}, max={cyto_counts.max()}')
