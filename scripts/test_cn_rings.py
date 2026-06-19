"""
Visualise Voronoi-constrained cytoplasmic rings for C/N ratio analysis.

For each nucleus:
  - Dilate outward by RING_RADIUS_UM in physical space (anisotropy-corrected)
  - Keep only voxels closer to this nucleus than to any other (Voronoi constraint)
  - Exclude the nuclear mask itself

Opens napari with:
  - nuclear channel (grey)
  - ERK channel (green)
  - nuclear masks (coloured by label ID)
  - cytoplasmic rings (same colours as nuclei, so ring = cytoplasm of that cell)

Run with: conda run -n napari_env python3 scripts/test_cn_rings.py
"""

import yaml
import numpy as np
import napari
import tifffile
from pathlib import Path
from scipy.ndimage import distance_transform_edt

REPO_ROOT   = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

VX_Z, VX_Y, VX_X = cfg['microscopy']['voxel_size_zyx']
SAMPLING          = (VX_Z, VX_Y, VX_X)
SCALE             = (VX_Z, VX_Y, VX_X)
RING_RADIUS_UM    = 1.0   # µm — change this to test different widths
# ERK background threshold: ring voxels below this are extracellular and excluded.
# Set to None to auto-estimate from the blank z-planes at the top of cam_short
# (z=0..2 are outside the embryo, so their mean ≈ camera background).
ERK_BG_THRESHOLD  = None

# ── Load t=0 ─────────────────────────────────────────────────────────────────

long_path  = sorted(Path(cfg['paths']['raw_dir']).glob(cfg['paths']['raw_glob']))[0]
short_path = sorted(Path(cfg['paths']['erk_dir']).glob(cfg['paths']['erk_glob']))[0]
label_path = sorted(Path(cfg['paths']['label_dir']).glob(cfg['paths']['label_glob']))[0]

print(f'Loading t=0...')
nuclear = tifffile.imread(long_path).astype(np.float32)
erk     = tifffile.imread(short_path).astype(np.float32)
labels  = tifffile.imread(label_path)

print(f'  nuclear: {nuclear.shape}  ERK: {erk.shape}  labels: {labels.shape}')
n_nuclei = len(np.unique(labels)) - 1  # exclude background
print(f'  {n_nuclei} nuclei at t=0')

# ── Compute Voronoi-constrained cytoplasmic rings ─────────────────────────────

print(f'Computing {RING_RADIUS_UM} µm Voronoi-constrained rings...')

background = (labels == 0)

# Distance from each background voxel to nearest nucleus surface (µm),
# and index of the nearest nucleus voxel for Voronoi assignment.
dist, nearest_idx = distance_transform_edt(
    background, sampling=SAMPLING, return_indices=True
)

# Which label does the nearest nucleus voxel belong to?
voronoi = labels[tuple(nearest_idx)]   # Voronoi assignment for every background voxel

# ERK foreground mask: exclude extracellular voxels from the ring.
# The blank z-planes at the top of cam_short (before the embryo appears) give a
# clean estimate of camera background — anything below 3× that is extracellular.
if ERK_BG_THRESHOLD is None:
    bg_level = erk[:3].mean()           # mean of top 3 z-planes = camera background
    ERK_BG_THRESHOLD = bg_level * 3.0
print(f'ERK background level: {erk[:3].mean():.1f}  →  threshold: {ERK_BG_THRESHOLD:.1f}')
in_cell = erk >= ERK_BG_THRESHOLD

# Ring: background voxels within RING_RADIUS_UM of their assigned nucleus,
# restricted to voxels with ERK signal (i.e., inside a cell)
ring_mask = background & (dist <= RING_RADIUS_UM) & in_cell

# Build a label image for the rings (same IDs as nuclear labels for matching colours)
ring_labels = np.where(ring_mask, voronoi, 0).astype(labels.dtype)

# Also build an unmasked ring for comparison
ring_mask_unmasked  = background & (dist <= RING_RADIUS_UM)
ring_labels_raw     = np.where(ring_mask_unmasked, voronoi, 0).astype(labels.dtype)

# ── Compute a quick C/N ratio summary ────────────────────────────────────────

label_ids = np.unique(labels)[1:]   # skip background
cn_ratios = []
for lid in label_ids:
    nuc_mean  = erk[labels == lid].mean()
    ring_vox  = erk[ring_labels == lid]
    if len(ring_vox) == 0:
        continue
    cyto_mean = ring_vox.mean()
    cn_ratios.append(cyto_mean / nuc_mean if nuc_mean > 0 else np.nan)

cn_ratios = np.array(cn_ratios)
print(f'\nC/N ratio summary at t=0 (ring={RING_RADIUS_UM} µm):')
print(f'  median: {np.nanmedian(cn_ratios):.3f}')
print(f'  range:  {np.nanmin(cn_ratios):.3f} – {np.nanmax(cn_ratios):.3f}')
print(f'  nuclei with no ring voxels: {n_nuclei - len(cn_ratios)}')

ring_vox_counts = [(ring_labels == lid).sum() for lid in label_ids]
print(f'  ring voxels per nucleus: min={min(ring_vox_counts)}, '
      f'median={int(np.median(ring_vox_counts))}, max={max(ring_vox_counts)}')

# ── Open napari ───────────────────────────────────────────────────────────────

print('\nOpening napari...')
viewer = napari.Viewer(title=f'C/N rings — t=0, ring={RING_RADIUS_UM} µm')

viewer.add_image(nuclear,     name='nuclear',          scale=SCALE,
                 colormap='gray',  opacity=0.7)
viewer.add_image(erk,         name='ERK (cam_short)',  scale=SCALE,
                 colormap='green', opacity=0.6, blending='additive')
viewer.add_labels(labels,     name='nuclear masks',    scale=SCALE, opacity=0.5)
viewer.add_labels(ring_labels,     name=f'cyto rings — masked ({RING_RADIUS_UM} µm)',
                  scale=SCALE, opacity=0.7)
viewer.add_labels(ring_labels_raw, name=f'cyto rings — raw ({RING_RADIUS_UM} µm)',
                  scale=SCALE, opacity=0.7, visible=False)

print('Layers: nuclear | ERK | nuclear masks | cyto rings (masked) | cyto rings (raw)')
print('Toggle "raw" vs "masked" rings to see the effect of the ERK threshold.')
print('Masked rings should not extend into extracellular background.')
napari.run()
