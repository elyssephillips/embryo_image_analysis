"""
Quick alignment test: load t=0 of cam_long, cam_short (z-cropped), and instance labels
into napari to visually verify the ERK channel aligns with the nuclear segmentation.

Run with:  conda run -n napari_env python3 scripts/test_erk_alignment.py
"""

import yaml
import numpy as np
import napari
import tifffile
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

VX_Z, VX_Y, VX_X = cfg['microscopy']['voxel_size_zyx']
SCALE = (VX_Z, VX_Y, VX_X)

long_path  = sorted(Path(cfg['paths']['raw_dir']).glob(cfg['paths']['raw_glob']))[0]
short_path = sorted(Path(cfg['paths']['erk_dir']).glob(cfg['paths']['erk_glob']))[0]
label_path = sorted(Path(cfg['paths']['label_dir']).glob(cfg['paths']['label_glob']))[0]

print(f'Loading t=0:')
print(f'  nuclear:  {long_path.name}')
print(f'  ERK:      {short_path.name}')
print(f'  labels:   {label_path.name}')

nuclear = tifffile.imread(long_path)                      # (61, Y, X)
erk    = tifffile.imread(short_path)                      # (61, Y, X)
labels = tifffile.imread(label_path)                      # (61, Y, X)

print(f'\nShapes after crop:')
print(f'  nuclear: {nuclear.shape}')
print(f'  ERK:     {erk.shape}')
print(f'  labels:  {labels.shape}')
assert nuclear.shape == erk.shape == labels.shape, \
    f"Shape mismatch! nuclear={nuclear.shape}, erk={erk.shape}, labels={labels.shape}"
print('  Shapes match ✓')

viewer = napari.Viewer(title='ERK alignment test — t=0')
viewer.add_image(nuclear, name='nuclear (cam_long)',  scale=SCALE,
                 colormap='gray',  opacity=0.8)
viewer.add_image(erk,     name='ERK (cam_short)',     scale=SCALE,
                 colormap='green', opacity=0.6, blending='additive')
viewer.add_labels(labels, name='segmentation',        scale=SCALE,
                  opacity=0.4)

print('\nnapari open. Check that:')
print('  1. ERK channel (green) shows cytoplasmic signal around nuclei')
print('  2. Segmentation labels sit inside the nuclear signal')
print('  3. ERK is excluded from (or lower in) labelled nuclei')
napari.run()
