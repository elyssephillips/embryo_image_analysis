"""
Open napari with flow field overlaid on raw image + segmentation.

Layers
------
  raw              — raw fluorescence stack
  labels           — track-relabeled stack (voxel value = track_id, consistent across time)
  tracks           — nucleus trajectories
  local flow speed — nuclei coloured by collective flow speed
  autonomous speed — nuclei coloured by autonomous (cell-intrinsic) speed
  local vectors    — arrows showing collective flow direction
  autonomous vectors — arrows showing autonomous motion direction

Usage
-----
  python 10_napari_flow_overlay.py                              # loads images (slow)
  python 10_napari_flow_overlay.py --no-images                  # flow data only (fast)
  python 10_napari_flow_overlay.py configs/tracking/control_2.yaml
"""

import sys
import argparse
from pathlib import Path
import glob
import yaml
import numpy as np
import pandas as pd
import napari
import tifffile
from skimage.io import imread

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# ── config ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('config', nargs='?',
                    default='configs/tracking/dataset001_implantation.yaml')
parser.add_argument('--no-images', action='store_true',
                    help='Skip loading raw + label stacks (faster startup)')
args = parser.parse_args()

CONFIG_PATH = REPO_ROOT / args.config
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

RAW_DIR          = Path(cfg['paths']['raw_dir'])
RAW_GLOB         = cfg['paths']['raw_glob']
TRACK_LABELS_TIF = Path(cfg['paths']['track_labels_tiff'])
OUTPUT_DIR       = Path(cfg['paths']['output_dir'])
VERSION    = cfg['tracking']['input_version']
N_T        = cfg['microscopy']['n_timepoints']
VX_Z, VX_Y, VX_X = cfg['microscopy']['voxel_size_zyx']
FRAME_MIN  = cfg['tracking']['frame_interval_min']

SCALE = (1, VX_Z, VX_Y, VX_X)   # T, Z, Y, X in µm

# ── load flow CSV ─────────────────────────────────────────────────────────────
flow_csv = OUTPUT_DIR / f'motion_flow_{VERSION}.csv'
print(f'Loading flow data: {flow_csv}')
flow_df = pd.read_csv(flow_csv)

has_orig  = 'z_um_orig' in flow_df.columns
z_col = 'z_um_orig' if has_orig else 'z_um'
y_col = 'y_um_orig' if has_orig else 'y_um'
x_col = 'x_um_orig' if has_orig else 'x_um'

# Rows with valid velocity (not first frame of track)
valid = flow_df.dropna(subset=['dy_um', 'dx_um', 'flow_alignment']).copy()

# ── build napari layer data ───────────────────────────────────────────────────
# Points: (N, 4) = [t, z, y, x] in physical µm — matches image scale
point_coords = valid[['t', z_col, y_col, x_col]].values.astype(float)

# Raw velocity vectors: (N, 2, 4) = [[t, z, y, x], [0, 0, dy, dx]]
# ARROW_SCALE: multiply µm displacement to get visible arrow length.
# Reduce to shorten arrows; increase to lengthen.
ARROW_SCALE = 1.5

def _make_velocity_vectors():
    starts     = valid[['t', z_col, y_col, x_col]].values.astype(float)
    directions = np.zeros_like(starts)
    directions[:, 2] = valid['dy_um'].values * ARROW_SCALE
    directions[:, 3] = valid['dx_um'].values * ARROW_SCALE
    return np.stack([starts, directions], axis=1)   # (N, 2, 4)

velocity_vectors = _make_velocity_vectors()

# Tracks layer: (N, 5) = [track_id, t, z, y, x]
track_data = (
    flow_df
    .sort_values(['track_id', 't'])
    [['track_id', 't', z_col, y_col, x_col]]
    .values
)

# ── load images (optional) ────────────────────────────────────────────────────
raw_stack   = None
label_stack = None

if not args.no_images:
    raw_files = sorted(glob.glob(str(RAW_DIR / RAW_GLOB)))[:N_T]
    print(f'Loading {len(raw_files)} raw tiffs...', flush=True)
    raw_stack = np.stack([imread(f) for f in raw_files])
    print(f'Raw:    {raw_stack.shape}')

    print(f'Loading track labels: {TRACK_LABELS_TIF}', flush=True)
    label_stack = tifffile.imread(str(TRACK_LABELS_TIF))
    print(f'Labels: {label_stack.shape}')
else:
    print('Skipping image loading (--no-images)')

# ── open napari ───────────────────────────────────────────────────────────────
viewer = napari.Viewer()

if raw_stack is not None:
    viewer.add_image(raw_stack,   name='raw',    scale=SCALE,
                     colormap='gray', opacity=0.6)
if label_stack is not None:
    viewer.add_labels(label_stack, name='labels', scale=SCALE, opacity=0.3)

viewer.add_tracks(
    track_data, name='tracks',
    tail_length=8, head_length=0, tail_width=2,
)

# Raw velocity arrows coloured by flow alignment
#   red  (+1) = arrow moving with the collective field
#   white (0) = perpendicular to field
#   blue (-1) = arrow moving against the field
viewer.add_vectors(
    velocity_vectors, name='velocity (alignment)',
    features={'alignment': valid['flow_alignment'].values},
    edge_color='alignment', edge_colormap='RdBu_r',
    edge_contrast_limits=(-1, 1),
    edge_width=2, length=1, opacity=0.8,
)

# Speed (hidden by default)
viewer.add_points(
    point_coords, name='speed',
    features={'speed': valid['speed_um_per_min'].values},
    face_color='speed', face_colormap='plasma',
    size=4, opacity=0.9, border_width=0,
    visible=False,
)

# Nuclear volume (hidden by default)
if 'area_um3' in valid.columns:
    viewer.add_points(
        point_coords, name='volume (µm³)',
        features={'volume': valid['area_um3'].values},
        face_color='volume', face_colormap='RdYlBu_r',
        size=4, opacity=0.9, border_width=0,
        visible=False,
    )

print('\nLayers loaded. Toggle visibility in the napari layer panel.')
print('  velocity arrows: red=with field  white=perpendicular  blue=against field')
print(f'  ARROW_SCALE = {ARROW_SCALE}  — edit near top of script to resize arrows')
napari.run()