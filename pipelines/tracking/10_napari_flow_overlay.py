"""
Open napari with flow field overlaid on raw image + segmentation.

Layers
------
  raw                  — raw fluorescence stack (optional)
  labels               — track-relabeled stack (voxel value = track_id, consistent across time)
  tracks               — nucleus trajectories
  velocity (alignment) — arrows showing each nucleus's motion direction,
                         coloured red (with field) → white (perpendicular) → blue (against field)
  speed                — nuclei coloured by instantaneous speed (hidden by default)
  volume (µm³)         — nuclei coloured by nuclear volume (hidden by default)

Tunable parameters (near top of script)
----------------------------------------
  SMOOTH_FRAMES  rolling window for velocity smoothing (default 4; set to 1 for raw)
  ARROW_SCALE    arrow length multiplier (default 1.5)

Usage
-----
  python 10_napari_flow_overlay.py                                          # loads images (slow)
  python 10_napari_flow_overlay.py --no-images                              # flow data only (fast)
  python 10_napari_flow_overlay.py configs/tracking/dataset001_implantation.yaml
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
ICM_ZYX    = (cfg.get('biology') or {}).get('icm_centroid_t30_zyx')  # None until set

SCALE = (1, VX_Z, VX_Y, VX_X)   # T, Z, Y, X in µm

# ── load flow CSV ─────────────────────────────────────────────────────────────
flow_csv = OUTPUT_DIR / f'motion_flow_{VERSION}.csv'
print(f'Loading flow data: {flow_csv}')
flow_df = pd.read_csv(flow_csv)

has_orig  = 'z_um_orig' in flow_df.columns
z_col = 'z_um_orig' if has_orig else 'z_um'
y_col = 'y_um_orig' if has_orig else 'y_um'
x_col = 'x_um_orig' if has_orig else 'x_um'

# ── smooth velocities per track ───────────────────────────────────────────────
# Rolling mean over the last SMOOTH_FRAMES frames reduces per-frame noise.
# Set to 1 to use raw single-frame displacements.
SMOOTH_FRAMES = 4

flow_df = flow_df.sort_values(['track_id', 't'])
for col in ['dy_um', 'dx_um', 'local_vy_um', 'local_vx_um']:
    flow_df[col] = (
        flow_df.groupby('track_id')[col]
        .transform(lambda x: x.rolling(SMOOTH_FRAMES, min_periods=1).mean())
    )

# Recompute flow_alignment from smoothed velocities
own_spd   = np.sqrt(flow_df['dy_um']**2       + flow_df['dx_um']**2)
loc_spd   = np.sqrt(flow_df['local_vy_um']**2 + flow_df['local_vx_um']**2)
dot       = flow_df['dy_um'] * flow_df['local_vy_um'] + flow_df['dx_um'] * flow_df['local_vx_um']
denom     = own_spd * loc_spd
flow_df['flow_alignment'] = np.where(denom > 0, dot / denom, np.nan)

print(f'Velocities smoothed over {SMOOTH_FRAMES} frames per track')

# ── ICM distance (per track, fixed at t=30 ± T_WINDOW) ───────────────────────
# Tracks not observed within the window get NaN — not classified.
T_REF    = 30
T_WINDOW = 2   # ±frames

if ICM_ZYX is not None:
    iz, iy, ix = ICM_ZYX
    within_window = flow_df[flow_df['t'].between(T_REF - T_WINDOW, T_REF + T_WINDOW)]
    ref_pos = (
        within_window
        .assign(_dt=lambda d: (d['t'] - T_REF).abs())
        .sort_values(['track_id', '_dt', 't'])
        .groupby('track_id')
        .first()
        .reset_index()
        [['track_id', z_col, y_col, x_col]]
    )
    ref_pos['icm_dist_um'] = np.sqrt(
        (ref_pos[z_col] - iz)**2 +
        (ref_pos[y_col] - iy)**2 +
        (ref_pos[x_col] - ix)**2
    )
    flow_df = flow_df.merge(ref_pos[['track_id', 'icm_dist_um']], on='track_id', how='left')
    n_total  = flow_df['track_id'].nunique()
    n_scored = ref_pos['track_id'].nunique()
    print(f'ICM centroid at t={T_REF}: Z={iz} Y={iy} X={ix} µm')
    print(f'  {n_scored}/{n_total} tracks scored — {n_total - n_scored} outside ±{T_WINDOW} frame window → NaN')
else:
    flow_df['icm_dist_um'] = np.nan
    print('ICM centroid not set — skipping distance layer (set biology.icm_centroid_t30_zyx in config)')

# Rows with valid smoothed velocity
valid = flow_df.dropna(subset=['dy_um', 'dx_um', 'flow_alignment']).copy()

# ── build napari layer data ───────────────────────────────────────────────────
# Points: (N, 4) = [t, z, y, x] in physical µm — matches image scale
point_coords = valid[['t', z_col, y_col, x_col]].values.astype(float)

# Velocity vectors: (N, 2, 4) = [[t, z, y, x], [0, 0, dy, dx]]
# ARROW_SCALE: multiply µm displacement to get visible arrow length.
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
viewer.dims.axis_labels = ['t', 'z', 'y', 'x']

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

# Z position — shows dorsal/ventral or apical/basal stratification
viewer.add_points(
    point_coords, name='Z position (µm)',
    features={'z': valid[z_col].values},
    face_color='z', face_colormap='RdYlBu',
    size=4, opacity=0.9, border_width=0,
    visible=False,
)

# Distance from ICM at t=30 — regional TE subpopulation proxy
# Uses a separate filtered coordinate array — NaN rows cause napari to render nothing
if ICM_ZYX is not None and valid['icm_dist_um'].notna().any():
    icm_valid  = valid[valid['icm_dist_um'].notna()]
    icm_coords = icm_valid[['t', z_col, y_col, x_col]].values.astype(float)
    viewer.add_points(
        icm_coords, name='ICM distance (µm)',
        features={'dist': icm_valid['icm_dist_um'].values},
        face_color='dist', face_colormap='magma',
        size=4, opacity=0.9, border_width=0,
        visible=False,
    )

print('\nLayers loaded. Toggle visibility in the napari layer panel.')
print('  velocity arrows: red=with field  white=perpendicular  blue=against field')
print(f'  ARROW_SCALE = {ARROW_SCALE}  — edit near top of script to resize arrows')
if ICM_ZYX is None:
    print('  ICM distance layer inactive — set biology.icm_centroid_t30_zyx in config')
napari.run()