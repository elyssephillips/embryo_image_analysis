"""
napari check of the ICM region from 01_extract_icm_region.py.

Layers (all in image space, µm-scaled, so they overlay):
  raw          nuclear channel
  TE nuclei    instance labels (tracked population)
  ICM nuclei   class-2 mask exactly as 01 builds it (same icm_mask function)
  ICM hull     convex hull of the ICM nuclei - the region distances are measured to
  TE centroids points coloured by icm_edge_dist_um (blue = at/inside the edge,
               red = far); hover a point to read all three distances
  ICM centroid

Frames are end-padded to the deepest one (only matters for stack5, whose
per-frame crops differ in depth; later stacks share one uniform z crop); the time slider steps through the selected timepoints in order
(mapping printed at startup and in the window title).

Usage
-----
  python pipelines/tracking/01_view_icm_region.py                  # t = 0 30 60 90 120
  python pipelines/tracking/01_view_icm_region.py --t 73 74 75 76  # specific frames
  python pipelines/tracking/01_view_icm_region.py --range 70 80    # inclusive range
  python pipelines/tracking/01_view_icm_region.py --ds 1           # full xy resolution
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import napari
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
icm = importlib.import_module('01_extract_icm_region')  # also loads the config (--config)
cfg, VOX, GRID_VOX = icm.cfg, icm.VOX, icm.GRID_VOX

parser = argparse.ArgumentParser()
parser.add_argument('--t', type=int, nargs='+', default=icm.QC_TIMEPOINTS)
parser.add_argument('--range', type=int, nargs=2, metavar=('START', 'END'))
parser.add_argument('--ds', type=int, default=2, help='xy downsample for raw/labels (memory)')
args = parser.parse_args()
timepoints = list(range(args.range[0], args.range[1] + 1)) if args.range else args.t

RAW_DIR = Path(cfg['paths']['raw_dir'])
raw_files   = sorted(RAW_DIR.glob(cfg['paths']['raw_glob']))
pred_files  = sorted(icm.PRED_DIR.glob(icm.PRED_GLOB))
label_files = sorted(icm.LABEL_DIR.glob(icm.LABEL_GLOB))

crop_info = icm.load_crop_info(cfg)
shift_um = {int(k): v.get('debris_shift_slices', 0) * VOX[0] for k, v in crop_info.items()}

feat = pd.read_csv(icm.FEATURES_CSV, usecols=['t', 'label_id', 'z_um', 'y_um', 'x_um'])
dist = pd.read_csv(icm.OUT_CSV)
pts_df = feat.merge(dist, on=['t', 'label_id'])

ds = args.ds
raw_l, te_l, icm_l, hull_l, points, props, centroids = [], [], [], [], [], [], []
for i, t in enumerate(timepoints):
    print(f'loading t={t} (slider {i})', flush=True)
    raw  = tifffile.imread(raw_files[t])
    pred = tifffile.imread(pred_files[t])
    te   = tifffile.imread(label_files[t])
    mask, _, _ = icm.icm_mask(pred, te)

    grid_shape = (pred.shape[0], -(-pred.shape[1] // icm.GRID_DS), -(-pred.shape[2] // icm.GRID_DS))
    if mask.sum() >= 4:
        mask_um = np.argwhere(mask) * VOX
        hull = icm.hull_inside_grid(ConvexHull(mask_um), grid_shape)
        centroids.append([i, *mask_um.mean(axis=0)])
    else:
        hull = np.zeros(grid_shape, dtype=bool)

    raw_l.append(raw[:, ::ds, ::ds]); te_l.append(te[:, ::ds, ::ds])
    icm_l.append(mask[:, ::ds, ::ds]); hull_l.append(hull)

    rows = pts_df[pts_df['t'] == t]
    zyx = rows[['z_um', 'y_um', 'x_um']].to_numpy() + [shift_um.get(t, 0.0), 0, 0]
    points.append(np.column_stack([np.full(len(rows), i), zyx]))
    props.append(rows)


def stack(frames, dtype=None):
    nz = max(f.shape[0] for f in frames)
    return np.stack([np.pad(f, [(0, nz - f.shape[0])] + [(0, 0)] * (f.ndim - 1)) for f in frames]
                    ).astype(dtype or frames[0].dtype)


img_scale  = (1, VOX[0], VOX[1] * ds, VOX[2] * ds)
hull_scale = (1, *GRID_VOX)
props_df = pd.concat(props, ignore_index=True)

viewer = napari.Viewer(title='ICM region  —  slider → t: ' +
                       ', '.join(f'{i}→{t}' for i, t in enumerate(timepoints)))
viewer.dims.axis_labels = ['slider', 'z', 'y', 'x']
viewer.add_image(stack(raw_l), name='raw', scale=img_scale, colormap='gray', blending='additive')
viewer.add_labels(stack(te_l), name='TE nuclei', scale=img_scale, opacity=0.25)
viewer.add_labels(stack(icm_l, np.uint8), name='ICM nuclei', scale=img_scale, opacity=0.6)
hull_layer = viewer.add_labels(stack(hull_l, np.uint8) * 2, name='ICM hull', scale=hull_scale, opacity=0.5)
hull_layer.contour = 1
viewer.add_points(np.concatenate(points), name='TE centroids', size=4,
                  features=props_df[['t', 'label_id', 'icm_edge_dist_um',
                                     'icm_nearest_dist_um', 'icm_centroid_dist_um']],
                  face_color='icm_edge_dist_um', face_colormap='coolwarm',
                  face_contrast_limits=(-20, 60), border_width=0)
if centroids:
    viewer.add_points(np.array(centroids), name='ICM centroid', size=8,
                      symbol='cross', face_color='yellow')

print('\nslider index → t: ' + ', '.join(f'{i}→{t}' for i, t in enumerate(timepoints)))
print('Hover a TE centroid to read its distances; toggle "ICM hull" contour (layer controls) '
      'to see the filled region.')
napari.run()
