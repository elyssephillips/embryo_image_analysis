"""
napari view of the nucleus meshes from 01_extract_meshes.py.

Layers:
  meshes   every nucleus surface, coloured by --color (a mesh_features.csv
           column, or 'label' for one Glasbey colour per label id -
           src/track_colors.py; labels layer matches)
  labels   instance labels (hidden) - toggle on to compare with the voxels
  raw      nuclear channel (hidden)

REGISTERED = True (default) puts the meshes in 02_register_centroids.py's
rigid-registered frame (transforms.npz, after undoing crop_info's debris
z-shift), which holds the embryo still for watching the whole movie. Images
aren't registered, so they can't be shown then. REGISTERED = False keeps each
frame's own image space (µm), so meshes overlay that frame's labels/raw
(SHOW_IMAGES = True) - but the embryo drifts and turns between frames (and on
stack5 each frame also has its own z-crop).

Opens in 3D. Images are end-padded to the deepest frame (only matters for
stack5's per-frame crops; later stacks share one uniform z crop); the time slider steps through the selected
timepoints in order (mapping printed at startup).

Usage
-----
Edit the SETTINGS block below and hit Run in VS Code - defaults: whole movie,
registered, meshes only, coloured by sphericity. Loads in ~10-15 s; then drag
the time slider or press its play button (right-click it to set the frame
rate). To save a video use Plugins > napari-animation (Wizard): capture a
keyframe at the first frame, another at the last, steps between = number of
frames, save as .mp4. Only the current frame is drawn, and meshes are simplified for display
(DISPLAY_GRID_UM) - set it to 0 for full detail when looking at a few frames.
Or override from a terminal:
  python pipelines/tracking/01_view_meshes.py --t 60 61 62            # specific frames
  python pipelines/tracking/01_view_meshes.py --range 70 80           # inclusive range
  python pipelines/tracking/01_view_meshes.py --step 2                # every 2nd frame
  python pipelines/tracking/01_view_meshes.py --unregistered --images --t 60   # overlay on raw/labels
  python pipelines/tracking/01_view_meshes.py --color mesh_solidity   # colour by a shape measure
Which dataset: pipelines/tracking/active_dataset.yaml (or --config X.yaml).
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import napari

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.tracking_config import load_tracking_config
from src import mesh_display
from src.track_colors import GLASBEY_CMAP, glasbey_values, label_colormap

cfg, CONFIG_PATH = load_tracking_config()

# ── SETTINGS - edit these and hit Run in VS Code ─────────────────────────────
TIMEPOINTS  = None               # None = whole movie; or a list, e.g. [60, 61, 62]
STEP        = 1                  # every Nth frame (whole movie / --range only)
REGISTERED  = True               # hold the embryo still (rigid registration)
COLOR       = 'mesh_sphericity'  # any mesh_features.csv column, or 'label' (Glasbey by label id)
SHOW_IMAGES = False              # raw + labels underneath (needs REGISTERED = False)
DISPLAY_GRID_UM = 1.0            # simplify meshes for display (µm); 0 = full resolution
DS          = 2                  # xy downsample for raw/labels (memory)
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--t', type=int, nargs='+', default=None)
parser.add_argument('--range', type=int, nargs=2, metavar=('START', 'END'))
parser.add_argument('--step', type=int, default=None)
parser.add_argument('--unregistered', action='store_true')
parser.add_argument('--color', default=None)
parser.add_argument('--images', action='store_true')
args = parser.parse_args()
if args.step is not None:   STEP = args.step
if args.unregistered:       REGISTERED = False
if args.color is not None:  COLOR = args.color
if args.images:             SHOW_IMAGES = True

N_T = cfg['microscopy']['n_timepoints']
if args.t is not None:
    timepoints = args.t
elif args.range is not None:
    timepoints = list(range(args.range[0], args.range[1] + 1, STEP))
elif TIMEPOINTS is not None:
    timepoints = list(TIMEPOINTS)
else:
    timepoints = list(range(0, N_T, STEP))
if REGISTERED and SHOW_IMAGES:
    print('SHOW_IMAGES ignored: images are not registered (set REGISTERED = False to overlay them)')
    SHOW_IMAGES = False

VX = np.array(cfg['microscopy']['voxel_size_zyx'], dtype=float)
FEAT_DIR = Path(cfg['paths']['features_csv']).parent
MESH_CSV = Path(cfg['paths'].get('mesh_features_csv') or FEAT_DIR / 'mesh_features.csv')
MESH_DIR = mesh_display.mesh_dir(cfg)

feats = pd.read_csv(MESH_CSV).set_index(['t', 'label_id'])
if COLOR != 'label' and COLOR not in feats.columns:
    sys.exit(f'--color must be "label" or one of: {", ".join(feats.columns)}')
M, shift_um = mesh_display.registration(cfg) if REGISTERED else (None, None)

# ── meshes: one (verts, faces, values) per frame, built once ─────────────────
# Only the current frame goes into the surface layer; moving the time slider
# swaps it (~0.01 s). One 4D surface holding every frame made napari search
# all ~25M vertices on each slider step (~1 s+), too slow to play the movie.
# Loading/registering/simplifying: src/mesh_display.py (shared with the 04
# curation notebook's registered-mesh view).
frames = []
for i, t in enumerate(timepoints):
    if not (MESH_DIR / f't{t:03d}.npz').exists():
        sys.exit(f'{MESH_DIR / f"t{t:03d}.npz"} missing - run 01_extract_meshes.py first')
    V, Fc, owner = mesh_display.load_frame(MESH_DIR, t, M, shift_um, DISPLAY_GRID_UM)
    lids = np.unique(owner)
    if COLOR == 'label':
        lut = dict(zip(lids, glasbey_values(lids)))
    else:
        lut = feats.loc[t, COLOR].to_dict()
    if len(timepoints) <= 10:
        print(f'slider {i} = t{t}: {len(lids)} meshes', flush=True)
    vals = pd.Series(owner).map(lut).to_numpy(np.float32)
    frames.append((V, Fc, vals))

print(f'{len(timepoints)} frames, t = {timepoints if len(timepoints) <= 10 else f"{timepoints[0]}..{timepoints[-1]}"} '
      f'(slider i = i-th of these); {sum(len(f[0]) for f in frames):,} vertices', flush=True)
all_values = np.concatenate([f[2] for f in frames])
lo, hi = (0.0, 1.0) if COLOR == 'label' else np.nanpercentile(all_values, [2, 98])
CLIM = (float(lo), float(hi))

frame = 'registered' if REGISTERED else 'image space'
viewer = napari.Viewer(ndisplay=3, title=f'meshes ({frame}) - {CONFIG_PATH.stem} - t {timepoints[0]}-{timepoints[-1]}')

# ── optional images underneath ───────────────────────────────────────────────
if SHOW_IMAGES:
    ds = DS
    raw_files   = sorted(Path(cfg['paths']['raw_dir']).glob(cfg['paths']['raw_glob']))
    label_files = sorted(Path(cfg['paths']['label_dir']).glob(cfg['paths']['label_glob']))
    raws, labs = [], []
    for t in timepoints:
        raws.append(tifffile.imread(raw_files[t])[:, ::ds, ::ds])
        labs.append(tifffile.imread(label_files[t])[:, ::ds, ::ds])
    zmax = max(a.shape[0] for a in raws)
    pad = lambda arrs: np.stack([np.pad(a, ((0, zmax - a.shape[0]), (0, 0), (0, 0))) for a in arrs])
    scale = (1, VX[0], VX[1] * ds, VX[2] * ds)
    viewer.add_image(pad(raws), name='raw', scale=scale, colormap='gray', visible=False)
    viewer.add_labels(pad(labs), name='labels', scale=scale, opacity=0.4, visible=False,
                      colormap=label_colormap(feats.index.get_level_values('label_id'), offset=0))

# Invisible 4D marker layer: one point per frame, just to give napari a time
# slider (and play button) over the selected frames.
centre = np.concatenate([f[0] for f in frames[:1]]).mean(0)
viewer.add_points(np.array([[i, *centre] for i in range(len(frames))]), name='time',
                  size=0.01, opacity=0.0)
surface = viewer.add_surface(frames[0], name=f'meshes ({COLOR})',
                             colormap=GLASBEY_CMAP if COLOR == 'label' else 'viridis',
                             contrast_limits=CLIM, shading='smooth')


def show_frame(event=None):
    i = min(max(int(viewer.dims.current_step[0]), 0), len(frames) - 1)
    surface.data = frames[i]
    surface.contrast_limits = CLIM
    # (layer name stays fixed - napari-animation finds layers by name)
    viewer.text_overlay.text = f't={timepoints[i]}  ·  {timepoints[i] * INTERVAL_MIN:g} min'


INTERVAL_MIN = cfg['tracking']['frame_interval_min']
viewer.text_overlay.visible = True
viewer.text_overlay.font_size = 12
viewer.dims.events.current_step.connect(show_frame)
viewer.dims.set_current_step(0, 0)
show_frame()

if COLOR != 'label':
    print(f'colour = {COLOR}: dark = {lo:.3g}, bright = {hi:.3g} (2nd-98th percentile)')

napari.run()
