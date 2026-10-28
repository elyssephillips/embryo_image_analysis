"""
3D tracking with Trackastra on the same segmentation masks as
03_run_btrack.py / 03_link_tracks.py.

Defaults run the production model: Trackastra's pretrained 3D "ctc" model
fine-tuned on the curated windows (trackastra_training/03_trackastra_finetune.py, TRAIN_FINAL),
tracking on registered centroids. No gap-closing afterwards: a wrong bridge
is harder to fix in curation than linking a new track to an old identity
(archive/03_stitch_trackastra.py did it). For the zero-shot baseline set MODEL_NAME = 'ctc',
REGISTER = False, VERSION = 'v1'.

Requires: pip install trackastra  (installed in napari_env)

Usage (settings below, or override them):
  python 03_run_trackastra.py [--config X.yaml] [--model RUN_DIR|ctc] [--version NAME]
                              [--t-range T0 T1] [--coords raw|registered] [--mesh-feats]
  --registered-csv PATH tracks on another registration's centroids (e.g. a
  training-only one, for scoring models trained with --registration v2).
  --mesh-feats swaps in mesh-based shape features (src/trackastra_mesh.py) -
  only for models fine-tuned with them (03_trackastra_finetune.py --mesh-feats).

Output:
  tracks_trackastra_{VERSION}.csv — track_id, t, z_um/y_um/x_um, label_id
  (z_um/y_um/x_um are registered µm with REGISTER, raw full-stack µm
  without; the curation notebooks recover display coords via the
  (t, label_id) join either way)
"""

import argparse
import glob
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.io import imread

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.log import log_run
from src.tracking_config import load_tracking_config, load_crop_info

cfg, CONFIG_PATH = load_tracking_config()   # --config X.yaml, else TRACKING_CONFIG / default

LABEL_DIR    = cfg['paths']['label_dir']
RAW_DIR      = cfg['paths']['raw_dir']
RAW_GLOB     = cfg['paths'].get('raw_glob', 'Cam_long_*.tif')
OUT_DIR      = Path(cfg['paths']['trackastra_output_dir'])
N_TIMEPOINTS = cfg['microscopy']['n_timepoints']
VX_Z, VX_Y, VX_X = cfg['microscopy']['voxel_size_zyx']

# Trackastra needs one consistent (T,Z,Y,X) array, so each frame is padded
# back into the original Z extent at its crop_info.json 'first_z' offset.
# For stacks prepared with prepare_inference.py's UNIFORM_Z_CROP (stack2,
# stack8 and later) first_z and depth are the same every frame, so this is
# just a constant offset. It matters for 250914_stack5, which was cropped per
# timepoint (different Z depth *and* origin per frame) - without the padding
# the same Z voxel index would mean a different physical depth in different
# frames there. first_z (not first_z + debris_shift_slices) is right: per
# scripts/fix_debris_crop_offsets.py, first_z still describes what is on disk.
# Padding to the full original depth (rather than the uniform crop) also keeps
# the voxel geometry identical to what the stack5-trained model saw.

# Pretrained name ('ctc' = only one with 3D support) or a fine-tuned run
# folder from trackastra_training/03_trackastra_finetune.py. Default: the
# model fine-tuned on 13 curated windows with the v2 registration (2026-10-22,
# ft_reg2_w13): stack5 A-G plus an early (tumbling) and a late (dense) window
# on each of stack2/8/11 - pair it with a v2-registered dataset (stack2/8/11;
# stack5's own analysis registration is v1, its tracks were made with
# runs/2026-10-06_11-41-38_ft_reg_final_all = 'ft_reg7_final'). Previous
# default: runs/2026-10-19_14-31-55_ft_reg2_final_all ('ft_reg2_final', A-G only).
MODEL_NAME = '/mnt/md0/elysse/nnUNet/inference/Dataset003_icm_te/250914_stack5/results/trackastra_train/runs/2026-10-22_16-44-48_ft_reg2_w13_final_all'
MODE       = 'greedy'   # or 'greedy_nodiv' / 'ilp'
# Greedy linking keeps an edge only if the model's link probability is above
# this (trackastra's own default is 0.5). Higher = fewer, surer links: on the
# 6 stack2/8/11 curated windows (2811 links, models held out by embryo,
# 2026-10-22) ft_reg2 at 0.5 missed 51 links and made 19 wrong/false ones, at
# 0.7: 66 / 4; the 13-window retrain (xs6 folds) at 0.8: 38 / 10. On stack5
# B/F/G (1220 links): ft_reg2 0.5 -> 47 / 19, xs6 0.8 -> 58 / 11.
LINK_THRESHOLD = 0.8
# Output name: tracks_trackastra_{VERSION}.csv. Held-out fold models go in as
# 'ft..._holdout_<window version>' so 03_score_tracks_vs_curated.py picks them up.
VERSION    = 'ft_reg2_w13_final'

# Track in drift-corrected coordinates, like btrack/03_link_tracks.py do:
# each nucleus's centroid is swapped for its registered position from
# features_registered.csv (02_register_centroids.py) after trackastra
# extracts its features. Images/masks aren't warped - trackastra only sees
# them through per-nucleus features, and the orientation-dependent one
# (inertia tensor) stays in the raw frame for every frame, so consecutive
# frames remain comparable. Output z_um/y_um/x_um are then registered µm,
# the same space btrack's CSV uses. Must match how the model was trained
# (03_trackastra_finetune.py's REGISTERED).
REGISTER = True
# Mesh-based equivalent_diameter_area / inertia_tensor (01_extract_meshes.py);
# must match how the model was trained.
MESH_FEATS = False

# Inclusive (t_start, t_end) subset in real t, or None for all N_TIMEPOINTS.
# Features are extracted frame by frame, so RAM stays at a few GB either way.
T_RANGE = None

# Command-line overrides for the settings above (e.g. scripted held-out runs).
parser = argparse.ArgumentParser()
parser.add_argument('--model', default=None, help='run folder or pretrained name')
parser.add_argument('--version', default=None, help='output name suffix')
parser.add_argument('--t-range', nargs=2, type=int, default=None, metavar=('T0', 'T1'))
parser.add_argument('--coords', choices=['raw', 'registered'], default=None,
                    help="'raw' = REGISTER off")
parser.add_argument('--mesh-feats', action='store_true', default=None)
parser.add_argument('--link-threshold', type=float, default=None, help='greedy edge threshold (LINK_THRESHOLD)')
parser.add_argument('--link-thresholds', nargs='+', type=float, default=[],
                    help='also re-link the same predictions at these greedy edge thresholds; '
                         'each writes tracks_trackastra_{VERSION}_thr{x}.csv')
parser.add_argument('--registered-csv', default=None,
                    help="registered centroids to track on (default: config's features_registered_csv)")
args = parser.parse_args()
REGISTERED_CSV = args.registered_csv or cfg['paths']['features_registered_csv']
if args.mesh_feats is not None: MESH_FEATS = args.mesh_feats
if args.model is not None:   MODEL_NAME = args.model
if args.link_threshold is not None: LINK_THRESHOLD = args.link_threshold
if args.version is not None: VERSION = args.version
if args.t_range is not None: T_RANGE = tuple(args.t_range)
if args.coords is not None:
    REGISTER = args.coords == 'registered'

# =============================================================================
#  LOAD
# =============================================================================

from trackastra.data import WRFeatures, build_windows
from trackastra.model import Trackastra
from trackastra.model.predict import predict_windows
from trackastra.tracking import build_graph, graph_to_napari_tracks, track_greedy
from trackastra.utils import normalize

label_files = sorted(glob.glob(f"{LABEL_DIR}*_instances_reclassified.tif"))[:N_TIMEPOINTS]
raw_files   = sorted(glob.glob(f"{RAW_DIR}{RAW_GLOB}"))[:N_TIMEPOINTS]

assert len(label_files) == N_TIMEPOINTS, \
    f"Expected {N_TIMEPOINTS} label files, found {len(label_files)}"
assert len(raw_files) == N_TIMEPOINTS, \
    f"Expected {N_TIMEPOINTS} raw files, found {len(raw_files)}"

T_START, T_END = T_RANGE if T_RANGE is not None else (0, N_TIMEPOINTS - 1)
label_files = label_files[T_START:T_END + 1]
raw_files   = raw_files[T_START:T_END + 1]
n_t = len(label_files)

crop_info = load_crop_info(cfg)

_TP_RE = re.compile(r'(\d{5})')


def _tp_id(path: str) -> str:
    m = _TP_RE.search(Path(path).stem)
    assert m, f'Could not parse a 5-digit timepoint id from {path}'
    return m.group(1)


_shapes = {tuple(v['original_shape']) for v in crop_info.values()}
assert len(_shapes) == 1, f'crop_info.json has inconsistent original_shape values: {_shapes}'
Z_FULL, Y_FULL, X_FULL = next(iter(_shapes))

for lf, rf in zip(label_files, raw_files):
    assert _tp_id(lf) == _tp_id(rf), f'label/raw timepoint mismatch: {lf} vs {rf}'

print(f'Tracking t{T_START}-{T_END} ({n_t} of {N_TIMEPOINTS} timepoints), padding each frame '
      f'to (Z,Y,X)=({Z_FULL},{Y_FULL},{X_FULL})', flush=True)

# =============================================================================
#  TRACK
# =============================================================================

if Path(MODEL_NAME).is_dir():
    print(f'Loading fine-tuned model from {MODEL_NAME}...', flush=True)
    model = Trackastra.from_folder(Path(MODEL_NAME), device='automatic')
else:
    print(f'Loading pretrained model "{MODEL_NAME}"...', flush=True)
    model = Trackastra.from_pretrained(MODEL_NAME, device='automatic')

assert model.train_args['features'] == 'wrfeat', \
    f"frame-by-frame features only implemented for 'wrfeat', model uses {model.train_args['features']!r}"

if REGISTER:
    _reg = pd.read_csv(REGISTERED_CSV)
    _reg_vox = {(int(t), int(l)): (z / VX_Z, y / VX_Y, x / VX_X)
                for t, l, z, y, x in _reg[['t', 'label_id', 'z_um_reg', 'y_um_reg', 'x_um_reg']].itertuples(index=False)}
    print(f'REGISTER: tracking on registered centroids from {REGISTERED_CSV}', flush=True)

if MESH_FEATS:
    from src.trackastra_mesh import mesh_wrfeat_table, COLS, swap_features

    _mesh_csv = cfg['paths'].get('mesh_features_csv') or \
        Path(cfg['paths']['features_csv']).parent / 'mesh_features.csv'
    _mt = mesh_wrfeat_table(_mesh_csv, VX_Y)
    _mlook = {(int(r[0]), int(r[1])): r[2:] for r in _mt[['t', 'label_id', *COLS]].to_numpy()}
    print(f'MESH_FEATS: mesh-based shape features from {_mesh_csv}', flush=True)

# Per-nucleus features, one frame at a time. This is what model.track() does
# internally (get_features' single-process loop -> build_windows ->
# predict_windows -> greedy/ilp), but model.track() needs the whole movie as
# (T,Z,Y,X) arrays - float32 imgs + uint16 masks, plus a zeros_like(masks)
# copy for the tracked masks we never used - ~175 GB peak, which OOM-killed
# the run on this shared box. Holding one frame keeps the peak at a few GB.
features = []
n_detections = []
for i, (lf, rf) in enumerate(zip(label_files, raw_files)):
    z0 = crop_info[_tp_id(lf)]['first_z']
    lab = imread(lf)
    assert lab.max() <= np.iinfo(np.uint16).max, f'{lf}: label ids exceed uint16'
    raw = imread(rf)
    assert lab.shape == raw.shape, f'Shape mismatch at {lf}: labels {lab.shape}, raw {raw.shape}'
    mask = np.zeros((1, Z_FULL, Y_FULL, X_FULL), dtype=np.uint16)
    img  = np.zeros((1, Z_FULL, Y_FULL, X_FULL), dtype=np.float32)
    mask[0, z0:z0 + lab.shape[0]] = lab
    # Normalize per frame on the unpadded crop (trackastra's own percentile
    # normalize), so padding zeros don't skew the percentiles; padding then
    # stays 0 = the frame's 1st-percentile background.
    img[0, z0:z0 + raw.shape[0]] = normalize(raw)
    f = WRFeatures.from_mask_img(mask=mask, img=img, t_start=i)  # model sees the subset as 0-based
    if REGISTER:
        f.coords = np.array([_reg_vox[(int(t) + T_START, int(l))] for t, l in zip(f.timepoints, f.labels)],
                            dtype=f.coords.dtype)
    if MESH_FEATS:
        swap_features(f, _mlook, lambda t, l: (t + T_START, l))
    features.append(f)
    n_detections.append(len(np.unique(lab)) - (1 if 0 in lab else 0))
    del lab, raw, mask, img
    if i % 10 == 0 or i == n_t - 1:
        print(f'  features t={T_START + i}', flush=True)

print(f'Tracking (mode="{MODE}", link threshold {LINK_THRESHOLD:g})...', flush=True)
model.transformer.eval()
windows = build_windows(features, window_size=model.transformer.config['window'], as_torch=True)
predictions = predict_windows(windows=windows, features=features, model=model.transformer,
                              edge_threshold=0.05, spatial_dim=3, batch_size=model.batch_size)


def link(threshold):
    """Greedy linking at a given edge threshold, with _track_from_predictions'
    candidate-graph defaults (identical to it at 0.5)."""
    cand = build_graph(nodes=predictions['nodes'], weights=predictions['weights'],
                       use_distance=False, max_distance=256, max_neighbors=10, delta_t=1)
    return track_greedy(cand, allow_divisions=MODE == 'greedy', threshold=threshold)


if MODE in ('greedy', 'greedy_nodiv'):
    track_graph = link(LINK_THRESHOLD)
else:
    assert LINK_THRESHOLD == 0.5, 'LINK_THRESHOLD only applies to the greedy modes'
    track_graph = model._track_from_predictions(predictions, mode=MODE)


def graph_to_df(graph):
    napari_tracks, _, props = graph_to_napari_tracks(graph, properties=['label'])
    # napari_tracks columns for 3D+time input: track_id, t, z, y, x (voxel units)
    df = pd.DataFrame(napari_tracks, columns=['track_id', 't', 'z', 'y', 'x'])
    df['track_id'] = df['track_id'].astype(int)
    df['t']        = df['t'].astype(int) + T_START  # model sees the subset as 0-based
    df['z_um']     = df['z'] * VX_Z
    df['y_um']     = df['y'] * VX_Y
    df['x_um']     = df['x'] * VX_X
    df['label_id'] = np.asarray(props['label'], dtype=int)  # original instance label, from the graph node
    df = df[['track_id', 't', 'z_um', 'y_um', 'x_um', 'label_id']]
    return df.sort_values(['track_id', 't']).reset_index(drop=True)


tracks_df = graph_to_df(track_graph)

# Same predictions re-linked at other edge thresholds, for comparison.
extra_links = {}
if args.link_thresholds:
    assert MODE in ('greedy', 'greedy_nodiv'), '--link-thresholds needs a greedy MODE'
    extra_links = {thr: graph_to_df(link(thr)) for thr in args.link_thresholds}

# =============================================================================
#  SAVE
# =============================================================================

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_csv = OUT_DIR / f'tracks_trackastra_{VERSION}.csv'
tracks_df.to_csv(out_csv, index=False)
for thr, df in extra_links.items():
    df.to_csv(OUT_DIR / f'tracks_trackastra_{VERSION}_thr{thr:g}.csv', index=False)
    print(f'Saved threshold {thr:g}: {len(df)} detections in tracks, {df["track_id"].nunique()} tracks', flush=True)

n_in  = sum(n_detections)
n_out = len(tracks_df)
print(f'\nDetections in: {n_in} | in output tracks: {n_out} '
      f'({n_in - n_out} dropped by trackastra, {100 * (n_in - n_out) / n_in:.1f}%)', flush=True)

lengths = tracks_df.groupby('track_id')['t'].count()
print(f'Trackastra ("{MODEL_NAME}", mode="{MODE}") found {len(lengths)} tracks '
      f'across {n_t} timepoints', flush=True)
print(f'Full timecourse ({n_t}): {(lengths == n_t).sum()}', flush=True)
print(f'>=50 frames:           {(lengths >= 50).sum()}', flush=True)
print(f'>=10 frames:           {(lengths >= 10).sum()}', flush=True)
print(f'1-2 frames:            {(lengths <= 2).sum()}', flush=True)
print(f'Median length:         {lengths.median():.0f}', flush=True)
print(f'\nSaved: {out_csv}')

log_run("tracking", CONFIG_PATH.stem, "03_run_trackastra.py",
        output_path=str(out_csv), detail="detailed",
        data_path=str(Path(LABEL_DIR)))
