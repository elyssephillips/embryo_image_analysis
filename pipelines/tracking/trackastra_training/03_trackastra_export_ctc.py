"""
Export hand-curated window tracks (04_curate_tracks_window.ipynb output) to
Cell Tracking Challenge (CTC) format, as training/validation data for
fine-tuning Trackastra (scripts/train.py in the trackastra repo).

Re-run whenever a window gets another curation pass - point EXPORTS at the
newest tracks_{version}.csv. Each export is written to
trackastra_train_dir/{name}/ and replaced wholesale on re-run.

Per exported CSV:
  {name}/t000.tif ...            raw nuclear channel, one frame per t
  {name}_GT/TRA/man_track000.tif  masks, voxel = CTC track label
  {name}_GT/TRA/man_track.txt     "label t_start t_end parent", 0-based t
  {name}_GT/TRA/registered_coords.csv  frame,label -> registered centroid (voxels)
  {name}_GT/TRA/registered_coords_v2.csv  the same file, for datasets whose
                                  production registration is already v2 (no
                                  training_registration_dir in the config:
                                  stack2/8/11) - so 03_trackastra_finetune.py
                                  --registration v2 can mix them with stack5's
                                  windows (whose v2 copy comes from
                                  03_trackastra_export_extras.py)

Other datasets: run with --config X.yaml --exports trackastra/tracks_....csv
(the windows saved by 04_curate_tracks.ipynb with WINDOW set); they land in
that dataset's own trackastra_train_dir.

Frames are padded to the full original Z extent at crop_info.json's first_z (a
constant offset on uniformly-cropped stacks, per-frame on stack5),
exactly like 03_run_trackastra.py, so training and inference see the same
voxel geometry. Raw padding is filled with the frame's 1st-percentile
intensity rather than 0: trackastra's loader percentile-normalizes each whole
padded frame, and zero-fill would drag the 1st percentile down to 0 and
rescale real signal differently than at inference (where padding is added
after normalizing the unpadded crop).

Silver fill (SILVER_SOURCE): curation only covers the window's z-band, so a
nucleus drifting out of the band looks like a track ending (and one drifting
in, like a track starting), and the scene is a thin slab of the embryo. With
SILVER_SOURCE set, nuclei within SILVER_Z_MARGIN_UM of the curated z-range
are added with that tracker's links as lower-confidence labels. Curated links
always win: a silver link is only added if neither end already has a curated
link in that direction, and never between two curated nuclei (the curator saw
both and chose not to link them). Output name gets a '_silver{margin}' suffix.

Curated-data caveats this format exposes:
  - Curation has no division concept (swap/reassign/break only), so a
    division shows up as one daughter continuing + one new track, never as a
    parent with two children.
  - Rows without a label_id (btrack gap-fill dummies) have no mask to export.
    They're dropped, and a curated track that loses frames that way is split
    into CTC segments linked parent -> child across the gap.
"""

import argparse
import glob
import re
import shutil
import sys
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.tracking_config import load_tracking_config, load_crop_info
cfg, CONFIG_PATH = load_tracking_config()   # --config X.yaml, else TRACKING_CONFIG / default

LABEL_DIR      = cfg['paths']['label_dir']
RAW_DIR        = cfg['paths']['raw_dir']
RAW_GLOB       = cfg['paths'].get('raw_glob', 'Cam_long_*.tif')
TRACKS_ROOT    = Path(cfg['paths']['btrack_output_dir']).parent
OUT_ROOT       = Path(cfg['paths']['trackastra_train_dir'])
FEATURES_CSV   = cfg['paths']['features_csv']
FEATURES_REG_CSV = cfg['paths']['features_registered_csv']
VX_Z, VX_Y, VX_X = cfg['microscopy']['voxel_size_zyx']

# Curated window CSVs to export, relative to results/tracks/. The window
# notebook saves into whichever TRACK_SOURCE's folder it was curating from.
EXPORTS = [
    'btrack/tracks_linked_c3_winA.csv',
    'btrack/tracks_linked_c3_winB.csv',
    'btrack/tracks_linked_c3_winC.csv',
    'btrack/tracks_linked_c2_winD.csv',
    # Full z, seeded from stitched trackastra (ft_reg_final_stitched).
    'trackastra/tracks_linked_c2_winE.csv',
    'trackastra/tracks_linked_c4_winF.csv',
    'trackastra/tracks_linked_c2_winG.csv',
]

# Tracker output (relative to results/tracks/) used to fill uncurated nuclei
# around the band, or None for curated nuclei only.
SILVER_SOURCE = None                # e.g. 'btrack/tracks_v1_raw.csv'
SILVER_Z_MARGIN_UM = 20.0           # how far beyond the curated z-range to fill; None = whole frame

# Load each export back through trackastra's own CTCData loader afterwards,
# to prove the trainer will accept it (slow-ish: normalizes every frame).
VALIDATE = True

parser = argparse.ArgumentParser()
parser.add_argument('--exports', nargs='+', default=None,
                    help='curated window CSVs relative to results/tracks/ (default: EXPORTS)')
args = parser.parse_args()
if args.exports is not None:
    EXPORTS = args.exports

# =============================================================================

crop_info = load_crop_info(cfg)
_shapes = {tuple(v['original_shape']) for v in crop_info.values()}
assert len(_shapes) == 1, f'crop_info.json has inconsistent original_shape values: {_shapes}'
Z_FULL, Y_FULL, X_FULL = next(iter(_shapes))

all_label_files = sorted(glob.glob(f"{LABEL_DIR}*_instances_reclassified.tif"))
all_raw_files   = sorted(glob.glob(f"{RAW_DIR}{RAW_GLOB}"))
_TP_RE = re.compile(r'(\d{5})')


def _tp_id(path: str) -> str:
    m = _TP_RE.search(Path(path).stem)
    assert m, f'Could not parse a 5-digit timepoint id from {path}'
    return m.group(1)


def _frame_files(t: int) -> tuple[str, str]:
    lf, rf = all_label_files[t], all_raw_files[t]
    assert _tp_id(lf) == _tp_id(rf) == f'{t:05d}', f't={t}: file mismatch {lf} / {rf}'
    return lf, rf


def _consecutive_links(df: pd.DataFrame) -> dict:
    """(t, label_id) -> (t+1, label_id) for each track's consecutive labeled rows."""
    d = df.dropna(subset=['label_id']).astype({'label_id': int, 't': int}).sort_values(['track_id', 't'])
    nxt = d.groupby('track_id')[['t', 'label_id']].shift(-1)
    ok = nxt['t'] == d['t'] + 1
    return {(t, l): (t + 1, int(l2)) for t, l, l2 in zip(d.loc[ok, 't'], d.loc[ok, 'label_id'], nxt.loc[ok, 'label_id'])}


def _silver_nodes(curated: pd.DataFrame, t0: int, t1: int) -> set:
    """Uncurated-candidate nuclei in t0..t1 within SILVER_Z_MARGIN_UM of the curated z-range,
    in the window notebook's absolute-z convention (z_um_abs)."""
    feat = pd.read_csv(FEATURES_CSV, usecols=['t', 'label_id', 'z_um'])
    feat = feat[feat['t'].between(t0, t1)]
    if SILVER_Z_MARGIN_UM is not None:
        assert 'z_um_abs' in curated.columns, 'curated CSV lacks z_um_abs - re-save it from the window notebook'
        abs_first_z = feat['t'].map(lambda t: crop_info[f'{t:05d}']['first_z']
                                    + crop_info[f'{t:05d}'].get('debris_shift_slices', 0))
        z_abs = feat['z_um'] + abs_first_z * VX_Z
        lo = curated['z_um_abs'].min() - SILVER_Z_MARGIN_UM
        hi = curated['z_um_abs'].max() + SILVER_Z_MARGIN_UM
        feat = feat[z_abs.between(lo, hi)]
    return set(zip(feat['t'].astype(int), feat['label_id'].astype(int)))


def export(csv_rel: str) -> Path:
    csv_path = TRACKS_ROOT / csv_rel
    version = csv_path.stem.removeprefix('tracks_')
    name = version if SILVER_SOURCE is None else \
        f'{version}_silver{"all" if SILVER_Z_MARGIN_UM is None else int(SILVER_Z_MARGIN_UM)}'
    curated = pd.read_csv(csv_path)

    n_no_label = curated['label_id'].isna().sum()
    df = curated.dropna(subset=['label_id']).astype({'label_id': int, 't': int, 'track_id': int})

    dup = df[df.duplicated(['t', 'label_id'], keep=False)]
    if len(dup):
        raise ValueError(f'{version}: same nucleus claimed by more than one track - fix in curation first:\n'
                         f'{dup[["track_id", "t", "label_id"]].sort_values(["t", "label_id"]).to_string()}')
    dup = df[df.duplicated(['track_id', 't'], keep=False)]
    if len(dup):
        raise ValueError(f'{version}: track has >1 nucleus at one t - fix in curation first:\n'
                         f'{dup[["track_id", "t", "label_id"]].sort_values(["track_id", "t"]).to_string()}')

    t0, t1 = df['t'].min(), df['t'].max()
    cur_nodes = set(zip(df['t'], df['label_id']))

    # Curated links; a curated track broken by dropped label-less rows keeps
    # its identity across the gap as a parent -> child pair.
    succ, gaps = {}, []
    for _, grp in df.sort_values('t').groupby('track_id'):
        pts = list(zip(grp['t'], grp['label_id']))
        for u, v in zip(pts, pts[1:]):
            (succ.__setitem__(u, v) if v[0] == u[0] + 1 else gaps.append((u, v)))
    has_succ = set(succ) | {u for u, _ in gaps}
    has_pred = set(succ.values()) | {v for _, v in gaps}

    nodes = set(cur_nodes)
    n_silver_links = 0
    if SILVER_SOURCE is not None:
        nodes |= _silver_nodes(curated, t0, t1)
        for u, v in _consecutive_links(pd.read_csv(TRACKS_ROOT / SILVER_SOURCE)).items():
            if (u in nodes and v in nodes and u not in has_succ and v not in has_pred
                    and not (u in cur_nodes and v in cur_nodes)):
                succ[u] = v
                n_silver_links += 1

    pred = {v: u for u, v in succ.items()}
    label_of, man_track = {}, []
    for n in sorted(nodes):
        if n in pred:
            continue
        chain = [n]
        while chain[-1] in succ:
            chain.append(succ[chain[-1]])
        lab_id = len(man_track) + 1
        for m in chain:
            label_of[m] = lab_id
        man_track.append([lab_id, chain[0][0] - t0, chain[-1][0] - t0, 0])
    for u, v in gaps:
        man_track[label_of[v] - 1][3] = label_of[u]
    man_track = pd.DataFrame(man_track, columns=['label', 't1', 't2', 'parent'])

    img_dir = OUT_ROOT / name
    tra_dir = OUT_ROOT / f'{name}_GT' / 'TRA'
    for d in (img_dir, tra_dir.parent):
        if d.exists():
            shutil.rmtree(d)
    img_dir.mkdir(parents=True)
    tra_dir.mkdir(parents=True)

    silver_msg = '' if SILVER_SOURCE is None else \
        f' | silver: +{len(nodes) - len(cur_nodes)} nuclei, +{n_silver_links} links from {SILVER_SOURCE}'
    print(f'\n{name}: t{t0}-{t1} ({t1 - t0 + 1} frames), {len(cur_nodes)} curated detections, '
          f'{df["track_id"].nunique()} curated tracks -> {len(man_track)} CTC segments '
          f'({n_no_label} rows without label_id dropped){silver_msg}', flush=True)

    by_t = pd.DataFrame([(t, l, label_of[(t, l)]) for t, l in nodes], columns=['t', 'label_id', 'ctc'])
    for i, t in enumerate(range(t0, t1 + 1)):
        lf, rf = _frame_files(t)
        z0 = crop_info[_tp_id(lf)]['first_z']
        lab = tifffile.imread(lf)
        raw = tifffile.imread(rf)
        assert lab.shape == raw.shape, f't={t}: labels {lab.shape} vs raw {raw.shape}'

        rows = by_t[by_t['t'] == t]
        missing = rows.loc[~rows['label_id'].isin(np.unique(lab)), 'label_id'].tolist()
        if missing:
            raise ValueError(f'{name} t={t}: label_id(s) {missing} not present in {Path(lf).name} '
                             f'- was the segmentation re-run after curation?')
        lut = np.zeros(int(lab.max()) + 1, dtype=np.uint16)
        lut[rows['label_id'].values] = rows['ctc'].values

        mask_full = np.zeros((Z_FULL, Y_FULL, X_FULL), dtype=np.uint16)
        mask_full[z0:z0 + lab.shape[0]] = lut[lab]

        bg = np.percentile(raw[:, ::4, ::4], 1)
        raw_full = np.full((Z_FULL, Y_FULL, X_FULL), bg, dtype=raw.dtype)
        raw_full[z0:z0 + raw.shape[0]] = raw

        tifffile.imwrite(img_dir / f't{i:03d}.tif', raw_full, compression='zlib')
        tifffile.imwrite(tra_dir / f'man_track{i:03d}.tif', mask_full, compression='zlib')

    man_track.to_csv(tra_dir / 'man_track.txt', sep=' ', header=False, index=False)

    # Registered centroid per exported nucleus, in voxel units - read by
    # 03_trackastra_train_registered.py to train on drift-corrected positions.
    reg = pd.read_csv(FEATURES_REG_CSV, usecols=['t', 'label_id', 'z_um_reg', 'y_um_reg', 'x_um_reg'])
    reg = reg.merge(by_t, on=['t', 'label_id'])
    assert len(reg) == len(by_t), f'{name}: {len(by_t) - len(reg)} exported nuclei missing from {FEATURES_REG_CSV}'
    pd.DataFrame({
        'frame': reg['t'] - t0, 'label': reg['ctc'],
        'z': reg['z_um_reg'] / VX_Z, 'y': reg['y_um_reg'] / VX_Y, 'x': reg['x_um_reg'] / VX_X,
    }).sort_values(['frame', 'label']).to_csv(tra_dir / 'registered_coords.csv', index=False)
    if not cfg['paths'].get('training_registration_dir'):
        shutil.copy(tra_dir / 'registered_coords.csv', tra_dir / 'registered_coords_v2.csv')

    print(f'  wrote {img_dir} and {tra_dir}', flush=True)
    return img_dir


if __name__ == '__main__':
    roots = [export(e) for e in EXPORTS]

    if VALIDATE:
        from trackastra.data import CTCData
        for root in roots:
            ds = CTCData(root=str(root), ndim=3, window_size=4, features='wrfeat')
            print(f'  CTCData OK: {root.name} -> {len(ds)} training windows', flush=True)
