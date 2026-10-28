"""
Run trackastra's scripts/train.py on drift-corrected nucleus positions.

Usage: same arguments as train.py - 03_trackastra_finetune.py calls this
instead of train.py when REGISTERED = True.

Each training/validation window exported by 03_trackastra_export_ctc.py has
{name}_GT/TRA/registered_coords.csv (frame, CTC label -> registered centroid
in voxels, from features_registered.csv). This swaps those in for the raw
mask centroids right after trackastra extracts per-nucleus features, the same
way 03_run_trackastra.py's REGISTER does at inference. Augmentations run
later, per sample, on the swapped coordinates.

Variants (environment variables, set by 03_trackastra_finetune.py):
  TRACKASTRA_COORDS_FILE  which centroids file to swap in (default
                          registered_coords.csv; registered_coords_v2.csv for
                          --registration v2)
  TRACKASTRA_MESH_FEATS=1 also swap equivalent_diameter_area / inertia_tensor
                          for mesh-based ones from mesh_feats.csv
                          (src/trackastra_mesh.py)

Pass a --cachedir separate from unregistered runs: trackastra caches built
datasets keyed on its arguments, not on these coordinates.
"""

import os
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from trackastra.data.data import CTCData

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.trackastra_mesh import COLS, swap_features

_build_windows = CTCData._build_windows_wrfeat
COORDS_FILE = os.environ.get('TRACKASTRA_COORDS_FILE', 'registered_coords.csv')
MESH_FEATS = os.environ.get('TRACKASTRA_MESH_FEATS') == '1'


def _build_windows_registered(self, features, det_masks, matching):
    reg_path = Path(f'{self.root}_GT') / 'TRA' / COORDS_FILE
    if not reg_path.exists():
        raise FileNotFoundError(f'{reg_path} missing - re-run 03_trackastra_export_ctc.py')
    reg = pd.read_csv(reg_path)
    lookup = {(int(f), int(l)): (z, y, x) for f, l, z, y, x in reg.itertuples(index=False)}
    for feats in features:
        feats.coords = np.array([lookup[(int(t), int(l))] for t, l in zip(feats.timepoints, feats.labels)],
                                dtype=feats.coords.dtype)
    msg = f'{COORDS_FILE} centroids'
    if MESH_FEATS:
        mf = pd.read_csv(Path(f'{self.root}_GT') / 'TRA' / 'mesh_feats.csv')
        mlook = {(int(r[0]), int(r[1])): r[2:] for r in mf[['frame', 'label', *COLS]].to_numpy()}
        for feats in features:
            swap_features(feats, mlook, lambda t, l: (t, l))
        msg += ' + mesh shape features'
    print(f'[registered] {self.root.name}: swapped {sum(len(f.labels) for f in features)} {msg}', flush=True)
    return _build_windows(self, features, det_masks, matching)


CTCData._build_windows_wrfeat = _build_windows_registered

TRAIN_SCRIPT = Path('/mnt/md0/elysse/code/trackastra/scripts/train.py')
sys.argv = [str(TRAIN_SCRIPT), *sys.argv[1:]]
sys.path.insert(0, str(TRAIN_SCRIPT.parent))
runpy.run_path(str(TRAIN_SCRIPT), run_name='__main__')
