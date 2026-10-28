"""
Add mesh-based shape features next to each exported training window's
registered_coords.csv (03_trackastra_export_ctc.py), for fine-tuning with
03_trackastra_finetune.py --mesh-feats:

  {window}_GT/TRA/mesh_feats.csv   frame,label -> mesh equivalent diameter +
                                   inertia tensor (src/trackastra_mesh.py)
  {window}_GT/TRA/registered_coords_v2.csv
                                   frame,label -> centroid (voxels) from the
                                   config's training_registration_dir, if set
                                   (03_trackastra_finetune.py --registration v2)

Additive only - images, masks, man_track.txt and registered_coords.csv are
left as they are. The export doesn't record each CTC label's original
(t, label_id), so it's recovered by matching registered_coords.csv exactly
against features_registered.csv (same numbers, so the match is exact).

Re-run after re-exporting a window, re-running 02_register_centroids.py or
01_extract_meshes.py.

Usage
-----
  python pipelines/tracking/trackastra_training/03_trackastra_export_extras.py [--config X.yaml]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.tracking_config import load_tracking_config
from src.trackastra_mesh import mesh_wrfeat_table, COLS

cfg, CONFIG_PATH = load_tracking_config()

TRAIN_DIR = Path(cfg['paths']['trackastra_train_dir'])
VX = np.array(cfg['microscopy']['voxel_size_zyx'], dtype=float)
FEAT_DIR = Path(cfg['paths']['features_csv']).parent

reg = pd.read_csv(cfg['paths']['features_registered_csv'],
                  usecols=['t', 'label_id', 'z_um_reg', 'y_um_reg', 'x_um_reg'])
reg_vox = reg[['z_um_reg', 'y_um_reg', 'x_um_reg']].to_numpy() / VX
mesh = mesh_wrfeat_table(cfg['paths'].get('mesh_features_csv') or FEAT_DIR / 'mesh_features.csv', VX[1])
V2_DIR = cfg['paths'].get('training_registration_dir')
reg_v2 = pd.read_csv(Path(V2_DIR) / 'features_registered.csv',
                     usecols=['t', 'label_id', 'z_um_reg', 'y_um_reg', 'x_um_reg']) if V2_DIR else None

windows = sorted(p.parent for p in TRAIN_DIR.glob('*_GT/TRA/registered_coords.csv'))
assert windows, f'no exported windows with registered_coords.csv under {TRAIN_DIR}'
for tra in windows:
    rc = pd.read_csv(tra / 'registered_coords.csv')
    # every exported nucleus, located among all nuclei of all frames
    by_t = {}
    for t, g in reg.groupby('t'):
        by_t[t] = (cKDTree(reg_vox[g.index]), g['label_id'].to_numpy())
    # the window's t0 is unknown - find the offset that matches frame 0 exactly
    f0 = rc[rc.frame == 0][['z', 'y', 'x']].to_numpy()
    t0 = min(by_t, key=lambda t: by_t[t][0].query(f0)[0].max())
    rows = []
    for frame, label, z, y, x in rc.itertuples(index=False):
        tree, lids = by_t[t0 + frame]
        d, i = tree.query([z, y, x])
        assert d < 1e-3, f'{tra}: frame {frame} label {label} has no exact match (d={d:.3g} vox)'
        rows.append((frame, label, t0 + frame, int(lids[i])))
    key = pd.DataFrame(rows, columns=['frame', 'label', 't', 'label_id'])

    mf = key.merge(mesh, on=['t', 'label_id'], how='left')
    assert mf[COLS].notna().all().all(), f'{tra}: nuclei missing from mesh_features'
    mf[['frame', 'label', *COLS]].to_csv(tra / 'mesh_feats.csv', index=False)
    written = ['mesh_feats.csv']
    if reg_v2 is not None:
        r2 = key.merge(reg_v2, on=['t', 'label_id'], how='left')
        assert r2['z_um_reg'].notna().all(), f'{tra}: nuclei missing from {V2_DIR}'
        pd.DataFrame({'frame': r2['frame'], 'label': r2['label'], 'z': r2['z_um_reg'] / VX[0],
                      'y': r2['y_um_reg'] / VX[1], 'x': r2['x_um_reg'] / VX[2]}
                     ).to_csv(tra / 'registered_coords_v2.csv', index=False)
        written.append('registered_coords_v2.csv')
    print(f'{tra.parent.name}: t0={t0}, {len(key)} nuclei -> {", ".join(written)}')
