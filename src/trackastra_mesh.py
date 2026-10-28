"""
Mesh-based replacements for two of trackastra's 12 per-nucleus 'wrfeat'
features, shared by training (trackastra_training/) and inference
(03_run_trackastra.py --mesh-feats).

Trackastra computes equivalent_diameter_area and inertia_tensor with
skimage regionprops in voxel-index units, so with 2.0 µm z steps and
0.208 µm xy pixels the shape terms mostly measure the voxel anisotropy.
These come from the 01_extract_meshes.py mesh instead - isotropic and
physically proportioned - expressed in xy-pixel units (µm / VX_XY) to keep
the magnitudes in the range the pretrained feature encoding expects.
The other features (intensity_mean, border_dist) are left alone.
"""

import numpy as np
import pandas as pd

INERTIA_COLS = [f'inertia_{i}{j}' for i in range(3) for j in range(3)]
COLS = ['equivalent_diameter_area', *INERTIA_COLS]


def mesh_wrfeat_table(mesh_csv, vx_xy: float) -> pd.DataFrame:
    """(t, label_id) -> equivalent_diameter_area + 9 inertia_tensor entries (row-major),
    matching skimage's definitions: d = (6V/π)^(1/3), I = tr(C)·Id - C."""
    m = pd.read_csv(mesh_csv)
    out = pd.DataFrame({'t': m['t'], 'label_id': m['label_id']})
    out['equivalent_diameter_area'] = (6 * m['mesh_volume_um3'] / np.pi) ** (1 / 3) / vx_xy
    C = np.zeros((len(m), 3, 3))
    for (i, j), k in {(0, 0): 'zz', (1, 1): 'yy', (2, 2): 'xx', (0, 1): 'zy', (0, 2): 'zx', (1, 2): 'yx'}.items():
        C[:, i, j] = C[:, j, i] = m[f'mesh_cov_{k}'] / vx_xy ** 2
    inertia = np.trace(C, axis1=1, axis2=2)[:, None, None] * np.eye(3) - C
    out[INERTIA_COLS] = inertia.reshape(-1, 9)
    return out


def swap_features(feats, lookup: dict, key_fn):
    """Overwrite a WRFeatures' equivalent_diameter_area / inertia_tensor in place.
    lookup: key -> 10 floats (COLS order); key_fn(timepoint, label) -> key."""
    vals = np.array([lookup[key_fn(int(t), int(l))] for t, l in zip(feats.timepoints, feats.labels)],
                    dtype=np.float32)
    assert feats.features['equivalent_diameter_area'].shape[1] == 1
    assert feats.features['inertia_tensor'].shape[1] == 9
    feats.features['equivalent_diameter_area'] = vals[:, :1]
    feats.features['inertia_tensor'] = vals[:, 1:]
