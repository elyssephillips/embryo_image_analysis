"""
Loading nucleus meshes (01_extract_meshes.py) for display - shared by
01_view_meshes.py and the 04 curation notebook's registered-mesh view.

Meshes are saved in each frame's own image space (µm). For a registered view
each frame is moved with 02_register_centroids.py's transforms.npz, after
undoing crop_info's debris z-shift (stack5 only: features.csv z was shifted by
-debris_shift_slices * VX_Z on some frames without re-cropping the tiffs, and
meshes follow the tiffs; uniformly-cropped stacks have no shift).
"""

from pathlib import Path

import numpy as np

from src.tracking_config import crop_info_path, load_crop_info


def mesh_dir(cfg):
    feat_dir = Path(cfg['paths']['features_csv']).parent
    return Path(cfg['paths'].get('mesh_dir') or feat_dir.parent / 'meshes')


def registration(cfg):
    """(transforms (T,4,4), {t: debris shift µm}) for registered display."""
    M = np.load(cfg['paths']['transforms_npz'])['transforms']
    shift_um = {}
    if crop_info_path(cfg).exists():
        vx_z = cfg['microscopy']['voxel_size_zyx'][0]
        shift_um = {int(k): v.get('debris_shift_slices', 0) * vx_z for k, v in load_crop_info(cfg).items()}
    return M, shift_um


def simplify(v, f, owner, grid):
    """Vertex clustering for display: snap each nucleus's vertices to a grid
    of `grid` µm, merge the ones in the same cell (mean position), drop
    collapsed triangles. ~5x fewer vertices at 1 µm. Returns (verts, faces,
    index of one original vertex per new vertex).
    Run it in the meshes' own (image) frame, where the marching-cubes
    vertices line up with the grid: on rotated (registered) meshes the cells
    cut the surface obliquely and ~2% of triangles fold over, which napari
    shades as dark slivers. The few that still fold are turned back to the
    original face's orientation, and duplicate faces dropped."""
    q = np.floor(v / grid).astype(np.int64)
    q -= q.min(0)
    span = q.max(0) + 1
    key = ((owner.astype(np.int64) * span[0] + q[:, 0]) * span[1] + q[:, 1]) * span[2] + q[:, 2]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    nv = np.zeros((len(cnt), 3))
    np.add.at(nv, inv, v)
    nv /= cnt[:, None]
    nf = inv[f]
    keep = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
    nf, of = nf[keep], f[keep]
    normal = lambda p, t: np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    folded = (normal(nv, nf) * normal(v, of)).sum(1) < 0
    nf[folded] = nf[folded][:, ::-1]
    _, uniq = np.unique(np.sort(nf, 1), axis=0, return_index=True)
    nf = nf[np.sort(uniq)]
    first = np.zeros(len(cnt), np.int64)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    return nv.astype(np.float32), nf, first


def load_frame(directory, t, transforms=None, shift_um=None, grid=1.0):
    """All nuclei of frame t as one mesh: (verts (N,3) µm, faces (F,3),
    owner (N,) label_id of each vertex). Registered if transforms is given;
    simplified for display if grid > 0."""
    m = np.load(Path(directory) / f't{t:03d}.npz')
    labels = sorted(int(k.split('_')[1]) for k in m.files if k.startswith('verts_'))
    vs, fs, owner, n = [], [], [], 0
    for lid in labels:
        v, f = m[f'verts_{lid}'], m[f'faces_{lid}']
        vs.append(v)
        fs.append(f + n)
        owner.append(np.full(len(v), lid))
        n += len(v)
    V, F, owner = np.concatenate(vs), np.concatenate(fs), np.concatenate(owner)
    if grid:                                  # before registering - see simplify()
        V, F, first = simplify(V, F, owner, grid)
        owner = owner[first]
    if transforms is not None:
        V = (V - [(shift_um or {}).get(t, 0.0), 0, 0]) @ transforms[t][:3, :3].T + transforms[t][:3, 3]
    return V.astype(np.float32), F, owner
