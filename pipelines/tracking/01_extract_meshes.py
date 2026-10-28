"""
Per-nucleus surface meshes (PyMCubes) and mesh-based shape descriptors.

The label voxels are 2.0 µm in z but 0.208 µm in xy, so a nucleus is only
~5-7 slices tall and meshing the raw mask gives a stair-stepped surface whose
shape measures mostly reflect the z steps. Per nucleus, at each timepoint:
  1. crop the mask (padded so the surface closes) and turn it into a signed
     distance field by shape-based interpolation: 2D signed distance (µm,
     + inside) per z slice, trilinearly resampled onto an isotropic ISO_UM
     grid. Between slices the outline morphs from one slice's to the next's
     (sloped walls) instead of jumping halfway (2 µm terraces); top/bottom
     caps dome up half a slice at the centre. mcubes.smooth can't do this -
     its constrained method keeps the surface within half a grid step of the
     binary boundary, and a z-only Gaussian leaves the 0.5 crossing at the
     slice midpoints whatever its sigma (2026-10-21 comparison)
  2. mcubes.marching_cubes at the zero level
  3. descriptors from the mesh, in µm:
       mesh_volume_um3, mesh_area_um2, mesh_sphericity
       mesh_solidity          volume / convex-hull volume (dents, merges)
       mesh_cov_{zz,yy,xx,zy,zx,yx}  covariance of the enclosed solid (µm²) -
                              the physical-units version of regionprops'
                              moments (trackastra's inertia_tensor = tr(C)·I - C)
       mesh_l1 >= l2 >= l3    its eigenvalues (principal second moments)
       mesh_elongation        sqrt(l1 / l3);  mesh_flatness  sqrt(l2 / l3)
       mesh_axis_{z,y,x}      major axis, raw image frame
       mesh_axis_reg_{z,y,x}  major axis rotated into 02's registered frame
                              (transforms.npz) - comparable across frames
     Axes are sign-ambiguous; they're flipped so the largest component is +.

Outputs (paths from config, defaulting next to features_csv):
  mesh_features_csv   one row per (t, label_id) - joins to features.csv / tracks
  mesh_dir/t{t:03d}.npz   verts (µm, z/y/x, raw image frame) + faces per label,
                          if SAVE_MESHES - load into napari with add_surface

Usage
-----
  python pipelines/tracking/01_extract_meshes.py [--config X.yaml] [--timepoints 0 60]
"""

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')      # one thread per worker - shared machine
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mcubes
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.tracking_config import load_tracking_config

cfg, CONFIG_PATH = load_tracking_config()

# =============================================================================
ISO_UM      = 0.5    # isotropic resampling grid (µm)
PAD         = 3      # ISO_UM grid steps of zero padding around each crop (closes the surface)
MIN_VOXELS  = 20     # skip fragments smaller than this
N_WORKERS   = 8      # one frame per worker, ~1-2 GB each
SAVE_MESHES = True
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--timepoints', nargs='+', type=int, default=None,
                    help='only these frames (for a quick look); default all')
args = parser.parse_args()

VX        = np.array(cfg['microscopy']['voxel_size_zyx'], dtype=float)
N_T       = cfg['microscopy']['n_timepoints']
LABEL_DIR = Path(cfg['paths']['label_dir'])
FEAT_DIR  = Path(cfg['paths']['features_csv']).parent
OUT_CSV   = Path(cfg['paths'].get('mesh_features_csv') or FEAT_DIR / 'mesh_features.csv')
MESH_DIR  = Path(cfg['paths'].get('mesh_dir') or FEAT_DIR.parent / 'meshes')
TRANSFORMS = Path(cfg['paths']['transforms_npz'])

label_files = sorted(LABEL_DIR.glob(cfg['paths']['label_glob']))[:N_T]
assert len(label_files) == N_T, f'Expected {N_T} label files, found {len(label_files)}'


def sdf_iso(mask):
    """Binary crop -> signed distance (µm, + inside) on an ISO_UM grid, by
    shape-based interpolation, zero-padded by PAD grid steps on every side.
    Grid point i sits at (i - PAD) * ISO_UM µm from the centre of the voxel
    one before the crop's first. (Padding on the resampled grid, not the raw
    one: PAD raw voxels is only PAD * 0.2 µm in xy, under one grid step, so
    the surface would touch the edge and come out open.)"""
    # One empty slice/pixel beyond the crop on every side. In xy the zero
    # crossing lands on the pixel faces (+-1 px either side of the edge). An
    # empty slice next to an occupied one gets -max(neighbour's SDF): the cap
    # then reaches the voxel face (half a slice out) at its centre and stays
    # on the slice at its rim.
    m = np.pad(mask, 1)
    sdf = np.empty(m.shape, np.float32)
    occupied = m.any((1, 2))
    for k in np.flatnonzero(occupied):
        sdf[k] = (ndi.distance_transform_edt(m[k], sampling=VX[1:])
                  - ndi.distance_transform_edt(~m[k], sampling=VX[1:]))
    for k in np.flatnonzero(~occupied):
        nb = [sdf[j].max() for j in (k - 1, k + 1) if 0 <= j < len(m) and occupied[j]]
        sdf[k] = -max(nb) if nb else -1e3
    extent = (np.array(m.shape) - 1) * VX
    axes = [np.arange(0, e + 1e-6, ISO_UM) / v for e, v in zip(extent, VX)]
    vol = ndi.map_coordinates(sdf, np.meshgrid(*axes, indexing='ij'), order=1)
    return np.pad(vol, PAD, constant_values=vol.min())


def orient(v):
    return v if v[np.argmax(np.abs(v))] >= 0 else -v


def describe(verts, faces):
    tri = verts[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    tri_area = 0.5 * np.linalg.norm(cross, axis=1)
    area = tri_area.sum()
    # Solid moments: sum over tetrahedra (o, a, b, c) from a reference point o
    # inside-ish (vertex mean) - signed volumes cancel outside the surface.
    o = verts.mean(0)
    p = tri - o
    vol_t = np.einsum('ij,ij->i', p[:, 0], np.cross(p[:, 1], p[:, 2])) / 6
    volume = vol_t.sum()
    sgn = 1.0 if volume >= 0 else -1.0                 # face winding -> sign
    s = p.sum(1)
    first = (vol_t[:, None] * s).sum(0) / 4
    second = (np.einsum('n,nki,nkj->ij', vol_t, p, p) + np.einsum('n,ni,nj->ij', vol_t, s, s)) / 20
    volume *= sgn
    c = first * sgn / volume
    cov = second * sgn / volume - np.outer(c, c)
    evals, evecs = np.linalg.eigh(cov)                 # ascending
    l3, l2, l1 = np.maximum(evals, 1e-12)
    try:
        hull_vol = ConvexHull(verts).volume
    except Exception:
        hull_vol = np.nan
    return {
        'mesh_volume_um3': volume,
        'mesh_area_um2':   area,
        'mesh_sphericity': np.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / area if area > 0 else np.nan,
        'mesh_solidity':   volume / hull_vol if hull_vol and hull_vol > 0 else np.nan,
        'mesh_cov_zz': cov[0, 0], 'mesh_cov_yy': cov[1, 1], 'mesh_cov_xx': cov[2, 2],
        'mesh_cov_zy': cov[0, 1], 'mesh_cov_zx': cov[0, 2], 'mesh_cov_yx': cov[1, 2],
        'mesh_l1': l1, 'mesh_l2': l2, 'mesh_l3': l3,
        'mesh_elongation': np.sqrt(l1 / l3),
        'mesh_flatness':   np.sqrt(l2 / l3),
    }, orient(evecs[:, 2])


def process_frame(t):
    labels = tifffile.imread(label_files[t])
    rot = np.load(TRANSFORMS)['transforms'][t][:3, :3] if TRANSFORMS.exists() else None
    if rot is not None:
        rot = rot / np.cbrt(np.linalg.det(rot))         # drop any similarity scale
    rows, meshes = [], {}
    for lid, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        mask = labels[sl] == lid
        if mask.sum() < MIN_VOXELS:
            continue
        verts, faces = mcubes.marching_cubes(sdf_iso(mask), 0.0)
        if len(faces) == 0:
            continue
        origin = (np.array([s.start for s in sl]) - 1) * VX - PAD * ISO_UM
        verts = verts * ISO_UM + origin                   # µm, raw image frame
        desc, axis = describe(verts, faces)
        row = {'t': t, 'label_id': lid, **desc,
               'mesh_axis_z': axis[0], 'mesh_axis_y': axis[1], 'mesh_axis_x': axis[2]}
        if rot is not None:
            a = orient(rot @ axis)
            row.update(mesh_axis_reg_z=a[0], mesh_axis_reg_y=a[1], mesh_axis_reg_x=a[2])
        rows.append(row)
        if SAVE_MESHES:
            meshes[f'verts_{lid}'] = verts.astype(np.float32)
            meshes[f'faces_{lid}'] = faces.astype(np.int32)
    if SAVE_MESHES:
        np.savez_compressed(MESH_DIR / f't{t:03d}.npz', **meshes)
    print(f't={t:03d}  {len(rows)} meshes', flush=True)
    return rows


if __name__ == '__main__':
    timepoints = args.timepoints if args.timepoints is not None else list(range(N_T))
    if not TRANSFORMS.exists():
        print(f'No {TRANSFORMS} (run 02_register_centroids.py) - skipping registered axes')
    if SAVE_MESHES:
        MESH_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(N_WORKERS) as ex:
        rows = [r for frame in ex.map(process_frame, timepoints) for r in frame]
    df = pd.DataFrame(rows).sort_values(['t', 'label_id'])

    out = OUT_CSV if args.timepoints is None else OUT_CSV.with_name(OUT_CSV.stem + '_subset.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'\nSaved {len(df)} rows ({df["t"].nunique()} timepoints) -> {out}')
    print(df[['mesh_volume_um3', 'mesh_sphericity', 'mesh_solidity', 'mesh_elongation',
              'mesh_flatness']].describe().round(3).to_string())
