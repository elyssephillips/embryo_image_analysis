"""
Per-timepoint ICM region from the Dataset003 nnUNet ICM class, and each TE
nucleus's distance to it.

00_preprocess_segmentation_dataset003.py drops ICM nuclei from the instance
labels, so every tracked nucleus is TE. The ICM itself is still in the raw
nnUNet prediction as class 2; prediction and label tiffs share the same
crop, so they align voxel-for-voxel and with features.csv centroids
(µm = voxel index × voxel size, frame-local). stack5 only:
scripts/fix_debris_crop_offsets.py shifted features.csv z_um by
-debris_shift_slices × VX_Z for some frames without re-cropping the tiffs,
so centroids are shifted back by crop_info.json's debris_shift_slices here
before measuring in image space (0 for uniformly-cropped stacks).

Per timepoint:
  1. ICM mask = class-2 voxels not inside a kept TE instance (mixed instances
     that 00 called TE stay TE), minus components < MIN_ICM_VOL_UM3 (debris).
  2. ICM region = convex hull of the ICM mask, rasterized on a coarse grid
     (GRID_DS × in xy ≈ 1 µm, native 2 µm in z) → signed distance field.
  3. For every TE nucleus in features.csv at that t:
       icm_edge_dist_um     signed distance from the nucleus centroid to the
                            hull surface (negative = inside the hull)
       icm_nearest_dist_um  distance to the nearest ICM nucleus voxel
                            (handles a concave ICM that the hull over-covers)
       icm_centroid_dist_um distance to the ICM voxel centroid

Nuclei-only caveat: the hull wraps ICM *nuclei*, so it sits inside the true
ICM cell boundary by roughly one cytoplasm thickness, and TE distances are
measured from centroids. Absolute values are biased high by a few µm; ranks
and changes over time are what's trustworthy until there's a membrane channel.
Check te_inside_hull in the summary - TE nuclei inside the hull mean the hull
is over-covering (concave ICM) and icm_nearest_dist_um is the safer measure.

Outputs (paths from config):
  icm_distance_csv  one row per (t, label_id) TE nucleus - joins to tracks CSVs
  icm_summary_csv   one row per t - ICM nucleus count, volumes, centroid, QC
  output_dir/icm_qc/  summary plot + XY/XZ projections at QC_TIMEPOINTS

Usage
-----
  python pipelines/tracking/01_extract_icm_region.py [--config X.yaml]
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.tracking_config import load_tracking_config, load_crop_info
from src.log import log_run

cfg, CONFIG_PATH = load_tracking_config()

# =============================================================================
#  CONFIG
# =============================================================================
ICM_CLASS       = 2       # nnUNet class id for ICM
MIN_ICM_VOL_UM3 = 100.0   # same floor as 00's MIN_INSTANCE_VOL_UM3
GRID_DS         = 5       # xy downsample for the hull distance grid (5 × 0.208 ≈ 1 µm)
N_WORKERS       = 8       # ~1 GB RAM per worker
QC_TIMEPOINTS   = [0, 30, 60, 90, 120]
# =============================================================================

PRED_DIR   = Path(cfg['paths']['nnunet_dir'])
PRED_GLOB  = cfg['paths']['nnunet_glob']
LABEL_DIR  = Path(cfg['paths']['label_dir'])
LABEL_GLOB = cfg['paths']['label_glob']
N_T        = cfg['microscopy']['n_timepoints']
VOX        = np.array(cfg['microscopy']['voxel_size_zyx'], dtype=float)
GRID_VOX   = VOX * np.array([1, GRID_DS, GRID_DS])

FEATURES_CSV = Path(cfg['paths']['features_csv'])
OUT_CSV      = Path(cfg['paths']['icm_distance_csv'])
SUMMARY_CSV  = Path(cfg['paths']['icm_summary_csv'])
QC_DIR       = Path(cfg['paths']['output_dir']) / 'icm_qc'


def icm_mask(pred, te):
    """ICM voxels: class 2, not inside a kept TE instance, components >= MIN_ICM_VOL_UM3.
    Returns (mask, kept component ids, total component count)."""
    icm = (pred == ICM_CLASS) & (te == 0)
    cc, n_cc = ndi.label(icm)
    vols_um3 = ndi.sum(icm, cc, np.arange(1, n_cc + 1)) * VOX.prod()
    keep = np.flatnonzero(vols_um3 >= MIN_ICM_VOL_UM3) + 1
    return np.isin(cc, keep), keep, n_cc


def hull_inside_grid(hull, shape):
    """Boolean grid (coarse) of voxels whose centres lie inside the convex hull."""
    nz, ny, nx = shape
    zz, yy, xx = np.meshgrid(np.arange(nz) * GRID_VOX[0],
                             np.arange(ny) * GRID_VOX[1],
                             np.arange(nx) * GRID_VOX[2], indexing='ij')
    pts = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
    normals, offsets = hull.equations[:, :3], hull.equations[:, 3]
    inside = np.ones(len(pts), dtype=bool)
    for start in range(0, len(pts), 500_000):
        chunk = pts[start:start + 500_000]
        inside[start:start + len(chunk)] = (chunk @ normals.T + offsets <= 1e-9).all(axis=1)
    return inside.reshape(shape)


def process_timepoint(args):
    t, pred_path, label_path, te_rows, z_shift_um = args
    pred = tifffile.imread(pred_path)
    te   = tifffile.imread(label_path)
    assert pred.shape == te.shape, f't={t}: prediction {pred.shape} vs labels {te.shape}'

    icm, keep, n_cc = icm_mask(pred, te)

    # features.csv → image space (see module docstring)
    te_um = te_rows[['z_um', 'y_um', 'x_um']].to_numpy() + np.array([z_shift_um, 0, 0])
    rows = te_rows[['t', 'label_id']].copy()
    summary = dict(t=t, debris_shift_um=z_shift_um, n_icm_components=len(keep), n_icm_dropped=n_cc - len(keep),
                   icm_nuclear_vol_um3=float(icm.sum() * VOX.prod()))
    qc = None

    if len(keep) < 1 or icm.sum() < 4:
        for c in ('icm_edge_dist_um', 'icm_nearest_dist_um', 'icm_centroid_dist_um'):
            rows[c] = np.nan
        return rows, summary, qc

    icm_um = np.argwhere(icm) * VOX
    centroid = icm_um.mean(axis=0)
    hull = ConvexHull(icm_um)

    grid_shape = (pred.shape[0], -(-pred.shape[1] // GRID_DS), -(-pred.shape[2] // GRID_DS))
    inside = hull_inside_grid(hull, grid_shape)
    sdf = np.where(inside,
                   -ndi.distance_transform_edt(inside, sampling=GRID_VOX),
                   ndi.distance_transform_edt(~inside, sampling=GRID_VOX))

    gi = np.rint(te_um / GRID_VOX).astype(int)
    gi = np.clip(gi, 0, np.array(grid_shape) - 1)
    rows['icm_edge_dist_um']     = sdf[gi[:, 0], gi[:, 1], gi[:, 2]]
    rows['icm_nearest_dist_um']  = cKDTree(icm_um).query(te_um)[0]
    rows['icm_centroid_dist_um'] = np.linalg.norm(te_um - centroid, axis=1)

    # centroid reported in features.csv's z frame so it lines up with z_um / z_um_orig
    summary.update(icm_hull_vol_um3=float(hull.volume),
                   icm_centroid_z_um=centroid[0] - z_shift_um, icm_centroid_y_um=centroid[1],
                   icm_centroid_x_um=centroid[2],
                   n_te=len(rows), te_inside_hull=int((rows['icm_edge_dist_um'] < 0).sum()))

    if t in QC_TIMEPOINTS:
        qc = dict(t=t, icm_xy=icm.max(axis=0)[::GRID_DS, ::GRID_DS], icm_xz=icm.max(axis=1)[:, ::GRID_DS],
                  hull_xy=inside.max(axis=0), hull_xz=inside.max(axis=1),
                  te_um=te_um, dist=rows['icm_edge_dist_um'].to_numpy())
    return rows, summary, qc


def plot_qc(qc_list, summary):
    QC_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(summary['t'], summary['n_icm_components'], '.-')
    axes[0].set_ylabel('ICM components\n(≈ nuclei)')
    axes[1].plot(summary['t'], summary['icm_nuclear_vol_um3'], '.-', label='ICM nuclei')
    axes[1].plot(summary['t'], summary['icm_hull_vol_um3'], '.-', label='hull')
    axes[1].set_ylabel('volume (µm³)'); axes[1].legend()
    axes[2].plot(summary['t'], summary['te_inside_hull'], '.-', color='C3')
    axes[2].set_ylabel('TE nuclei\ninside hull'); axes[2].set_xlabel('t')
    fig.tight_layout(); fig.savefig(QC_DIR / 'icm_summary.png', dpi=120); plt.close(fig)

    gy, gx = GRID_VOX[1], GRID_VOX[2]
    for qc in qc_list:
        fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(14, 6),
                                           gridspec_kw=dict(width_ratios=[1, 1]))
        ny, nx = qc['hull_xy'].shape
        ax_xy.imshow(qc['icm_xy'], cmap='Greys', extent=[0, nx * gx, ny * gy, 0])
        ax_xy.contour(np.arange(nx) * gx, np.arange(ny) * gy, qc['hull_xy'], [0.5], colors='C1')
        sc = ax_xy.scatter(qc['te_um'][:, 2], qc['te_um'][:, 1], c=qc['dist'], s=14,
                           cmap='coolwarm', vmin=-20, vmax=60)
        ax_xy.set_title(f't={qc["t"]}  XY max-projection'); ax_xy.set_xlabel('x (µm)'); ax_xy.set_ylabel('y (µm)')

        nz = qc['hull_xz'].shape[0]
        ax_xz.imshow(qc['icm_xz'], cmap='Greys', aspect='auto', extent=[0, nx * gx, nz * VOX[0], 0])
        ax_xz.contour(np.arange(nx) * gx, np.arange(nz) * VOX[0], qc['hull_xz'], [0.5], colors='C1')
        ax_xz.scatter(qc['te_um'][:, 2], qc['te_um'][:, 0], c=qc['dist'], s=14,
                      cmap='coolwarm', vmin=-20, vmax=60)
        ax_xz.set_title('XZ max-projection'); ax_xz.set_xlabel('x (µm)'); ax_xz.set_ylabel('z (µm)')
        fig.colorbar(sc, ax=[ax_xy, ax_xz], label='TE centroid → ICM hull edge (µm)')
        fig.savefig(QC_DIR / f'icm_region_t{qc["t"]:03d}.png', dpi=120); plt.close(fig)


def main():
    pred_files  = sorted(PRED_DIR.glob(PRED_GLOB))[:N_T]
    label_files = sorted(LABEL_DIR.glob(LABEL_GLOB))[:N_T]
    assert len(pred_files) == len(label_files) == N_T, \
        f'Expected {N_T} prediction + label files, found {len(pred_files)} / {len(label_files)}'

    feat = pd.read_csv(FEATURES_CSV, usecols=['t', 'label_id', 'z_um', 'y_um', 'x_um'])
    crop_info = load_crop_info(cfg)
    shift_um = {int(k): v.get('debris_shift_slices', 0) * VOX[0] for k, v in crop_info.items()}
    jobs = [(t, p, l, feat[feat['t'] == t], shift_um.get(t, 0.0))
            for t, (p, l) in enumerate(zip(pred_files, label_files))]

    rows, summaries, qcs = [], [], []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for r, s, qc in pool.map(process_timepoint, jobs):
            print(f"t={s['t']:03d}  ICM components={s['n_icm_components']:3d}  "
                  f"TE inside hull={s.get('te_inside_hull', '-')}/{s.get('n_te', len(r))}", flush=True)
            rows.append(r); summaries.append(s)
            if qc is not None:
                qcs.append(qc)

    out = pd.concat(rows, ignore_index=True)
    summary = pd.DataFrame(summaries)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f'\nSaved: {OUT_CSV}  ({len(out)} TE nucleus-timepoints)')
    print(f'Saved: {SUMMARY_CSV}')

    plot_qc(qcs, summary)
    print(f'Saved QC plots: {QC_DIR}')

    print('\n--- distances (µm) ---')
    print(out[['icm_edge_dist_um', 'icm_nearest_dist_um', 'icm_centroid_dist_um']]
          .describe().round(1).to_string())
    print(f"\nFrames with TE nuclei inside the hull: {(summary['te_inside_hull'] > 0).sum()}/{len(summary)}")

    log_run('tracking', CONFIG_PATH.stem, '01_extract_icm_region.py',
            output_path=str(OUT_CSV), detail='light', data_path=str(OUT_CSV.parent))


if __name__ == '__main__':
    main()
