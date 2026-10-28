"""
Rigid registration of the nucleus point clouds over time - removes whole-
embryo drift and tumbling, so the same nucleus sits in about the same place
from frame to frame. Rotation + translation only: distances are preserved,
so registered coordinates are real µm and are used for analysis too.

For each t -> t+1 pair, in the raw (camera) coordinates of both frames:
  0. size-normalise: scale frame t+1's cloud about its centroid to the same
     RMS radius as frame t's, for fitting only. The blastocyst collapses and
     re-expands (radius changes of 10-40% between frames on stack5/stack8);
     without this, the fit at those steps is arbitrary (e.g. stack8 t18->19
     came out 67° instead of ~7°). Output stays rotation + translation.
  1. coarse search: centre both clouds and try ~4500 rotations up to
     MAX_STEP_DEG; score each by the trimmed (best TRIM fraction) mean
     nearest-neighbour distance from frame t+1 to frame t
  2. refine the TOP_K best starts with ICP (mutual nearest neighbours within
     ICP_MAX_UM + Kabsch), keep the refined fit with the lowest trimmed cost
     plus a small prior against large rotations (ANGLE_PRIOR per degree)
  3. translation for the unscaled cloud from the matched pairs, then chain:
     M[t+1] = M[t] @ step
  4. re-reference to REFERENCE_T: every frame maps into that frame's camera
     coordinates. Default 'last': by the end the embryo has attached and
     flattened on the dish, so late frames stay as they sit on the dish and
     registered z ≈ height above it; earlier tumbling is undone relative to
     that. (t=0 instead would hold the early embryo still but show the
     attached disc tilted on its side.)

Why the search: the embryo can turn ~30° in one 15-min frame, moving nuclei
further than the ~17 µm spacing, so plain nearest-neighbour matching (the
previous version, archive/02_register_centroids_v1.py) gets most pairs
wrong exactly when it matters. On stack5's curated tracks, rotation left
between frames 10 apart: v1 16° (= no better than raw), this 1.6°; 40
apart: 41° -> 7°. Nearest-neighbour errors on curated links 53 -> 35.

QC: steps whose fit is still poor (cost > 2x median, or < 50% of nuclei
matched) are listed at the end - check those frames in the raw data. The QC
file's size_ratio column (frame t RMS radius / frame t+1) also marks collapse
(> 1) and re-expansion (< 1) events.

Output:
  features_registered.csv  original features + z_um_reg, y_um_reg, x_um_reg
  transforms.npz           (T, 4, 4): M[t] maps features.csv µm at t -> registered µm
                           (M[REFERENCE_T] = identity)
  registration_qc.csv      per step: rotation, size ratio, fit cost, fraction matched, flag
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.tracking_config import load_tracking_config

cfg, CONFIG_PATH = load_tracking_config()   # pipelines/tracking/active_dataset.yaml, or --config

FEATURES_CSV = Path(cfg['paths']['features_csv'])
OUT_CSV      = Path(cfg['paths']['features_registered_csv'])
OUT_NPZ      = Path(cfg['paths']['transforms_npz'])
OUT_QC       = OUT_CSV.with_name('registration_qc.csv')

# =============================================================================
MAX_STEP_DEG = 60      # largest rotation searched per frame step
GRID_STEP    = 4       # degrees between rotation angles in the coarse search
N_AXES       = 300     # rotation axes in the coarse search
TRIM         = 0.8     # fraction of nuclei scored (ignores appearing/vanishing ones)
TOP_K        = 20      # coarse candidates refined with ICP
ICP_MAX_UM   = 10.0    # ICP match distance
ANGLE_PRIOR  = 0.02    # µm of cost per degree - prefer smaller rotations on near-ties
REFERENCE_T  = 'last'  # frame whose camera coordinates everything maps into: 'last', or an int t
# =============================================================================


def rotation_grid():
    i = np.arange(N_AXES) + 0.5
    phi = np.arccos(1 - 2 * i / N_AXES)
    th = np.pi * (1 + 5 ** 0.5) * i
    axes = np.column_stack([np.cos(phi), np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th)])
    rots = [np.eye(3)]
    for ang in np.arange(GRID_STEP, MAX_STEP_DEG + 1e-9, GRID_STEP):
        rots.extend(Rotation.from_rotvec(axes * np.radians(ang)).as_matrix())
    return np.array(rots)


GRID = rotation_grid()


def angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def trimmed_cost(tree, pts):
    d = np.sort(tree.query(pts)[0])
    return d[:max(1, int(len(d) * TRIM))].mean()


def kabsch(src, dst):
    """R, t minimising |R src + t - dst|."""
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - cd).T @ (src - cs))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, cd - R @ cs


def icp(A, B, R, t, treeA):
    ok, j = np.zeros(len(B), bool), np.zeros(len(B), int)
    for _ in range(30):
        Bt = B @ R.T + t
        d, j = treeA.query(Bt)
        back = cKDTree(Bt).query(A)[1]
        ok = (d < ICP_MAX_UM) & (back[j] == np.arange(len(B)))
        if ok.sum() < 6:
            break
        Rn, tn = kabsch(B[ok], A[j[ok]])
        if np.allclose(Rn, R, atol=1e-7) and np.allclose(tn, t, atol=1e-6):
            break
        R, t = Rn, tn
    return R, t, ok, j


def rms_radius(P):
    return np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean())


def register_step(A, B):
    """4x4 rigid step mapping frame-t+1 points B onto frame-t points A,
    plus (size ratio, trimmed fit cost µm, fraction of nuclei matched)."""
    ca, cb = A.mean(0), B.mean(0)
    size_ratio = rms_radius(A) / rms_radius(B)
    Bs = (B - cb) * size_ratio + cb                      # size-normalised, for fitting only
    tree_c = cKDTree(A - ca)
    costs = np.array([trimmed_cost(tree_c, (Bs - cb) @ R.T) for R in GRID])
    treeA = cKDTree(A)
    best = None
    for k in np.argsort(costs)[:TOP_K]:
        R, t, ok, j = icp(A, Bs, GRID[k], ca - GRID[k] @ cb, treeA)
        fit = trimmed_cost(treeA, Bs @ R.T + t)
        score = fit + ANGLE_PRIOR * angle_deg(R)
        if best is None or score < best[0]:
            best = (score, R, fit, ok, j)
    _, R, fit, ok, j = best
    # translation for the real (unscaled) cloud: line up the matched pairs' centroids
    t = (A[j[ok]].mean(0) - R @ B[ok].mean(0)) if ok.sum() >= 6 else ca - R @ cb
    S = np.eye(4)
    S[:3, :3], S[:3, 3] = R, t
    return S, size_ratio, fit, ok.sum() / min(len(A), len(B))


def main():
    df = pd.read_csv(FEATURES_CSV).sort_values(['t', 'label_id']).reset_index(drop=True)
    ts = sorted(df['t'].unique())
    raw = {t: df.loc[df['t'] == t, ['z_um', 'y_um', 'x_um']].to_numpy() for t in ts}
    print(f'Loaded {len(df)} detections across {len(ts)} timepoints')

    M = np.zeros((max(ts) + 1, 4, 4))
    M[ts[0]] = np.eye(4)
    qc = []
    for t0, t1 in zip(ts[:-1], ts[1:]):
        S, size_ratio, fit, frac = register_step(raw[t0], raw[t1])
        M[t1] = M[t0] @ S
        qc.append({'t': t0, 'step_deg': angle_deg(S[:3, :3]), 'size_ratio': size_ratio,
                   'fit_cost_um': fit, 'frac_matched': frac})
        print(f'  t={t0}→{t1}: rotation {qc[-1]["step_deg"]:5.1f}°, size ratio {size_ratio:4.2f}, '
              f'fit {fit:5.2f} µm, {100 * frac:3.0f}% matched', flush=True)

    ref = ts[-1] if REFERENCE_T == 'last' else int(REFERENCE_T)
    M[ts] = np.linalg.inv(M[ref]) @ M[ts]
    print(f'Reference frame: t={ref} (registered = its camera coordinates)')
    reg = np.vstack([raw[t] @ M[t][:3, :3].T + M[t][:3, 3] for t in ts])
    df[['z_um_reg', 'y_um_reg', 'x_um_reg']] = reg
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    np.savez_compressed(OUT_NPZ, transforms=M)

    qc = pd.DataFrame(qc)
    qc['flag'] = (qc['fit_cost_um'] > 2 * qc['fit_cost_um'].median()) | (qc['frac_matched'] < 0.5)
    qc.to_csv(OUT_QC, index=False)

    print(f'\nSaved registered features: {OUT_CSV}')
    print(f'Saved transforms:          {OUT_NPZ}')
    print(f'Saved per-step QC:         {OUT_QC}')
    print(f'\nStep rotation median {qc.step_deg.median():.1f}°, max {qc.step_deg.max():.1f}°; '
          f'fit cost median {qc.fit_cost_um.median():.2f} µm')
    size = qc[(qc['size_ratio'] - 1).abs() > 0.1]
    if len(size):
        print(f'\n{len(size)} step(s) where the embryo changed size by >10% '
              f'(size_ratio > 1 = shrank, e.g. collapse; < 1 = re-expanded):')
        print(size[['t', 'size_ratio', 'step_deg', 'frac_matched']].round(2).to_string(index=False))
    bad = qc[qc['flag']]
    if len(bad):
        print(f'\n{len(bad)} step(s) with a poor fit - check these frames in the raw data:')
        print(bad.round(2).to_string(index=False))
    else:
        print('\nNo poorly fitting steps.')


if __name__ == '__main__':
    main()
