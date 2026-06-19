"""
Register centroid point clouds between consecutive timepoints using a
similarity transform (rotation + translation + optional uniform scale).

For each t → t+1 pair:
  1. Match centroids using Hungarian algorithm (nearest-neighbour cost)
  2. Estimate rigid/similarity transform via Umeyama algorithm (SVD)
  3. Apply transform to accumulate all frames into a common reference (t=0)

Output:
  features_registered.csv  — original features + registered coords (z_um_reg, y_um_reg, x_um_reg)
  transforms.npz            — (101, 4, 4) cumulative affine matrices
                              M[t] maps original µm coords at t → registered frame
                              np.linalg.inv(M[t]) maps registered → original at t

The original coordinates are kept in the CSV so we can recover image-space
positions for napari display without needing to apply inverse transforms manually.
"""

import yaml
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

FEATURES_CSV = Path(cfg['paths']['features_csv'])
OUT_CSV      = Path(cfg['paths']['features_registered_csv'])
OUT_NPZ      = Path(cfg['paths']['transforms_npz'])

# --- Registration parameters ---
# Maximum distance (µm) to allow a centroid match during Hungarian assignment.
# Should be less than inter-nuclear spacing (~17.7 µm) to avoid false matches.
MAX_MATCH_DIST = 15.0

# Use similarity (rotation + scale + translation) vs rigid (rotation + translation only).
# Rigid is safer for long timeseries to avoid scale drift accumulation.
WITH_SCALE = False

# Minimum number of matched pairs required to trust a transform.
# If fewer pairs are matched, fall back to translation-only.
MIN_MATCHES = 10


def hungarian_match(src, dst, max_dist):
    """
    Match src to dst using Hungarian algorithm on Euclidean distance.
    Returns matched (src_pts, dst_pts) arrays, filtered by max_dist.
    """
    dist = np.sqrt(((src[:, None] - dst[None, :]) ** 2).sum(axis=2))
    row_ind, col_ind = linear_sum_assignment(dist)
    valid = dist[row_ind, col_ind] < max_dist
    return src[row_ind[valid]], dst[col_ind[valid]]


def umeyama(src, dst, with_scale=False):
    """
    Estimate transform that maps src → dst (Umeyama 1991).
    Returns (R 3x3, t 3-vec, c scalar).
    For rigid: c = 1.0
    """
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = (src_c ** 2).sum() / n
    if var_src < 1e-10:
        return np.eye(3), mu_dst - mu_src, 1.0

    C = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(C)

    # Handle reflections
    det_sign = np.linalg.det(U) * np.linalg.det(Vt)
    S = np.diag([1.0, 1.0, float(np.sign(det_sign))])

    R = U @ S @ Vt
    c = float(np.trace(np.diag(D) @ S) / var_src) if with_scale else 1.0
    t = mu_dst - c * R @ mu_src

    return R, t, c


def make_affine(R, t, c=1.0):
    """Build 4×4 homogeneous affine matrix from R, t, scale c."""
    M = np.eye(4)
    M[:3, :3] = c * R
    M[:3, 3] = t
    return M


def apply_affine(M, pts):
    """Apply 4×4 affine M to (N, 3) points."""
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    return (M @ pts_h.T).T[:, :3]


def main():
    df = pd.read_csv(FEATURES_CSV)
    n_frames = df['t'].nunique()
    print(f'Loaded {len(df)} detections across {n_frames} timepoints')

    # Cumulative transforms: M[t] maps original coords at t → registered frame (t=0)
    transforms = np.zeros((n_frames, 4, 4))
    transforms[0] = np.eye(4)  # t=0 is the reference

    # Registered coordinates (start with t=0 unchanged)
    reg_coords = {}  # t → (N, 3) registered coords

    coords_t0 = df[df['t'] == 0][['z_um', 'y_um', 'x_um']].values
    reg_coords[0] = coords_t0.copy()

    current_M = np.eye(4)  # cumulative transform

    for t in range(n_frames - 1):
        # Source: registered coords at t (in reference frame)
        src = reg_coords[t]

        # Target: original coords at t+1 (not yet registered)
        dst_raw = df[df['t'] == t + 1][['z_um', 'y_um', 'x_um']].values

        # Match
        src_m, dst_m = hungarian_match(src, dst_raw, MAX_MATCH_DIST)
        n_matches = len(src_m)

        if n_matches < MIN_MATCHES:
            print(f'  t={t}→{t+1}: only {n_matches} matches — using translation only')
            if n_matches >= 3:
                t_vec = src_m.mean(axis=0) - dst_m.mean(axis=0)
                R, c = np.eye(3), 1.0
            else:
                # Fall back to global centroid shift
                t_vec = src.mean(axis=0) - dst_raw.mean(axis=0)
                R, c = np.eye(3), 1.0
            step_M = make_affine(R, t_vec, c)
        else:
            # Umeyama: src is target (reference), dst is source to transform
            # We want transform T such that T(dst_raw) ≈ src
            R, t_vec, c = umeyama(dst_m, src_m, with_scale=WITH_SCALE)
            step_M = make_affine(R, t_vec, c)
            if n_matches > 0:
                residual = np.mean(np.linalg.norm(
                    apply_affine(step_M, dst_m) - src_m, axis=1))
                print(f'  t={t}→{t+1}: {n_matches} matches, '
                      f'residual {residual:.2f} µm, scale {c:.4f}')

        # Compose: cumulative transform for t+1
        current_M = step_M @ current_M
        transforms[t + 1] = current_M

        # Apply step_M to all t+1 original coords to get registered coords
        reg_coords[t + 1] = apply_affine(step_M, dst_raw)

    # Build output DataFrame — add registered coords alongside originals
    rows = []
    for t in range(n_frames):
        sub = df[df['t'] == t].copy().reset_index(drop=True)
        reg = reg_coords[t]
        sub['z_um_reg'] = reg[:, 0]
        sub['y_um_reg'] = reg[:, 1]
        sub['x_um_reg'] = reg[:, 2]
        rows.append(sub)

    out_df = pd.concat(rows, ignore_index=True)
    out_df.to_csv(OUT_CSV, index=False)
    np.savez_compressed(OUT_NPZ, transforms=transforms)

    print(f'\nSaved registered features: {OUT_CSV}')
    print(f'Saved transforms:          {OUT_NPZ}')

    # Sanity check: residuals after registration
    residuals = []
    for t in range(n_frames - 1):
        src = reg_coords[t]
        dst = reg_coords[t + 1]
        src_m, dst_m = hungarian_match(src, dst, MAX_MATCH_DIST)
        if len(src_m) > 0:
            residuals.extend(np.linalg.norm(src_m - dst_m, axis=1).tolist())

    residuals = np.array(residuals)
    print(f'\nPost-registration nearest-neighbour residuals:')
    print(f'  median:  {np.median(residuals):.2f} µm')
    print(f'  90th pct: {np.percentile(residuals, 90):.2f} µm')
    print(f'  max:     {residuals.max():.2f} µm')


if __name__ == '__main__':
    main()
