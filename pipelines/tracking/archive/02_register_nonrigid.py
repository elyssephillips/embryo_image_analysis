"""
Non-rigid (local-flow) registration on top of 02_register_centroids.py's
rigid registration - removes twisting / uneven deformation of the embryo
that a single rotation + translation can't.

Frame by frame (t -> t+1), in the rigid-registered space:
  1. match centroids (Hungarian, ≤ MAX_MATCH_DIST µm), as 02 does
  2. each matched pair gives a displacement: frame t+1 position -> its
     partner's position in the already-warped frame t
  3. every nucleus at t+1 is moved by the Gaussian-weighted (SIGMA_UM) mean
     displacement of the *other* matched nuclei - leave-one-out, so the warp
     predicts a nucleus's motion from its neighbours and removes collective
     motion on scales ≳ SIGMA_UM without erasing the nucleus's own movement
Warps chain from t=0, so the result is one consistent frame for tracking.

Not physical positions: the accumulated warp distorts long-range geometry, so
use these for linking only - analysis stays on the rigid *_reg coordinates.

Proxy check on stack5's curated links (true successor = nearest nucleus at
t+1): raw 165 wrong / 2103, rigid 53, this with σ=25-35 µm 39; 90th-pct step
6.5 -> 4.9 µm. σ=12 is worse again (warp starts absorbing real motion).

Output:
  features_nonrigid_csv   t, label_id, z_um_nr, y_um_nr, x_um_nr

Usage
-----
  python pipelines/tracking/02_register_nonrigid.py [--config X.yaml] [--sigma 30]
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.tracking_config import load_tracking_config

cfg, CONFIG_PATH = load_tracking_config()

SIGMA_UM       = 30.0   # warp smoothness: collective motion over ≳ this scale is removed
MAX_MATCH_DIST = 15.0   # same as 02_register_centroids.py

parser = argparse.ArgumentParser()
parser.add_argument('--sigma', type=float, default=None)
args = parser.parse_args()
if args.sigma is not None:
    SIGMA_UM = args.sigma

IN_CSV  = Path(cfg['paths']['features_registered_csv'])
OUT_CSV = Path(cfg['paths'].get('features_nonrigid_csv')
               or IN_CSV.with_name('features_nonrigid.csv'))
REG = ['z_um_reg', 'y_um_reg', 'x_um_reg']


def match(a, b):
    cost = np.linalg.norm(a[:, None] - b[None], axis=2)
    r, c = linear_sum_assignment(np.where(cost > MAX_MATCH_DIST, 1e6, cost))
    keep = cost[r, c] <= MAX_MATCH_DIST
    return r[keep], c[keep]


feat = pd.read_csv(IN_CSV).sort_values(['t', 'label_id'])
frames = [g for _, g in feat.groupby('t')]

warped = [frames[0][REG].to_numpy()]
print(f'σ = {SIGMA_UM:g} µm')
for prev, cur in zip(frames[:-1], frames[1:]):
    cur_rigid = cur[REG].to_numpy()
    ia, ib = match(prev[REG].to_numpy(), cur_rigid)
    disp = warped[-1][ia] - cur_rigid[ib]
    w = np.exp(-((cur_rigid[:, None] - cur_rigid[ib][None]) ** 2).sum(2) / (2 * SIGMA_UM ** 2))
    w[ib, np.arange(len(ib))] = 0                       # leave own match out
    wsum = w.sum(1, keepdims=True)
    fallback = disp.mean(0) if len(disp) else np.zeros(3)
    field = np.where(wsum > 1e-12, (w @ disp) / np.maximum(wsum, 1e-12), fallback)
    warped.append(cur_rigid + field)
    t = int(cur['t'].iloc[0])
    if t % 10 == 0:
        print(f'  t={t:03d}: {len(ib)} matches, mean |warp| {np.linalg.norm(field, axis=1).mean():.2f} µm')

out = pd.concat(frames)[['t', 'label_id']].reset_index(drop=True)
out[['z_um_nr', 'y_um_nr', 'x_um_nr']] = np.vstack(warped)
out.to_csv(OUT_CSV, index=False)
print(f'Saved {len(out)} rows -> {OUT_CSV}')
