"""
Extract per-nucleus features from instance label tiffs + raw tiffs.

For each nucleus at each timepoint, records:
  - centroid in µm (z_um, y_um, x_um)
  - volume (area_vox, area_um3)
  - equivalent spherical diameter in µm
  - intensity from raw signal (mean, max, sum, std)
  - elongation: ratio of major to minor principal axis (from inertia tensor)
  - sphericity: how close to a perfect sphere (1.0 = sphere, <1 = irregular)

Output: features.csv
"""

import glob
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.io import imread
from skimage.measure import regionprops, marching_cubes, mesh_surface_area

REPO_ROOT   = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.log import log_run

CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# --- Paths ---
LABEL_DIR    = cfg['paths']['label_dir']
RAW_DIR      = cfg['paths']['raw_dir']
OUT_CSV      = cfg['paths']['features_csv']
N_TIMEPOINTS = cfg['microscopy']['n_timepoints']

# --- Voxel sizes ---
VX_Z  = 2.0    # µm per z-step
VX_Y  = 0.208  # µm per y-pixel
VX_X  = 0.208  # µm per x-pixel
VOX_VOL = VX_Z * VX_Y * VX_X  # µm³ per voxel

label_files = sorted(glob.glob(f"{LABEL_DIR}*_instances_reclassified.tif"))[:N_TIMEPOINTS]
raw_files   = sorted(glob.glob(f"{RAW_DIR}Cam_long_*.tif"))[:N_TIMEPOINTS]

assert len(label_files) == N_TIMEPOINTS, \
    f"Expected {N_TIMEPOINTS} label files, found {len(label_files)}"
assert len(raw_files) == N_TIMEPOINTS, \
    f"Expected {N_TIMEPOINTS} raw files, found {len(raw_files)}"

records = []

for t, (lf, rf) in enumerate(zip(label_files, raw_files)):
    print(f"t={t:03d}  {lf.split('/')[-1]}", flush=True)
    labels = imread(lf)
    raw    = imread(rf).astype(np.float32)

    assert labels.shape == raw.shape, \
        f"Shape mismatch at t={t}: labels {labels.shape}, raw {raw.shape}"

    for prop in regionprops(labels, intensity_image=raw):
        z_um = prop.centroid[0] * VX_Z
        y_um = prop.centroid[1] * VX_Y
        x_um = prop.centroid[2] * VX_X

        area_vox  = int(prop.area)
        area_um3  = area_vox * VOX_VOL

        # Equivalent diameter of a sphere with the same volume (in µm)
        equiv_diam_um = 2.0 * ((3.0 * area_um3) / (4.0 * np.pi)) ** (1.0 / 3.0)

        # Elongation from inertia tensor eigenvalues (λ1 ≤ λ2 ≤ λ3)
        # Semi-axes of equivalent ellipsoid ∝ sqrt(λj + λk - λi)
        # elongation = major / minor axis ratio (1.0 = sphere)
        lam = np.array(prop.inertia_tensor_eigvals)
        a2 = max(lam[1] + lam[2] - lam[0], 0.0)
        c2 = max(lam[0] + lam[1] - lam[2], 0.0)
        elongation = np.sqrt(a2 / c2) if c2 > 0 else np.nan

        # Sphericity: ratio of sphere surface area (same volume) to actual surface area
        # Uses marching cubes on the nucleus binary image within its bounding box
        try:
            verts, faces, _, _ = marching_cubes(
                prop.image.astype(np.float32), level=0.5,
                spacing=(VX_Z, VX_Y, VX_X)
            )
            surface_um2 = mesh_surface_area(verts, faces)
            sphere_surface = np.pi ** (1/3) * (6 * area_um3) ** (2/3)
            sphericity = sphere_surface / surface_um2 if surface_um2 > 0 else np.nan
        except Exception:
            sphericity = np.nan

        records.append({
            't':              t,
            'label_id':       int(prop.label),
            'z_um':           z_um,
            'y_um':           y_um,
            'x_um':           x_um,
            'area_vox':       area_vox,
            'area_um3':       area_um3,
            'equiv_diam_um':  equiv_diam_um,
            'mean_intensity': float(prop.intensity_mean),
            'max_intensity':  float(prop.intensity_max),
            'sum_intensity':  float(prop.intensity_mean * area_vox),
            'std_intensity':  float(prop.intensity_std),
            'elongation':     float(elongation) if not np.isnan(elongation) else np.nan,
            'sphericity':     float(sphericity) if not np.isnan(sphericity) else np.nan,
        })

df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)

print(f"\nDone. Saved {len(df)} rows ({df['t'].nunique()} timepoints) to {OUT_CSV}")
print(f"Nuclei per timepoint: min={df.groupby('t').size().min()}, "
      f"max={df.groupby('t').size().max()}, "
      f"mean={df.groupby('t').size().mean():.1f}")
print(df.head())

log_run("tracking", CONFIG_PATH.stem, "01_extract_features.py",
        output_path=str(OUT_CSV), detail="detailed",
        data_path=str(Path(cfg['paths']['raw_dir'])))
