#!/home/elysse/miniforge3/envs/napari_env/bin/python
"""
Compute ERK cytoplasm-to-nucleus (C/N) ratio for a single timepoint and
display in napari: ERK channel + nuclei coloured by C/N ratio.

Edit the paths and settings in the CONFIG section below, then run.
"""

import sys
from pathlib import Path

import numpy as np
import tifffile
import napari
from napari.utils.colormaps import DirectLabelColormap
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_fill_holes

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these before running
# ═════════════════════════════════════════════════════════════════════════════

# One entry per condition: "name": (erk_path, seg_path)
CONDITIONS = {
    "control": (Path("/mnt/md1/elysse/for r21/erk/control_erk.tif"),
                Path("/mnt/md1/elysse/for r21/segmentations/control_seg.tif")),
    "meki":    (Path("/mnt/md1/elysse/for r21/erk/meki_erk.tif"),
                Path("/mnt/md1/elysse/for r21/segmentations/meki_seg.tif")),
    "fgf":     (Path("/mnt/md1/elysse/for r21/erk/fgf_erk.tif"),
                Path("/mnt/md1/elysse/for r21/segmentations/fgf_seg.tif")),
}

# Which condition's ratio distribution sets the colour scale for all others.
REFERENCE_CONDITION = "control"

ERK_CHANNEL = 0  # channel index if ERK TIFF is CZYX; 0 for single-channel ZYX

VOXEL_ZYX         = (2.0, 0.208, 0.208)  # z, y, x in µm
RING_RADIUS_UM    = 1.0   # outer edge of cytoplasmic ring from nucleus surface (µm)
RING_INNER_UM     = 0.0   # inner edge — set > 0 to skip voxels right at the surface
NUC_EROSION_UM    = 1.0   # erode nucleus inward to avoid edge contamination
BG_MULTIPLIER     = 3.0   # ring voxels with ERK < bg_mean * this are excluded
MIN_NUC_BG_FACTOR = 1.5   # nuclei with mean ERK < bg_mean * this are excluded as artifacts
EMBRYO_FILL_UM    = 3.0   # gaps smaller than this are filled to define embryo interior
OUTLIER_PERCENTILE_LO = 0.0   # exclude nuclei below this percentile (0 = keep all)
OUTLIER_PERCENTILE_HI = 90.0  # exclude nuclei above this percentile (90 = drop top 10%)

# ═════════════════════════════════════════════════════════════════════════════


def load_erk(path, channel):
    raw = tifffile.imread(path).astype(np.float32)
    if raw.ndim == 4:
        return raw[channel]
    if raw.ndim == 3:
        return raw
    sys.exit(f"Unexpected shape {raw.shape} in {path}; expected ZYX or CZYX.")


def compute_cn_ratio(erk, labels, voxel_zyx, ring_radius_um, ring_inner_um,
                     nuc_erosion_um, bg_multiplier, min_nuc_bg_factor, embryo_fill_um):
    background = (labels == 0)

    bg_level      = float(erk[background].mean())
    threshold     = bg_level * bg_multiplier
    min_nuc_level = bg_level * min_nuc_bg_factor
    print(f"  bg_mean={bg_level:.1f}  ring_threshold={threshold:.1f}  nuc_min={min_nuc_level:.1f}")

    dist_from_bg, nearest_idx = distance_transform_edt(
        background, sampling=voxel_zyx, return_indices=True
    )
    voronoi = labels[tuple(nearest_idx)]

    nuc_dilated  = (~background) | (dist_from_bg <= embryo_fill_um)
    embryo_mask  = np.stack([binary_fill_holes(nuc_dilated[z]) for z in range(nuc_dilated.shape[0])])

    nuc_interior_dist = distance_transform_edt(~background, sampling=voxel_zyx)

    in_cell   = erk >= threshold
    ring_mask = (background
                 & (dist_from_bg <= ring_radius_um)
                 & (dist_from_bg >  ring_inner_um)
                 & in_cell
                 & embryo_mask)
    nuc_core  = nuc_interior_dist >= nuc_erosion_um

    ratios = {}
    n_excluded = 0
    for lid in np.unique(labels):
        if lid == 0:
            continue
        nuc_vox  = erk[(labels == lid) & nuc_core]
        ring_vox = erk[ring_mask & (voronoi == lid)]

        if len(nuc_vox) == 0:
            nuc_vox = erk[labels == lid]

        nuc_mean = float(nuc_vox.mean())
        if nuc_mean < min_nuc_level or len(ring_vox) == 0:
            ratios[int(lid)] = np.nan
            n_excluded += 1
        else:
            ratios[int(lid)] = float(ring_vox.mean()) / nuc_mean

    print(f"  {n_excluded} nuclei excluded (artifact or no ring)")
    return ratios


def clip_outliers(ratios, lo_pct, hi_pct):
    valid = [v for v in ratios.values() if not np.isnan(v)]
    if not valid:
        return ratios
    lo = float(np.percentile(valid, lo_pct))
    hi = float(np.percentile(valid, hi_pct))
    n_clipped = 0
    for lid, val in ratios.items():
        if not np.isnan(val) and (val < lo or val > hi):
            ratios[lid] = np.nan
            n_clipped += 1
    print(f"  Outlier clip [{lo_pct}–{hi_pct}th pct]: {n_clipped} greyed out  ({lo:.3f}–{hi:.3f})")
    return ratios


def make_colormap(ratios, vmin, vmax, cmap_name="coolwarm"):
    cmap = plt.get_cmap(cmap_name)
    color_dict = {0: np.array([0, 0, 0, 0], dtype=np.float64)}
    for lid, val in ratios.items():
        if np.isnan(val):
            color_dict[lid] = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            t = np.clip((val - vmin) / (vmax - vmin + 1e-9), 0, 1)
            color_dict[lid] = np.array(cmap(t))
    return DirectLabelColormap(color_dict=color_dict)


# ── compute ratios for all conditions ─────────────────────────────────────────
all_ratios = {}
all_labels = {}
all_erk    = {}

for name, (erk_path, seg_path) in CONDITIONS.items():
    print(f"\n── {name} ──")
    erk    = load_erk(erk_path, ERK_CHANNEL)
    labels = tifffile.imread(seg_path)

    if erk.shape != labels.shape:
        sys.exit(f"Shape mismatch for {name}: ERK {erk.shape} vs labels {labels.shape}.")

    print(f"  {int((np.unique(labels) != 0).sum())} nuclei")
    ratios = compute_cn_ratio(erk, labels, VOXEL_ZYX, RING_RADIUS_UM, RING_INNER_UM,
                              NUC_EROSION_UM, BG_MULTIPLIER, MIN_NUC_BG_FACTOR, EMBRYO_FILL_UM)

    valid = [v for v in ratios.values() if not np.isnan(v)]
    if valid:
        print(f"  C/N  median={np.median(valid):.3f}  min={np.min(valid):.3f}  max={np.max(valid):.3f}")

    ratios = clip_outliers(ratios, OUTLIER_PERCENTILE_LO, OUTLIER_PERCENTILE_HI)

    all_ratios[name] = ratios
    all_labels[name] = labels
    all_erk[name]    = erk

# ── derive colour scale from reference condition ───────────────────────────────
ref_valid = [v for v in all_ratios[REFERENCE_CONDITION].values() if not np.isnan(v)]
vmin = float(np.percentile(ref_valid, 2))
vmax = float(np.percentile(ref_valid, 98))
print(f"\nColour scale from '{REFERENCE_CONDITION}': {vmin:.3f} – {vmax:.3f}")

# ── open in napari ────────────────────────────────────────────────────────────
viewer = napari.Viewer(title="ERK C/N ratio")

for name in CONDITIONS:
    viewer.add_image(
        all_erk[name],
        name=f"ERK  {name}",
        scale=VOXEL_ZYX,
        colormap="gray",
        blending="additive",
        visible=False,
    )
    cmap = make_colormap(all_ratios[name], vmin, vmax)
    viewer.add_labels(
        all_labels[name],
        name=f"C/N  {name}  [{vmin:.2f}–{vmax:.2f}]",
        scale=VOXEL_ZYX,
        colormap=cmap,
        opacity=0.8,
        visible=False,
    )

# Make reference condition visible by default
viewer.layers[f"ERK  {REFERENCE_CONDITION}"].visible   = True
viewer.layers[f"C/N  {REFERENCE_CONDITION}  [{vmin:.2f}–{vmax:.2f}]"].visible = True

print("\nnapari open — toggle layers to compare conditions. Close window to exit.")
napari.run()
