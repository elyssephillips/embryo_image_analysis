#!/home/elysse/miniforge3/envs/blastospim-tf/bin/python
"""
2D ERK C:N ratio from a MIP of a chosen Z range.

Workflow per condition:
  1. Max-project the ERK volume over the chosen Z range
  2. Segment nuclei in 2D with StarDist2D "2D_versatile_fluo"
  3. Compute C:N ratio (eroded nuclear core vs. cytoplasmic ring)
  4. Save coloured PNG overlay

The colour scale is fixed to the reference condition so conditions are comparable.

Edit the CONFIG section below, then run.
"""

import sys
from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from csbdeep.utils import normalize
from stardist.models import StarDist2D

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

# One entry per condition: "name": (erk_path, nuc_path)
# erk_path  — used for the C:N ratio measurement
# nuc_path  — used for 2D segmentation (separate nuclear/H2B image)
#             set to None to segment on the ERK channel instead
CONDITIONS = {
    "control": (Path("/mnt/md1/elysse/for r21/erk/control_erk2.tif"),
                Path("/mnt/md1/elysse/for r21/nuclei/control_nuc2.tif")),
    "meki":    (Path("/mnt/md1/elysse/for r21/erk/meki_erk.tif"),
                Path("/mnt/md1/elysse/for r21/nuclei/meki_nuc.tif")),
    "fgf":     (Path("/mnt/md1/elysse/for r21/erk/fgf_erk.tif"),
                Path("/mnt/md1/elysse/for r21/nuclei/fgf_nuc.tif")),
    "early":   (Path("/mnt/md1/elysse/for r21/erk/early_erk.tif"), None),
}

REFERENCE_CONDITION = "control"   # colour scale is derived from this condition
OUT_DIR = Path("/mnt/md1/elysse/for r21/cn_2d_withearly")

# Z range to MIP: (start_fraction, end_fraction) of the total Z stack.
#   Middle third:  (0.33, 0.67)
#   Bottom half:   (0.00, 0.50)
#   Top half:      (0.50, 1.00)
Z_RANGE = (0.2, 0.7)

ERK_CHANNEL = 0   # channel index if ERK TIFF is multi-channel; 0 for single-channel
NUC_CHANNEL = 0   # channel index if nuclear TIFF is multi-channel; 0 for single-channel

VOXEL_YX_UM  = 0.208   # y/x pixel size in µm

RING_RADIUS_UM = 1.0   # outer edge of cytoplasmic ring (µm)
RING_INNER_UM  = 0.0   # inner edge — set > 0 to skip voxels right at nucleus surface
NUC_EROSION_UM = 1.0   # erode nucleus inward before sampling nuclear intensity

BG_MULTIPLIER     = 3.0   # ring voxels with ERK < bg_mean * this are excluded
MIN_NUC_BG_FACTOR = 1.5   # nuclei with mean ERK < bg_mean * this are excluded as artifacts
EMBRYO_FILL_UM    = 3.0   # fill intercellular gaps up to this size to define embryo interior

OUTLIER_PERCENTILE_LO = 0.0    # exclude nuclei below this percentile
OUTLIER_PERCENTILE_HI = 90.0   # exclude nuclei above this percentile (90 = drop top 10%)

MIN_NUC_AREA_UM2 = 50.0   # nuclei smaller than this (µm²) are removed as fragments — tune me
NMS_THRESH = 0.5          # StarDist NMS overlap threshold (default ~0.4); higher = keep more overlapping detections

# ═════════════════════════════════════════════════════════════════════════════


def load_channel(path, channel):
    raw = tifffile.imread(path).astype(np.float32)
    if raw.ndim == 4:
        return raw[channel]
    if raw.ndim == 3:
        return raw
    sys.exit(f"Unexpected shape {raw.shape} in {path}")


def z_mip(vol, z_range):
    nz = vol.shape[0]
    z0 = int(np.round(z_range[0] * nz))
    z1 = int(np.round(z_range[1] * nz))
    z0, z1 = max(0, z0), min(nz, z1)
    print(f"  MIP over Z {z0}–{z1} of {nz}  ({z_range[0]:.0%}–{z_range[1]:.0%})")
    return vol[z0:z1].max(axis=0)


def segment_2d(mip, model, min_area_um2, yx_um):
    img_norm = normalize(mip, 1, 99.8)
    labels, _ = model.predict_instances(img_norm, nms_thresh=NMS_THRESH)
    labels = labels.astype(np.int32)

    min_area_px = min_area_um2 / (yx_um ** 2)
    ids, counts = np.unique(labels, return_counts=True)
    small = {lid for lid, n in zip(ids, counts) if lid != 0 and n < min_area_px}
    if small:
        labels[np.isin(labels, list(small))] = 0
        print(f"  Removed {len(small)} fragments < {min_area_um2} µm²")
    return labels


def compute_cn_ratio_2d(erk_mip, labels, yx_um, ring_radius_um, ring_inner_um,
                         nuc_erosion_um, bg_multiplier, min_nuc_bg_factor, embryo_fill_um):
    sampling = (yx_um, yx_um)
    background = (labels == 0)

    bg_level      = float(erk_mip[background].mean())
    threshold     = bg_level * bg_multiplier
    min_nuc_level = bg_level * min_nuc_bg_factor
    print(f"  bg_mean={bg_level:.1f}  ring_thresh={threshold:.1f}  nuc_min={min_nuc_level:.1f}")

    dist_from_bg, nearest_idx = distance_transform_edt(
        background, sampling=sampling, return_indices=True
    )
    voronoi = labels[tuple(nearest_idx)]

    # Embryo interior: dilate nuclei, fill holes in 2D
    nuc_dilated  = (~background) | (dist_from_bg <= embryo_fill_um)
    embryo_mask  = binary_fill_holes(nuc_dilated)

    nuc_interior_dist = distance_transform_edt(~background, sampling=sampling)

    in_cell   = erk_mip >= threshold
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
        nuc_vox  = erk_mip[(labels == lid) & nuc_core]
        ring_vox = erk_mip[ring_mask & (voronoi == lid)]

        if len(nuc_vox) == 0:
            nuc_vox = erk_mip[labels == lid]

        nuc_mean = float(nuc_vox.mean())
        if nuc_mean < min_nuc_level or len(ring_vox) == 0:
            ratios[int(lid)] = np.nan
            n_excluded += 1
        else:
            ratios[int(lid)] = float(ring_vox.mean()) / nuc_mean

    print(f"  {n_excluded}/{len(ratios)} nuclei excluded")
    return ratios


def clip_outliers(ratios, lo_pct, hi_pct):
    valid = [v for v in ratios.values() if not np.isnan(v)]
    if not valid:
        return ratios
    lo = float(np.percentile(valid, lo_pct))
    hi = float(np.percentile(valid, hi_pct))
    for lid, val in list(ratios.items()):
        if not np.isnan(val) and (val < lo or val > hi):
            ratios[lid] = np.nan
    n_clipped = sum(1 for v in valid if v < lo or v > hi)
    print(f"  Outlier clip: {n_clipped} removed  (kept {lo:.3f}–{hi:.3f})")
    return ratios


def save_overlay(erk_mip, labels, ratios, vmin, vmax, out_path, name):
    cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Build RGBA label image
    ratio_img = np.full(labels.shape, np.nan, dtype=np.float32)
    for lid, val in ratios.items():
        if not np.isnan(val):
            ratio_img[labels == lid] = val

    rgba = cmap(norm(ratio_img))
    rgba[np.isnan(ratio_img), 3] = 0.0   # transparent for excluded nuclei

    # ERK background (inverted: dark signal on white)
    p_lo, p_hi = np.percentile(erk_mip, (1, 99))
    erk_norm = np.clip((erk_mip - p_lo) / (p_hi - p_lo + 1e-9), 0, 1)

    valid = [v for v in ratios.values() if not np.isnan(v)]
    title = f"{name}  |  n={len(valid)} nuclei  |  C/N median={np.median(valid):.2f}"

    def _add_colorbar(fig, ax):
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03, shrink=0.6)
        cbar.set_label("ERK C/N ratio", fontsize=13, labelpad=8)
        cbar.ax.tick_params(labelsize=11)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── ERK overlay on white background ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.imshow(erk_norm, cmap="gray_r")   # inverted: bright signal → dark
    ax.imshow(rgba, alpha=0.8)
    _add_colorbar(fig, ax)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # ── segments only on white background ────────────────────────────────────
    labels_path = out_path.with_name(out_path.stem + "_labels.png")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    # composite RGBA over white
    rgb_white = np.ones((*labels.shape, 3), dtype=np.float32)
    alpha = rgba[..., 3:4]
    rgb_white = rgba[..., :3] * alpha + rgb_white * (1 - alpha)
    ax.imshow(rgb_white)
    _add_colorbar(fig, ax)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.savefig(labels_path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {labels_path}")


def save_channel_png(img, out_path):
    p_lo, p_hi = np.percentile(img, (1, 99))
    img_norm = np.clip((img - p_lo) / (p_hi - p_lo + 1e-9), 0, 1)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    fig.patch.set_facecolor("white")
    ax.imshow(img_norm, cmap="gray_r")
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── load StarDist2D model ──────────────────────────────────────────────────────
print("Loading StarDist2D model...")
model = StarDist2D.from_pretrained("2D_versatile_fluo")

# ── process each condition ─────────────────────────────────────────────────────
all_ratios = {}
all_erks   = {}
all_labels = {}

for name, (erk_path, nuc_path) in CONDITIONS.items():
    print(f"\n── {name} ──")
    erk_vol = load_channel(erk_path, ERK_CHANNEL)
    nuc_vol = load_channel(nuc_path, NUC_CHANNEL) if nuc_path is not None else erk_vol

    erk_mip = z_mip(erk_vol, Z_RANGE)
    nuc_mip = z_mip(nuc_vol, Z_RANGE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(OUT_DIR / f"{name}_erk_mip.tif"), erk_mip)
    save_channel_png(erk_mip, OUT_DIR / f"{name}_erk_mip.png")
    if nuc_path is not None:
        tifffile.imwrite(str(OUT_DIR / f"{name}_nuc_mip.tif"), nuc_mip)
        save_channel_png(nuc_mip, OUT_DIR / f"{name}_nuc_mip.png")
        print(f"  Saved MIPs: {name}_erk_mip.tif/png, {name}_nuc_mip.tif/png")
    else:
        print(f"  Saved MIPs: {name}_erk_mip.tif/png  (no nuclear channel — segmenting on ERK)")

    print("  Segmenting...")
    labels = segment_2d(nuc_mip, model, MIN_NUC_AREA_UM2, VOXEL_YX_UM)
    print(f"  {int((np.unique(labels) != 0).sum())} nuclei detected")

    ratios = compute_cn_ratio_2d(
        erk_mip, labels, VOXEL_YX_UM,
        RING_RADIUS_UM, RING_INNER_UM, NUC_EROSION_UM,
        BG_MULTIPLIER, MIN_NUC_BG_FACTOR, EMBRYO_FILL_UM,
    )

    valid = [v for v in ratios.values() if not np.isnan(v)]
    if valid:
        print(f"  C/N  median={np.median(valid):.3f}  min={np.min(valid):.3f}  max={np.max(valid):.3f}")

    ratios = clip_outliers(ratios, OUTLIER_PERCENTILE_LO, OUTLIER_PERCENTILE_HI)

    all_ratios[name] = ratios
    all_erks[name]   = erk_mip
    all_labels[name] = labels

# ── colour scale from reference condition ──────────────────────────────────────
ref_valid = [v for v in all_ratios[REFERENCE_CONDITION].values() if not np.isnan(v)]
vmin = float(np.percentile(ref_valid, 2))
vmax = float(np.percentile(ref_valid, 98))
print(f"\nColour scale from '{REFERENCE_CONDITION}': {vmin:.3f} – {vmax:.3f}")

# ── save overlays ──────────────────────────────────────────────────────────────
for name in CONDITIONS:
    save_overlay(
        all_erks[name], all_labels[name], all_ratios[name],
        vmin, vmax,
        OUT_DIR / f"{name}_cn_2d.png",
        name,
    )

print("\nDone.")
