"""
Per-timepoint 3D vector from the substrate-contact footprint of the membrane
channel to the center of the ICM (oct4) channel, for an implanting embryo.

Biological question: as the embryo implants, where does the ICM sit relative
to where the embryo is actually touching the dish? The membrane channel's
"center" is deliberately restricted to a thin slab at the dish-contact end of
the stack (BOTTOM_SLAB_UM) rather than the whole embryo, so it represents the
substrate-contact footprint, not the whole membrane centroid. The ICM
channel's center uses the full 3D volume, since the ICM is a compact 3D
cluster of cells wherever it currently sits.

Channel mapping for this dataset (confirmed 2026-08-07, not from metadata --
the raw json's "camera" field says "long" for both channels here, so it does
not disambiguate short/long-camera the way it did for other datasets):
  channel 0 ("far red")  = membrane
  channel 1 ("ch1")      = ICM marker (oct4)

Z orientation: this dataset's stacks are obj_bottom with Z=max at the dish
surface/coverslip (confirmed 2026-08-07) -- i.e. the LAST few Z slices are
the substrate-contact end, not the first.

Method per channel/timepoint (revised 2026-08-07 after two failed attempts,
see below): percentile-intensity threshold -> connected-component -> pick
whichever component is CLOSEST (in XY) to where that same structure was in
the PREVIOUS timepoint (frame-to-frame nearest-centroid tracking, seeded by
a manually-confirmed t=0 position) -> intensity-weighted center of mass
within that chosen component, weighted by (intensity - threshold) so voxels
near background contribute ~0. No dedicated segmentation model -- meant as a
first-pass validation on one embryo before scaling to the rest of the
config.

Why not simpler alternatives (both tried and failed on real data here):
  - Otsu threshold: too permissive for this data -- on stack_9-fgf__embryo2
    t=0, Otsu (thresh=231) left the ENTIRE visible field (target + neighbor
    embryos + background haze) as ONE 13-million-voxel connected component,
    so there was no separation to choose between at all. A stricter
    percentile threshold (e.g. 99th pct) does break the field into
    distinguishable pieces.
  - Largest-component / closest-to-frame-center: both assume the target
    embryo is the biggest blob, or stays centered in its crop. Neither holds
    -- embryos drift substantially over a 10.5h timelapse, and a brighter
    y neighbor embryo's bleed-through can easily be bigger than the real
    target signal (confirmed on stack_9-fgf__embryo2, which also turned out
    to have NO real ICM signal at all -- membrane-only -- explaining why
    every method kept landing on the neighbor's blob instead).
Not every embryo in this config has real signal in the ICM channel (several
are membrane-only per the yaml notes) -- those must be skipped explicitly,
not auto-detected, since a percentile threshold always finds *something*
whether or not it's real signal.

Edit the CONFIG section below, then run (VS Code Run button -- no CLI args).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, label as cc_label

from src.conversion import get_config_value, load_yaml_config

# ============================== EDIT THESE ==============================
CONFIG_PATH = PROJECT_ROOT / "configs" / "other live images" / "260721_e45c_fgf_oct4_snap_2.yaml"
STACK_FOLDER = "stack_10_embryo1"   # subfolder under live_timecourse.output_dir
                    # (confirmed double-positive validation embryo, 2026-08-07;
                    # stack_9-fgf__embryo2, tried first, turned out membrane-only)

CHANNEL_MEMBRANE = 0   # "far red"
CHANNEL_ICM = 1        # "ch1" (oct4)

BOTTOM_SLAB_UM = 10.0  # thickness of the dish-contact slab for the membrane centroid
DISH_SURFACE_AT_ZMAX = True  # True: last Z slices are the dish surface; False: first Z slices are

PERCENTILE_THRESHOLD = 99.0  # per-timepoint intensity percentile used to threshold
                    # each channel before connected-component labeling (Otsu was
                    # too permissive here -- see module docstring)
MIN_COMPONENT_VOXEL_FRACTION = 0.02  # ignore connected components smaller than this
                    # fraction of total foreground voxels (speckle noise) when
                    # deciding which component to track

# t=0 seed (y, x) pixel coordinates for the ICM centroid, confirmed by eye
# against analysis/icm_membrane_vector/<STACK_FOLDER>/seed_pick_t0.png.
# Required -- there's no previous timepoint to anchor tracking at t=0.
ICM_SEED_YX_T0 = (800.0, 650.0)
MEMBRANE_SEED_YX_T0 = None  # None = default to the bottom-slab frame center at t=0

OUT_DIR = PROJECT_ROOT.parent  # overridden below from config's paths.output_dir
QC_N_TIMEPOINTS = 5   # number of evenly-spaced QC overlay panels to save
# ==========================================================================


def load_channel_volumes(tiff_path):
    arr = tifffile.imread(str(tiff_path))  # (C, Z, Y, X)
    return arr[CHANNEL_MEMBRANE].astype(np.float32), arr[CHANNEL_ICM].astype(np.float32)


def pick_component_mask(volume, ref_centroid_yx):
    """Percentile-threshold a (Z,Y,X) volume, connected-component it, and pick
    whichever component's centroid is closest (in XY) to ref_centroid_yx --
    rather than assuming it's the largest blob or stays centered in frame
    (both failed in practice, see module docstring). Returns
    (component_mask, thresh), or (None, thresh) if nothing is above
    threshold. Shared by tracked_intensity_weighted_centroid (ref_centroid_yx
    = previous timepoint's accepted position) and the napari footprint
    viewer (ref_centroid_yx = this timepoint's already-recorded position, to
    recover which component produced it -- see
    view_icm_membrane_vector_napari.py)."""
    thresh = np.percentile(volume, PERCENTILE_THRESHOLD)
    mask = volume > thresh
    if not mask.any():
        return None, thresh

    labeled, _ = cc_label(mask, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # background label
    candidate_labels = np.nonzero(sizes >= sizes.sum() * MIN_COMPONENT_VOXEL_FRACTION)[0]
    if len(candidate_labels) == 0:
        candidate_labels = np.array([int(sizes.argmax())])

    py, px = ref_centroid_yx
    candidate_centroids = center_of_mass(mask, labeled, index=candidate_labels)
    dists_xy = [np.hypot(c[1] - py, c[2] - px) for c in candidate_centroids]
    chosen_label = int(candidate_labels[int(np.argmin(dists_xy))])
    return labeled == chosen_label, thresh


def tracked_intensity_weighted_centroid(volume, prev_centroid_yx, z_offset=0):
    """Returns the (z,y,x) voxel centroid of pick_component_mask's chosen
    component, weighted by (intensity - thresh) so near-background voxels
    near its edge barely pull the centroid. z_offset is added back onto the
    z coordinate (for sub-volume slabs). Returns None if the volume has no
    variation at all (constant array)."""
    component_mask, thresh = pick_component_mask(volume, prev_centroid_yx)
    if component_mask is None:
        return None

    weights = np.where(component_mask, volume - thresh, 0.0)
    weights[weights < 0] = 0.0
    if weights.sum() <= 0:
        return None
    cz, cy, cx = center_of_mass(weights)
    return np.array([cz + z_offset, cy, cx])


def membrane_bottom_centroid(membrane_vol, bottom_slab_voxels, prev_centroid_yx):
    z_total = membrane_vol.shape[0]
    if DISH_SURFACE_AT_ZMAX:
        z_start = max(z_total - bottom_slab_voxels, 0)
        slab = membrane_vol[z_start:]
        return tracked_intensity_weighted_centroid(slab, prev_centroid_yx, z_offset=z_start)
    else:
        z_end = min(bottom_slab_voxels, z_total)
        slab = membrane_vol[:z_end]
        return tracked_intensity_weighted_centroid(slab, prev_centroid_yx, z_offset=0)


def save_qc_panel(membrane_vol, icm_vol, mem_centroid_vox, icm_centroid_vox,
                   bottom_slab_voxels, out_path, title):
    z_total = membrane_vol.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), dpi=150)

    # XY MIP: membrane gray, ICM magenta, both centroids projected
    mem_mip = membrane_vol.max(axis=0)
    icm_mip = icm_vol.max(axis=0)
    ax = axes[0]
    p_lo, p_hi = np.percentile(mem_mip, (1, 99.5))
    ax.imshow(np.clip((mem_mip - p_lo) / (p_hi - p_lo + 1e-9), 0, 1), cmap="gray")
    p_lo, p_hi = np.percentile(icm_mip, (1, 99.5))
    icm_norm = np.clip((icm_mip - p_lo) / (p_hi - p_lo + 1e-9), 0, 1)
    ax.imshow(icm_norm, cmap="spring", alpha=icm_norm)
    if mem_centroid_vox is not None:
        ax.scatter(mem_centroid_vox[2], mem_centroid_vox[1], c="cyan", marker="o",
                   s=80, edgecolor="black", label="membrane (bottom slab)")
    if icm_centroid_vox is not None:
        ax.scatter(icm_centroid_vox[2], icm_centroid_vox[1], c="yellow", marker="*",
                   s=150, edgecolor="black", label="ICM")
    ax.set_title("XY (Z-MIP)")
    ax.legend(loc="lower right", fontsize=8)
    ax.axis("off")

    # YZ MIP: shows the Z separation relative to the dish surface
    mem_yz = membrane_vol.max(axis=2)  # (Z, Y): Z is already the row axis, Y the column axis
    icm_yz = icm_vol.max(axis=2)
    ax = axes[1]
    p_lo, p_hi = np.percentile(mem_yz, (1, 99.5))
    ax.imshow(np.clip((mem_yz - p_lo) / (p_hi - p_lo + 1e-9), 0, 1), cmap="gray", aspect="auto")
    p_lo, p_hi = np.percentile(icm_yz, (1, 99.5))
    icm_yz_norm = np.clip((icm_yz - p_lo) / (p_hi - p_lo + 1e-9), 0, 1)
    ax.imshow(icm_yz_norm, cmap="spring", alpha=icm_yz_norm, aspect="auto")
    if mem_centroid_vox is not None:
        ax.scatter(mem_centroid_vox[1], mem_centroid_vox[0], c="cyan", marker="o",
                   s=80, edgecolor="black")
    if icm_centroid_vox is not None:
        ax.scatter(icm_centroid_vox[1], icm_centroid_vox[0], c="yellow", marker="*", s=150, edgecolor="black")
    dish_z = z_total - 1 if DISH_SURFACE_AT_ZMAX else 0
    ax.axhline(dish_z, color="red", linestyle="--", linewidth=1, label="dish surface")
    if DISH_SURFACE_AT_ZMAX:
        ax.axhline(z_total - bottom_slab_voxels, color="orange", linestyle=":", linewidth=1, label="slab edge")
    else:
        ax.axhline(bottom_slab_voxels, color="orange", linestyle=":", linewidth=1, label="slab edge")
    ax.set_title("YZ (X-MIP)  -- red = dish surface")
    ax.legend(loc="upper right", fontsize=7)
    ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def process_embryo(stack_folder, icm_seed_yx_t0, output_dir, analysis_dir, voxel_zyx,
                    membrane_seed_yx_t0=None, qc_n_timepoints=QC_N_TIMEPOINTS):
    """Runs the full per-timepoint vector pipeline for one embryo folder and
    writes its CSV + QC panels + timecourse plot to
    analysis_dir/icm_membrane_vector/<stack_folder>/. Returns the CSV path.
    Shared by this script's standalone __main__ (one embryo, constants below)
    and batch_icm_membrane_vector.py (loops over many embryos, seeds/skips
    read from an overrides file rather than hardcoded here)."""
    out_dir = analysis_dir / "icm_membrane_vector" / stack_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    stack_path = output_dir / stack_folder
    tiff_files = sorted(stack_path.glob("t*.tif"))
    if not tiff_files:
        raise FileNotFoundError(f"No t*.tif files found in {stack_path}.")
    n_t = len(tiff_files)
    print(f"{stack_folder}: {n_t} timepoints, voxel_zyx_um={tuple(voxel_zyx)}")

    bottom_slab_voxels = max(int(round(BOTTOM_SLAB_UM / voxel_zyx[0])), 1)
    print(f"BOTTOM_SLAB_UM={BOTTOM_SLAB_UM} -> {bottom_slab_voxels} Z slices "
          f"({'at Z=max' if DISH_SURFACE_AT_ZMAX else 'at Z=0'})")

    qc_indices = set(np.linspace(0, n_t - 1, qc_n_timepoints, dtype=int)) if qc_n_timepoints > 0 else set()

    if icm_seed_yx_t0 is None:
        raise ValueError(
            f"icm_seed_yx_t0 must be set for {stack_folder!r} -- there's no previous timepoint "
            f"to anchor tracking at t=0. Check analysis/icm_membrane_vector/{stack_folder}/seed_pick_t0.png."
        )
    prev_icm_yx = icm_seed_yx_t0
    prev_mem_yx = None  # set from the first volume's shape below, once we know it

    rows = []
    for t, tiff_path in enumerate(tiff_files):
        membrane_vol, icm_vol = load_channel_volumes(tiff_path)

        if prev_mem_yx is None:
            prev_mem_yx = membrane_seed_yx_t0 or (
                (membrane_vol.shape[1] - 1) / 2.0, (membrane_vol.shape[2] - 1) / 2.0
            )

        mem_centroid_vox = membrane_bottom_centroid(membrane_vol, bottom_slab_voxels, prev_mem_yx)
        icm_centroid_vox = tracked_intensity_weighted_centroid(icm_vol, prev_icm_yx)

        row = {"timepoint": t, "file": tiff_path.name}
        if mem_centroid_vox is None or icm_centroid_vox is None:
            print(f"  t={t}: no signal in {'membrane slab' if mem_centroid_vox is None else 'ICM channel'}, skipping "
                  f"(tracking anchor NOT updated, carried forward from last valid timepoint)")
            row.update({k: np.nan for k in [
                "mem_z_um", "mem_y_um", "mem_x_um", "icm_z_um", "icm_y_um", "icm_x_um",
                "vec_z_um", "vec_y_um", "vec_x_um", "vec_magnitude_um", "vec_angle_from_dish_deg",
            ]})
        else:
            prev_mem_yx = (mem_centroid_vox[1], mem_centroid_vox[2])
            prev_icm_yx = (icm_centroid_vox[1], icm_centroid_vox[2])

            mem_um = mem_centroid_vox * voxel_zyx
            icm_um = icm_centroid_vox * voxel_zyx
            vec_um = icm_um - mem_um
            vec_magnitude = float(np.linalg.norm(vec_um))

            # Signed elevation angle of the vector above the dish plane: 0 deg = vector lies
            # flat along the dish surface, +90 deg = ICM directly away from the dish relative
            # to the membrane footprint, -90 deg = ICM directly toward/into the dish side.
            # "Away from the dish" is -Z when the dish is at Z=max, +Z when it's at Z=0.
            dish_away_sign = -1.0 if DISH_SURFACE_AT_ZMAX else 1.0
            z_away_from_dish = dish_away_sign * vec_um[0]
            angle_from_dish_deg = (
                float(np.degrees(np.arcsin(np.clip(z_away_from_dish / vec_magnitude, -1.0, 1.0))))
                if vec_magnitude > 1e-9 else np.nan
            )

            row.update({
                "mem_z_um": mem_um[0], "mem_y_um": mem_um[1], "mem_x_um": mem_um[2],
                "icm_z_um": icm_um[0], "icm_y_um": icm_um[1], "icm_x_um": icm_um[2],
                "vec_z_um": vec_um[0], "vec_y_um": vec_um[1], "vec_x_um": vec_um[2],
                "vec_magnitude_um": vec_magnitude,
                "vec_angle_from_dish_deg": angle_from_dish_deg,
            })
            print(f"  t={t}: vec_zyx_um=({vec_um[0]:.1f}, {vec_um[1]:.1f}, {vec_um[2]:.1f})  "
                  f"|vec|={vec_magnitude:.1f} um  angle_from_dish={angle_from_dish_deg:.1f} deg")

        rows.append(row)

        if t in qc_indices:
            save_qc_panel(
                membrane_vol, icm_vol, mem_centroid_vox, icm_centroid_vox,
                bottom_slab_voxels, out_dir / f"qc_t{t:04d}.png",
                f"{stack_folder}  t={t}",
            )

    table = pd.DataFrame(rows)
    csv_path = out_dir / f"{stack_folder}_icm_membrane_vector.csv"
    table.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    fig, axes = plt.subplots(3, 1, figsize=(8, 9.5), dpi=150, sharex=True)
    axes[0].plot(table["timepoint"], table["vec_magnitude_um"], marker="o", color="steelblue")
    axes[0].set_ylabel("|vector| (um)")
    axes[0].set_title(f"{stack_folder}: ICM center relative to membrane dish-contact footprint")

    axes[1].plot(table["timepoint"], table["vec_angle_from_dish_deg"], marker="o", color="darkorange")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].axhline(90, color="gray", linewidth=0.6, linestyle=":")
    axes[1].axhline(-90, color="gray", linewidth=0.6, linestyle=":")
    axes[1].set_ylim(-95, 95)
    axes[1].set_ylabel("angle from dish (deg)")
    axes[1].set_title("+90 = ICM directly away from dish, 0 = flat along dish, -90 = toward dish")

    for col, label, color in [("vec_z_um", "Z", "tab:red"), ("vec_y_um", "Y", "tab:green"), ("vec_x_um", "X", "tab:blue")]:
        axes[2].plot(table["timepoint"], table[col], marker=".", label=label, color=color)
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[2].set_ylabel("vector component (um)")
    axes[2].set_xlabel("timepoint")
    axes[2].legend()
    fig.tight_layout()
    plot_path = out_dir / f"{stack_folder}_vector_timecourse.png"
    fig.savefig(plot_path, facecolor="white")
    plt.close(fig)
    print(f"Saved: {plot_path}")

    print(f"\nDone with {stack_folder}. Check the QC overlay PNGs (qc_t####.png) to confirm the "
          "membrane centroid sits in the bottom slab and the ICM centroid "
          "looks right before trusting the vector numbers.")
    return csv_path


def main():
    config = load_yaml_config(CONFIG_PATH) if CONFIG_PATH.exists() else {}
    live_cfg = config.get("live_timecourse") or {}
    output_dir = Path(get_config_value(live_cfg, ["output_dir"]) or ".")
    voxel_zyx = get_config_value(config, ["microscopy", "voxel_size_zyx_um"]) or [1.0, 1.0, 1.0]
    voxel_zyx = np.array(voxel_zyx, dtype=np.float64)
    analysis_dir = Path(get_config_value(config, ["paths", "output_dir"]) or (PROJECT_ROOT / "analysis"))

    process_embryo(
        STACK_FOLDER, ICM_SEED_YX_T0, output_dir, analysis_dir, voxel_zyx,
        membrane_seed_yx_t0=MEMBRANE_SEED_YX_T0, qc_n_timepoints=QC_N_TIMEPOINTS,
    )


if __name__ == "__main__":
    main()
