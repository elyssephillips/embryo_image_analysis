"""
Interactive napari viewer for icm_membrane_vector_3d.py's output: both raw
channels for one embryo (lazy per-timepoint loading, same as
view_timepoint_channels.py) plus the membrane bottom-slab centroid, ICM
centroid, and the vector connecting them, as scrollable-through-time napari
Points/Vectors layers.

Run icm_membrane_vector_3d.py first to generate the CSV this reads.

Points/Vectors carry a leading T coordinate (matching the image's T axis),
so napari only renders the point/vector for the timepoint currently on the
slider -- standard convention in this codebase, see e.g.
pipelines/tracking/10_napari_flow_overlay.py.

To run: set CONFIG_PATH / STACK_FOLDER below, then click VS Code's "Run
Python File" button (or Ctrl+F5) -- no CLI args. Needs a real display
(X11/VNC) since it opens a napari window.
"""
import functools
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dask
import dask.array as da
import napari
import numpy as np
import pandas as pd
import tifffile

from src.conversion import get_config_value, load_yaml_config
from icm_membrane_vector_3d import (
    pick_component_mask, CHANNEL_MEMBRANE, CHANNEL_ICM,
    BOTTOM_SLAB_UM, DISH_SURFACE_AT_ZMAX,
)

# ============================== EDIT THESE ==============================
CONFIG_PATH = PROJECT_ROOT / "configs" / "other live images" / "260721_e45c_fgf_oct4_snap_2.yaml"
STACK_FOLDER = "stack_10_embryo1"   # must match icm_membrane_vector_3d.py's STACK_FOLDER
TIMEPOINT_CACHE_SIZE = 6
ARROW_LENGTH_MULTIPLIER = 1.0  # napari Vectors `length` scaling, in case the raw vector is hard to see

SHOW_SLAB_HIGHLIGHT = True   # translucent layer showing what counts as the dish-contact slab
SHOW_FOOTPRINT_MASKS = True  # thresholded connected-component footprint actually used for each
                    # centroid (hidden by default in the layer list) -- lets you judge how well
                    # the percentile threshold traces the real embryo edges
# ==========================================================================

DEFAULT_COLORS = {"dapi": "blue"}
FALLBACK_PALETTE = ["green", "magenta", "yellow", "cyan", "red"]


def assign_colors(channel_names: list) -> list:
    colors = []
    fallback_iter = iter(FALLBACK_PALETTE)
    for name in channel_names:
        color = DEFAULT_COLORS.get(name.lower())
        if color is None:
            color = next(fallback_iter, "gray")
        colors.append(color)
    return colors


def compute_contrast_limits(channel_volume, stride: int = 4) -> tuple:
    sample = np.asarray(channel_volume[::stride, ::stride, ::stride])
    lo, hi = np.percentile(sample, (1, 99.5))
    return float(lo), float(hi)


def main():
    config = load_yaml_config(CONFIG_PATH) if CONFIG_PATH.exists() else {}
    live_cfg = config.get("live_timecourse") or {}
    output_dir = Path(get_config_value(live_cfg, ["output_dir"]) or ".")
    channel_name_by_index = {
        c["index"]: c.get("name")
        for c in (get_config_value(config, ["microscopy", "channels"]) or [])
        if isinstance(c, dict) and "index" in c
    }
    voxel_size_zyx = get_config_value(config, ["microscopy", "voxel_size_zyx_um"]) or [1.0, 1.0, 1.0]

    analysis_dir = Path(get_config_value(config, ["paths", "output_dir"]) or (PROJECT_ROOT / "analysis"))
    csv_path = analysis_dir / "icm_membrane_vector" / STACK_FOLDER / f"{STACK_FOLDER}_icm_membrane_vector.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found -- run icm_membrane_vector_3d.py for STACK_FOLDER={STACK_FOLDER!r} first."
        )
    vec_df = pd.read_csv(csv_path)
    print(f"Loaded {len(vec_df)} timepoint rows from {csv_path}")

    path = output_dir / STACK_FOLDER
    tiff_files = sorted(path.glob("t*.tif"))
    if not tiff_files:
        raise FileNotFoundError(f"No t*.tif files found in {path}.")
    n_timepoints = len(tiff_files)
    print(f"Loading {n_timepoints} timepoint(s) from {path} (lazy, {TIMEPOINT_CACHE_SIZE} cached)...")

    with tifffile.TiffFile(str(tiff_files[0])) as tif:
        shape = tif.series[0].shape
        dtype = tif.series[0].dtype

    @functools.lru_cache(maxsize=TIMEPOINT_CACHE_SIZE)
    def load_timepoint(idx: int) -> np.ndarray:
        return tifffile.imread(str(tiff_files[idx]))

    delayed_timepoints = [
        da.from_delayed(dask.delayed(load_timepoint)(t), shape=shape, dtype=dtype)
        for t in range(n_timepoints)
    ]
    stack = da.stack(delayed_timepoints, axis=0)  # (T, C, Z, Y, X)
    n_channels = stack.shape[1]
    names = [channel_name_by_index.get(i) or f"ch{i}" for i in range(n_channels)]
    colors = assign_colors(names)

    first_volume = load_timepoint(0)
    contrast_limits = [compute_contrast_limits(first_volume[c]) for c in range(n_channels)]

    valid = vec_df.dropna(subset=["mem_z_um", "icm_z_um"]).copy()
    n_skipped = len(vec_df) - len(valid)
    if n_skipped:
        print(f"  {n_skipped} timepoint(s) had no signal in one channel -- skipped in points/vectors layers")

    mem_coords = valid[["timepoint", "mem_z_um", "mem_y_um", "mem_x_um"]].values.astype(float)
    icm_coords = valid[["timepoint", "icm_z_um", "icm_y_um", "icm_x_um"]].values.astype(float)

    starts = mem_coords.copy()
    directions = np.zeros_like(starts)
    directions[:, 1:] = valid[["vec_z_um", "vec_y_um", "vec_x_um"]].values.astype(float) * ARROW_LENGTH_MULTIPLIER
    vectors = np.stack([starts, directions], axis=1)  # (N, 2, 4)

    viewer = napari.Viewer(title=STACK_FOLDER)
    viewer.dims.axis_labels = ["t", "z", "y", "x"]
    viewer.add_image(
        stack,
        name=names,
        channel_axis=1,
        colormap=colors,
        contrast_limits=contrast_limits,
        blending="additive",
        scale=[1.0] + list(voxel_size_zyx),
    )
    viewer.add_points(
        mem_coords, name="membrane centroid (bottom slab)",
        face_color="cyan", size=6, opacity=0.9, border_width=0,
    )
    viewer.add_points(
        icm_coords, name="ICM centroid",
        face_color="yellow", size=6, opacity=0.9, border_width=0,
    )
    viewer.add_vectors(
        vectors, name="ICM vector (from membrane footprint)",
        edge_color="orange", edge_width=1.5, length=1, opacity=0.9,
    )

    if SHOW_FOOTPRINT_MASKS:
        vec_by_t = vec_df.set_index("timepoint")
        z_total = shape[1]
        bottom_slab_voxels = max(int(round(BOTTOM_SLAB_UM / voxel_size_zyx[0])), 1)

        @functools.lru_cache(maxsize=TIMEPOINT_CACHE_SIZE)
        def compute_footprint_masks(t):
            """Replays pick_component_mask against the centroid already
            recorded in the CSV for timepoint t, to recover which connected
            component produced it (see icm_membrane_vector_3d.py's
            pick_component_mask docstring)."""
            volume = load_timepoint(t)
            membrane_vol = volume[CHANNEL_MEMBRANE].astype(np.float32)
            icm_vol = volume[CHANNEL_ICM].astype(np.float32)
            row = vec_by_t.loc[t]

            mem_mask_full = np.zeros_like(membrane_vol, dtype=np.uint8)
            icm_mask_full = np.zeros_like(icm_vol, dtype=np.uint8)

            if not np.isnan(row["mem_y_um"]):
                ref_yx = (row["mem_y_um"] / voxel_size_zyx[1], row["mem_x_um"] / voxel_size_zyx[2])
                if DISH_SURFACE_AT_ZMAX:
                    z_start = max(z_total - bottom_slab_voxels, 0)
                    slab = membrane_vol[z_start:]
                else:
                    z_start = 0
                    slab = membrane_vol[:min(bottom_slab_voxels, z_total)]
                comp_mask, _ = pick_component_mask(slab, ref_yx)
                if comp_mask is not None:
                    mem_mask_full[z_start:z_start + comp_mask.shape[0]] = comp_mask

            if not np.isnan(row["icm_y_um"]):
                ref_yx = (row["icm_y_um"] / voxel_size_zyx[1], row["icm_x_um"] / voxel_size_zyx[2])
                comp_mask, _ = pick_component_mask(icm_vol, ref_yx)
                if comp_mask is not None:
                    icm_mask_full = comp_mask.astype(np.uint8)

            return mem_mask_full, icm_mask_full

        mask_shape = shape[1:]  # (Z, Y, X)
        mem_mask_stack = da.stack([
            da.from_delayed(dask.delayed(lambda t=t: compute_footprint_masks(t)[0])(),
                             shape=mask_shape, dtype=np.uint8)
            for t in range(n_timepoints)
        ], axis=0)
        icm_mask_stack = da.stack([
            da.from_delayed(dask.delayed(lambda t=t: compute_footprint_masks(t)[1])(),
                             shape=mask_shape, dtype=np.uint8)
            for t in range(n_timepoints)
        ], axis=0)
        viewer.add_labels(
            mem_mask_stack, name="membrane footprint (thresholded)",
            scale=[1.0] + list(voxel_size_zyx), opacity=0.5, visible=False,
        )
        viewer.add_labels(
            icm_mask_stack, name="ICM footprint (thresholded)",
            scale=[1.0] + list(voxel_size_zyx), opacity=0.5, visible=False,
        )
        print("  footprint masks added (hidden by default -- toggle on in the layer list)")

    if SHOW_SLAB_HIGHLIGHT:
        z_total = shape[1]
        bottom_slab_voxels = max(int(round(BOTTOM_SLAB_UM / voxel_size_zyx[0])), 1)
        slab_mask_zyx = np.zeros(shape[1:], dtype=np.uint8)  # (Z, Y, X)
        if DISH_SURFACE_AT_ZMAX:
            slab_mask_zyx[max(z_total - bottom_slab_voxels, 0):] = 1
        else:
            slab_mask_zyx[:min(bottom_slab_voxels, z_total)] = 1
        slab_mask = da.broadcast_to(slab_mask_zyx[np.newaxis], (n_timepoints,) + slab_mask_zyx.shape)
        viewer.add_image(
            slab_mask, name=f"bottom slab ({BOTTOM_SLAB_UM} um)",
            colormap="yellow", opacity=0.15, blending="additive",
            contrast_limits=(0, 1), scale=[1.0] + list(voxel_size_zyx),
        )
        print(f"  bottom slab highlight: {bottom_slab_voxels} Z slices "
              f"({'at Z=max' if DISH_SURFACE_AT_ZMAX else 'at Z=0'})")

    print("\nLayers loaded. Scrub the T slider to step through timepoints.")
    napari.run()


if __name__ == "__main__":
    main()
