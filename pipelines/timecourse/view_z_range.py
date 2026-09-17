"""Open a folder of per-timepoint TIFFs (written by convert_h5_timecourse_to_tiff.py
with per_timepoint=true) in napari, showing only a chosen Z range, with each
channel as its own colored layer overlaid additively.

Same napari-layering problem as view_timepoint_channels.py, but for cases
where you only care about a subset of Z slices (e.g. to skim through a
volume faster, or to avoid loading slices with nothing in them). Each file
is opened via tifffile.memmap, which memory-maps the TIFF instead of
decoding it, so slicing out Z_RANGE before touching the array means only
the requested planes are ever read from disk — the full volume is never
pulled into RAM.

To run: set CONFIG_PATH / STACK_FOLDER / Z_RANGE below, then click VS
Code's "Run Python File" button (or Ctrl+F5) — no terminal command needed.
Leave STACK_FOLDER as None to list the available stack/embryo folders in
output_dir and exit. Needs a real display (X11/VNC) since it opens a
napari window.
"""
import functools
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dask
import dask.array as da
import napari
import numpy as np
import tifffile

from src.conversion import get_config_value, load_yaml_config

# ============================== EDIT THESE ==============================
CONFIG_PATH = PROJECT_ROOT / "configs" / "other live images" / "260804_c_meki_h2b_snap_2.yaml"  # path to the dataset's config.yaml
STACK_FOLDER = "stack_3-ctrl_4"     # folder name under live_timecourse.output_dir
                    # (or a full path to a t*.tif folder anywhere else).
                    # Leave as None to list available folders and exit.
Z_RANGE = (55, 58)  # (start, stop) Z indices to load, stop-exclusive.
                    # Set to None to load the full Z range.
TIMEPOINT_CACHE_SIZE = 6   # number of Z-cropped timepoint volumes kept in RAM at once.
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
    """Fast percentile estimate from a strided (Z, Y, X) sample of one timepoint."""
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

    if STACK_FOLDER is None:
        folders = sorted(p for p in output_dir.iterdir() if p.is_dir()) if output_dir.is_dir() else []
        print(f"STACK_FOLDER is not set. Available folders in {output_dir}:\n")
        for f in folders:
            print(f"  {f.name}")
        print("\nSet STACK_FOLDER to one of the names above (or a full path) and re-run.")
        return

    path = Path(STACK_FOLDER)
    if not path.is_absolute() and not path.exists():
        path = output_dir / STACK_FOLDER
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a folder of per-timepoint TIFFs.")

    tiff_files = sorted(path.glob("t*.tif"))
    if not tiff_files:
        raise FileNotFoundError(f"No t*.tif files found in {path}.")
    n_timepoints = len(tiff_files)

    with tifffile.TiffFile(str(tiff_files[0])) as tif:
        full_shape = tif.series[0].shape  # header-only, no pixel data read
        dtype = tif.series[0].dtype
    if len(full_shape) != 4:
        raise ValueError(f"Expected (C, Z, Y, X) per timepoint, got shape {full_shape} for {tiff_files[0]}.")
    n_channels, n_z, height, width = full_shape

    z_start, z_stop = Z_RANGE if Z_RANGE is not None else (0, n_z)
    z_start = max(0, z_start)
    z_stop = min(n_z, z_stop)
    if z_start >= z_stop:
        raise ValueError(f"Z_RANGE {Z_RANGE} is empty for a stack with {n_z} Z slices.")
    shape = (n_channels, z_stop - z_start, height, width)

    print(f"Loading {n_timepoints} timepoint(s) from {path}, Z slices [{z_start}:{z_stop}] of {n_z} "
          f"(memory-mapped, {TIMEPOINT_CACHE_SIZE} cached at a time)...")

    @functools.lru_cache(maxsize=TIMEPOINT_CACHE_SIZE)
    def load_timepoint(idx: int) -> np.ndarray:
        volume = tifffile.memmap(str(tiff_files[idx]))
        return np.asarray(volume[:, z_start:z_stop])

    delayed_timepoints = [
        da.from_delayed(dask.delayed(load_timepoint)(t), shape=shape, dtype=dtype)
        for t in range(n_timepoints)
    ]
    stack = da.stack(delayed_timepoints, axis=0)  # (T, C, Z, Y, X)
    print(f"  shape (T, C, Z, Y, X) = {stack.shape}")

    names = [channel_name_by_index.get(i) or f"ch{i}" for i in range(n_channels)]
    colors = assign_colors(names)
    print("  Computing contrast limits from timepoint 0 (strided sample; avoids "
          "eagerly loading other timepoints just for this)...")
    first_volume = load_timepoint(0)  # (C, Z, Y, X)
    contrast_limits = [compute_contrast_limits(first_volume[c]) for c in range(n_channels)]
    for name, color, cl in zip(names, colors, contrast_limits):
        print(f"    {name}: colormap={color}, contrast_limits={cl}")

    viewer = napari.Viewer(title=f"{path.name} [z {z_start}:{z_stop}]")
    viewer.add_image(
        stack,
        name=names,
        channel_axis=1,
        colormap=colors,
        contrast_limits=contrast_limits,
        blending="additive",
        scale=[1.0] + list(voxel_size_zyx),  # leading axis is T (unscaled), then Z, Y, X
    )
    napari.run()


if __name__ == "__main__":
    main()
