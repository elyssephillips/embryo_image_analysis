"""Open a folder of per-timepoint TIFFs (written by convert_h5_timecourse_to_tiff.py
with per_timepoint=true) in napari with each channel as its own layer,
colored and named from the dataset's config, overlaid additively.

Needed because napari's builtin folder-of-tiffs reader stacks the files into
one generic (T, C, Z, Y, X) layer without knowing axis 1 is "channel" —
dropping the folder straight into napari gives a single layer with plain
number sliders, not two overlaid colored channels. Napari's own "Split
Stack" action only splits axis 0 (timepoints here), so it doesn't help
either without manually reordering axes first.

Each timepoint's file is the whole 3D volume for both channels (no way
around reading it in full for a real 3D view), so this loads one timepoint
at a time — lazily, via dask, one whole-volume chunk per timepoint — and
explicitly caches the last TIMEPOINT_CACHE_SIZE decoded volumes in RAM
(functools.lru_cache) so revisiting a timepoint, or scrubbing Z within one
already-loaded timepoint, is instant instead of re-reading from disk. Plain
dask does not cache computed chunks between separate .compute() calls on its
own, so this cache is explicit rather than assumed.

To run: set CONFIG_PATH / STACK_FOLDER below, then click VS Code's "Run
Python File" button (or Ctrl+F5) — no terminal command needed. Leave
STACK_FOLDER as None to list the available stack/embryo folders in
output_dir and exit. Needs a real display (X11/VNC) since it opens a napari
window.
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
CONFIG_PATH = PROJECT_ROOT / "configs" / "other live images" / "260804_c_meki_h2b_snap_2.yaml"  # path to the dataset's config.yaml (or a full path to any other config.yaml)
STACK_FOLDER = "stack_7-meki_4"     # folder name under live_timecourse.output_dir
                    # (or a full path to a t*.tif folder anywhere else).
                    # Leave as None to list available folders and exit.
TIMEPOINT_CACHE_SIZE = 6   # number of full timepoint volumes kept in RAM at once
                    # (each is ~1-4GB depending on the stack's crop size — pick
                    # this based on how much RAM you're willing to spend).
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
    print(f"Loading {n_timepoints} timepoint(s) from {path} "
          f"(lazy, one whole-volume chunk per timepoint, {TIMEPOINT_CACHE_SIZE} cached at a time)...")

    with tifffile.TiffFile(str(tiff_files[0])) as tif:
        shape = tif.series[0].shape  # header-only, no pixel data read
        dtype = tif.series[0].dtype

    @functools.lru_cache(maxsize=TIMEPOINT_CACHE_SIZE)
    def load_timepoint(idx: int) -> np.ndarray:
        return tifffile.imread(str(tiff_files[idx]))

    delayed_timepoints = [
        da.from_delayed(dask.delayed(load_timepoint)(t), shape=shape, dtype=dtype)
        for t in range(n_timepoints)
    ]
    stack = da.stack(delayed_timepoints, axis=0)  # (T, C, Z, Y, X), or (T, Z, Y, X) if each file is single-channel
    print(f"  shape (T, ...) = {stack.shape}")

    if stack.ndim == 5:
        n_channels = stack.shape[1]
    elif stack.ndim == 4:
        stack = stack[:, np.newaxis]
        n_channels = 1
    else:
        raise ValueError(f"Unexpected stack shape {stack.shape} for {path}.")

    names = [channel_name_by_index.get(i) or f"ch{i}" for i in range(n_channels)]
    colors = assign_colors(names)
    print("  Computing contrast limits from timepoint 0 (strided sample; avoids "
          "eagerly loading other timepoints just for this)...")
    first_volume = load_timepoint(0)  # (C, Z, Y, X), or (Z, Y, X) for a single channel
    if first_volume.ndim == 3:
        first_volume = first_volume[np.newaxis]
    contrast_limits = [compute_contrast_limits(first_volume[c]) for c in range(n_channels)]
    for name, color, cl in zip(names, colors, contrast_limits):
        print(f"    {name}: colormap={color}, contrast_limits={cl}")

    viewer = napari.Viewer(title=path.name)
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
