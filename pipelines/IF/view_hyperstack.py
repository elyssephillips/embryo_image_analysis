"""Open a converted multi-channel IF TIFF in napari as a proper hyperstack —
each channel as its own layer, colored and named from configs/IF/config.yaml,
scaled by the real voxel size. Optionally overlays the matching StarDist
segmentation (see SHOW_SEGMENTATION below) at the same scale.

Needed because napari's builtin reader never parses the ImageJ hyperstack
metadata that convert_h5_to_tiff.py writes — dropping the file straight into
napari loads it as one flat (pages, Y, X) stack instead of splitting it into
Z and channel. This script reads the embedded `channels`/`slices` metadata
itself, reshapes the stack accordingly (memory-mapped, so multi-GB files
don't get fully loaded into RAM), and adds it to napari with an explicit
channel_axis.

The same applies to segmentation labels: 00_segment.py writes them with
tifffile.imwrite and no calibration at all, and napari's builtin reader
defaults any dragged-in file to scale=(1,1,1) regardless. Dragging a
segmentation TIFF in next to a properly-scaled hyperstack puts the two layers
on different coordinate systems even though they share the same voxel grid
(StarDist's internal `scale=` correction resamples for inference but resizes
the output labels back to the original input shape). Loading it here instead
applies the identical scale= to both, so there's nothing to match by hand.

To run: set TIFF_PATH below, then click VS Code's "Run Python File" button
(or Ctrl+F5) — no terminal command needed. Leave TIFF_PATH as None to just
list the available TIFFs in raw_data_dir and exit. Needs a real display
(X11/VNC) since it opens a napari window.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import napari
import numpy as np
import tifffile

from src.conversion import get_config_value, load_hyperstack_czyx, load_yaml_config

CONFIG_PATH = PROJECT_ROOT / "configs" / "IF" / "config.yaml"

# ============================== EDIT THESE ==============================
TIFF_PATH = "stack_017_embryo4.tif"   # e.g. "stack_5-ctrl_embryo1.tif" (resolved against raw_data_dir)
                    # or a full path to a TIFF anywhere else (e.g. in rotated_dir).
                    # Leave as None to list available TIFFs in raw_data_dir and exit.
VOXEL_SIZE_ZYX_OVERRIDE = None  # e.g. [0.55, 0.122666664, 0.122666664] to test a
                    # different Z calibration without touching configs/IF/config.yaml
                    # (which 02_extract_intensities.py / 04_plot_intensities.py also read).
                    # Leave as None to use the config's voxel_size_zyx as-is.
SHOW_SEGMENTATION = True  # auto-find and overlay {stem}_segmentation.tif from
                    # config's segmentation_dir_raw (dapi at the top level, other
                    # channels in their own subfolder — see 00_segment.py) at the
                    # same scale as the raw channels. Set False to skip.
# ==========================================================================

# Default LUTs, keyed by channel_names entries (case-insensitive); anything
# unmatched cycles through the fallback palette in order.
DEFAULT_COLORS = {"dapi": "blue", "hoechst": "blue"}
FALLBACK_PALETTE = ["green", "magenta", "yellow", "cyan", "red"]


def assign_colors(channel_names: list[str]) -> list[str]:
    colors = []
    fallback_iter = iter(FALLBACK_PALETTE)
    for name in channel_names:
        color = DEFAULT_COLORS.get(name.lower())
        if color is None:
            color = next(fallback_iter, "gray")
        colors.append(color)
    return colors


def compute_contrast_limits(channel_stack: np.ndarray, stride: int = 4) -> tuple[float, float]:
    """Fast percentile estimate from a strided sample instead of the full memmap."""
    sample = channel_stack[::stride, ::stride, ::stride]
    lo, hi = np.percentile(sample, (1, 99.5))
    return float(lo), float(hi)


def find_segmentations(seg_dir: Path, stem: str, channel_names: list[str]) -> list[tuple[str, Path]]:
    """Return [(channel_name, path), ...] for every {stem}_segmentation.tif found.

    Mirrors 00_segment.py's output layout: dapi's segmentation sits directly in
    seg_dir, every other channel gets its own seg_dir/{channel}/ subfolder.
    """
    found = []
    dapi_path = seg_dir / f"{stem}_segmentation.tif"
    if dapi_path.exists():
        found.append(("dapi", dapi_path))
    for name in channel_names:
        if name == "dapi":
            continue
        p = seg_dir / name / f"{stem}_segmentation.tif"
        if p.exists():
            found.append((name, p))
    return found


def main():
    config = load_yaml_config(CONFIG_PATH) if CONFIG_PATH.exists() else {}
    raw_data_dir = Path(get_config_value(config, ["raw_data_dir"]) or ".")
    channel_names = get_config_value(config, ["microscopy", "channel_names"]) or []
    voxel_size_zyx = VOXEL_SIZE_ZYX_OVERRIDE or get_config_value(config, ["microscopy", "voxel_size_zyx"]) or [1.0, 1.0, 1.0]
    print(f"  Using voxel_size_zyx = {voxel_size_zyx}"
          f"{' (override)' if VOXEL_SIZE_ZYX_OVERRIDE else ' (from config.yaml)'}")

    if TIFF_PATH is None:
        tiffs = sorted(raw_data_dir.glob("*.tif"))
        print(f"TIFF_PATH is not set. Available TIFFs in {raw_data_dir}:\n")
        for t in tiffs:
            print(f"  {t.name}")
        print(f"\nSet TIFF_PATH to one of the names above (or a full path) and re-run.")
        return

    path = Path(TIFF_PATH)
    if not path.is_absolute() and not path.exists():
        path = raw_data_dir / TIFF_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    print(f"Loading {path} ...")
    arr, meta = load_hyperstack_czyx(path)
    n_slices, n_channels = arr.shape[:2]
    print(f"  shape (Z, C, Y, X) = {arr.shape}  [{meta}]")

    if len(channel_names) != n_channels:
        print(
            f"  WARNING: config has {len(channel_names)} channel_names but this file has "
            f"{n_channels} channels — falling back to ch0..ch{n_channels - 1}."
        )
        names = [f"ch{i}" for i in range(n_channels)]
    else:
        names = list(channel_names)

    colors = assign_colors(names)
    print("  Computing contrast limits per channel...")
    contrast_limits = [compute_contrast_limits(arr[:, c]) for c in range(n_channels)]
    for name, color, cl in zip(names, colors, contrast_limits):
        print(f"    {name}: colormap={color}, contrast_limits={cl}")

    viewer = napari.Viewer(title=path.name, ndisplay=3)
    viewer.add_image(
        arr,
        name=names,
        channel_axis=1,
        colormap=colors,
        contrast_limits=contrast_limits,
        blending="additive",
        scale=voxel_size_zyx,
    )

    if SHOW_SEGMENTATION:
        seg_dir = get_config_value(config, ["segmentation_dir_raw"])
        if seg_dir is None:
            print("  SHOW_SEGMENTATION=True but segmentation_dir_raw isn't set in config.yaml — skipping.")
        else:
            segmentations = find_segmentations(Path(seg_dir), path.stem, names)
            if not segmentations:
                print(f"  No segmentation found for {path.stem!r} under {seg_dir}.")
            for seg_name, seg_path in segmentations:
                print(f"  Loading segmentation [{seg_name}] from {seg_path} ...")
                labels = tifffile.memmap(str(seg_path))
                viewer.add_labels(labels, name=f"seg:{seg_name}", scale=voxel_size_zyx)

    napari.run()


if __name__ == "__main__":
    main()
