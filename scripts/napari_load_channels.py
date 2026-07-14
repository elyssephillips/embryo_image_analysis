"""Open a multi-channel ImageJ-hyperstack TIFF in napari with each channel as
its own layer.

Needed because napari's builtin reader (no OME-aware plugin installed in
napari_env) never parses ImageJ hyperstack metadata — dropping a file like
this straight into napari loads it as one flat (pages, Y, X) stack instead of
splitting channels. This script reads the ImageJ `channels`/`slices`
metadata itself, reshapes the stack accordingly, and adds it with an explicit
channel_axis so each channel becomes a separate, independently controllable
layer.

Usage
-----
  python scripts/napari_load_channels.py "/path/to/stack.tif"
  python scripts/napari_load_channels.py "/path/to/stack.tif" --channel-names DAPI ZO1 DMKN FMNL2
  python scripts/napari_load_channels.py "/path/to/stack.tif" --channel-axis 0  # override auto-detection
"""
import argparse
from pathlib import Path

import napari
import numpy as np
import tifffile


def load_hyperstack(path: Path, channel_axis: int | None):
    with tifffile.TiffFile(str(path)) as tif:
        arr = tif.asarray()
        meta = tif.imagej_metadata

    if channel_axis is not None:
        return arr, channel_axis

    if not meta or "channels" not in meta:
        raise ValueError(
            f"No ImageJ channel metadata found in {path.name} (shape {arr.shape}). "
            "Pass --channel-axis explicitly to split it manually."
        )

    n_channels = int(meta["channels"])
    n_slices = int(meta.get("slices", arr.shape[0] // n_channels))
    if n_channels * n_slices != arr.shape[0]:
        raise ValueError(
            f"channels ({n_channels}) * slices ({n_slices}) != page count ({arr.shape[0]}) "
            f"in {path.name}. Pass --channel-axis explicitly to split it manually."
        )

    # Files written by the current convert_h5_channels_to_tiff.py pipeline are
    # z-major, channel-fastest (page index = z * n_channels + c).
    arr = arr.reshape(n_slices, n_channels, *arr.shape[1:])
    return arr, 1


def main():
    parser = argparse.ArgumentParser(description="Open a multi-channel TIFF in napari, split into per-channel layers.")
    parser.add_argument("path", type=Path, help="Path to the multi-channel TIFF.")
    parser.add_argument("--channel-axis", type=int, default=None, help="Axis to split into layers (default: auto-detect from ImageJ metadata).")
    parser.add_argument("--channel-names", nargs="+", default=None, help="Name for each channel layer, in order.")
    args = parser.parse_args()

    arr, channel_axis = load_hyperstack(args.path, args.channel_axis)
    print(f"Loaded {args.path.name}: shape {arr.shape}, splitting on axis {channel_axis}")

    viewer = napari.Viewer()
    viewer.add_image(
        arr,
        channel_axis=channel_axis,
        name=args.channel_names,
    )
    napari.run()


if __name__ == "__main__":
    main()
