"""Standalone, occasional-use fix for cross-channel drift within a stack —
NOT part of the normal new_if_config.py -> convert_h5_to_tiff.py flow, since
this doesn't usually happen on this acquisition setup. Run this by hand only
on specific files where you've actually noticed channels sitting offset from
each other (e.g. visible in view_hyperstack.py), then re-point raw_data_dir
at the output (or copy the corrected files over the originals) if you want
the rest of the pipeline to use the corrected version.

Runs on the already-converted, already-cropped TIFFs in raw_data_dir (not the
raw HDF5 — this needs to work even after the raw acquisition folders are gone,
which is the normal state once a dataset has been converted and cleaned up).
Estimates each channel's (dz, dy, dx) drift vs REFERENCE_CHANNEL by phase
correlation directly on the loaded array, then re-writes the file with each
channel read from its own shifted window carved out of the same array — this
only works because the crop already has some padding margin around the real
signal (pad in h5_conversion) for the shift to land in; the output ends up
very slightly smaller than the input (shrunk by the drift margin on each
side), not padded/interpolated.

To run: set TIFF_NAMES below to the specific file(s) you want fixed (or flip
PROCESS_ALL_TIFFS on to do every TIFF in raw_data_dir), then click VS Code's
"Run Python File" button (or Ctrl+F5) — no terminal command needed.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.conversion import (
    compute_channel_drift_from_array,
    get_config_value,
    load_hyperstack_czyx,
    load_yaml_config,
    write_tiff_czyx_from_array,
    _shift_bounds_clamped,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "IF" / "config.yaml"

# ============================== EDIT THESE ==============================
TIFF_NAMES = []  # e.g. ["stack_0-ctrl 1 2 3_embryo1.tif"] -- filenames (resolved
                  # against raw_data_dir) or full paths, for the specific file(s)
                  # you want drift-corrected. Ignored if PROCESS_ALL_TIFFS is True.
PROCESS_ALL_TIFFS = True  # True = process every *.tif in raw_data_dir instead of
                    # just TIFF_NAMES. Still has to be flipped on deliberately —
                    # this isn't the default, since drift correction is meant to
                    # be applied to specific files you've actually checked, not
                    # the whole dataset automatically.
REFERENCE_CHANNEL = "dapi"  # every other channel is aligned to this one
MAX_SHIFT_FRAC = 0.15  # reject (leave uncorrected) any channel whose estimated
                    # shift on any axis exceeds this fraction of that axis's
                    # array size -- guards against phase correlation locking
                    # onto a spurious peak instead of the true small stage
                    # drift, which otherwise collapses the crop (seen on this
                    # dataset: one channel's estimate was over 500 px).
OUTPUT_DIR = None  # None = sibling folder "<raw_data_dir.name>_drift_corrected"
                    # next to raw_data_dir. Set an explicit path to override.
# ==========================================================================


def main():
    config = load_yaml_config(CONFIG_PATH)
    raw_data_dir = Path(get_config_value(config, ["raw_data_dir"]))
    channel_names = get_config_value(config, ["microscopy", "channel_names"]) or []
    if REFERENCE_CHANNEL not in channel_names:
        raise ValueError(f"REFERENCE_CHANNEL={REFERENCE_CHANNEL!r} not in microscopy.channel_names {channel_names}.")
    reference_idx = channel_names.index(REFERENCE_CHANNEL)

    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else raw_data_dir.parent / f"{raw_data_dir.name}_drift_corrected"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output -> {output_dir}")

    if PROCESS_ALL_TIFFS:
        tiff_paths = sorted(raw_data_dir.glob("*.tif"))
        if not tiff_paths:
            raise ValueError(f"No .tif files found in {raw_data_dir}")
        print(f"PROCESS_ALL_TIFFS is True — found {len(tiff_paths)} file(s) in {raw_data_dir}")
    elif TIFF_NAMES:
        tiff_paths = []
        for tiff_name in TIFF_NAMES:
            p = Path(tiff_name)
            if not p.is_absolute() and not p.exists():
                p = raw_data_dir / tiff_name
            if not p.exists():
                raise FileNotFoundError(f"{p} does not exist.")
            tiff_paths.append(p)
    else:
        raise ValueError(
            "Set TIFF_NAMES to the specific file(s) you want drift-corrected, "
            "or PROCESS_ALL_TIFFS = True to do every TIFF in raw_data_dir."
        )

    for path in tiff_paths:
        print(f"\n=== {path.name} ===")
        output_path = output_dir / path.name
        if output_path.exists():
            print(f"  Skipping: {output_path} already exists.")
            continue

        try:
            arr, _meta = load_hyperstack_czyx(path)
            nz, _n_channels, ny, nx = arr.shape
            print(f"  shape (Z,C,Y,X) = {arr.shape}")

            drift = compute_channel_drift_from_array(
                arr, reference_idx=reference_idx, max_shift_frac=MAX_SHIFT_FRAC
            )

            # Shrink to an inner window with enough margin on every side to fit
            # every channel's shift without running off the edge of the array —
            # see _shift_bounds_clamped's docstring for why margin = max(|shift|)
            # per axis is exactly enough (not more, not less).
            margin_z = max((abs(d[0]) for d in drift), default=0)
            margin_y = max((abs(d[1]) for d in drift), default=0)
            margin_x = max((abs(d[2]) for d in drift), default=0)
            if margin_z * 2 >= nz or margin_y * 2 >= ny or margin_x * 2 >= nx:
                raise ValueError(
                    f"Measured drift {drift} is too large relative to this crop's size {(nz, ny, nx)} "
                    "to correct without running out of margin — check the drift numbers above look sane."
                )
            inner_bounds = (margin_z, nz - margin_z, margin_y, ny - margin_y, margin_x, nx - margin_x)
            print(f"  drift margin (z,y,x) = ({margin_z}, {margin_y}, {margin_x}); "
                  f"output shape shrinks from {(nz, ny, nx)} to "
                  f"{(nz - 2*margin_z, ny - 2*margin_y, nx - 2*margin_x)}")

            full_shape = (nz, ny, nx)
            per_channel_bounds = [_shift_bounds_clamped(inner_bounds, s, full_shape) for s in drift]

            print(f"  Writing drift-corrected TIFF -> {output_path}")
            write_tiff_czyx_from_array(output_path, arr, per_channel_bounds, dtype=str(arr.dtype))
        except Exception as e:
            if not PROCESS_ALL_TIFFS:
                raise
            print(f"  SKIPPED {path.name} due to error: {e}")
            continue

    print("\nDone.")


if __name__ == "__main__":
    main()
