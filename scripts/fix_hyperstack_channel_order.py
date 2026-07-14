"""Fix multi-channel ImageJ hyperstack TIFFs written with channel-major page
order (all Z of channel 0, then all Z of channel 1, ...) but no explicit
"order" field, so ImageJ/Fiji/napari assume the default channel-fastest order
and split every 4th page into a "channel" — mixing all real channels together.

Affects TIFFs written by convert_h5_channels_to_tiff.py before the ordering
fix in write_tiff_czyx_streaming (commit d1ef9e1, 2026-06-18). Files written
after that commit are already correct and are skipped automatically.

Auto-detects the on-disk order per file (by comparing how smoothly per-page
means vary under each hypothesis — real Z-adjacent slices of one channel vary
smoothly, real channel boundaries don't) and only rewrites files that are
actually channel-major. Originals are left untouched; fixed copies are
written to an output directory.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile


def detect_order(page_means: np.ndarray, n_channels: int, n_slices: int) -> str:
    """Return 'channel_major' (all Z per channel) or 'channel_fast' (already correct)."""
    fast = page_means.reshape(n_slices, n_channels)   # hypothesis: c fastest, z slowest
    major = page_means.reshape(n_channels, n_slices)  # hypothesis: z fastest, c slowest

    # Whichever hypothesis makes each channel's own Z-run vary smoothly is the real one.
    smoothness_fast = np.mean(np.abs(np.diff(fast, axis=0)))
    smoothness_major = np.mean(np.abs(np.diff(major, axis=1)))

    return "channel_major" if smoothness_major < smoothness_fast else "channel_fast"


def fix_file(input_path: Path, output_path: Path, dry_run: bool = False) -> str:
    with tifffile.TiffFile(str(input_path)) as tif:
        meta = tif.imagej_metadata
        if not meta or "channels" not in meta or "slices" not in meta:
            return f"SKIP (no channels/slices metadata): {input_path.name}"

        n_channels = int(meta["channels"])
        n_slices = int(meta["slices"])
        n_pages = len(tif.pages)
        if n_channels * n_slices != n_pages:
            return (
                f"SKIP (page count {n_pages} != channels*slices "
                f"{n_channels}*{n_slices}): {input_path.name}"
            )

        arr = tif.asarray()  # (n_pages, Y, X)

    page_means = arr.reshape(n_pages, -1).mean(axis=1)
    order = detect_order(page_means, n_channels, n_slices)

    if order == "channel_fast":
        return f"SKIP (already correct order): {input_path.name}"

    if dry_run:
        return f"WOULD FIX (channel-major detected): {input_path.name}"

    # Actual on-disk layout: index = c * n_slices + z. Reorder to z-major,
    # channel-fastest (index = z * n_channels + c) to match the default
    # ImageJ hyperstack order the metadata already declares.
    major = arr.reshape(n_channels, n_slices, *arr.shape[1:])
    fast = np.ascontiguousarray(major.transpose(1, 0, 2, 3))  # (n_slices, n_channels, Y, X)

    imagej_desc = (
        "ImageJ=1.11a\n"
        f"images={n_channels * n_slices}\n"
        f"channels={n_channels}\n"
        f"slices={n_slices}\n"
        "hyperstack=true\n"
        "mode=composite\n"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(str(output_path), bigtiff=True) as tif_out:
        first = True
        for z in range(n_slices):
            for c in range(n_channels):
                sl = fast[z, c]
                if first:
                    tif_out.write(sl, contiguous=True, description=imagej_desc)
                    first = False
                else:
                    tif_out.write(sl, contiguous=True)

    return f"FIXED -> {output_path}"


def main():
    parser = argparse.ArgumentParser(
        description="Fix channel-major hyperstack TIFFs so channels split correctly in Fiji/napari."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing the mis-ordered TIFFs.")
    parser.add_argument(
        "output_dir", type=Path, nargs="?", default=None,
        help="Directory to write fixed TIFFs (default: <input_dir>_fixed alongside input_dir).",
    )
    parser.add_argument("--pattern", default="*.tif", help="Glob pattern for input files (default: *.tif).")
    parser.add_argument("--dry-run", action="store_true", help="Only report detected order, write nothing.")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir.parent / f"{input_dir.name}_fixed"

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        print(f"No files matching {args.pattern} in {input_dir}")
        sys.exit(1)

    print(f"Found {len(files)} file(s). Output dir: {output_dir}\n")
    for f in files:
        out_path = output_dir / f.name
        print(f"[{f.name}]", flush=True)
        result = fix_file(f, out_path, dry_run=args.dry_run)
        print(f"  {result}", flush=True)


if __name__ == "__main__":
    main()
