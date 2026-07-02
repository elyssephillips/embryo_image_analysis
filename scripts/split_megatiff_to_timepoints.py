"""Split a TCZYX BigTIFF into per-timepoint TIFFs for Napari.

Usage:
    python scripts/split_megatiff_to_timepoints.py <input.tif> [output_dir] --n-timepoints N

If output_dir is omitted, a folder named after the TIFF is created next to it.
--n-timepoints is required when the file's ImageJ metadata lacks explicit frames/slices
(common with large BigTIFF hyperstacks written by tifffile).

Napari can then open the output folder as a lazy TZYX stack via drag-and-drop.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.log import log_run


def split_to_timepoints(input_path: Path, output_dir: Path | None, n_timepoints: int | None) -> None:
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent / input_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(str(input_path)) as tif:
        series = tif.series[0]
        axes = series.axes
        shape = series.shape
        dtype = series.dtype
        n_pages = len(tif.pages)
        ij = tif.imagej_metadata or {}

        print(f"  Source : axes={axes}  shape={shape}  dtype={dtype}")
        print(f"  Pages  : {n_pages}")
        print(f"  ImageJ metadata: {ij}")

        # --- Determine n_t -------------------------------------------------------
        if n_timepoints is not None:
            n_t = n_timepoints
        elif 'T' in axes:
            n_t = shape[axes.index('T')]
        elif 'frames' in ij and int(ij['frames']) > 1:
            n_t = int(ij['frames'])
        else:
            raise ValueError(
                "Cannot determine timepoint count from file metadata. "
                "Pass --n-timepoints N explicitly."
            )

        if n_pages % n_t != 0:
            raise ValueError(f"Total pages {n_pages} is not divisible by n_timepoints={n_t}.")

        pages_per_t = n_pages // n_t
        ny, nx = shape[-2], shape[-1]
        nz = pages_per_t  # 1 channel; if multi-channel this would be pages_per_t // n_channels

        print(f"  n_timepoints={n_t}, pages_per_t={pages_per_t}  →  ZYX ({nz}, {ny}, {nx})")
        print(f"  Writing to {output_dir}/")

        pages = tif.pages

        for t in range(n_t):
            if t % 10 == 0:
                print(f"  t {t}/{n_t}...", flush=True)

            start = t * pages_per_t
            vol = np.stack([pages[start + i].asarray() for i in range(pages_per_t)]).astype(dtype)
            # vol shape: (pages_per_t, ny, nx) = ZYX

            tifffile.imwrite(
                str(output_dir / f"t{t:04d}.tif"),
                vol,
                photometric="minisblack",
                metadata={"axes": "ZYX"},
            )

    print(f"  Done. {n_t} files written to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Split a TCZYX BigTIFF into per-timepoint TIFFs.")
    parser.add_argument("input", type=Path, help="Input BigTIFF file.")
    parser.add_argument("output_dir", type=Path, nargs="?", default=None,
                        help="Output directory (default: sibling folder named after the TIFF).")
    parser.add_argument("--n-timepoints", type=int, default=None,
                        help="Number of timepoints (required if not in file metadata).")
    args = parser.parse_args()
    output_dir = args.output_dir or args.input.parent / args.input.stem
    split_to_timepoints(args.input, output_dir, args.n_timepoints)
    dataset_id = args.input.parent.name
    log_run("preprocessing", dataset_id, "split_megatiff_to_timepoints.py",
            output_path=str(output_dir), detail="light",
            data_path=str(args.input.parent))


if __name__ == "__main__":
    main()
