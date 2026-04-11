from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import tifffile
import yaml

FOLDER_RE = re.compile(r"^(stack_[^_]+)_channel_(\d+)(?:-(.+))?$", re.IGNORECASE)


def parse_folder_name(folder_name: str):
    match = FOLDER_RE.match(folder_name)
    if not match:
        return None
    stack_id = match.group(1)
    channel_index = int(match.group(2))
    channel_name = match.group(3) or ""
    return stack_id, channel_index, channel_name


def find_h5_file(folder: Path) -> Path:
    files = list(folder.glob("*.h5")) + list(folder.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No .h5/.hdf5 file found in folder: {folder}")
    if len(files) > 1:
        raise ValueError(f"Expected one HDF5 file in {folder}, found: {files}")
    return files[0]


def find_first_dataset(group: Any):
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            return item
        if isinstance(item, h5py.Group):
            dataset = find_first_dataset(item)
            if dataset is not None:
                return dataset
    return None


def load_h5_image(file_path: Path, dataset_path: str = None) -> np.ndarray:
    with h5py.File(file_path, "r") as f:
        if dataset_path:
            dataset = f[dataset_path]
        else:
            dataset = find_first_dataset(f)
            if dataset is None:
                raise ValueError(f"No dataset found inside {file_path}")
        data = dataset[()]
    data = np.asarray(data)
    if data.ndim == 2:
        return data[np.newaxis, ...]
    if data.ndim == 3:
        return data
    if data.ndim == 4 and data.shape[0] == 1:
        return data[0]
    raise ValueError(
        f"Unsupported HDF5 image shape {data.shape} in {file_path}. Expected 2D or 3D data."
    )


def crop_volume(volume: np.ndarray, crop_bounds: Tuple[int, int, int, int, int, int]) -> np.ndarray:
    z0, z1, y0, y1, x0, x1 = crop_bounds
    return volume[z0:z1, y0:y1, x0:x1]


def parse_crop_arg(arg: str) -> Tuple[int, ...]:
    values = [int(v) for v in arg.split(":") if v != ""]
    if len(values) not in (4, 6):
        raise argparse.ArgumentTypeError(
            "Crop must be given as y0:y1:x0:x1 or z0:z1:y0:y1:x0:x1"
        )
    if len(values) == 4:
        return (values[0], values[1], values[2], values[3])
    return tuple(values)


def auto_crop_bounds(stack: np.ndarray, pad: int = 0) -> Tuple[int, int, int, int, int, int]:
    mask = np.any(stack != 0, axis=0)
    if not np.any(mask):
        raise ValueError("Auto-crop found no nonzero pixels across the combined channels.")
    coord = np.argwhere(mask)
    z0, y0, x0 = coord.min(axis=0)
    z1, y1, x1 = coord.max(axis=0) + 1
    z0 = max(0, z0 - pad)
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    z1 = min(stack.shape[1], z1 + pad)
    y1 = min(stack.shape[2], y1 + pad)
    x1 = min(stack.shape[3], x1 + pad)
    return z0, z1, y0, y1, x0, x1


def compute_mip_streaming(
    channel_h5_files: list,
    dataset_path: str | None,
    crop_bounds: Tuple[int, int, int, int, int, int],
) -> list:
    """Return a full-field max-Z projection (Y, X) for each channel.

    Uses the Z crop range from *crop_bounds* for efficiency, but keeps the
    full XY extent so the crop rectangle can be overlaid on the preview.
    Reads one Z-slice at a time.
    """
    cz0, cz1 = crop_bounds[0], crop_bounds[1]
    mips = []
    for h5_file in channel_h5_files:
        with h5py.File(h5_file, "r") as f:
            ds = _get_h5_dataset(f, dataset_path)
            mip = None
            for z in range(cz0, cz1):
                sl = np.asarray(_read_zslice(ds, z), dtype=np.float32)
                mip = sl if mip is None else np.maximum(mip, sl)
        mips.append(mip)
    return mips


def confirm_autocrop(
    stack_id: str,
    bounds: Tuple[int, int, int, int, int, int],
    original_shape: Tuple[int, ...],
    cropped_shape: Tuple[int, ...],
    channel_h5_files: list | None = None,
    dataset_path: str | None = None,
    channel_names: list | None = None,
) -> bool:
    z0, z1, y0, y1, x0, x1 = bounds
    print(f"\nAuto-crop for stack {stack_id}")
    print(f"  original shape: {original_shape}")
    print(f"  crop bounds: z={z0}:{z1}, y={y0}:{y1}, x={x0}:{x1}")
    print(f"  cropped shape: {cropped_shape}")

    if channel_h5_files:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            mips = compute_mip_streaming(channel_h5_files, dataset_path, bounds)
            n = len(mips)
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
            fig.suptitle(
                f"Max-Z MIP — {stack_id}  |  yellow box = crop  (z={z0}:{z1}, y={y0}:{y1}, x={x0}:{x1})",
                fontsize=10,
            )
            for i, mip in enumerate(mips):
                ax = axes[0, i]
                p_low, p_high = np.percentile(mip, (1, 99))
                ax.imshow(mip, cmap="gray", vmin=p_low, vmax=p_high, origin="upper")
                # Overlay crop rectangle: imshow x-axis = X, y-axis = Y
                rect = mpatches.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    linewidth=1.5, edgecolor="yellow", facecolor="none",
                )
                ax.add_patch(rect)
                label = channel_names[i] if channel_names and i < len(channel_names) else f"ch{i}"
                ax.set_title(label)
                ax.axis("off")
            plt.tight_layout()
            plt.show(block=True)
        except Exception as e:
            print(f"  (MIP preview unavailable: {e})")

    while True:
        choice = input("Approve and save this cropped TIFF? [y/N]: ").strip().lower()
        if choice in {"y", "yes"}:
            return True
        if choice in {"n", "no", ""}:
            return False
        print("Please enter 'y' or 'n'.")


def write_combined_tiff(stack: np.ndarray, output_path: Path) -> None:
    tifffile.imwrite(str(output_path), stack, metadata={"axes": "CZYX"})


# ---------------------------------------------------------------------------
# Helpers for slice-level streaming (no full volume ever loaded into RAM)
# ---------------------------------------------------------------------------

def _get_h5_dataset(f: h5py.File, dataset_path: str | None) -> h5py.Dataset:
    if dataset_path:
        return f[dataset_path]
    ds = find_first_dataset(f)
    if ds is None:
        raise ValueError("No dataset found in HDF5 file.")
    return ds


def _effective_zyx_shape(ds: h5py.Dataset) -> Tuple[int, int, int]:
    """Return (nz, ny, nx) without loading data (handles 2-D, 3-D, and (1,Z,Y,X))."""
    s = ds.shape
    if len(s) == 2:
        return (1, s[0], s[1])
    if len(s) == 3:
        return (s[0], s[1], s[2])
    if len(s) == 4 and s[0] == 1:
        return (s[1], s[2], s[3])
    raise ValueError(f"Unsupported HDF5 dataset shape {s}. Expected 2-D or 3-D.")


def _read_zslice(ds: h5py.Dataset, z: int) -> np.ndarray:
    """Read one Z-slice (Y, X) without touching the rest of the dataset."""
    s = ds.shape
    if len(s) == 2:
        return ds[:]
    if len(s) == 3:
        return ds[z]
    if len(s) == 4 and s[0] == 1:
        return ds[0, z]
    raise ValueError(f"Unsupported HDF5 dataset shape {s}.")


def compute_autocrop_bounds_streaming(
    channel_h5_files: list,
    dataset_path: str | None,
    pad: int = 0,
    threshold: float = 0,
    threshold_percentile: float | None = None,
    blur_sigma: float = 0,
) -> Tuple[int, int, int, int, int, int]:
    """Compute auto-crop bounds using a MIP-based algorithm.

    Steps per channel:
      1. Stream Z-slices to build the max-Z projection (MIP) in one pass.
         The MIP naturally suppresses single-Z-plane artifacts that would
         otherwise drag the bounding box outward.
      2. Optionally blur the MIP (blur_sigma > 0) with a Gaussian — this
         fills in dim signal at the embryo edges so the threshold is less
         sensitive to patchy illumination.
      3. Determine the threshold: if *threshold_percentile* is given, use
         that percentile of the (blurred) MIP so the cutoff adapts to the
         actual signal level in each stack; otherwise use the absolute
         *threshold* value.
      4. For Z bounds: do a second pass to find which Z-slices have any
         pixel above the threshold (cheap — just reads scalar per-slice max).

    Peak memory: one (Y, X) MIP per channel being scanned (~42 MB each).
    """
    try:
        from scipy.ndimage import gaussian_filter as _gf
    except ImportError:
        _gf = None
        if blur_sigma > 0:
            print("  Warning: scipy not available; blur_sigma ignored.", flush=True)

    ref_shape: Tuple[int, int, int] | None = None
    union_yx: np.ndarray | None = None   # accumulated signal mask (ny, nx)
    z_any: np.ndarray | None = None

    n_ch = len(channel_h5_files)
    for ch_idx, h5_file in enumerate(channel_h5_files):
        with h5py.File(h5_file, "r") as f:
            ds = _get_h5_dataset(f, dataset_path)
            nz, ny, nx = _effective_zyx_shape(ds)
            if ref_shape is None:
                ref_shape = (nz, ny, nx)
                z_any = np.zeros(nz, dtype=bool)
            elif (nz, ny, nx) != ref_shape:
                raise ValueError(
                    f"Shape mismatch: {ref_shape} vs {(nz, ny, nx)} in {h5_file}"
                )

            # Pass 1: build MIP for this channel
            print(f"  Scanning ch {ch_idx + 1}/{n_ch} — building MIP ({nz} slices)...", flush=True)
            mip = np.zeros((ny, nx), dtype=np.float32)
            for z in range(nz):
                if z % 50 == 0:
                    print(f"    z {z}/{nz}", flush=True)
                sl = np.asarray(_read_zslice(ds, z), dtype=np.float32)
                np.maximum(mip, sl, out=mip)

        # Optionally blur the MIP
        if blur_sigma > 0 and _gf is not None:
            mip_proc = _gf(mip, sigma=blur_sigma)
        else:
            mip_proc = mip

        # Determine threshold for this channel
        if threshold_percentile is not None:
            eff_threshold = float(np.percentile(mip_proc, threshold_percentile))
            print(f"    threshold={eff_threshold:.1f} (p{threshold_percentile} of MIP)", flush=True)
        else:
            eff_threshold = float(threshold)

        signal_yx = mip_proc > eff_threshold
        union_yx = signal_yx if union_yx is None else (union_yx | signal_yx)

        # Pass 2: Z bounds — which slices have any pixel above threshold
        with h5py.File(h5_file, "r") as f:
            ds = _get_h5_dataset(f, dataset_path)
            print(f"    Computing Z bounds...", flush=True)
            for z in range(nz):
                if not z_any[z]:
                    sl = np.asarray(_read_zslice(ds, z), dtype=np.float32)
                    if sl.max() > eff_threshold:
                        z_any[z] = True

    if union_yx is None or not union_yx.any():
        raise ValueError(
            "Auto-crop found no signal. Try lowering threshold or threshold_percentile."
        )

    # Reduce to the largest connected component so that isolated bright pixels
    # outside the embryo don't drag the bounding box to the image edges.
    try:
        from scipy.ndimage import label as _label
        labeled, n_components = _label(union_yx)
        if n_components > 1:
            sizes = np.bincount(labeled.ravel())[1:]  # index 0 = background
            largest = int(np.argmax(sizes)) + 1
            kept_frac = sizes[largest - 1] / union_yx.sum()
            print(
                f"  Found {n_components} signal components; keeping largest "
                f"({sizes[largest - 1]:,} px, {kept_frac:.0%} of signal).",
                flush=True,
            )
            union_yx = labeled == largest
    except ImportError:
        print("  scipy not available; skipping connected-component filtering.", flush=True)

    y_any = union_yx.any(axis=1)
    x_any = union_yx.any(axis=0)

    y0 = int(np.argmax(y_any))
    y1 = int(len(y_any) - 1 - np.argmax(y_any[::-1])) + 1
    x0 = int(np.argmax(x_any))
    x1 = int(len(x_any) - 1 - np.argmax(x_any[::-1])) + 1
    z0 = int(np.argmax(z_any))
    z1 = int(len(z_any) - 1 - np.argmax(z_any[::-1])) + 1

    nz, ny, nx = ref_shape
    z0 = max(0, z0 - pad)
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    z1 = min(nz, z1 + pad)
    y1 = min(ny, y1 + pad)
    x1 = min(nx, x1 + pad)
    return z0, z1, y0, y1, x0, x1


def write_tiff_czyx_streaming(
    output_path: Path,
    channel_h5_files: list,
    dataset_path: str | None,
    dtype: str,
    crop_bounds=None,
) -> None:
    """Write a CZYX TIFF one (Y, X) slice at a time.

    Reads from each h5 file Z-slice by Z-slice so the peak in-memory footprint
    is a single (Y, X) plane (≈ 21 MB for 2368 × 4432 float32) regardless of
    the number of channels or Z depth.  Uses BigTIFF for files > 4 GB and
    writes an ImageJ-compatible hyperstack description so the result opens
    correctly in Fiji and tifffile.
    """
    np_dtype = np.dtype(dtype)
    n_channels = len(channel_h5_files)

    with h5py.File(channel_h5_files[0], "r") as f:
        nz_full, ny_full, nx_full = _effective_zyx_shape(_get_h5_dataset(f, dataset_path))

    if crop_bounds is not None:
        cz0, cz1, cy0, cy1, cx0, cx1 = crop_bounds
    else:
        cz0, cz1 = 0, nz_full
        cy0, cy1 = 0, ny_full
        cx0, cx1 = 0, nx_full

    nz = cz1 - cz0
    imagej_desc = (
        "ImageJ=1.11a\n"
        f"channels={n_channels}\n"
        f"slices={nz}\n"
        f"images={n_channels * nz}\n"
        "hyperstack=true\n"
        "mode=composite\n"
    )

    first_page = True
    with tifffile.TiffWriter(str(output_path), bigtiff=True) as tif:
        for ch_idx, h5_file in enumerate(channel_h5_files):
            print(f"  Writing ch {ch_idx + 1}/{n_channels} ({nz} Z-slices)...", flush=True)
            with h5py.File(h5_file, "r") as f:
                ds = _get_h5_dataset(f, dataset_path)
                nz_ch, ny_ch, nx_ch = _effective_zyx_shape(ds)
                if (nz_ch, ny_ch, nx_ch) != (nz_full, ny_full, nx_full):
                    raise ValueError(
                        f"Shape mismatch: expected {(nz_full, ny_full, nx_full)}, "
                        f"got {(nz_ch, ny_ch, nx_ch)} in {h5_file}"
                    )
                for z in range(cz0, cz1):
                    if z % 50 == 0:
                        print(f"    z {z - cz0}/{nz}", flush=True)
                    sl = np.asarray(_read_zslice(ds, z)[cy0:cy1, cx0:cx1], dtype=np_dtype)
                    if first_page:
                        tif.write(sl, contiguous=True, description=imagej_desc)
                        first_page = False
                    else:
                        tif.write(sl, contiguous=True)


def build_stack_groups(root_dir: Path) -> Dict[str, list[tuple[int, Path]]]:
    groups: Dict[str, list[tuple[int, Path]]] = {}
    for folder in sorted(root_dir.iterdir()):
        if not folder.is_dir():
            continue
        parsed = parse_folder_name(folder.name)
        if parsed is None:
            continue
        stack_id, channel_index, channel_name = parsed
        groups.setdefault(stack_id, []).append((channel_index, folder))
    return groups


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_config_value(config: Dict[str, Any], key_path: list[str], default=None):
    if config is None:
        return default
    current = config
    for key in key_path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def get_h5_conversion_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if config is None:
        return {}
    h5_config = config.get("h5_conversion")
    return h5_config if isinstance(h5_config, dict) else {}


def resolve_relative_path(base: Path, path_value):
    if path_value is None:
        return None
    p = Path(path_value)
    return p if p.is_absolute() else (base / p)


def load_pipeline_raw_dir(pipeline_config_path: Path) -> Path:
    pipeline_config = load_yaml_config(pipeline_config_path)
    raw_dir = get_config_value(pipeline_config, ["raw_data_dir"])
    if raw_dir is None:
        raw_dir = get_config_value(pipeline_config, ["paths", "raw_data_dir"])
    if raw_dir is None:
        raise ValueError(
            f"Could not find raw_data_dir in pipeline config: {pipeline_config_path}"
        )
    return Path(raw_dir)
