from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import tifffile
import yaml

FOLDER_RE = re.compile(r"^(stack_[^_]+)_channel_(\d+)(?:[_-](.+))?$", re.IGNORECASE)

# Live-imaging folders: stack_N[-label]_channel_M[_suffix] where the label may
# contain spaces/dashes but no underscores (e.g. "stack_1-control plus fgf_channel_0_obj_bottom").
LIVE_FOLDER_RE = re.compile(r"^(stack_.+?)_channel_(\d+)(?:[_-](.+))?$", re.IGNORECASE)


def parse_live_folder_name(folder_name: str):
    match = LIVE_FOLDER_RE.match(folder_name)
    if not match:
        return None
    stack_id = match.group(1)
    channel_index = int(match.group(2))
    suffix = match.group(3) or ""
    return stack_id, channel_index, suffix


def find_h5_files_sorted(folder: Path) -> list[Path]:
    """Return all .h5/.hdf5 files in *folder* sorted by filename (= timepoint order)."""
    files = sorted(
        set(folder.glob("*.h5")) | set(folder.glob("*.hdf5")),
        key=lambda p: p.name,
    )
    if not files:
        raise FileNotFoundError(f"No .h5/.hdf5 files found in: {folder}")
    return files


def build_live_stack_groups(root_dir: Path) -> Dict[str, list[tuple[int, Path]]]:
    """Group channel folders by stack (embryo) for live-imaging data.

    Returns {stack_id: [(channel_index, folder), ...]} sorted by channel index.
    """
    groups: Dict[str, list[tuple[int, Path]]] = {}
    for folder in sorted(root_dir.iterdir()):
        if not folder.is_dir():
            continue
        parsed = parse_live_folder_name(folder.name)
        if parsed is None:
            continue
        stack_id, channel_index, _ = parsed
        groups.setdefault(stack_id, []).append((channel_index, folder))
    return groups


def write_tiff_tczyx_streaming(
    output_path: Path,
    channel_timepoint_files: list[list[Path]],
    dataset_path: str | None,
    dtype: str,
    crop_bounds=None,
) -> None:
    """Write a TCZYX TIFF one (Y, X) plane at a time.

    channel_timepoint_files[c][t] is the h5 file for channel c at timepoint t.
    Page order: T outer → Z middle → C inner (ImageJ hyperstack convention).
    Peak memory: n_channels × one (Y, X) plane regardless of T or Z depth.
    """
    np_dtype = np.dtype(dtype)
    n_channels = len(channel_timepoint_files)
    n_timepoints = len(channel_timepoint_files[0])

    with h5py.File(channel_timepoint_files[0][0], "r") as f:
        nz_full, ny_full, nx_full = _effective_zyx_shape(_get_h5_dataset(f, dataset_path))

    if crop_bounds is not None:
        cz0, cz1, cy0, cy1, cx0, cx1 = crop_bounds
    else:
        cz0, cz1 = 0, nz_full
        cy0, cy1 = 0, ny_full
        cx0, cx1 = 0, nx_full

    nz = cz1 - cz0

    ny = cy1 - cy0
    nx = cx1 - cx0

    imagej_desc = (
        "ImageJ=1.11a\n"
        f"images={n_timepoints * n_channels * nz}\n"
        f"channels={n_channels}\n"
        f"frames={n_timepoints}\n"
        f"slices={nz}\n"
        "hyperstack=true\n"
        "mode=composite\n"
    )
    first_page = True
    with tifffile.TiffWriter(str(output_path), bigtiff=True) as tif:
        for t in range(n_timepoints):
            if t % 10 == 0:
                print(f"  t {t}/{n_timepoints}...", flush=True)
            h5_handles = [
                h5py.File(str(channel_timepoint_files[c][t]), "r")
                for c in range(n_channels)
            ]
            try:
                datasets = [_get_h5_dataset(h, dataset_path) for h in h5_handles]
                for z in range(cz0, cz1):
                    for ds in datasets:
                        sl = np.asarray(
                            _read_zslice(ds, z)[cy0:cy1, cx0:cx1], dtype=np_dtype
                        )
                        if first_page:
                            tif.write(sl, contiguous=True, photometric='minisblack',
                                      description=imagej_desc)
                            first_page = False
                        else:
                            tif.write(sl, contiguous=True, photometric='minisblack')
            finally:
                for h in h5_handles:
                    h.close()


def write_tiff_per_timepoint_streaming(
    output_dir: Path,
    channel_timepoint_files: list[list[Path]],
    dataset_path: str | None,
    dtype: str,
    crop_bounds=None,
) -> None:
    """Write one CZYX (or ZYX for single channel) TIFF per timepoint into output_dir.

    Files are named t0000.tif, t0001.tif, etc.  Napari loads a folder of
    same-shape TIFFs as a lazy TZYX/TCZYX stack via drag-and-drop.
    """
    np_dtype = np.dtype(dtype)
    n_channels = len(channel_timepoint_files)
    n_timepoints = len(channel_timepoint_files[0])

    with h5py.File(channel_timepoint_files[0][0], "r") as f:
        nz_full, ny_full, nx_full = _effective_zyx_shape(_get_h5_dataset(f, dataset_path))

    if crop_bounds is not None:
        cz0, cz1, cy0, cy1, cx0, cx1 = crop_bounds
    else:
        cz0, cz1 = 0, nz_full
        cy0, cy1 = 0, ny_full
        cx0, cx1 = 0, nx_full

    nz = cz1 - cz0
    ny = cy1 - cy0
    nx = cx1 - cx0

    output_dir.mkdir(parents=True, exist_ok=True)

    for t in range(n_timepoints):
        if t % 10 == 0:
            print(f"  t {t}/{n_timepoints}...", flush=True)
        vol = np.empty((n_channels, nz, ny, nx), dtype=np_dtype)
        h5_handles = [
            h5py.File(str(channel_timepoint_files[c][t]), "r")
            for c in range(n_channels)
        ]
        try:
            datasets = [_get_h5_dataset(h, dataset_path) for h in h5_handles]
            for c, ds in enumerate(datasets):
                for zi, z in enumerate(range(cz0, cz1)):
                    vol[c, zi] = np.asarray(
                        _read_zslice(ds, z)[cy0:cy1, cx0:cx1], dtype=np_dtype
                    )
        finally:
            for h in h5_handles:
                h.close()

        out_vol = vol[0] if n_channels == 1 else vol
        axes = "ZYX" if n_channels == 1 else "CZYX"
        out_path = output_dir / f"t{t:04d}.tif"
        tifffile.imwrite(str(out_path), out_vol, photometric="minisblack",
                         metadata={"axes": axes})


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

    Steps:
      1. Per channel, stream Z-slices to build the max-Z projection (MIP).
      2. Optionally blur the MIP (blur_sigma > 0) with a Gaussian — this
         fills in dim signal at the embryo edges so the threshold is less
         sensitive to patchy illumination.
      3. Determine each channel's threshold: if *threshold_percentile* is
         given, use that percentile of the (blurred) MIP so the cutoff
         adapts to the actual signal level in each stack; otherwise use the
         absolute *threshold* value.
      4. Union all channels' thresholded MIPs into one XY signal mask, then
         reduce it to its largest connected component — this is what keeps
         a stray bright pixel or hot pixel from dragging the XY bounding
         box out to the image edges.
      5. For Z bounds: a second pass finds which Z-slices have any
         thresholded pixel *within that same connected XY footprint*. This
         must be checked against the footprint, not the whole frame —
         checking the whole frame means a single background/hot pixel
         anywhere at all (common with a channel that has a flat, nonzero
         baseline) marks the slice in-bounds, so Z never actually crops
         down from the full raw acquisition depth.

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
    channel_thresholds: list[float] = []

    n_ch = len(channel_h5_files)
    for ch_idx, h5_file in enumerate(channel_h5_files):
        with h5py.File(h5_file, "r") as f:
            ds = _get_h5_dataset(f, dataset_path)
            nz, ny, nx = _effective_zyx_shape(ds)
            if ref_shape is None:
                ref_shape = (nz, ny, nx)
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
        channel_thresholds.append(eff_threshold)

        signal_yx = mip_proc > eff_threshold
        union_yx = signal_yx if union_yx is None else (union_yx | signal_yx)

    if union_yx is None or not union_yx.any():
        raise ValueError(
            "Auto-crop found no signal. Try lowering threshold or threshold_percentile."
        )

    # Reduce to the largest connected component so that isolated bright pixels
    # outside the embryo don't drag the bounding box to the image edges. The
    # Z pass below reuses this same footprint, so it inherits the same
    # protection.
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

    nz, ny, nx = ref_shape
    z_any = np.zeros(nz, dtype=bool)

    # Pass 2: Z bounds — a slice counts as "in" only if it has a thresholded
    # pixel *inside the embryo's own connected XY footprint* (union_yx), not
    # just anywhere in the frame. See docstring point 5.
    for ch_idx, h5_file in enumerate(channel_h5_files):
        eff_threshold = channel_thresholds[ch_idx]
        with h5py.File(h5_file, "r") as f:
            ds = _get_h5_dataset(f, dataset_path)
            print(f"  Computing Z bounds against ch {ch_idx + 1}/{n_ch} footprint...", flush=True)
            for z in range(nz):
                if not z_any[z]:
                    sl = np.asarray(_read_zslice(ds, z), dtype=np.float32)
                    if (sl[union_yx] > eff_threshold).any():
                        z_any[z] = True

    y_any = union_yx.any(axis=1)
    x_any = union_yx.any(axis=0)

    y0 = int(np.argmax(y_any))
    y1 = int(len(y_any) - 1 - np.argmax(y_any[::-1])) + 1
    x0 = int(np.argmax(x_any))
    x1 = int(len(x_any) - 1 - np.argmax(x_any[::-1])) + 1
    z0 = int(np.argmax(z_any))
    z1 = int(len(z_any) - 1 - np.argmax(z_any[::-1])) + 1

    z0 = max(0, z0 - pad)
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    z1 = min(nz, z1 + pad)
    y1 = min(ny, y1 + pad)
    x1 = min(nx, x1 + pad)
    return z0, z1, y0, y1, x0, x1


def autocrop_bounds_from_timepoints(
    channel_timepoint_files: list,
    timepoint_indices: list,
    dataset_path: str | None,
    pad: int,
    threshold: float,
    threshold_percentile: float | None,
    blur_sigma: float,
    auto_crop_channel: int | None,
) -> Tuple[int, int, int, int, int, int]:
    """Auto-crop bounds as the spatial union over several timepoints of a live-imaging stack.

    The union ensures a fixed crop window covers the embryo at every sampled
    timepoint, accommodating drift across the movie. *channel_timepoint_files*
    is `[channel][timepoint] -> Path` as built by find_h5_files_sorted per channel.
    """
    n_ch = len(channel_timepoint_files)
    n_tp = len(channel_timepoint_files[0])

    union_bounds = None
    for t in timepoint_indices:
        t = min(t, n_tp - 1)
        if auto_crop_channel is not None:
            crop_files = [channel_timepoint_files[auto_crop_channel][t]]
            print(f"  Autocrop t={t}: using channel {auto_crop_channel}", flush=True)
        else:
            crop_files = [channel_timepoint_files[c][t] for c in range(n_ch)]
            print(f"  Autocrop t={t}: using all {n_ch} channel(s)", flush=True)

        bounds = compute_autocrop_bounds_streaming(
            crop_files, dataset_path,
            pad=pad,
            threshold=threshold or 0,
            threshold_percentile=threshold_percentile,
            blur_sigma=blur_sigma or 0,
        )
        if union_bounds is None:
            union_bounds = list(bounds)
        else:
            union_bounds[0] = min(union_bounds[0], bounds[0])  # z0
            union_bounds[1] = max(union_bounds[1], bounds[1])  # z1
            union_bounds[2] = min(union_bounds[2], bounds[2])  # y0
            union_bounds[3] = max(union_bounds[3], bounds[3])  # y1
            union_bounds[4] = min(union_bounds[4], bounds[4])  # x0
            union_bounds[5] = max(union_bounds[5], bounds[5])  # x1

    return tuple(union_bounds)


def compute_channel_drift(
    channel_h5_files: list,
    dataset_path: str | None,
    reference_idx: int = 0,
    z_stride: int = 2,
    xy_stride: int = 8,
) -> list[Tuple[int, int, int]]:
    """Estimate each channel's (dz, dy, dx) drift, in raw voxels, relative to
    channel `reference_idx`.

    Sequential per-channel acquisition on a fixed sample can drift a
    meaningful fraction of a nucleus diameter over the course of one
    stack's imaging (stage drift accumulating between channels) — this
    corrects for that so per-nucleus intensity comparisons across channels
    stay valid. write_tiff_czyx_streaming applies the correction by
    offsetting each channel's own crop window rather than resampling
    pixels, so only an integer-voxel shift is needed here, not sub-pixel.

    Uses 3D phase correlation (skimage) on a coarse, strided sub-sample of
    each channel's full raw volume — cheap enough to hold two such volumes
    in memory at once regardless of the full frame size (~2368x4432 raw),
    and upsample_factor keeps the coarse-grid shift estimate accurate
    despite the large stride.
    """
    from skimage.registration import phase_cross_correlation

    with h5py.File(channel_h5_files[0], "r") as f:
        nz_full, ny_full, nx_full = _effective_zyx_shape(_get_h5_dataset(f, dataset_path))

    def _coarse_volume(h5_file: Path) -> np.ndarray:
        with h5py.File(h5_file, "r") as f:
            ds = _get_h5_dataset(f, dataset_path)
            z_indices = list(range(0, nz_full, z_stride))
            vol = np.empty(
                (len(z_indices), len(range(0, ny_full, xy_stride)), len(range(0, nx_full, xy_stride))),
                dtype=np.float64,
            )
            for i, z in enumerate(z_indices):
                sl = np.asarray(_read_zslice(ds, z), dtype=np.float64)
                vol[i] = sl[::xy_stride, ::xy_stride]
        return vol

    print(f"  Estimating channel drift (reference=ch{reference_idx})...", flush=True)
    ref_vol = _coarse_volume(channel_h5_files[reference_idx])

    shifts: list[Tuple[int, int, int]] = []
    for ch_idx, h5_file in enumerate(channel_h5_files):
        if ch_idx == reference_idx:
            shifts.append((0, 0, 0))
            continue
        vol = _coarse_volume(h5_file)
        # normalization=None isn't passed: it's newer-skimage-only and, unlike the
        # raw-float32 case this was needed for elsewhere, these coarse volumes are
        # float64 (headroom against the overflow that motivated it) and the
        # discarded error/diffphase outputs aren't used here anyway.
        shift_coarse, _, _ = phase_cross_correlation(ref_vol, vol, upsample_factor=10)
        # shift_coarse is what moving needs to be shifted BY (via ndimage.shift
        # semantics) to align with the reference. We instead move the *read
        # window*, which needs the opposite sign: the aligned sample for this
        # channel at output position i lives at this channel's own raw index
        # (i - shift), so the window itself must start at (bounds - shift).
        dz = -int(round(shift_coarse[0] * z_stride))
        dy = -int(round(shift_coarse[1] * xy_stride))
        dx = -int(round(shift_coarse[2] * xy_stride))
        print(f"    ch{ch_idx}: window offset (dz,dy,dx) = ({dz}, {dy}, {dx}) voxels", flush=True)
        shifts.append((dz, dy, dx))
    return shifts


def _shift_bounds_clamped(
    bounds: Tuple[int, int, int, int, int, int],
    shift: Tuple[int, int, int],
    full_shape: Tuple[int, int, int],
) -> Tuple[int, int, int, int, int, int]:
    """Offset a (z0,z1,y0,y1,x0,x1) window by (dz,dy,dx), sliding it back
    within [0, full_shape) if the shift would push it out of bounds, rather
    than shrinking it — keeps every channel's window the same size so their
    outputs stay the same shape."""
    z0, z1, y0, y1, x0, x1 = bounds
    dz, dy, dx = shift
    nz_full, ny_full, nx_full = full_shape

    sz0, sz1 = z0 + dz, z1 + dz
    if sz0 < 0:
        sz1 -= sz0; sz0 = 0
    if sz1 > nz_full:
        sz0 -= (sz1 - nz_full); sz1 = nz_full

    sy0, sy1 = y0 + dy, y1 + dy
    if sy0 < 0:
        sy1 -= sy0; sy0 = 0
    if sy1 > ny_full:
        sy0 -= (sy1 - ny_full); sy1 = ny_full

    sx0, sx1 = x0 + dx, x1 + dx
    if sx0 < 0:
        sx1 -= sx0; sx0 = 0
    if sx1 > nx_full:
        sx0 -= (sx1 - nx_full); sx1 = nx_full

    return (max(0, sz0), sz1, max(0, sy0), sy1, max(0, sx0), sx1)


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

    crop_bounds may be a single (z0,z1,y0,y1,x0,x1) tuple shared by every
    channel (the original behavior), or a list of one such tuple per channel
    — the latter is how drift correction is applied: each channel reads from
    its own, independently-offset window (see compute_channel_drift /
    _shift_bounds_clamped) so the outputs line up spatially despite the
    channels having drifted relative to each other during acquisition. All
    per-channel windows must be the same size (only their position differs).
    """
    np_dtype = np.dtype(dtype)
    n_channels = len(channel_h5_files)

    with h5py.File(channel_h5_files[0], "r") as f:
        nz_full, ny_full, nx_full = _effective_zyx_shape(_get_h5_dataset(f, dataset_path))

    if crop_bounds is None:
        per_channel_bounds = [(0, nz_full, 0, ny_full, 0, nx_full)] * n_channels
    elif isinstance(crop_bounds[0], (tuple, list)):
        per_channel_bounds = list(crop_bounds)
        if len(per_channel_bounds) != n_channels:
            raise ValueError(f"crop_bounds has {len(per_channel_bounds)} entries, expected {n_channels}")
    else:
        per_channel_bounds = [tuple(crop_bounds)] * n_channels

    cz0, cz1, cy0, cy1, cx0, cx1 = per_channel_bounds[0]
    nz = cz1 - cz0
    ny = cy1 - cy0
    nx = cx1 - cx0
    for b in per_channel_bounds[1:]:
        if (b[1] - b[0], b[3] - b[2], b[5] - b[4]) != (nz, ny, nx):
            raise ValueError(f"Per-channel crop windows must all be the same size: {per_channel_bounds}")

    # Open all channel files up front so we can interleave channels per Z-plane.
    # OME-TIFF (ome=True) gives napari explicit axis labels so it always splits
    # channels correctly regardless of viewer version.
    h5_handles = [h5py.File(str(h5_file), "r") for h5_file in channel_h5_files]
    try:
        datasets = []
        for ch_idx, (h5_file, h5f) in enumerate(zip(channel_h5_files, h5_handles)):
            ds = _get_h5_dataset(h5f, dataset_path)
            nz_ch, ny_ch, nx_ch = _effective_zyx_shape(ds)
            if (nz_ch, ny_ch, nx_ch) != (nz_full, ny_full, nx_full):
                raise ValueError(
                    f"Shape mismatch: expected {(nz_full, ny_full, nx_full)}, "
                    f"got {(nz_ch, ny_ch, nx_ch)} in {h5_file}"
                )
            datasets.append(ds)

        # ImageJ hyperstack description written on the first page.
        # Pages must be in TZC order (Z-major: z0c0, z0c1, z1c0, ...) so that
        # tifffile reconstructs the (Z, C, Y, X) series correctly on read.
        imagej_desc = (
            "ImageJ=1.11a\n"
            f"images={n_channels * nz}\n"
            f"channels={n_channels}\n"
            f"slices={nz}\n"
            "hyperstack=true\n"
            "mode=composite\n"
        )
        first_page = True
        with tifffile.TiffWriter(str(output_path), bigtiff=True) as tif:
            for zi in range(nz):
                if zi % 50 == 0:
                    print(f"  z {zi}/{nz} ({n_channels} ch)...", flush=True)
                for ds, (bz0, _bz1, by0, by1, bx0, bx1) in zip(datasets, per_channel_bounds):
                    sl = np.asarray(_read_zslice(ds, bz0 + zi)[by0:by1, bx0:bx1], dtype=np_dtype)
                    if first_page:
                        tif.write(sl, contiguous=True, description=imagej_desc)
                        first_page = False
                    else:
                        tif.write(sl, contiguous=True)
    finally:
        for h5f in h5_handles:
            h5f.close()


def load_hyperstack_czyx(path: Path) -> Tuple[np.ndarray, dict]:
    """Read a TIFF written by write_tiff_czyx_streaming back out as (Z, C, Y, X).

    Older tifffile versions' series reader didn't recognize the ImageJ
    hyperstack metadata written above (it saw a flat, unlabeled page stack),
    so `tifffile.imread(path)`/`tifffile.memmap(path)` silently returned a 3D
    (pages, Y, X) array that needed manually reshaping via the `channels`/
    `slices` counts from the embedded ImageJ description tag. Newer tifffile
    (confirmed as of 2026.3.3) recognizes the series itself and hands back an
    already-correct 4D (Z, C, Y, X) array directly — reshaping that using the
    old pages-based math corrupts it (arr.shape[0] is now Z, not the page
    count), so both cases are handled here based on what memmap returns.

    Uses tifffile.memmap rather than a full read, so multi-GB files aren't
    loaded into RAM just to be looked at.

    Returns (array, imagej_metadata). Raises ValueError if the file wasn't
    written by write_tiff_czyx_streaming (no ImageJ channel metadata, or the
    shape tifffile returns doesn't factor into channels * slices).
    """
    with tifffile.TiffFile(str(path)) as tif:
        meta = tif.imagej_metadata

    arr = tifffile.memmap(str(path))  # (pages, Y, X) or, on newer tifffile, already (Z, C, Y, X)

    if not meta or "channels" not in meta:
        raise ValueError(
            f"No ImageJ channel metadata found in {path}. (shape {arr.shape}). "
            "This file wasn't written by write_tiff_czyx_streaming."
        )

    n_channels = int(meta["channels"])

    if arr.ndim == 4:
        # tifffile already parsed the ImageJ hyperstack series -- nothing to reshape.
        if arr.shape[1] != n_channels:
            raise ValueError(
                f"tifffile returned {arr.ndim}D shape {arr.shape} whose channel axis "
                f"doesn't match metadata channels ({n_channels}) in {path}."
            )
        return arr, meta

    # Flat (pages, Y, X): pages are z-major, channel-fastest (page index = z * n_channels + c).
    n_slices = int(meta.get("slices", arr.shape[0] // n_channels))
    if n_channels * n_slices != arr.shape[0]:
        raise ValueError(
            f"channels ({n_channels}) * slices ({n_slices}) != page count ({arr.shape[0]}) in {path}."
        )
    arr = arr.reshape(n_slices, n_channels, *arr.shape[1:])
    return arr, meta


def compute_channel_drift_from_array(
    arr: np.ndarray, reference_idx: int = 0, max_shift_frac: float = 0.15
) -> list[Tuple[int, int, int]]:
    """Like compute_channel_drift, but for an already-assembled (Z, C, Y, X)
    array (e.g. from load_hyperstack_czyx) instead of separate raw per-channel
    HDF5 files. Use this to correct drift after the fact on data whose raw
    HDF5 source is no longer around — the array is already cropped down from
    the full sensor frame, so unlike compute_channel_drift this reads each
    channel at full resolution rather than a coarse strided sub-sample; it's
    cheap enough at this size and more accurate.

    Whole-volume phase correlation between two channels with genuinely
    different spatial content (e.g. a nuclear stain vs. a membrane marker)
    occasionally locks onto a spurious peak instead of the true small stage
    drift, producing an implausible triple/quadruple-digit-voxel "shift" —
    seen in practice on this dataset, e.g. (dz,dy,dx) = (30, -542, -180)
    where every other channel in the same stack landed under 50 voxels. Since
    _shift_bounds_clamped grows the crop margin to fit the largest shift on
    each axis, one bad estimate silently collapses the whole file's crop
    (one file shrank to 12 px tall). Any axis whose |shift| exceeds
    max_shift_frac of that axis's own array size is treated as a failed
    estimate: the whole channel's shift is zeroed (left uncorrected) rather
    than trusted, and a warning is printed so the file/channel can be
    checked by hand if needed.
    """
    from skimage.registration import phase_cross_correlation

    n_channels = arr.shape[1]
    nz, ny, nx = arr.shape[0], arr.shape[2], arr.shape[3]
    max_shift_zyx = (max_shift_frac * nz, max_shift_frac * ny, max_shift_frac * nx)
    print(f"  Estimating channel drift (reference=ch{reference_idx})...", flush=True)
    ref_vol = np.asarray(arr[:, reference_idx], dtype=np.float64)

    shifts: list[Tuple[int, int, int]] = []
    for c in range(n_channels):
        if c == reference_idx:
            shifts.append((0, 0, 0))
            continue
        vol = np.asarray(arr[:, c], dtype=np.float64)
        shift, _, _ = phase_cross_correlation(ref_vol, vol, upsample_factor=10)
        # Same sign flip as compute_channel_drift: phase_cross_correlation
        # returns what the moving image needs to be shifted BY to align; the
        # window instead needs to be shifted by the opposite amount.
        dz, dy, dx = (-int(round(s)) for s in shift)
        print(f"    ch{c}: window offset (dz,dy,dx) = ({dz}, {dy}, {dx}) voxels", flush=True)
        if any(abs(d) > cap for d, cap in zip((dz, dy, dx), max_shift_zyx)):
            print(
                f"    ch{c}: WARNING shift ({dz}, {dy}, {dx}) exceeds "
                f"{max_shift_frac:.0%} of array size {(nz, ny, nx)} — treating as a "
                "failed estimate and leaving this channel uncorrected (0, 0, 0)."
            )
            shifts.append((0, 0, 0))
            continue
        shifts.append((dz, dy, dx))
    return shifts


def write_tiff_czyx_from_array(
    output_path: Path, arr: np.ndarray, per_channel_bounds: list, dtype: str
) -> None:
    """Write a CZYX TIFF from an already-assembled (Z, C, Y, X) array (e.g.
    from load_hyperstack_czyx), one (Y, X) plane at a time, with each channel
    read from its own crop window — the array-source counterpart to
    write_tiff_czyx_streaming, for the same drift-correction-after-the-fact
    case as compute_channel_drift_from_array. per_channel_bounds is a list of
    one (z0,z1,y0,y1,x0,x1) window per channel, all the same size.
    """
    np_dtype = np.dtype(dtype)
    n_channels = arr.shape[1]
    if len(per_channel_bounds) != n_channels:
        raise ValueError(f"per_channel_bounds has {len(per_channel_bounds)} entries, expected {n_channels}")

    bz0, bz1, by0, by1, bx0, bx1 = per_channel_bounds[0]
    nz, ny, nx = bz1 - bz0, by1 - by0, bx1 - bx0
    for b in per_channel_bounds[1:]:
        if (b[1] - b[0], b[3] - b[2], b[5] - b[4]) != (nz, ny, nx):
            raise ValueError(f"Per-channel crop windows must all be the same size: {per_channel_bounds}")

    imagej_desc = (
        "ImageJ=1.11a\n"
        f"images={n_channels * nz}\n"
        f"channels={n_channels}\n"
        f"slices={nz}\n"
        "hyperstack=true\n"
        "mode=composite\n"
    )
    first_page = True
    with tifffile.TiffWriter(str(output_path), bigtiff=True) as tif:
        for zi in range(nz):
            if zi % 50 == 0:
                print(f"  z {zi}/{nz} ({n_channels} ch)...", flush=True)
            for c, (cbz0, _cbz1, cby0, cby1, cbx0, cbx1) in enumerate(per_channel_bounds):
                sl = np.asarray(arr[cbz0 + zi, c, cby0:cby1, cbx0:cbx1], dtype=np_dtype)
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


def detect_h5_layout(root_dir: Path) -> tuple[str, dict]:
    """Inspect root_dir and figure out which conversion strategy applies.

    Returns (mode, info):
      "flat"       - root_dir directly contains stack_*_channel_* folders, one h5 file each
                     (single fixed-sample acquisition) -> convert_h5_channels_to_tiff.py
      "wells"      - root_dir contains subfolders that each have their own raw/ folder of
                     stack_*_channel_* folders (fixed samples imaged well-by-well)
      "timecourse" - channel folders contain multiple h5 files (one per timepoint) ->
                     this is a live-imaging layout, not a fixed IF one; use
                     convert_h5_timecourse_to_tiff.py instead
      "unknown"    - nothing recognizable found
    """
    flat_groups = build_stack_groups(root_dir)
    if flat_groups:
        sample_folder = next(iter(flat_groups.values()))[0][1]
        h5_files = list(sample_folder.glob("*.h5")) + list(sample_folder.glob("*.hdf5"))
        if len(h5_files) > 1:
            return "timecourse", {"stack_ids": sorted(flat_groups.keys())}
        return "flat", {"stack_ids": sorted(flat_groups.keys())}

    wells = {}
    for child in sorted(root_dir.iterdir()):
        if child.is_dir() and (child / "raw").is_dir():
            groups = build_stack_groups(child / "raw")
            if groups:
                wells[child.name] = sorted(groups.keys())
    if wells:
        return "wells", {"wells": wells}

    return "unknown", {}


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
