"""Convert live-imaging HDF5 folders to TCZYX TIFF hyperstacks.

Folder layout expected under root_dir:
    stack_N[-label]_channel_M[_suffix]/
        Cam_short_00000.lux.h5   # timepoint 0
        Cam_short_00001.lux.h5   # timepoint 1
        ...

Each h5 file is one 3-D volume (Z, Y, X) for that timepoint and channel.
Folders with the same stack_N prefix are grouped as channels of the same embryo.
One TCZYX BigTIFF is written per embryo.

Crop bounds (--auto-crop or --crop) are computed once from the first and last
timepoints (union of both boxes, covering drift) and applied uniformly across all T.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

# ── Dev override ──────────────────────────────────────────────────────────────
# Set this to run the script directly (play button / F5) without CLI args.
# Set to None to require CLI args instead.
DEV_CONFIG = Path("configs/other live images/20260519_mtmg_fgf_e45.yaml")
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.log import log_conversion
from src.conversion import (
    build_live_stack_groups,
    compute_autocrop_bounds_streaming,
    find_h5_files_sorted,
    get_config_value,
    load_yaml_config,
    parse_crop_arg,
    write_tiff_tczyx_streaming,
    write_tiff_per_timepoint_streaming,
    _effective_zyx_shape,
    _get_h5_dataset,
)
import h5py


def _get_live_config(config):
    if config is None:
        return {}
    val = config.get("live_timecourse")
    return val if isinstance(val, dict) else {}


def _autocrop_from_timepoints(
    channel_timepoint_files, timepoint_indices, dataset_path,
    pad, threshold, threshold_percentile, blur_sigma, auto_crop_channel,
):
    """Compute crop bounds as the spatial union over the given timepoint indices.

    The union ensures the fixed crop window covers the embryo at every sampled
    timepoint, accommodating drift across the movie.
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert live-imaging HDF5 folders to TCZYX TIFF hyperstacks."
    )
    parser.add_argument("root_dir", type=Path, nargs="?",
                        help="Folder containing stack_N_channel_M subdirectories.")
    parser.add_argument("output_dir", type=Path, nargs="?",
                        help="Directory to write TIFF files.")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config file (live_timecourse section).")
    parser.add_argument("--dataset-path", default=None,
                        help="HDF5 dataset path inside each file (auto-detected if omitted).")
    parser.add_argument("--auto-crop", action="store_true",
                        help="Compute a spatial crop from the first and last timepoints and apply to all T.")
    parser.add_argument("--auto-crop-n-timepoints", type=int, default=None,
                        help="Use N evenly-spaced timepoints for autocrop instead of the default first+last.")
    parser.add_argument("--auto-crop-channel", type=int, default=None,
                        help="Channel index to use for auto-crop (default: all channels).")
    parser.add_argument("--auto-crop-threshold", type=float, default=None,
                        help="Absolute intensity threshold for auto-crop signal detection.")
    parser.add_argument("--auto-crop-threshold-percentile", type=float, default=None,
                        help="MIP percentile threshold for auto-crop (overrides --auto-crop-threshold).")
    parser.add_argument("--auto-crop-blur-sigma", type=float, default=None,
                        help="Gaussian blur sigma applied to MIP before thresholding.")
    parser.add_argument("--pad", type=int, default=None,
                        help="Padding (voxels) added around auto-crop bounds.")
    parser.add_argument("--crop", type=str, default=None,
                        help="Manual crop as y0:y1:x0:x1 or z0:z1:y0:y1:x0:x1.")
    parser.add_argument("--dtype", default=None,
                        help="Output TIFF dtype (default: uint16).")
    parser.add_argument("--per-timepoint", action="store_true",
                        help="Write one ZYX/CZYX TIFF per timepoint into a subfolder "
                             "instead of a single TCZYX mega-TIFF. "
                             "Easier to load in Napari for large datasets.")
    parser.add_argument("--stacks", nargs="+", default=None,
                        help="Process only these stack IDs (default: all).")
    args = parser.parse_args()

    # Config loading
    config_path = args.config or DEV_CONFIG

    config = load_yaml_config(config_path) if config_path else None
    live_cfg = _get_live_config(config)

    def _cv(keys, default=None):
        return get_config_value(live_cfg, keys if isinstance(keys, list) else [keys], default)

    root_dir = (
        args.root_dir
        or (Path(v) if (v := _cv("root_dir")) else None)
    )
    output_dir = (
        args.output_dir
        or (Path(v) if (v := _cv("output_dir")) else None)
    )
    dataset_path = args.dataset_path or _cv("dataset_path")
    auto_crop = args.auto_crop or _cv("auto_crop", False)
    n_tp_for_crop = (
        args.auto_crop_n_timepoints
        if args.auto_crop_n_timepoints is not None
        else _cv("auto_crop_n_timepoints")  # None means use first+last
    )
    auto_crop_channel = (
        args.auto_crop_channel
        if args.auto_crop_channel is not None
        else _cv("auto_crop_channel")
    )
    threshold = (
        args.auto_crop_threshold
        if args.auto_crop_threshold is not None
        else _cv("auto_crop_threshold", 0)
    )
    threshold_percentile = (
        args.auto_crop_threshold_percentile
        if args.auto_crop_threshold_percentile is not None
        else _cv("auto_crop_threshold_percentile")
    )
    blur_sigma = (
        args.auto_crop_blur_sigma
        if args.auto_crop_blur_sigma is not None
        else _cv("auto_crop_blur_sigma", 0)
    )
    pad = args.pad if args.pad is not None else _cv("pad", 0)
    crop = args.crop or _cv("crop")
    dtype = args.dtype or _cv("dtype", "uint16")
    per_timepoint = args.per_timepoint or _cv("per_timepoint", False)
    stacks_include = args.stacks or (_cv("stacks") and list(_cv("stacks")))
    skip_stacks = set(_cv("skip_stacks") or [])

    if root_dir is None or output_dir is None:
        parser.error("root_dir and output_dir must be provided via CLI or config.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load per-stack crop overrides (same format as fixed pipeline)
    overrides_path = config_path.parent / "crop_overrides.yaml" if config_path else None
    crop_overrides: dict = {}
    if overrides_path and overrides_path.exists():
        with open(overrides_path) as _f:
            crop_overrides = yaml.safe_load(_f) or {}
        print(f"Loaded crop overrides for {len(crop_overrides)} stack(s).")

    groups = build_live_stack_groups(root_dir)
    if not groups:
        raise ValueError(f"No channel folders found under {root_dir}. Check folder naming.")

    stack_ids = sorted(groups.keys())
    if stacks_include:
        stack_ids = [s for s in stack_ids if s in stacks_include]
        if not stack_ids:
            raise ValueError(f"None of the requested stacks found. Available: {sorted(groups)}")
    if skip_stacks:
        stack_ids = [s for s in stack_ids if s not in skip_stacks]

    # Determine the expected channel set from the most common across stacks
    all_ch_sets = {sid: frozenset(ci for ci, _ in groups[sid]) for sid in sorted(groups)}
    expected_channels = Counter(all_ch_sets.values()).most_common(1)[0][0]
    print(f"Found {len(stack_ids)} stack(s). Expected channels: {sorted(expected_channels)}")

    bad_stacks = []
    for sid in stack_ids:
        missing = sorted(expected_channels - all_ch_sets[sid])
        extra = sorted(all_ch_sets[sid] - expected_channels)
        if missing or extra:
            bad_stacks.append(sid)
            msg = f"  WARNING: {sid}"
            if missing:
                msg += f" — missing channels {missing}"
            if extra:
                msg += f" — extra channels {extra}"
            print(msg)
    if bad_stacks:
        print(f"{len(bad_stacks)} stack(s) with channel mismatches will be skipped.")

    n_stacks = len(stack_ids)
    for stack_num, stack_id in enumerate(stack_ids, 1):
        if stack_id in bad_stacks:
            print(f"\n[{stack_num}/{n_stacks}] {stack_id} — SKIPPED (channel mismatch)", flush=True)
            continue

        print(f"\n[{stack_num}/{n_stacks}] {stack_id}", flush=True)

        # Build channel_timepoint_files[c][t]
        items_sorted = sorted(groups[stack_id], key=lambda x: x[0])
        channel_timepoint_files = []
        for ch_idx, folder in items_sorted:
            tp_files = find_h5_files_sorted(folder)
            channel_timepoint_files.append(tp_files)
            print(f"  ch{ch_idx}: {len(tp_files)} timepoints in {folder.name}", flush=True)

        # Warn if timepoint counts differ across channels
        tp_counts = [len(f) for f in channel_timepoint_files]
        if len(set(tp_counts)) > 1:
            print(f"  WARNING: unequal timepoint counts across channels {tp_counts}; using minimum.")
        n_tp = min(tp_counts)
        channel_timepoint_files = [files[:n_tp] for files in channel_timepoint_files]

        # Determine crop bounds
        stack_entry = crop_overrides.get(stack_id) or {}
        override_crop_str = stack_entry.get("crop")

        def _parse_crop_override(cs):
            spec = parse_crop_arg(cs)
            if len(spec) == 6:
                return tuple(spec)
            y0o, y1o, x0o, x1o = spec
            with h5py.File(channel_timepoint_files[0][0], "r") as _f:
                nzo, _, _ = _effective_zyx_shape(_get_h5_dataset(_f, dataset_path))
            return (0, nzo, y0o, y1o, x0o, x1o)

        if override_crop_str:
            bounds = _parse_crop_override(override_crop_str)
            z0, z1, y0, y1, x0, x1 = bounds
            print(f"  Crop override: z={z0}:{z1}, y={y0}:{y1}, x={x0}:{x1}", flush=True)

        elif auto_crop:
            if n_tp_for_crop is not None:
                import numpy as np
                tp_indices = [int(i) for i in np.linspace(0, n_tp - 1, n_tp_for_crop, dtype=int)]
                print(f"  Computing autocrop from {n_tp_for_crop} evenly-spaced timepoints: {tp_indices}", flush=True)
            else:
                tp_indices = [0, n_tp - 1]
                print(f"  Computing autocrop from first (t=0) and last (t={n_tp - 1}) timepoints...", flush=True)
            bounds = _autocrop_from_timepoints(
                channel_timepoint_files, tp_indices, dataset_path,
                pad, threshold, threshold_percentile, blur_sigma, auto_crop_channel,
            )
            z0, z1, y0, y1, x0, x1 = bounds
            print(f"  Autocrop bounds: z={z0}:{z1}, y={y0}:{y1}, x={x0}:{x1}", flush=True)

        elif crop:
            spec = parse_crop_arg(crop)
            if len(spec) == 4:
                y0c, y1c, x0c, x1c = spec
                with h5py.File(channel_timepoint_files[0][0], "r") as _f:
                    nz, _, _ = _effective_zyx_shape(_get_h5_dataset(_f, dataset_path))
                bounds = (0, nz, y0c, y1c, x0c, x1c)
            else:
                bounds = tuple(spec)

        else:
            bounds = None

        out_name = stack_id.replace(" ", "_")
        if per_timepoint:
            tp_dir = output_dir / out_name
            print(f"  Writing per-timepoint TIFFs ({n_tp}t × {len(channel_timepoint_files)}c) → {tp_dir}/", flush=True)
            write_tiff_per_timepoint_streaming(tp_dir, channel_timepoint_files, dataset_path, dtype, bounds)
        else:
            output_path = output_dir / f"{out_name}.tif"
            print(f"  Writing TCZYX TIFF ({n_tp}t × {len(channel_timepoint_files)}c) → {output_path}", flush=True)
            write_tiff_tczyx_streaming(output_path, channel_timepoint_files, dataset_path, dtype, bounds)
        print(f"  Done. [{stack_num}/{n_stacks}]", flush=True)

    _proj = (config or {}).get("project", {})
    dataset_id       = _proj.get("dataset", config_path.stem if config_path else "unknown")
    acquisition_date = _proj.get("date", "")
    condition        = _proj.get("name", "")
    log_conversion(dataset_id, raw_path=str(root_dir), output_path=str(output_dir),
                   n_stacks=len(stack_ids), acquisition_date=acquisition_date,
                   condition=condition)


if __name__ == "__main__":
    main()
