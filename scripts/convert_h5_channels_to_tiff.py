import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.conversion import (
    build_stack_groups,
    compute_autocrop_bounds_streaming,
    confirm_autocrop as _confirm_autocrop_fn,
    find_h5_file,
    get_config_value,
    get_h5_conversion_config,
    load_pipeline_raw_dir,
    load_yaml_config,
    parse_crop_arg,
    resolve_relative_path,
    write_tiff_czyx_streaming,
    _effective_zyx_shape,
    _get_h5_dataset,
)
import h5py  # used directly for the manual-crop Z-extent lookup
import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Convert per-channel HDF5 folders into combined multi-channel TIFF stacks."
    )
    parser.add_argument("root_dir", type=Path, nargs="?", help="Root folder containing channel folders.")
    parser.add_argument("output_dir", type=Path, nargs="?", help="Directory to write combined TIFFs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file to define input/output paths and crop settings.",
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=None,
        help="Optional pipeline YAML config file to read raw_data_dir and feed TIFFs directly into the pipeline.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional HDF5 dataset path to load if the file contains multiple datasets.",
    )
    parser.add_argument(
        "--auto-crop",
        action="store_true",
        help="Automatically crop the combined volume to non-zero data across all channels.",
    )
    parser.add_argument(
        "--confirm-autocrop",
        action="store_true",
        help="Ask for approval before saving each auto-cropped TIFF.",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=None,
        help="Padding to add around auto-crop bounds.",
    )
    parser.add_argument(
        "--crop",
        type=str,
        default=None,
        help=(
            "Manual crop bounds as y0:y1:x0:x1 or z0:z1:y0:y1:x0:x1. "
            "If 4 values are given, all z slices are kept."
        ),
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Target TIFF dtype (default: float32)."
    )
    args = parser.parse_args()

    default_config_path = None
    for candidate in [Path("configs/config.yaml"), Path("config.yaml")]:
        if candidate.exists():
            default_config_path = candidate
            break
    if default_config_path is None:
        found = list(Path("configs").glob("**/config.yaml")) if Path("configs").exists() else []
        if len(found) == 1:
            default_config_path = found[0]
        elif len(found) > 1:
            print(f"Multiple config.yaml files found: {found}. Use --config to specify one.")
            sys.exit(1)

    config_path = args.config or default_config_path
    config = load_yaml_config(config_path) if config_path else None
    config_base = config_path.parent if config_path else Path.cwd()
    h5_config = get_h5_conversion_config(config)

    pipeline_config_path = (
        args.pipeline_config
        or get_config_value(h5_config, ["pipeline_config"])
        or get_config_value(config, ["pipeline_config"])
    )
    if pipeline_config_path is not None:
        pipeline_config_path = resolve_relative_path(config_base, pipeline_config_path)

    root_dir = (
        args.root_dir
        or (Path(v) if (v := get_config_value(h5_config, ["root_dir"])) else None)
        or (Path(v) if (v := get_config_value(config, ["input", "root_dir"])) else None)
        or (Path(v) if (v := get_config_value(config, ["root_dir"])) else None)
    )
    output_dir = (
        args.output_dir
        or get_config_value(h5_config, ["output_dir"])
        or get_config_value(config, ["raw_data_dir"])
    )
    dataset_path = (
        args.dataset_path
        or get_config_value(h5_config, ["dataset_path"])
        or get_config_value(config, ["input", "dataset_path"])
    )
    auto_crop = args.auto_crop or get_config_value(h5_config, ["auto_crop"], False)
    pad = args.pad if args.pad is not None else get_config_value(h5_config, ["pad"], 0)
    crop = args.crop if args.crop is not None else get_config_value(h5_config, ["crop"])
    do_confirm_autocrop = args.confirm_autocrop or get_config_value(h5_config, ["confirm_autocrop"], False)
    dtype = args.dtype or get_config_value(h5_config, ["dtype"], "float32")
    channel_names = get_config_value(config, ["microscopy", "channel_names"])
    auto_crop_threshold = get_config_value(h5_config, ["auto_crop_threshold"], 0)
    auto_crop_threshold_percentile = get_config_value(h5_config, ["auto_crop_threshold_percentile"], None)
    auto_crop_blur_sigma = get_config_value(h5_config, ["auto_crop_blur_sigma"], 0)
    auto_crop_channel = get_config_value(h5_config, ["auto_crop_channel"], None)

    if output_dir is None and pipeline_config_path is not None:
        output_dir = load_pipeline_raw_dir(pipeline_config_path)

    if root_dir is None or output_dir is None:
        raise ValueError(
            "root_dir and output_dir must be specified either on the command line, in the config file, or via pipeline_config."
        )

    output_dir = Path(output_dir)

    if pipeline_config_path is not None and args.output_dir is not None:
        pipeline_raw = load_pipeline_raw_dir(pipeline_config_path)
        if Path(args.output_dir).resolve() != pipeline_raw.resolve():
            print(
                f"Warning: output_dir {args.output_dir} differs from pipeline raw_data_dir {pipeline_raw}. "
                "The pipeline will not automatically use this output unless output_dir matches the pipeline config."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load per-stack crop overrides written by inspect_crop_bounds.py
    overrides_path = config_path.parent / "crop_overrides.yaml" if config_path else None
    crop_overrides: dict = {}
    if overrides_path and overrides_path.exists():
        with open(overrides_path) as _f:
            crop_overrides = yaml.safe_load(_f) or {}
        print(f"Loaded crop overrides for {len(crop_overrides)} stack(s) from {overrides_path}")

    groups = build_stack_groups(root_dir)
    if not groups:
        raise ValueError(f"No channel folders found under {args.root_dir}. Check folder naming.")

    stack_ids = sorted(groups.keys())
    n_stacks = len(stack_ids)
    print(f"Found {n_stacks} stack(s) to convert.")

    # Determine expected channel indices from the most common set across stacks.
    all_channel_sets = {sid: frozenset(ci for ci, _ in groups[sid]) for sid in stack_ids}
    from collections import Counter
    expected_channels = Counter(all_channel_sets.values()).most_common(1)[0][0]
    print(f"Expected channels: {sorted(expected_channels)}")

    # Warn about any stacks that deviate.
    bad_stacks = []
    for sid in stack_ids:
        ch_set = all_channel_sets[sid]
        missing = sorted(expected_channels - ch_set)
        extra   = sorted(ch_set - expected_channels)
        if missing or extra:
            bad_stacks.append(sid)
            msg = f"  WARNING: {sid}"
            if missing:
                msg += f" — missing channels {missing}"
            if extra:
                msg += f" — unexpected extra channels {extra}"
            print(msg)
    if bad_stacks:
        print(f"\n{len(bad_stacks)} stack(s) have channel mismatches (listed above). "
              "They will be skipped to avoid silently corrupt output.")

    for stack_num, stack_id in enumerate(stack_ids, 1):
        items = groups[stack_id]
        items_sorted = sorted(items, key=lambda x: x[0])
        if not items_sorted:
            continue

        if stack_id in bad_stacks:
            print(f"\n[{stack_num}/{n_stacks}] {stack_id} — SKIPPED (channel mismatch)", flush=True)
            continue

        print(f"\n[{stack_num}/{n_stacks}] {stack_id}", flush=True)
        channel_h5_files = [find_h5_file(folder) for _, folder in items_sorted]

        # ------------------------------------------------------------------
        # Determine crop bounds — all reads are one Z-slice at a time so the
        # peak in-memory footprint is a single (Y, X) plane (~21 MB here).
        #
        # crops_to_write: list of (output_stem, crop_bounds_or_None)
        # Multiple entries arise when inspect_crop_bounds saved several boxes
        # for the same stack (e.g. two embryos in one field of view).
        # ------------------------------------------------------------------
        stack_entry = crop_overrides.get(stack_id) or {}
        override_crops_list = stack_entry.get("crops")   # list of crop strings
        override_crop_str   = stack_entry.get("crop")    # single crop string

        def _parse_crop_str(cs):
            spec = parse_crop_arg(cs)
            if len(spec) == 6:
                return tuple(spec)
            y0o, y1o, x0o, x1o = spec
            with h5py.File(channel_h5_files[0], "r") as _f:
                nzo, _, _ = _effective_zyx_shape(_get_h5_dataset(_f, dataset_path))
            return (0, nzo, y0o, y1o, x0o, x1o)

        if override_crops_list:
            crops_to_write = []
            for idx, cs in enumerate(override_crops_list, 1):
                bounds = _parse_crop_str(cs)
                z0, z1, y0, y1, x0, x1 = bounds
                print(f"  Embryo {idx}: z={z0}:{z1}, y={y0}:{y1}, x={x0}:{x1}", flush=True)
                crops_to_write.append((f"{stack_id}_embryo{idx}", bounds))

        elif override_crop_str:
            bounds = _parse_crop_str(override_crop_str)
            z0, z1, y0, y1, x0, x1 = bounds
            print(f"  Using manual override: z={z0}:{z1}, y={y0}:{y1}, x={x0}:{x1}", flush=True)
            crops_to_write = [(stack_id, bounds)]

        elif auto_crop:
            if auto_crop_channel is not None:
                crop_files = [channel_h5_files[auto_crop_channel]]
                ch_label = (channel_names[auto_crop_channel]
                            if channel_names and auto_crop_channel < len(channel_names)
                            else f"ch{auto_crop_channel}")
                print(f"  Auto-cropping using {ch_label} (threshold={auto_crop_threshold})...", flush=True)
            else:
                crop_files = channel_h5_files
                print(f"  Auto-cropping all channels (threshold={auto_crop_threshold})...", flush=True)
            bounds = compute_autocrop_bounds_streaming(
                crop_files, dataset_path, pad=pad,
                threshold=auto_crop_threshold or 0,
                threshold_percentile=auto_crop_threshold_percentile,
                blur_sigma=auto_crop_blur_sigma,
            )
            z0, z1, y0, y1, x0, x1 = bounds
            print(f"  Crop bounds: z={z0}:{z1}, y={y0}:{y1}, x={x0}:{x1}", flush=True)

            if do_confirm_autocrop:
                with h5py.File(channel_h5_files[0], "r") as _f:
                    ref_shape = _effective_zyx_shape(_get_h5_dataset(_f, dataset_path))
                full_shape = (len(channel_h5_files), *ref_shape)
                cropped_shape = (len(channel_h5_files), z1 - z0, y1 - y0, x1 - x0)
                if not _confirm_autocrop_fn(
                    stack_id, bounds, full_shape, cropped_shape,
                    channel_h5_files=channel_h5_files,
                    dataset_path=dataset_path,
                    channel_names=channel_names,
                ):
                    print(f"Skipped saving {stack_id} because auto-crop was not approved.")
                    continue

            crops_to_write = [(stack_id, bounds)]

        elif crop:
            crop_spec = parse_crop_arg(crop)
            if len(crop_spec) == 4:
                y0c, y1c, x0c, x1c = crop_spec
                with h5py.File(channel_h5_files[0], "r") as _f:
                    nz, _, _ = _effective_zyx_shape(_get_h5_dataset(_f, dataset_path))
                bounds = (0, nz, y0c, y1c, x0c, x1c)
            else:
                bounds = tuple(crop_spec)
            crops_to_write = [(stack_id, bounds)]

        else:
            crops_to_write = [(stack_id, None)]

        # ------------------------------------------------------------------
        # Write: streams one (Y, X) slice at a time from h5py to tifffile.
        # Peak memory ≈ one Z-plane regardless of number of channels or depth.
        # ------------------------------------------------------------------
        for out_stem, crop_bounds in crops_to_write:
            output_path = output_dir / f"{out_stem}.tif"
            print(f"  Writing TIFF → {output_path}", flush=True)
            write_tiff_czyx_streaming(output_path, channel_h5_files, dataset_path, dtype, crop_bounds)
        print(f"  Done. [{stack_num}/{n_stacks}]", flush=True)


if __name__ == "__main__":
    main()
