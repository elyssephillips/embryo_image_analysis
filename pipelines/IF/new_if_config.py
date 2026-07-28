"""Generate configs/IF/config.yaml for a new dataset from the standard template.

Every new IF dataset needs the same ~10 fields updated together (raw_data_dir,
rotated_dir, segmentation_dir, segmentation_dir_raw, output_dir, rotation_log,
datasets, h5_conversion.root_dir, ...). Doing this by hand via copy-paste is
how stale fields (wrong dataset name, wrong rotation_log path, mismatched
channel_names) end up left over from the previous dataset. This script fills
in everything that's mechanically derivable from a dataset name + base path,
carries forward the h5_conversion/microscopy tuning from the current config,
archives the current config.yaml first (so it isn't lost), and leaves clearly
marked TODOs for the handful of things that genuinely require a human
(channel identity, voxel size, which metadata JSON to point at).

This is the first step for any new dataset: fill in the settings below, run it,
then fill in the TODOs it leaves in configs/IF/config.yaml, then run
pipelines/IF/convert_h5_to_tiff.py (it auto-detects whether this dataset needs
the flat or well-by-well conversion path — no need to choose). Every step gets
logged to logs/IF.md automatically.


"""
import re
import sys
from datetime import date
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.conversion import detect_h5_layout
from src.log import log_run

CONFIG_DIR = PROJECT_ROOT / "configs" / "IF"
ACTIVE_CONFIG = CONFIG_DIR / "config.yaml"

# ============================== EDIT THESE ==============================
DATASET_NAME = "20260604_meki_cdx2_ppmlc_gata3"          # e.g. "20260604_fixed" — required
BASE_DIR = "/mnt/md1/elysse/20260604_fixed"              # e.g. "/mnt/md1/elysse/20260604_fixed" — required
H5_ROOT = None             # h5_conversion.root_dir; defaults to BASE_DIR if left None
NAME = None                # human-readable experiment name; defaults to carrying over the previous config's name
# ==========================================================================

TEMPLATE = """name: "{name}"

raw_data_dir: "{raw_data_dir}"
rotated_dir: "{rotated_dir}"
segmentation_dir: "{segmentation_dir}"
segmentation_dir_raw: "{segmentation_dir_raw}"
output_dir: "{output_dir}"
metadata_json: "/mnt/md1/elysse/20260604_fixed/2026-06-04_121950/raw/stack_0-ctrl 1 2 3_channel_0-DAPI_obj_bottom/Cam_short_00000.json"  # TODO: point at a Cam_*.json from this dataset's DAPI channel folder
rotation_log: "{rotation_log}"

file_extension: ".tif"

datasets: "{datasets}"

microscopy:
  voxel_size_zyx: {voxel_size_zyx}  # TODO: verify against this dataset's Cam_*.json processingInformation.voxel_size_um
  channels:
{channels_block}
  channel_names: {channel_names}  # TODO: verify these are the actual stains for THIS dataset, not carried over from the template

single_file_mode: false

# Optional HDF5 conversion settings. Add these to the same config file for a pipeline-style setup.
h5_conversion:
  root_dir: "{h5_root_dir}"
  dataset_path: null  # optional HDF5 dataset path, e.g. "/images/data"

  auto_crop: {auto_crop}
  auto_crop_channel: {auto_crop_channel}           # channel index for crop bounds (null = all channels; 0 = DAPI)
  auto_crop_threshold: {auto_crop_threshold}      # absolute intensity cutoff (null = use threshold_percentile instead)
  auto_crop_threshold_percentile: {auto_crop_threshold_percentile}  # percentile of the DAPI MIP to use as threshold
  auto_crop_blur_sigma: {auto_crop_blur_sigma}       # gaussian blur applied to MIP before thresholding
  pad: {pad}
  crop: null  # manual crop: y0:y1:x0:x1 or z0:z1:y0:y1:x0:x1
  dtype: "{dtype}"
  confirm_autocrop: {confirm_autocrop}
"""

DEFAULT_H5_CONVERSION = {
    "auto_crop": True,
    "auto_crop_channel": 0,
    "auto_crop_threshold": None,
    "auto_crop_threshold_percentile": 90,
    "auto_crop_blur_sigma": 10,
    "pad": 150,
    "dtype": "float32",
    "confirm_autocrop": True,
}


def yaml_scalar(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def yaml_list(v):
    parts = [f'"{x}"' if isinstance(x, str) else str(x) for x in v]
    return "[" + ", ".join(parts) + "]"


def main():
    if not DATASET_NAME or not BASE_DIR:
        raise ValueError("Set DATASET_NAME and BASE_DIR at the top of this file before running.")

    base_dir = BASE_DIR.rstrip("/")
    h5_root = (H5_ROOT or base_dir).rstrip("/")

    template_source = {}
    if ACTIVE_CONFIG.exists():
        with open(ACTIVE_CONFIG) as f:
            template_source = yaml.safe_load(f) or {}

        if template_source.get("datasets") == DATASET_NAME:
            raise ValueError(
                f"configs/IF/config.yaml is already set up for '{DATASET_NAME}'. Re-running this "
                "script would reset metadata_json back to blank (it's always cleared, unlike "
                "voxel_size_zyx/channel_names which carry over) and could overwrite TODOs you've "
                "already filled in. Edit configs/IF/config.yaml directly instead — this script is "
                "only for bootstrapping a NEW dataset."
            )

        archive_date = None
        old_datasets = template_source.get("datasets", "")
        m = re.match(r"^(\d{8})", str(old_datasets))
        if m:
            archive_date = m.group(1)
        else:
            archive_date = date.today().strftime("%Y%m%d")
        archive_path = CONFIG_DIR / f"config{archive_date}.yaml"
        if not archive_path.exists():
            archive_path.write_text(ACTIVE_CONFIG.read_text())
            print(f"Archived current config.yaml -> {archive_path.relative_to(PROJECT_ROOT)}")
        else:
            print(f"Archive {archive_path.name} already exists, not overwriting; current config.yaml will still be replaced.")

    microscopy = template_source.get("microscopy", {})
    channels = microscopy.get("channels", {"dapi": 0, "gfp": 1, "rfp": 2, "cy5": 3})
    channels_block = "\n".join(f"    {k}: {v}" for k, v in channels.items())

    h5c = {**DEFAULT_H5_CONVERSION, **template_source.get("h5_conversion", {})}

    rendered = TEMPLATE.format(
        name=NAME or template_source.get("name", DATASET_NAME),
        raw_data_dir=f"{base_dir}/cropped",
        rotated_dir=f"{base_dir}/cropped/rotated",
        segmentation_dir=f"{base_dir}/nucleimask/cleaned",
        segmentation_dir_raw=f"{base_dir}/nucleimask",
        output_dir=f"{base_dir}/analysis",
        metadata_json="",
        rotation_log=f"{base_dir}/analysis/rotation_log.csv",
        datasets=DATASET_NAME,
        voxel_size_zyx=yaml_list(microscopy.get("voxel_size_zyx", [1.0, 0.122666664, 0.122666664])),
        channels_block=channels_block,
        channel_names=yaml_list(microscopy.get("channel_names", ["dapi", "ch1", "ch2", "ch3"])),
        h5_root_dir=h5_root,
        auto_crop=yaml_scalar(h5c["auto_crop"]),
        auto_crop_channel=yaml_scalar(h5c["auto_crop_channel"]),
        auto_crop_threshold=yaml_scalar(h5c["auto_crop_threshold"]),
        auto_crop_threshold_percentile=yaml_scalar(h5c["auto_crop_threshold_percentile"]),
        auto_crop_blur_sigma=yaml_scalar(h5c["auto_crop_blur_sigma"]),
        pad=yaml_scalar(h5c["pad"]),
        dtype=h5c["dtype"],
        confirm_autocrop=yaml_scalar(h5c["confirm_autocrop"]),
    )

    ACTIVE_CONFIG.write_text(rendered)
    print(f"Wrote {ACTIVE_CONFIG.relative_to(PROJECT_ROOT)} for dataset '{DATASET_NAME}'.")
    print("Still need to fill in by hand: metadata_json, voxel_size_zyx, channel_names (marked TODO in the file).")

    mode, info = detect_h5_layout(Path(h5_root))
    if mode == "flat":
        print(f"\nDetected layout: flat ({len(info['stack_ids'])} stack(s)). "
              f"Next: if any stack has multiple embryos in one field of view, run "
              f"pipelines/IF/inspect_crop_bounds.py to crop them individually — "
              f"then run pipelines/IF/convert_h5_to_tiff.py.")
    elif mode == "wells":
        print(f"\nDetected layout: wells ({len(info['wells'])} well(s)). "
              f"Next: if any stack has multiple embryos in one field of view, run "
              f"pipelines/IF/inspect_crop_bounds.py to crop them individually — "
              f"then run pipelines/IF/convert_h5_to_tiff.py (it converts each well automatically).")
    elif mode == "timecourse":
        print(f"\nDetected layout: timecourse (live-imaging, not fixed IF). "
              f"Use scripts/convert_h5_timecourse_to_tiff.py instead.")
    else:
        print(f"\nWARNING: couldn't detect a usable HDF5 layout under {h5_root}. "
              f"Check H5_ROOT points at the right folder.")

    log_run("IF", DATASET_NAME, "new_if_config.py",
            output_path=str(ACTIVE_CONFIG), data_path=h5_root, detail="detailed")


if __name__ == "__main__":
    main()
