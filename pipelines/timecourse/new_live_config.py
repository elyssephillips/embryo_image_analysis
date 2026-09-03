"""Generate a configs/other live images/{dataset}.yaml from a live-imaging acquisition.

Bootstraps the same kind of config as configs/other live images/20260519_mtmg_fgf_e45.yaml:
a documentation-style project/microscopy/acquisition/stacks block on top of the
live_timecourse block that pipelines/timecourse/convert_h5_timecourse_to_tiff.py actually reads.

Everything mechanically derivable comes straight from the acquisition's own
Cam_*.json sidecars (voxel size, image size, instrument/serial/firmware,
detection NA/magnification, per-timepoint timestamps -> interval/duration) and
from the stack_N[-label]_channel_M folder names (stack list, per-stack
condition guessed from the "-label" suffix). Auto-crop/dtype/per_timepoint
tuning carries forward from the most recently modified config already in
configs/other live images/, the same way pipelines/IF/new_if_config.py carries
forward h5_conversion tuning. Fields that need a human (channel identities,
conditions with no "-label" to guess from, objective model/immersion) are left
as clearly marked TODOs.

This is the first step for a new live-imaging dataset: fill in the settings
below, run it, fill in the TODOs it leaves in the generated yaml, then run
pipelines/timecourse/convert_h5_timecourse_to_tiff.py against that config (edit its
DEV_CONFIG or pass --config).

To run: edit the settings below, then click VS Code's "Run Python File"
button (or Ctrl+F5) — no terminal args needed.
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import h5py
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.conversion import build_live_stack_groups, find_h5_files_sorted, find_first_dataset
from src.log import log_run

CONFIG_DIR = PROJECT_ROOT / "configs" / "other live images"
TIMEPOINT_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{6})$")
LABEL_RE = re.compile(r"^stack_[^-]*-(.+)$")

# ============================== EDIT THESE ==============================
DATASET_NAME = "260804_c_meki_h2b_snap_3"  # required
BASE_DIR = "/mnt/md1/elysse/260804_c_meki_h2b_snap/run 3"
ACQUISITION_DIR = None
                          # leave None to auto-detect it (requires exactly one under BASE_DIR)
NAME = None               # human-readable experiment name; defaults to DATASET_NAME
NOTES = ""                # treatment details, litter info, etc.
# ==========================================================================

DEFAULT_LIVE_TIMECOURSE = {
    "dtype": "uint16",
    "per_timepoint": True,
    "auto_crop": True,
    "auto_crop_channel": 0,
    "auto_crop_threshold_percentile": 80,
    "auto_crop_blur_sigma": 2,
    "pad": 20,
}

TEMPLATE = """project:
  name: "{name}"
  date: "{date}"
  dataset: "{dataset}"
  notes: "{notes}"

paths:
  raw_data_dir: "{raw_data_dir}"
  output_dir: "{output_dir}"

microscopy:
  instrument: "{instrument}"
  serial_number: "{serial_number}"
  software_version: "{software_version}"

  # ZYX voxel size in microns (from processingInformation.voxel_size_um)
  voxel_size_zyx_um: {voxel_size_zyx_um}
  image_size_zyx: {image_size_zyx}

  detection_objective:
    model: ""  # TODO: not in Cam_*.json — fill in from the scope's objective config
    magnification: {magnification}
    na: {na}
    immersion: ""  # TODO
    wd_mm: null  # TODO

  channels:
{channels_block}

acquisition:
  n_timepoints: {n_timepoints}
  interval_s: {interval_s}
  interval_min: {interval_min}
  total_duration_h: {total_duration_h}
  start_time: "{start_time}"   # first frame, {reference_stack_id}
  end_time: "{end_time}"     # last frame, {reference_stack_id}

stacks:
{stacks_block}

# ── Timecourse TIFF conversion ────────────────────────────────────────────────
# Run: python pipelines/timecourse/convert_h5_timecourse_to_tiff.py --config "{config_rel_path}"
live_timecourse:
  root_dir: "{root_dir}"
  output_dir: "{live_output_dir}"
  dataset_path: "{dataset_path}"
  dtype: "{dtype}"

  per_timepoint: {per_timepoint}          # write one ZYX TIFF per timepoint per stack subfolder
  skip_stacks: []    # TODO: any stack_ids to exclude, e.g. ["stack_0"]

  # Autocrop: computed from first + last timepoint (union), applied to all T.
  auto_crop: {auto_crop}
  auto_crop_channel: {auto_crop_channel}
  auto_crop_threshold_percentile: {auto_crop_threshold_percentile}
  auto_crop_blur_sigma: {auto_crop_blur_sigma}
  pad: {pad}
"""


def yaml_scalar(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def yaml_list(v):
    parts = [f'"{x}"' if isinstance(x, str) else str(x) for x in v]
    return "[" + ", ".join(parts) + "]"


def find_acquisition_dir(base_dir: Path) -> Path:
    candidates = [d for d in sorted(base_dir.iterdir()) if d.is_dir() and TIMEPOINT_RE.match(d.name)]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            f"No YYYY-MM-DD_HHMMSS acquisition folder found under {base_dir}. "
            "Set ACQUISITION_DIR explicitly at the top of this script."
        )
    raise ValueError(
        f"Multiple acquisition folders found under {base_dir}: {[c.name for c in candidates]}. "
        "Set ACQUISITION_DIR explicitly at the top of this script."
    )


def read_json(folder: Path, cam_json_name: str) -> dict:
    with open(folder / cam_json_name) as f:
        return json.load(f)


def read_timestamp(folder: Path, cam_json_name: str) -> datetime | None:
    try:
        data = read_json(folder, cam_json_name)
        ts = data["metaData"]["microscopeInfo"]["timeStamp"]
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except (KeyError, ValueError, FileNotFoundError, json.JSONDecodeError):
        return None


def slugify(label: str) -> str:
    return re.sub(r"\s+", "_", label.strip().lower())


def natural_stack_key(stack_id: str):
    m = re.match(r"^stack_(\d+)", stack_id)
    return int(m.group(1)) if m else 0


def detect_dataset_path(h5_file: Path) -> str:
    with h5py.File(h5_file, "r") as f:
        if "Data" in f and isinstance(f["Data"], h5py.Dataset):
            return "Data"
        dataset = find_first_dataset(f)
        if dataset is None:
            raise ValueError(f"No dataset found inside {h5_file}")
        print(f"  WARNING: no top-level 'Data' dataset in {h5_file.name}; "
              f"using first dataset found instead: {dataset.name}")
        return dataset.name.lstrip("/")


def load_most_recent_live_timecourse(exclude_name: str) -> dict:
    if not CONFIG_DIR.is_dir():
        return {}
    existing = [p for p in CONFIG_DIR.glob("*.yaml") if p.stem != exclude_name]
    if not existing:
        return {}
    latest = max(existing, key=lambda p: p.stat().st_mtime)
    with open(latest) as f:
        config = yaml.safe_load(f) or {}
    live_cfg = config.get("live_timecourse")
    if not isinstance(live_cfg, dict):
        return {}
    print(f"  Carrying forward live_timecourse tuning from {latest.name}")
    return live_cfg


def main():
    if not DATASET_NAME or not BASE_DIR:
        raise ValueError("Set DATASET_NAME and BASE_DIR at the top of this file before running.")

    config_path = CONFIG_DIR / f"{DATASET_NAME}.yaml"
    if config_path.exists():
        raise ValueError(
            f"{config_path} already exists. Edit it directly, or delete it first "
            "if you want to regenerate it from scratch."
        )

    base_dir = Path(BASE_DIR.rstrip("/"))
    acquisition_dir = Path(ACQUISITION_DIR) if ACQUISITION_DIR else find_acquisition_dir(base_dir)
    raw_dir = acquisition_dir / "raw"
    if not raw_dir.is_dir():
        raise ValueError(f"No raw/ folder under {acquisition_dir}.")

    groups = build_live_stack_groups(raw_dir)
    if not groups:
        raise ValueError(f"No stack_N_channel_M folders found under {raw_dir}.")

    stack_ids = sorted(groups, key=natural_stack_key)
    reference_stack_id = stack_ids[0]
    reference_items = sorted(groups[reference_stack_id], key=lambda x: x[0])
    reference_folder = reference_items[0][1]

    json_files = sorted(reference_folder.glob("Cam_*.json"))
    if not json_files:
        raise ValueError(f"No Cam_*.json sidecars found in {reference_folder}.")
    ref_data = read_json(reference_folder, json_files[0].name)
    pi = ref_data["processingInformation"]
    acq0 = pi["acquisition"][0]

    voxel = pi["voxel_size_um"]
    voxel_size_zyx_um = [voxel["depth"], voxel["height"], voxel["width"]]
    size = pi["image_size_vx"]
    image_size_zyx = [size["depth"], size["height"], size["width"]]
    detection = acq0.get("detection", {})

    h5_files = find_h5_files_sorted(reference_folder)
    n_timepoints = len(h5_files)
    t0 = read_timestamp(reference_folder, json_files[0].name)
    t_last = read_timestamp(reference_folder, json_files[-1].name)
    if t0 and t_last and n_timepoints > 1:
        total_seconds = (t_last - t0).total_seconds()
        interval_s = round(total_seconds / (n_timepoints - 1), 1)
        interval_min = round(interval_s / 60, 2)
        total_duration_h = round(total_seconds / 3600, 2)
    else:
        interval_s = interval_min = total_duration_h = ""
    start_time = t0.strftime("%Y-%m-%d %H:%M:%S") if t0 else ""
    end_time = t_last.strftime("%Y-%m-%d %H:%M:%S") if t_last else ""

    # Channels: union across all stacks, keyed by channel index.
    channel_indices = sorted({ci for items in groups.values() for ci, _ in items})
    channel_lines = []
    for ci in channel_indices:
        name = None
        for items in groups.values():
            match = next((folder for idx, folder in items if idx == ci), None)
            if match is None:
                continue
            jsons = sorted(match.glob("Cam_*.json"))
            if not jsons:
                continue
            desc = read_json(match, jsons[0].name)["processingInformation"].get("channel_description")
            if desc:
                name = desc
                break
        channel_lines.append(
            f'    - index: {ci}\n'
            f'      name: "{name or f"ch{ci}"}"'  # TODO: verify — not in metadata for every channel
            + ("  # TODO: verify channel identity" if not name else "")
            + "\n      laser_nm: null  # TODO: not in Cam_*.json\n"
            "      exposure_ms: null  # TODO: not in Cam_*.json"
        )
    channels_block = "\n".join(channel_lines)

    # Stacks: condition guessed from the "-label" suffix in the stack_id, if present.
    stack_lines = []
    for stack_id in stack_ids:
        items = sorted(groups[stack_id], key=lambda x: x[0])
        folder_name = items[0][1].name
        label_match = LABEL_RE.match(stack_id)
        condition = slugify(label_match.group(1)) if label_match else ""
        condition_comment = "" if condition else "  # TODO: no label in folder name to guess from"
        stack_lines.append(
            f'\n  - id: "{stack_id}"\n'
            f'    condition: "{condition}"{condition_comment}\n'
            f'    folder: "{folder_name}"\n'
            f'    notes: ""'
        )
    stacks_block = "".join(stack_lines)

    output_dir = f"{base_dir}/tifs"
    dataset_path = detect_dataset_path(h5_files[0])
    live_cfg = {**DEFAULT_LIVE_TIMECOURSE, **load_most_recent_live_timecourse(DATASET_NAME)}

    rendered = TEMPLATE.format(
        name=NAME or DATASET_NAME,
        date=acquisition_dir.name[:10],
        dataset=DATASET_NAME,
        notes=NOTES,
        raw_data_dir=output_dir,
        output_dir=f"{base_dir}/analysis",
        live_output_dir=output_dir,
        instrument=acq0.get("microscope_type", ""),
        serial_number=acq0.get("serial_number", ""),
        software_version=acq0.get("embedded_version", ""),
        voxel_size_zyx_um=yaml_list(voxel_size_zyx_um),
        image_size_zyx=yaml_list(image_size_zyx),
        magnification=yaml_scalar(detection.get("magnification")),
        na=yaml_scalar(detection.get("numerical_aperture")),
        channels_block=channels_block,
        n_timepoints=n_timepoints,
        interval_s=yaml_scalar(interval_s),
        interval_min=yaml_scalar(interval_min),
        total_duration_h=yaml_scalar(total_duration_h),
        start_time=start_time,
        end_time=end_time,
        reference_stack_id=reference_stack_id,
        stacks_block=stacks_block,
        config_rel_path=config_path.relative_to(PROJECT_ROOT),
        root_dir=str(raw_dir),
        dataset_path=dataset_path,
        dtype=live_cfg["dtype"],
        per_timepoint=yaml_scalar(live_cfg["per_timepoint"]),
        auto_crop=yaml_scalar(live_cfg["auto_crop"]),
        auto_crop_channel=yaml_scalar(live_cfg["auto_crop_channel"]),
        auto_crop_threshold_percentile=yaml_scalar(live_cfg["auto_crop_threshold_percentile"]),
        auto_crop_blur_sigma=yaml_scalar(live_cfg["auto_crop_blur_sigma"]),
        pad=yaml_scalar(live_cfg["pad"]),
    )

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path.write_text(rendered)
    print(f"\nWrote {config_path.relative_to(PROJECT_ROOT)} for dataset '{DATASET_NAME}'.")
    print(f"  {len(stack_ids)} stack(s), {len(channel_indices)} channel(s), {n_timepoints} timepoint(s).")
    print("Still need to fill in by hand: detection_objective.model/immersion/wd_mm, "
          "channel identities/laser info marked TODO, and any stack conditions left blank.")
    print(f"\nNext: edit the TODOs above, then run pipelines/timecourse/convert_h5_timecourse_to_tiff.py "
          f"(point its DEV_CONFIG at {config_path.relative_to(PROJECT_ROOT)}, or pass --config).")

    log_run("preprocessing", DATASET_NAME, "new_live_config.py",
            output_path=str(config_path), data_path=str(raw_dir), detail="detailed")


if __name__ == "__main__":
    main()
