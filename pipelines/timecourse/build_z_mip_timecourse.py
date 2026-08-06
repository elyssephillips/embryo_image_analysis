"""Assemble a per-stack timecourse movie from the scope software's own Z-MIP previews.

During acquisition, the scope software writes a running MIP preview into every
stack/channel folder, one set of files per timepoint:

    <ACQUISITION_DIR>/                          <- e.g. 2026-07-21_184317 (acquisition start)
        raw/
            stack_0_channel_0-far red_obj_bottom/
                Cam_long_00000.json
                Cam_long_00000.lux.h5
                ...
                mip/
                    Cam_long_00000.max.x.jpg   Cam_long_00000.max.x.tiff
                    Cam_long_00000.max.y.jpg   Cam_long_00000.max.y.tiff
                    Cam_long_00000.max.z.jpg   Cam_long_00000.max.z.tiff  <- the one we want
                    Cam_long_00001.max.z.tiff
                    ...
            stack_0_channel_1_obj_bottom/
                ...mip/...
            stack_1_channel_0-far red_obj_bottom/
                ...mip/...
            ...

This script groups stack_*_channel_* folders under ROOT_DIR by their shared
stack (embryo/position), picks out the max.z.tiff (falling back to .jpg if no
TIFF) for each timepoint index of every channel, and writes one MP4 per stack:
each timepoint's channels are contrast-stretched to 8-bit and stacked as rows
into a single composite frame. Contrast limits are computed once per channel
across the whole movie (not per-frame) so relative brightness changes over
time stay visible instead of every frame being auto-leveled to look the same.
A companion CSV records each frame's timepoint index and, when the matching
Cam_*.json sidecar has metaData.microscopeInfo.timeStamp, its real acquisition
timestamp and elapsed seconds since the first frame.

Getting the data local first
-----------------------------
ROOT_DIR must be a local path — this script doesn't reach across SMB. Pull
just the mip/ folders down (skip the much larger raw .h5 stacks) with
something like:

    rsync -av --include='*/' --include='mip/***' --exclude='*' \\
        '/path/to/mounted/share/<experiment>/<acquisition>/raw/' /local/dest/raw/

(mount the SMB share first, e.g. via your file manager or `mount -t cifs`).

To run: edit ROOT_DIR / OUTPUT_DIR below, then click VS Code's "Run Python
File" button (or Ctrl+F5) — no terminal args needed.
"""
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import tifffile
from skimage import io as skio

from src.conversion import build_live_stack_groups

# ============================== EDIT THESE ==============================
# Either the acquisition's raw/ folder itself, or its parent (the folder
# containing raw/) — both are accepted.
ROOT_DIR = Path("/mnt/md0/elysse/260721_e45c_fgf_oct4_snap_2/2026-07-22_122157/raw")
OUTPUT_DIR = Path("/mnt/md0/elysse/260721_e45c_fgf_oct4_snap_2/2026-07-22_122157/z_mip_timecourses")

STACKS_INCLUDE = None   # e.g. ["stack_0"] to only process some stacks (all their channels); None = all

FPS = 8                        # playback frame rate of the output movie
BRIGHTNESS_PERCENTILES = (1, 99.5)  # (low, high) percentile of each channel's whole movie mapped to (black, white)
# ==========================================================================

Z_MIP_RE = re.compile(r"^(?P<stem>.+)\.max\.z\.(?P<ext>tiff?|jpe?g)$", re.IGNORECASE)
INDEX_RE = re.compile(r"(\d+)$")
EXT_PRIORITY = {"tiff": 0, "tif": 0, "jpg": 1, "jpeg": 1}


def resolve_raw_dir(root_dir: Path) -> Path:
    if root_dir.name.lower() == "raw" or not (root_dir / "raw").is_dir():
        return root_dir
    return root_dir / "raw"


def natural_stack_key(stack_id: str):
    m = re.match(r"^stack_(\d+)", stack_id)
    return int(m.group(1)) if m else 0


def find_z_mip_frames(mip_dir: Path) -> dict[int, tuple[str, Path]]:
    """Return {timepoint_index: (stem, path)}, preferring TIFF over JPG when
    both exist for the same timepoint."""
    best: dict[str, Path] = {}
    for path in mip_dir.iterdir():
        match = Z_MIP_RE.match(path.name)
        if not match:
            continue
        stem = match.group("stem")
        ext = match.group("ext").lower()
        current = best.get(stem)
        if current is None or EXT_PRIORITY[ext] < EXT_PRIORITY[Z_MIP_RE.match(current.name).group("ext").lower()]:
            best[stem] = path

    frames = {}
    for stem, path in best.items():
        idx_match = INDEX_RE.search(stem)
        if idx_match is None:
            print(f"    WARNING: can't parse a timepoint index from {path.name} — skipping.")
            continue
        frames[int(idx_match.group(1))] = (stem, path)
    return frames


def read_json(folder: Path, stem: str) -> dict | None:
    json_path = folder / f"{stem}.json"
    if not json_path.exists():
        return None
    try:
        with open(json_path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def read_timestamp(folder: Path, stem: str) -> datetime | None:
    data = read_json(folder, stem)
    if data is None:
        return None
    try:
        ts = data["metaData"]["microscopeInfo"]["timeStamp"]
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except (KeyError, ValueError):
        return None


def channel_label(folder: Path, channel_index: int, stem: str) -> str:
    data = read_json(folder, stem)
    desc = data["processingInformation"].get("channel_description") if data else None
    return desc or f"ch{channel_index}"


def load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        return tifffile.imread(str(path))
    return skio.imread(str(path))


def normalize_to_uint8(imgs: list[np.ndarray], percentiles: tuple[float, float]) -> list[np.ndarray]:
    stacked = np.stack(imgs)
    lo, hi = np.percentile(stacked[:, ::4, ::4], percentiles)
    scale = 255.0 / max(hi - lo, 1e-6)
    return [np.clip((img.astype(np.float32) - lo) * scale, 0, 255).astype(np.uint8) for img in imgs]


def main():
    raw_dir = resolve_raw_dir(ROOT_DIR)
    if not raw_dir.is_dir():
        raise ValueError(f"ROOT_DIR does not resolve to a raw/ folder: {raw_dir} — edit ROOT_DIR at the top of this script.")

    if (raw_dir / "complete" / "EXPERIMENT_ABORTED").exists():
        print("NOTE: raw/complete/EXPERIMENT_ABORTED found — this acquisition was aborted; "
              "stacks may have uneven timepoint counts.")

    groups = build_live_stack_groups(raw_dir)
    if STACKS_INCLUDE:
        groups = {k: v for k, v in groups.items() if k in STACKS_INCLUDE}
    if not groups:
        raise ValueError(f"No stack_*_channel_* folders found under {raw_dir}. Check ROOT_DIR and STACKS_INCLUDE.")

    stack_ids = sorted(groups, key=natural_stack_key)
    print(f"Found {len(stack_ids)} stack(s) under {raw_dir}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, stack_id in enumerate(stack_ids, 1):
        print(f"\n[{i}/{len(stack_ids)}] {stack_id}", flush=True)
        channel_items = sorted(groups[stack_id], key=lambda x: x[0])

        channels = []
        for channel_index, folder in channel_items:
            mip_dir = folder / "mip"
            if not mip_dir.is_dir():
                print(f"  WARNING: {folder.name} has no mip/ subfolder — skipping this channel.")
                continue
            frames_by_idx = find_z_mip_frames(mip_dir)
            if not frames_by_idx:
                print(f"  WARNING: no max.z.* files found in {mip_dir} — skipping this channel.")
                continue

            # Drop timepoints whose frame shape doesn't match this channel's first frame.
            ref_shape = None
            clean_frames = {}
            for idx in sorted(frames_by_idx):
                stem, path = frames_by_idx[idx]
                img = load_image(path)
                if ref_shape is None:
                    ref_shape = img.shape
                elif img.shape != ref_shape:
                    print(f"  WARNING: {path.name} shape {img.shape} != {ref_shape} — skipping this frame.")
                    continue
                clean_frames[idx] = (stem, path, img)

            label = channel_label(folder, channel_index, next(iter(clean_frames.values()))[0]) if clean_frames else f"ch{channel_index}"
            channels.append({"index": channel_index, "folder": folder, "frames": clean_frames, "label": label})

        if not channels:
            print(f"  WARNING: no usable channels for {stack_id} — skipping.")
            continue

        common_indices = sorted(set.intersection(*(set(c["frames"]) for c in channels)))
        if not common_indices:
            print(f"  WARNING: channels share no common timepoints for {stack_id} — skipping.")
            continue
        for c in channels:
            missing = len(c["frames"]) - len(common_indices)
            if missing:
                print(f"  NOTE: {c['folder'].name} has {missing} timepoint(s) not shared with other channels; using the {len(common_indices)} shared timepoints.")

        # Normalize each channel independently across its shared frames.
        normalized_rows = {}
        for c in channels:
            imgs = [c["frames"][idx][2] for idx in common_indices]
            normalized_rows[c["index"]] = normalize_to_uint8(imgs, BRIGHTNESS_PERCENTILES)

        composite_frames = []
        for f_i in range(len(common_indices)):
            rows = []
            for c in channels:
                bgr = cv2.cvtColor(normalized_rows[c["index"]][f_i], cv2.COLOR_GRAY2BGR)
                cv2.putText(bgr, c["label"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                rows.append(bgr)
            composite_frames.append(np.vstack(rows))

        out_name = stack_id.replace(" ", "_")
        out_path = OUTPUT_DIR / f"{out_name}.mp4"
        h, w = composite_frames[0].shape[:2]
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        try:
            for frame in composite_frames:
                writer.write(frame)
        finally:
            writer.release()

        manifest_rows = []
        t0 = None
        for f_i, idx in enumerate(common_indices):
            stem0 = channels[0]["frames"][idx][0]
            ts = read_timestamp(channels[0]["folder"], stem0)
            if t0 is None and ts is not None:
                t0 = ts
            row = {
                "frame_index": f_i,
                "timepoint_index": idx,
                "timestamp": ts.isoformat() if ts else "",
                "elapsed_seconds": (ts - t0).total_seconds() if (ts and t0) else "",
            }
            for c in channels:
                row[f"source_file_ch{c['index']}"] = c["frames"][idx][1].name
            manifest_rows.append(row)

        manifest_path = OUTPUT_DIR / f"{out_name}_manifest.csv"
        with open(manifest_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer_csv.writeheader()
            writer_csv.writerows(manifest_rows)

        print(f"  Wrote {len(composite_frames)} frames ({len(channels)} channel row(s)) -> {out_path}")
        print(f"  Wrote manifest -> {manifest_path}")

    print(f"\nDone. Timecourses written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
