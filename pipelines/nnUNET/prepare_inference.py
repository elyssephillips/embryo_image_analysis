"""
Prepare Cam_long / Cam_short stacks for nnUNet inference.

For each timepoint:
  1. Load Cam_long (nuclei channel) and detect first signal slice from intensity
  2. Subtract SAFETY_SLICES to keep a buffer above the signal
  3. Crop both Cam_long and Cam_short from that z onwards (same crop for both)
  4. Save Cam_long crop to imagesTs/ as Dataset001_XXXXX_0000.tif (nnUNet input)
  5. Save Cam_short crop to cam_short_cropped/ for downstream biosensor analysis
  6. Write crop_info.json with per-timepoint offsets for mapping predictions back

Signal detection uses an adaptive threshold:
  threshold = np.percentile(per_slice_max, BACKGROUND_PERCENTILE) * SIGNAL_MULTIPLIER
  Raise SIGNAL_MULTIPLIER if background slices are being included as signal.
  Lower it if real signal is being missed.
"""

import os
import re
import json
import numpy as np
import tifffile

INPUT_DIR    = "/mnt/md0/elysse/250914_stack_5"
OUTPUT_DIR   = "/mnt/md0/elysse/nnUNet/inference/Dataset001_implantation"
DATASET_NAME = "Dataset001"

SAFETY_SLICES        = 5    # slices to keep before first detected signal
BACKGROUND_PERCENTILE = 10  # percentile of per-slice maxima used to estimate background
SIGNAL_MULTIPLIER     = 2.0 # threshold = background * this; raise if over-including background

images_out    = os.path.join(OUTPUT_DIR, "imagesTs")
cam_short_out = os.path.join(OUTPUT_DIR, "cam_short_cropped")
os.makedirs(images_out, exist_ok=True)
os.makedirs(cam_short_out, exist_ok=True)

long_files = sorted(
    f for f in os.listdir(INPUT_DIR)
    if f.startswith("Cam_long_") and f.endswith("_cropped.tif") and not f.startswith("._")
)
print(f"Found {len(long_files)} Cam_long timepoints\n")

crop_info = {}

for lf in long_files:
    match = re.search(r'Cam_long_(\d+)_cropped\.tif', lf)
    if not match:
        print(f"  WARNING: unexpected filename {lf}, skipping")
        continue
    tp_id = match.group(1)

    sf = f"Cam_short_{tp_id}_cropped.tif"
    lpath = os.path.join(INPUT_DIR, lf)
    spath = os.path.join(INPUT_DIR, sf)

    if not os.path.exists(spath):
        print(f"  WARNING: no matching Cam_short for {lf}, skipping")
        continue

    print(f"Processing timepoint {tp_id}")
    long_arr  = tifffile.imread(lpath)   # (Z, Y, X)
    short_arr = tifffile.imread(spath)   # (Z, Y, X)

    if long_arr.shape != short_arr.shape:
        print(f"  WARNING: shape mismatch Cam_long {long_arr.shape} vs Cam_short {short_arr.shape}, skipping")
        continue

    # Adaptive signal detection from nuclei channel
    per_slice_max = np.array([long_arr[z].max() for z in range(long_arr.shape[0])])
    background_level = np.percentile(per_slice_max, BACKGROUND_PERCENTILE)
    threshold = background_level * SIGNAL_MULTIPLIER
    signal_slices = np.where(per_slice_max > threshold)[0]

    if len(signal_slices) == 0:
        print(f"  WARNING: no signal detected (threshold={threshold:.1f}), skipping")
        continue

    first_signal_z = int(signal_slices[0])
    first_z = max(0, first_signal_z - SAFETY_SLICES)
    print(f"  Background ~{background_level:.0f}, threshold={threshold:.0f}")
    print(f"  Signal starts z={first_signal_z}, cropping from z={first_z} (safety={SAFETY_SLICES})")

    long_crop  = long_arr[first_z:]
    short_crop = short_arr[first_z:]
    print(f"  Original shape: {long_arr.shape} -> Cropped: {long_crop.shape}")

    spacing = {"spacing": [2.0, 0.208, 0.208]}  # [z, y, x] in µm

    img_fname   = f"{DATASET_NAME}_{tp_id}_0000.tif"
    json_fname  = f"{DATASET_NAME}_{tp_id}.json"
    short_fname = f"Cam_short_{tp_id}_cropped.tif"

    tifffile.imwrite(os.path.join(images_out, img_fname), long_crop)
    with open(os.path.join(images_out, json_fname), "w") as f:
        json.dump(spacing, f)

    tifffile.imwrite(os.path.join(cam_short_out, short_fname), short_crop)

    crop_info[tp_id] = {
        "cam_long_source": lf,
        "cam_short_source": sf,
        "original_shape": list(long_arr.shape),
        "first_signal_z": first_signal_z,
        "first_z": first_z,
        "cropped_shape": list(long_crop.shape),
    }

    print(f"  Saved: {img_fname}, {short_fname}\n")

crop_info_path = os.path.join(OUTPUT_DIR, "crop_info.json")
with open(crop_info_path, "w") as f:
    json.dump(crop_info, f, indent=4)

print(f"Wrote crop offsets to {crop_info_path}")
print(f"Done. {len(crop_info)} timepoints prepared.")
