"""
Prepare Cam_long / Cam_short stacks for nnUNet inference.

For each timepoint:
  1. Load Cam_long (nuclei channel) and detect the first/last slice with real
     signal from intensity
  2. Pad by SAFETY_SLICES on both ends
  3. Crop both Cam_long and Cam_short to that z-range (same crop for both)
  4. Save Cam_long crop to imagesTs/ as Dataset001_XXXXX_0000.tif (nnUNet input)
  5. Save Cam_short crop to cam_short_cropped/ for downstream biosensor analysis
  6. Write crop_info.json with per-timepoint offsets for mapping predictions back
  7. If N_GPU_SPLITS > 1, also copy imagesTs into imagesTs_gpu0, imagesTs_gpu1, ...
     (contiguous chunks by sorted timepoint) for running nnUNetv2_predict in
     parallel across multiple GPUs - see inference_commands.md. All splits can
     safely share one -o predictions dir since timepoint ranges never overlap.

Signal detection uses an adaptive intensity threshold:
  threshold = np.percentile(per_slice_max, BACKGROUND_PERCENTILE) * SIGNAL_MULTIPLIER
  Raise SIGNAL_MULTIPLIER if background slices are being included as signal.
  Lower it if real signal is being missed.

A slice only counts as "signal" if at least MIN_SIGNAL_PIXELS pixels clear
that threshold, not just the single brightest pixel - checked against dataset003
250914_stack5, a lone debris speck (as few as 2 px) above threshold in an
otherwise-empty slice was enough to fool a max-only test into cropping from
40+ slices too early, while real embryo signal was never under ~800 px. The
gap between the two was large and unambiguous in every case checked, so
MIN_SIGNAL_PIXELS trades essentially no risk of missing real signal for
immunity to single/few-pixel artifacts.
"""

import os
import re
import json
import shutil
import numpy as np
import tifffile

LONG_DIR     = "/mnt/md0/elysse/250914_stack_5/long"
SHORT_DIR    = "/mnt/md0/elysse/250914_stack_5/short"
OUTPUT_DIR   = "/mnt/md0/elysse/nnUNet/inference/Dataset003_icm_te/250914_stack5"
DATASET_NAME = "Dataset003"

N_GPU_SPLITS = 2  # number of imagesTs_gpuN folders to create for parallel nnUNetv2_predict; 1 to disable

SAFETY_SLICES        = 5    # slices to keep as buffer beyond first/last detected signal
BACKGROUND_PERCENTILE = 10  # percentile of per-slice maxima used to estimate background
SIGNAL_MULTIPLIER     = 2.0 # threshold = background * this; raise if over-including background
MIN_SIGNAL_PIXELS     = 50  # min pixels above threshold for a slice to count as real signal (not debris)

images_out    = os.path.join(OUTPUT_DIR, "imagesTs")
cam_short_out = os.path.join(OUTPUT_DIR, "cam_short_cropped")
os.makedirs(images_out, exist_ok=True)
os.makedirs(cam_short_out, exist_ok=True)

long_files = sorted(
    f for f in os.listdir(LONG_DIR)
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
    lpath = os.path.join(LONG_DIR, lf)
    spath = os.path.join(SHORT_DIR, sf)

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
    per_slice_area = np.array([(long_arr[z] > threshold).sum() for z in range(long_arr.shape[0])])
    signal_slices = np.where(per_slice_area > MIN_SIGNAL_PIXELS)[0]

    if len(signal_slices) == 0:
        print(f"  WARNING: no signal detected (threshold={threshold:.1f}), skipping")
        continue

    first_signal_z = int(signal_slices[0])
    last_signal_z  = int(signal_slices[-1])
    first_z = max(0, first_signal_z - SAFETY_SLICES)
    last_z  = min(long_arr.shape[0], last_signal_z + SAFETY_SLICES + 1)
    print(f"  Background ~{background_level:.0f}, threshold={threshold:.0f}")
    print(f"  Signal z={first_signal_z}-{last_signal_z}, cropping to z={first_z}-{last_z} (safety={SAFETY_SLICES})")

    long_crop  = long_arr[first_z:last_z]
    short_crop = short_arr[first_z:last_z]
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
        "last_signal_z": last_signal_z,
        "first_z": first_z,
        "last_z": last_z,
        "cropped_shape": list(long_crop.shape),
    }

    print(f"  Saved: {img_fname}, {short_fname}\n")

crop_info_path = os.path.join(OUTPUT_DIR, "crop_info.json")
with open(crop_info_path, "w") as f:
    json.dump(crop_info, f, indent=4)

print(f"Wrote crop offsets to {crop_info_path}")
print(f"Done. {len(crop_info)} timepoints prepared.")

# Split imagesTs into N_GPU_SPLITS contiguous chunks (by sorted timepoint) so
# nnUNetv2_predict can run in parallel across GPUs - see module docstring.
if N_GPU_SPLITS > 1:
    tp_ids = sorted(crop_info.keys())
    chunks = np.array_split(tp_ids, N_GPU_SPLITS)
    for gpu_idx, chunk in enumerate(chunks):
        split_dir = os.path.join(OUTPUT_DIR, f"imagesTs_gpu{gpu_idx}")
        os.makedirs(split_dir, exist_ok=True)
        for tp_id in chunk:
            base = f"{DATASET_NAME}_{tp_id}"
            shutil.copy2(os.path.join(images_out, f"{base}_0000.tif"), split_dir)
            shutil.copy2(os.path.join(images_out, f"{base}.json"), split_dir)
        print(f"  imagesTs_gpu{gpu_idx}: {len(chunk)} timepoints ({chunk[0]}-{chunk[-1]})")
