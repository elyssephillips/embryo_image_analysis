"""
Salvage fix for dataset003 (250914_stack5): a handful of timepoints had their
prepare_inference.py z-crop start triggered by a debris speck (a few bright
pixels) rather than real embryo signal, offsetting first_z by dozens of
slices for just those frames - see prepare_inference.py's MIN_SIGNAL_PIXELS
fix for the root cause.

Rather than re-running nnUNet inference / 01_extract_features.py (expensive,
and the already-saved crops/predictions/features are still valid - they just
contain extra debris-included padding at the front for the affected frames),
this re-scans the already-saved cropped images with the corrected detector to
find the real signal start per timepoint, then:
  1. Records the correction in crop_info.json under corrected_first_signal_z /
     corrected_last_signal_z / debris_shift_slices, without touching the
     existing first_z/last_z/cropped_shape fields (those still describe what
     is actually saved on disk - changing them without re-cropping the actual
     tiffs would make them lie).
  2. Shifts z_um for the affected timepoints' rows in features.csv by
     debris_shift_slices * VX_Z, so z is expressed relative to the same
     real-signal-relative origin across every timepoint instead of a
     debris-corrupted one for a few frames.

features_registered.csv / tracks CSVs are NOT touched here - registration is
centroid-drift-based, so it needs to be recomputed from the corrected
features.csv (02_register_centroids.py) rather than patched by hand.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

OUTPUT_DIR = Path("/mnt/md0/elysse/nnUNet/inference/Dataset003_icm_te/250914_stack5")
IMAGES_DIR = OUTPUT_DIR / "imagesTs" / "tifs"
CROP_INFO_PATH = OUTPUT_DIR / "crop_info.json"
FEATURES_CSV = OUTPUT_DIR / "results" / "features" / "features.csv"

VX_Z = 2.0  # µm/slice, matches prepare_inference.py's spacing

# Must match prepare_inference.py's fixed detector exactly, or the two won't agree.
BACKGROUND_PERCENTILE = 10
SIGNAL_MULTIPLIER = 2.0
MIN_SIGNAL_PIXELS = 50
SAFETY_SLICES = 5

# A frame only needs correcting if the debris-robust scan disagrees with the
# already-saved crop's start by more than this - real per-frame drift in
# first_z is normally a few slices; anything past this is a debris jump, not
# refinement of the same detection.
SHIFT_TOLERANCE_SLICES = 3

with open(CROP_INFO_PATH) as f:
    crop_info = json.load(f)

features_df = pd.read_csv(FEATURES_CSV)

corrections = {}  # tp_id -> shift_slices (front) actually applied

for tp_id, entry in sorted(crop_info.items()):
    img_path = IMAGES_DIR / f"Dataset003_{tp_id}_0000.tif"
    if not img_path.exists():
        print(f"t={tp_id}: WARNING saved crop not found at {img_path}, skipping")
        continue

    arr = tifffile.imread(img_path)  # already cropped to entry['first_z']:entry.get('last_z', end)
    per_slice_max = arr.max(axis=(1, 2))
    background = np.percentile(per_slice_max, BACKGROUND_PERCENTILE)
    threshold = background * SIGNAL_MULTIPLIER
    per_slice_area = (arr > threshold).sum(axis=(1, 2))
    signal_slices = np.where(per_slice_area > MIN_SIGNAL_PIXELS)[0]

    if len(signal_slices) == 0:
        print(f"t={tp_id}: WARNING no signal found on re-scan, leaving untouched")
        continue

    new_first_local = int(signal_slices[0])
    new_last_local = int(signal_slices[-1])
    front_trim = max(0, new_first_local - SAFETY_SLICES)

    if front_trim < SHIFT_TOLERANCE_SLICES:
        continue  # already correct (or within normal per-frame variation)

    old_first_z = entry["first_z"]
    corrected_first_signal_z = old_first_z + new_first_local
    corrected_last_signal_z = old_first_z + new_last_local

    entry["corrected_first_signal_z"] = corrected_first_signal_z
    entry["corrected_last_signal_z"] = corrected_last_signal_z
    entry["debris_shift_slices"] = front_trim

    corrections[tp_id] = front_trim
    print(
        f"t={tp_id}: DEBRIS FOUND - old first_signal_z={entry['first_signal_z']}, "
        f"corrected={corrected_first_signal_z} (shift={front_trim} slices = "
        f"{front_trim * VX_Z:.1f} um)"
    )

print(f"\n{len(corrections)} timepoint(s) corrected: {sorted(corrections.keys())}")

if corrections:
    backup_path = CROP_INFO_PATH.with_suffix(".json.bak")
    shutil.copy2(CROP_INFO_PATH, backup_path)
    with open(CROP_INFO_PATH, "w") as f:
        json.dump(crop_info, f, indent=4)
    print(f"crop_info.json patched (backup at {backup_path})")

    features_backup = FEATURES_CSV.with_suffix(".csv.bak")
    shutil.copy2(FEATURES_CSV, features_backup)
    for tp_id, shift_slices in corrections.items():
        t = int(tp_id)
        shift_um = shift_slices * VX_Z
        mask = features_df["t"] == t
        n_rows = mask.sum()
        features_df.loc[mask, "z_um"] -= shift_um
        print(f"  features.csv: t={t}, {n_rows} row(s), z_um -= {shift_um:.1f}")
    features_df.to_csv(FEATURES_CSV, index=False)
    print(f"features.csv patched (backup at {features_backup})")
else:
    print("No corrections needed.")
