"""
Convert hand-annotated label/raw tif pairs into nnUNet v2 Dataset format.

Steps per pair:
  1. Find first non-empty slice in the label stack
  2. Crop both raw and label from that slice onwards (matching)
  3. Binarize label: >0 -> 1 (uint8)
  4. Save raw as imagesTr/Dataset001_XXXXX_0000.tif
  5. Save binary label as labelsTr/Dataset001_XXXXX.tif

Case ID is taken from the original numeric suffix in the source filename.
Also writes dataset.json.
"""

import os
import re
import json
import numpy as np
import tifffile

LABEL_DIR = "/mnt/md0/elysse/training/labels"
RAW_DIR   = "/mnt/md0/elysse/training/raw"
OUT_DIR   = "/mnt/md0/elysse/training/nnUNet_raw/Dataset001_implantation"
DATASET_NAME = "Dataset001"

images_out = os.path.join(OUT_DIR, "imagesTr")
labels_out = os.path.join(OUT_DIR, "labelsTr")
os.makedirs(images_out, exist_ok=True)
os.makedirs(labels_out, exist_ok=True)

# Collect valid label files (skip macOS resource forks)
label_files = sorted(
    f for f in os.listdir(LABEL_DIR) if f.endswith(".tif") and not f.startswith("._")
)

print(f"Found {len(label_files)} label files\n")

case_id = 0
training_cases = []

for lf in label_files:
    lpath = os.path.join(LABEL_DIR, lf)
    rname = lf.replace("_label", "").replace(".label", "")
    rpath = os.path.join(RAW_DIR, rname)

    if not os.path.exists(rpath):
        print(f"  WARNING: no matching raw for {lf}, skipping")
        continue

    # Extract original numeric ID from filename (e.g. "Cam_long_00049" -> "00049")
    match = re.search(r'(\d+)', lf)
    if not match:
        print(f"  WARNING: could not extract numeric ID from {lf}, skipping")
        continue
    orig_id = match.group(1)

    print(f"Processing {lf} (id: {orig_id})")
    larr = tifffile.imread(lpath)   # (Z, Y, X) uint16 instance labels
    rarr = tifffile.imread(rpath)   # (Z, Y, X) uint16 raw intensity

    # Find first non-empty label slice
    nonempty = np.array([larr[z].max() > 0 for z in range(larr.shape[0])])
    if not nonempty.any():
        print(f"  WARNING: all label slices are empty, skipping")
        continue
    first_z = int(np.argmax(nonempty))
    print(f"  Cropping from slice {first_z} (removing {first_z} empty slices at start)")

    larr_crop = larr[first_z:]
    rarr_crop = rarr[first_z:]
    print(f"  Cropped shape: {larr_crop.shape}")

    # Binarize labels
    binary_label = (larr_crop > 0).astype(np.uint8)
    print(f"  Binary label unique values: {np.unique(binary_label)}")

    img_fname  = f"{DATASET_NAME}_{orig_id}_0000.tif"
    lbl_fname  = f"{DATASET_NAME}_{orig_id}.tif"

    spacing = {"spacing": [2.0, 0.208, 0.208]}  # [z, y, x] in µm

    tifffile.imwrite(os.path.join(images_out, img_fname), rarr_crop)
    img_json_fname = f"{DATASET_NAME}_{orig_id}.json"  # strip _0000, per nnUNet tif_reader_writer.py:53
    with open(os.path.join(images_out, img_json_fname), "w") as f:
        json.dump(spacing, f)

    tifffile.imwrite(os.path.join(labels_out, lbl_fname), binary_label)
    with open(os.path.join(labels_out, lbl_fname.replace(".tif", ".json")), "w") as f:
        json.dump(spacing, f)

    print(f"  Saved: {img_fname}, {lbl_fname}\n")

    training_cases.append({"image": f"./imagesTr/{img_fname}", "label": f"./labelsTr/{lbl_fname}"})
    case_id += 1

# Write dataset.json
dataset_json = {
    "channel_names": {
        "0": "nuclei"
    },
    "labels": {
        "background": 0,
        "nucleus": 1
    },
    "numTraining": len(training_cases),
    "file_ending": ".tif",
    "training": training_cases,
    "description": "Hand-annotated nuclei labels, implantation dataset",
    "name": "Dataset001_implantation",
    "reference": "",
    "licence": "",
    "release": "0.0"
}

json_path = os.path.join(OUT_DIR, "dataset.json")
with open(json_path, "w") as f:
    json.dump(dataset_json, f, indent=4)
print(f"Wrote {json_path}")
print(f"\nDone. {case_id} cases prepared.")
