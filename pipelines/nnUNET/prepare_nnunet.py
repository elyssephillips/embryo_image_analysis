"""
Convert hand-annotated label/raw tif pairs into nnUNet v2 Dataset format.

Steps per pair:
  1. Find first non-empty slice in the label stack
  2. Crop both raw and label from that slice onwards (matching)
  3. Remove the touching interface between any two different instances
     (see separate_touching_instances)
  4. Assign each remaining instance its class - background=0, TE=1, ICM=2 -
     using the {label_id: "ICM"} sidecar written by scripts/hand_label_icm_te.py
     (IDs absent from that sidecar are TE; see that script's docstring)
  5. Save raw as imagesTr/Dataset001_XXXXX_0000.tif
  6. Save 3-class label as labelsTr/Dataset001_XXXXX.tif

Case ID is taken from the original numeric suffix in the source filename.
Also writes dataset.json.

Every label file needs a matching *.icm_te.json sidecar (same basename with
the .tif swapped for .icm_te.json, e.g. Cam_long_00074.label.tif ->
Cam_long_00074.label.icm_te.json) in LABEL_DIR - produced locally by
scripts/hand_label_icm_te.py and copied over alongside the label tif. A
missing sidecar is a hard error rather than a silent all-TE fallback, since
the latter would quietly corrupt the training target instead of just being
imprecise.

Why step 3 isn't a plain `labels != 0` binarization
-----------------------------------------------------
Hand-drawn instances never overlap, but they're routinely face-adjacent -
in Z, Y, or X alike. A straight binarization fuses every touching pair into
one blob before nnUNet ever sees per-instance identity, so the network
faithfully learns to reproduce that fusion.

An earlier attempt fixed this with a fixed-margin XY-only erosion per
instance (see pipelines/tracking/segmentation_notes.md). Re-validated
directly against this dataset, that approach left a large fraction of
touching pairs still fused even at a generous margin, and provably could
never resolve any pair whose only contact was across a Z-slice boundary
(erosion is per-slice, so it never touches Z-adjacency at all - about a
third of all touching pairs in this dataset are Z-only).

separate_touching_instances instead removes exactly the 1-voxel contact
layer between two *different* instances, wherever it occurs, before
binarizing - not a margin around each instance's whole surface. This:
  - fully separates every touching pair, regardless of axis or how broad
    the contact is (unlike a margin, which fails once contact is wider
    than the eroded rind)
  - never touches voxels on a surface that faces background or doesn't
    border another instance, so it costs far less real volume than
    uniform per-instance erosion
  - needs no margin parameter

At inference, connected-components on the predicted binary mask will
naturally have a thin gap wherever training taught it one; grow instances
back out with skimage.segmentation.expand_labels (a small distance, e.g.
2-3 voxels) to recover the true volume without re-merging neighbors.
"""

import os
import re
import json
from pathlib import Path
import numpy as np
import tifffile

LABEL_DIR = "/mnt/md0/elysse/training/labels"
RAW_DIR   = "/mnt/md0/elysse/training/raw"
OUT_DIR   = "/mnt/md0/elysse/training/nnUNet_raw/Dataset001_implantation"
DATASET_NAME = "Dataset001"


def separate_touching_instances(larr):
    """Zero out the 1-voxel contact layer between any two different, nonzero
    instance IDs, checked along all three axes independently. Guarantees a
    background gap at every point two instances touch, however the contact
    is shaped, without shrinking any surface that doesn't border another
    instance.

    Side effect: if one instance touches two different neighbors close
    together (e.g. near its own thin point), removing both contact layers
    can nick off a few stray voxels of that same instance. Always trivial
    (single digits to low tens of voxels vs. thousands in the main body,
    checked directly against this dataset) - drop_stray_fragments cleans
    these up so they don't show up as noise in the binary training target.
    """
    to_remove = np.zeros(larr.shape, dtype=bool)
    for axis in range(3):
        sl_left = [slice(None)] * 3
        sl_right = [slice(None)] * 3
        sl_left[axis] = slice(0, -1)
        sl_right[axis] = slice(1, None)
        sl_left, sl_right = tuple(sl_left), tuple(sl_right)
        left = larr[sl_left]
        right = larr[sl_right]
        interface = (left != 0) & (right != 0) & (left != right)
        to_remove[sl_left] |= interface
        to_remove[sl_right] |= interface
    separated = larr.copy()
    separated[to_remove] = 0
    return separated


def drop_stray_fragments(larr):
    """For each instance ID, keep only its largest connected fragment. Cleans
    up the trivial nicks separate_touching_instances can leave behind."""
    import scipy.ndimage as ndi

    ids = np.unique(larr)
    ids = ids[ids != 0]
    objs = ndi.find_objects(larr)
    cleaned = larr.copy()
    for iid in ids:
        sl = objs[int(iid) - 1]
        if sl is None:
            continue
        pad = tuple(slice(max(0, s.start - 1), min(dim, s.stop + 1)) for s, dim in zip(sl, larr.shape))
        sub = cleaned[pad]
        mask = sub == iid
        lbl, n = ndi.label(mask)
        if n <= 1:
            continue
        sizes = ndi.sum(mask, lbl, index=range(1, n + 1))
        keep = int(np.argmax(sizes)) + 1
        drop_mask = mask & (lbl != keep)
        sub[drop_mask] = 0
    return cleaned

def load_icm_ids(lpath):
    """Load the {label_id: "ICM"} sidecar written by scripts/hand_label_icm_te.py
    for this label file. IDs not listed are TE by convention (see that
    script's docstring). A missing sidecar is a hard error - see module
    docstring for why."""
    class_path = Path(lpath).with_suffix(".icm_te.json")
    if not class_path.exists():
        raise FileNotFoundError(
            f"No ICM/TE sidecar for {lpath} (expected {class_path}). Run "
            f"scripts/hand_label_icm_te.py on it first, or remove it from LABEL_DIR."
        )
    with open(class_path) as f:
        assignments = json.load(f)
    return set(int(k) for k in assignments.keys())


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

    # Extract original numeric ID from filename (e.g. "Cam_long_00049_label.tif" -> "00049").
    # Anchored to "Cam_long_" specifically, not a bare \d+ search - a generic search
    # would grab the "001" out of "Dataset001" for any differently-prefixed file and
    # silently collide two different volumes onto the same case ID.
    match = re.search(r'Cam_long_(\d+)', lf)
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

    # Separate touching instances (any axis), clean up stray nicks
    separated = separate_touching_instances(larr_crop)
    separated = drop_stray_fragments(separated)
    n_removed = int((larr_crop > 0).sum() - (separated > 0).sum())
    print(f"  Removed {n_removed} voxels (touching interfaces + stray nicks) to guarantee separation")

    # Assign class: background=0, TE=1, ICM=2 (see load_icm_ids / module docstring)
    icm_ids = load_icm_ids(lpath)
    class_label = np.zeros(separated.shape, dtype=np.uint8)
    class_label[separated > 0] = 1
    if icm_ids:
        class_label[np.isin(separated, list(icm_ids))] = 2
    ids_in_crop = set(np.unique(separated).tolist()) - {0}
    n_icm = len(ids_in_crop & icm_ids)
    n_te = len(ids_in_crop) - n_icm
    print(f"  Classes: {n_icm} ICM, {n_te} TE instances (unique values: {np.unique(class_label)})")

    img_fname  = f"{DATASET_NAME}_{orig_id}_0000.tif"
    lbl_fname  = f"{DATASET_NAME}_{orig_id}.tif"

    spacing = {"spacing": [2.0, 0.208, 0.208]}  # [z, y, x] in µm

    tifffile.imwrite(os.path.join(images_out, img_fname), rarr_crop)
    img_json_fname = f"{DATASET_NAME}_{orig_id}.json"  # strip _0000, per nnUNet tif_reader_writer.py:53
    with open(os.path.join(images_out, img_json_fname), "w") as f:
        json.dump(spacing, f)

    tifffile.imwrite(os.path.join(labels_out, lbl_fname), class_label)
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
        "TE": 1,
        "ICM": 2
    },
    "numTraining": len(training_cases),
    "file_ending": ".tif",
    "training": training_cases,
    "description": "Hand-annotated nuclei labels (TE/ICM), implantation dataset",
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
