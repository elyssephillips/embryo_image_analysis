#!/home/elysse/miniforge3/envs/blastospim-tf/bin/python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import tifffile as tiff
import numpy as np
from pathlib import Path
from csbdeep.utils import normalize
from stardist.models import StarDist3D
from src.conversion import load_hyperstack_czyx
from src.io import load_config, get_image_paths

MODEL_BASEDIR = "/mnt/md1/elysse/code/blastospim model files"
MODEL_NAME = "late_blastocyst_model"

# BlastoSPIM's published training voxel size (Ker et al. 2023, PMC10055179;
# blastospim.flatironinstitute.org). StarDist predicts nucleus shape directly in
# raw voxel units, so if this dataset's acquisition voxel size differs from what
# the model trained on, real nuclei span a different number of raw voxels than
# "one nucleus" to the model — causing systematic over/under-segmentation
# regardless of channel. No scale correction was ever applied here before.
BLASTOSPIM_VOXEL_ZYX_UM = (2.0, 0.208, 0.208)

# ============================== EDIT THESE ==============================
SEGMENTATION_CHANNEL = "gata3"  # or "cdx2" — must be one of configs/IF/config.yaml's
                    # microscopy.channel_names. Note the model was trained on DAPI
                    # nuclear stains; cdx2 segmentation quality hasn't been validated
                    # against it.
PROB_THRESH = 0.65  # None = use the model's own thresholds.json default (0.531).
                    # Raise (e.g. 0.6-0.8) to drop lower-confidence false positives;
                    # lower to catch more/dimmer real nuclei at the cost of more junk.
NMS_THRESH = 0.22   # None = model default (0.3). Lower = more aggressive merging of
                    # overlapping candidate detections.
# ==========================================================================


def run_segmentation():
    config = load_config('configs/IF/config.yaml')
    raw_dir = Path(config['raw_data_dir'])
    channel_names = config['microscopy']['channel_names']

    our_voxel_zyx = config['microscopy']['voxel_size_zyx']
    scale_zyx = tuple(our_voxel_zyx[i] / BLASTOSPIM_VOXEL_ZYX_UM[i] for i in range(3))
    our_anisotropy = our_voxel_zyx[0] / our_voxel_zyx[1]
    trained_anisotropy = BLASTOSPIM_VOXEL_ZYX_UM[0] / BLASTOSPIM_VOXEL_ZYX_UM[1]
    print(f"  voxel_size_zyx: this dataset={our_voxel_zyx} um, BlastoSPIM training={list(BLASTOSPIM_VOXEL_ZYX_UM)} um")
    print(f"  predict_instances(scale={scale_zyx})")
    print(f"  z:xy anisotropy: this dataset={our_anisotropy:.2f}, BlastoSPIM training={trained_anisotropy:.2f}"
          f"{'  <-- MISMATCH, verify voxel_size_zyx' if abs(our_anisotropy - trained_anisotropy) / trained_anisotropy > 0.25 else ''}")

    if SEGMENTATION_CHANNEL not in channel_names:
        raise ValueError(
            f"SEGMENTATION_CHANNEL={SEGMENTATION_CHANNEL!r} not in "
            f"config.yaml's microscopy.channel_names {channel_names}."
        )
    channel_idx = channel_names.index(SEGMENTATION_CHANNEL)

    # Keep the default (dapi) output path unchanged so 02_extract_intensities.py's
    # f"{identifier}_segmentation.tif" lookup keeps working without edits. Any other
    # channel goes in its own subfolder so it can't collide with or silently shadow
    # a dapi segmentation of the same file.
    seg_dir = Path(config['segmentation_dir_raw'])
    if SEGMENTATION_CHANNEL != "dapi":
        seg_dir = seg_dir / SEGMENTATION_CHANNEL
    os.makedirs(seg_dir, exist_ok=True)

    print(f"Loading BlastoSPIM model: {MODEL_NAME}")
    model = StarDist3D(None, name=MODEL_NAME, basedir=MODEL_BASEDIR)
    print(f"  prob_thresh={PROB_THRESH if PROB_THRESH is not None else 'model default'}, "
          f"nms_thresh={NMS_THRESH if NMS_THRESH is not None else 'model default'}")

    image_files = get_image_paths(raw_dir, extension=".tif")

    for img_path in image_files:
        identifier = img_path.stem
        out_path = seg_dir / f"{identifier}_segmentation.tif"
        if out_path.exists():
            print(f"Skipping {identifier}: segmentation already exists.")
            continue

        print(f"--- Processing: {identifier} ---")
        arr, _ = load_hyperstack_czyx(img_path)  # (Z, C, Y, X)
        if channel_idx >= arr.shape[1]:
            print(f"  {identifier} only has {arr.shape[1]} channels, no index {channel_idx}. Skipping.")
            continue
        channel_vol = np.asarray(arr[:, channel_idx], dtype=np.float32)

        # Percentile normalization expected by BlastoSPIM
        vol_norm = normalize(channel_vol, 1, 99.8, axis=(0, 1, 2))

        # StarDist3D expects (Z, Y, X, C)
        vol_norm = vol_norm[..., np.newaxis]

        # Tile to fit in GPU memory; target ~[48, 256, 256] per tile (ZYX)
        tile_z, tile_y, tile_x = 48, 256, 256
        z, y, x = vol_norm.shape[:3]
        n_tiles = (
            int(np.ceil(z / tile_z)),
            int(np.ceil(y / tile_y)),
            int(np.ceil(x / tile_x)),
            1,  # channel
        )
        print(f"Running segmentation on {identifier} [{SEGMENTATION_CHANNEL}] (shape {vol_norm.shape}, tiles {n_tiles})...")
        labels, _ = model.predict_instances(
            vol_norm, n_tiles=n_tiles, scale=(*scale_zyx, 1),
            prob_thresh=PROB_THRESH, nms_thresh=NMS_THRESH,
        )

        tiff.imwrite(str(out_path), labels.astype(np.uint16))
        print(f"Saved: {out_path.name} | {labels.max()} nuclei detected")


if __name__ == "__main__":
    run_segmentation()
