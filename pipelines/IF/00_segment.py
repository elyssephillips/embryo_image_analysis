import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import tifffile as tiff
import numpy as np
from pathlib import Path
from csbdeep.utils import normalize
from stardist.models import StarDist3D
from src.io import load_config, get_image_paths, process_image_loading

MODEL_BASEDIR = "/mnt/md1/elysse/code/blastospim model files"
MODEL_NAME = "late_blastocyst_model"


def run_segmentation():
    config = load_config('configs/IF/config.yaml')
    raw_dir = Path(config['raw_data_dir'])
    seg_dir = Path(config['segmentation_dir_raw'])
    os.makedirs(seg_dir, exist_ok=True)

    dapi_channel = config['microscopy']['channels']['dapi']

    print(f"Loading BlastoSPIM model: {MODEL_NAME}")
    model = StarDist3D(None, name=MODEL_NAME, basedir=MODEL_BASEDIR)

    image_files = get_image_paths(raw_dir, extension=".tif")

    for img_path in image_files:
        img, _, identifier = process_image_loading(img_path)

        out_path = seg_dir / f"{identifier}_segmentation.tif"
        if out_path.exists():
            print(f"Skipping {identifier}: segmentation already exists.")
            continue

        # TIFFs are written as (C*Z, Y, X) without axes metadata — reshape to (C, Z, Y, X)
        n_channels = len(config['microscopy']['channels'])
        if img.ndim == 3 and img.shape[0] % n_channels == 0:
            img = img.reshape(n_channels, img.shape[0] // n_channels, img.shape[1], img.shape[2])

        # Extract DAPI channel: (C, Z, Y, X) -> (Z, Y, X)
        if img.ndim == 4:
            dapi = img[dapi_channel].astype(np.float32)
        elif img.ndim == 3:
            dapi = img.astype(np.float32)
        else:
            print(f"Unexpected image shape {img.shape} for {identifier}, skipping.")
            continue

        # Percentile normalization expected by BlastoSPIM
        dapi_norm = normalize(dapi, 1, 99.8, axis=(0, 1, 2))

        # StarDist3D expects (Z, Y, X, C)
        dapi_norm = dapi_norm[..., np.newaxis]

        # Tile to fit in GPU memory; target ~[48, 256, 256] per tile (ZYX)
        tile_z, tile_y, tile_x = 48, 256, 256
        z, y, x = dapi_norm.shape[:3]
        n_tiles = (
            int(np.ceil(z / tile_z)),
            int(np.ceil(y / tile_y)),
            int(np.ceil(x / tile_x)),
            1,  # channel
        )
        print(f"Running segmentation on {identifier} (shape {dapi_norm.shape}, tiles {n_tiles})...")
        labels, _ = model.predict_instances(dapi_norm, n_tiles=n_tiles)

        tiff.imwrite(str(out_path), labels.astype(np.uint16))
        print(f"Saved: {out_path.name} | {labels.max()} nuclei detected")


if __name__ == "__main__":
    run_segmentation()
