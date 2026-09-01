import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import numpy as np
import tifffile as tiff
from src.io import load_config, get_image_paths, log_rotation, get_storage_note, get_config_notes, get_config_n_conditions, summarize_config_metadata
from src.log import sync_dataset_fields, sync_notes
from src.conversion import load_hyperstack_czyx
from src.image import get_user_rotation, rotate_full_stack


#note: rotation direction is counter clockwise!

def run_rotation():
    config = load_config()
    _dataset_id = config.get("datasets", "")
    sync_dataset_fields("IF", _dataset_id, storage=get_storage_note(),
                         data_path=config.get("raw_data_dir", ""),
                         n_conditions=get_config_n_conditions(), **summarize_config_metadata(config))
    sync_notes("IF", _dataset_id, get_config_notes())
    raw_dir = Path(config['raw_data_dir'])
    rotated_dir = Path(config['rotated_dir'])
    os.makedirs(rotated_dir, exist_ok=True)

    image_files = get_image_paths(raw_dir, extension=".tif")
    dapi_channel = config['microscopy']['channels']['dapi']

    for img_path in image_files:
        identifier = Path(img_path).stem
        print(f"--- Processing: {identifier} ---")
        img, _meta = load_hyperstack_czyx(img_path)  # (Z, C, Y, X)
        print(f"Shape: {img.shape} | Dtype: {img.dtype}")
        dapi_mip = np.max(img[:, dapi_channel], axis=0)
        angle = get_user_rotation(dapi_mip, identifier)
        rotated = rotate_full_stack(img, angle)
        out_path = rotated_dir / f"{identifier}_rotated.tif"
        tiff.imwrite(str(out_path), rotated, imagej=True)
        print(f"Saved: {out_path.name}")
        log_rotation(str(Path(config['rotation_log']).parent), identifier, angle)


if __name__ == "__main__":
    run_rotation()
