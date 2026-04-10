import os
import tifffile as tiff
from pathlib import Path
from src.io import load_config, get_image_paths, process_image_loading, log_rotation
from src.image import get_user_rotation, rotate_full_stack


def run_rotation():
    config = load_config()
    raw_dir = Path(config['raw_data_dir'])
    rotated_dir = Path(config['rotated_dir'])
    os.makedirs(rotated_dir, exist_ok=True)

    image_files = get_image_paths(raw_dir, extension=".tif")
    dapi_channel = config['microscopy']['channels']['dapi']

    for img_path in image_files:
        img, mip, identifier = process_image_loading(img_path)
        # mip is (C, Y, X) for a 4D stack — grab just the DAPI channel
        dapi_mip = mip[dapi_channel] if mip.ndim == 3 else mip
        angle = get_user_rotation(dapi_mip, identifier)
        rotated = rotate_full_stack(img, angle)
        out_path = rotated_dir / f"{identifier}_rotated.tif"
        tiff.imwrite(str(out_path), rotated)
        print(f"Saved: {out_path.name}")
        log_rotation(str(rotated_dir), identifier, angle)


if __name__ == "__main__":
    run_rotation()
