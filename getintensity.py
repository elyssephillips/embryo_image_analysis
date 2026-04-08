import os
import tifffile as tiff
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import rotate
from src.utils import (
    load_config, 
    get_image_paths, 
    get_masks, 
    get_voxel_size_from_json, 
    erode_labels_optimized, 
    load_rotation_log,
    measure_nuclear_intensities
)

def run_extraction():
    config = load_config()
    voxel_size = get_voxel_size_from_json(config['metadata_json'])
    
    # Setup Paths
    rotated_dir = Path(config['rotated_dir'])
    seg_dir = Path(config['segmentation_dir'])
    output_dir = Path(config['output_dir'])
    os.makedirs(output_dir, exist_ok=True)

    # Calculate Erosion (1.0µm XY, 3.2µm Z)
    erode_z_px = int(round(3.2 / voxel_size[0]))
    erode_xy_px = int(round(1.0 / voxel_size[1]))
    
    # Load Data Tools
    image_files = get_image_paths(rotated_dir, extension="_rotated.tif")
    all_masks = get_masks(seg_dir)
    mask_map = {m.stem.replace('_segmentation', ''): m for m in all_masks}
    angle_map = load_rotation_log(config['rotation_log'])

    all_results = []

    for img_path in image_files:
        img_id = img_path.stem.replace('_rotated', '')
        mask_path = mask_map.get(img_id)
        angle = angle_map.get(img_id)
        
        if not mask_path or angle is None:
            print(f"Skipping {img_id}: Missing mask or rotation angle.")
            continue

        print(f"--- Processing: {img_id} (Angle: {angle}) ---")
        
        # Load Image and Mask
        img = tiff.imread(img_path)
        labels_unrotated = tiff.imread(mask_path)
        
        # Rotate Mask to match Image
        labels_rotated = rotate(labels_unrotated, angle=angle, axes=(1, 2), order=0, reshape=True)

        # Safety Check: Do they match?
        if labels_rotated.shape != (img.shape[0], img.shape[2], img.shape[3]):
            print(f"Shape mismatch for {img_id}. Skipping.")
            continue
        
        # Erode and Measure
        labels_eroded = erode_labels_optimized(labels_rotated, erode_xy_px, erode_z_px)
        nucleus_data = measure_nuclear_intensities(img, labels_eroded, config['microscopy']['channels'])
        
        for entry in nucleus_data:
            entry['image_id'] = img_id
            all_results.append(entry)

        # Save eroded mask for visual verification
        tiff.imwrite(output_dir / f"{img_id}_eroded_seg.tif", labels_eroded.astype(np.uint16))

    # Save RAW data
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / "nuclear_intensities_raw.csv", index=False)
    print(f"Raw data saved.")

if __name__ == "__main__":
    run_extraction()