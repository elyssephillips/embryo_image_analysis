from src.utils import load_config, get_image_paths, process_image_loading, get_user_rotation, rotate_full_stack, log_rotation
import os
import tifffile as tiff
import numpy as np
from tqdm import tqdm

def run_pipeline():
    config = load_config()
   # 1. Define dapi_idx 
    channels = config['microscopy']['channels']
    dapi_idx = channels.get('dapi', 0) 
    
    image_files = get_image_paths(config['raw_data_dir'], config['file_extension'])
    print(f"Found {len(image_files)} images.")

    rotated_path = config['rotated_dir']
    os.makedirs(rotated_path, exist_ok=True)

    for f_path in image_files:
        # img shape is (Z, C, Y, X) -> (163, 4, 1044, 1182)
        img, mip, img_id = process_image_loading(f_path)
        
        # 2. Extract DAPI using the correct axis
        # ":" means "take everything in this dimension"
        dapi_stack = img[:, dapi_idx, :, :] 
        
        # 3. Create a MIP of just the DAPI for the rotation check
        dapi_mip = np.max(dapi_stack, axis=0)
        
        # 4. Interaction & Rotation
        angle = get_user_rotation(dapi_mip, img_id)
        
        # Note: In nd.rotate, specify axes=(2, 3) for Y and X
        rotated_img = rotate_full_stack(img, angle)
        
        # 5. Save and Log
        save_path = os.path.join(rotated_path, f"{img_id}_rotated.tif")
        tiff.imwrite(save_path, rotated_img.astype(img.dtype), imagej=True)
        log_rotation(config['output_dir'], img_id, angle)

if __name__ == "__main__":
    run_pipeline()