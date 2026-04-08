from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import os
import yaml
import tifffile as tiff
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import pandas as pd
import json
import scipy.ndimage as ndi


def load_config(config_path='configs/config.yaml'):
    # Get the path of the directory where main.py is located
    base_path = Path(__file__).parent.parent 
    full_path = base_path / config_path
    
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)

def get_image_paths(directory, extension=".tif"):
    """
    Finds all files with a specific extension in a given directory.
    Default extension is '.tif' if none is provided.
    """
    path = Path(directory)
    # This looks for files ending in whatever extension you pass (e.g., _rotated.tif)
    return sorted(list(path.glob(f"*{extension}")))

def get_masks(directory):
    """
    Finds all .tif files in the provided directory.
    """
    path = Path(directory) # 'directory' is already the folder path string
    return sorted(list(path.glob("*.tif")))


def process_image_loading(file_path):
    """Loads image, prints stats, and generates a MIP."""
    img = tiff.imread(file_path)
    identifier = Path(file_path).stem
    
    print(f"--- Processing: {identifier} ---")
    print(f"Shape: {img.shape} | Dtype: {img.dtype}")
    
    # Generate Maximum Intensity Projection (MIP) if 3D/4D
    # Assumes Z is the first or second dimension; adjust as needed
    mip = np.max(img, axis=0) if img.ndim >= 3 else img
    
    return img, mip, identifier

def get_user_rotation(dapi_mip, identifier):
    """Displays the DAPI MIP and asks the user for a rotation angle."""
    plt.imshow(dapi_mip, cmap='gray')
    plt.title(f"Image: {identifier}\nEnter rotation angle in terminal (0-360):")
    plt.show(block=False) # Shows the window but doesn't freeze the script
    
    angle = float(input(f"Enter rotation angle for {identifier} (or 0 to skip): "))
    plt.close() # Close the image window after input
    return angle

def rotate_full_stack(img, angle):
    """Rotates the full 3D/4D stack on the Y-X plane."""
    if angle == 0:
        return img
        
    # Microscopy axes: (C, Z, Y, X) -> rotate Y and X (axes 2 and 3)
    # If your image is (Z, Y, X) -> rotate axes 1 and 2
    axes = (2, 3) if img.ndim == 4 else (1, 2)
    
    # reshape=False keeps the original pixel dimensions
    return nd.rotate(img, angle, axes=axes, reshape=True, order=1)

def log_rotation(output_path, identifier, angle):
    """Saves the rotation angle for a specific image to a CSV file."""
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, "rotation_log.csv")
    
    # Create the data entry
    new_data = pd.DataFrame({'identifier': [identifier], 'rotation_angle': [angle]})
    
    # If the file doesn't exist, write it with headers
    if not os.path.exists(log_file):
        new_data.to_csv(log_file, index=False)
    else:
        # Append without writing the header again
        new_data.to_csv(log_file, mode='a', header=False, index=False)
    
    print(f"Logged {angle}° for {identifier} in rotation_log.csv")

def get_voxel_size_from_json(json_path):
    z_default, xy_default = 0.8, 0.1226
    try:
        with open(json_path, 'r') as f:
            # If your JSON is a LIST of dictionaries, we need to grab the first one [0]
            data = json.load(f)
            if isinstance(data, list):
                data = data[0]
            
            v_size = data.get('voxel_size_um', {})
            z_um = v_size.get('depth', z_default)
            xy_um = v_size.get('width', xy_default)
            
            return [z_um, xy_um, xy_um]
    except Exception as e:
        return [z_default, xy_default, xy_default]
    

from skimage.measure import regionprops

def make_anisotropic_selem(erode_xy_px, erode_z_px):
    """
    Creates a 3D ellipsoid (structuring element) for erosion.
    If Z erosion is 0, it creates a 2D disk inside a 3D volume.
    """
    zz, rr = int(max(0, erode_z_px)), int(max(0, erode_xy_px))
    
    # Create a coordinate grid
    z, y, x = np.arange(-zz, zz+1), np.arange(-rr, rr+1), np.arange(-rr, rr+1)
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    
    # Ellipsoid formula: (z/az)^2 + (y/ay)^2 + (x/ax)^2 <= 1
    # We use denominators of 1 if the radius is 0 to avoid dividing by zero
    denom_z = zz if zz > 0 else 1
    denom_r = rr if rr > 0 else 1
    
    return (Z / denom_z)**2 + (Y / denom_r)**2 + (X / denom_r)**2 <= 1.0

def erode_labels_optimized(labels_zyx, erode_xy_px, erode_z_px):
    """
    Erodes each nucleus ID individually using bounding boxes for speed.
    This prevents nuclei from 'bleeding' into each other during erosion.
    """
    out = np.zeros_like(labels_zyx)
    selem = make_anisotropic_selem(erode_xy_px, erode_z_px)
    
    # regionprops finds the bounding box for every unique nucleus ID
    props = regionprops(labels_zyx)
    
    for prop in props:
        label_id = prop.label
        # Extract the bounding box coordinates [min_z, min_y, min_x, max_z, max_y, max_x]
        z0, y0, x0, z1, y1, x1 = prop.bbox
        
        # 1. Crop the specific nucleus (with a 1-pixel buffer for safety)
        # We add/subtract 1 to ensure the erosion has 'room' to work
        z_start, z_end = max(0, z0-1), min(labels_zyx.shape[0], z1+1)
        y_start, y_end = max(0, y0-1), min(labels_zyx.shape[1], y1+1)
        x_start, x_end = max(0, x0-1), min(labels_zyx.shape[2], x1+1)
        
        crop = labels_zyx[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # 2. Create a binary mask of just THIS nucleus
        mask = (crop == label_id)
        
        # 3. Erode just that small binary mask
        eroded_mask = ndi.binary_erosion(mask, structure=selem)
        
        # 4. Paste the result back into the full-sized output array
        out[z_start:z_end, y_start:y_end, x_start:x_end][eroded_mask] = label_id
        
    return out


def load_rotation_log(log_path):
    """
    Reads the rotation_log.csv.
    Columns: 'identifier', 'rotation_angle'
    """
    try:
        df = pd.read_csv(log_path)
        # Ensure 'identifier' is a string and remove '.tif' if it exists in the name
        df['identifier'] = df['identifier'].astype(str).str.replace('.tif', '', regex=False)
        
        # Create a dictionary for fast lookup: {'c_001': 45.2, 'c_002': -12.0}
        return dict(zip(df['identifier'], df['rotation_angle']))
    except Exception as e:
        print(f" ERROR: Could not read rotation log at {log_path}: {e}")
        return {}
    

def measure_nuclear_intensities(img_stack, labels, channels_dict):
    """
    Measures raw and DAPI-normalized intensities for all nuclei.
    img_stack: (Z, C, Y, X)
    labels: (Z, Y, X)
    channels_dict: {'dapi': 0, 'gata3': 1, ...}
    """
    results = []
    props = regionprops(labels)
    
    for prop in props:
        z0, y0, x0, z1, y1, x1 = prop.bbox
        mask = prop.image
        
        stats = {
            'nucleus_id': prop.label,
            'volume_voxels': prop.area,
            'center_z': prop.centroid[0],
            'center_y': prop.centroid[1],
            'center_x': prop.centroid[2]
        }
        
        # 1. Get DAPI value first
        d_idx = channels_dict.get('dapi', 0)
        d_crop = img_stack[z0:z1, d_idx, y0:y1, x0:x1]
        
        if d_crop.shape != mask.shape:
            continue
            
        dapi_mean = np.mean(d_crop[mask])
        stats['dapi_mean'] = dapi_mean
        
        # 2. Get other channels and normalize
        for name, idx in channels_dict.items():
            if name == 'dapi': continue
            
            ch_crop = img_stack[z0:z1, idx, y0:y1, x0:x1]
            if ch_crop.shape != mask.shape:
                continue
                
            mean_val = np.mean(ch_crop[mask])
            stats[f'{name}_mean'] = mean_val
            stats[f'{name}_dapi_norm'] = mean_val / (dapi_mean + 1e-6)
            
        results.append(stats)
        
    return results

def normalize_by_dapi(df, dapi_col='dapi_mean'):
    """
    Finds all columns ending in '_mean' (except dapi) 
    and creates new '_dapi_norm' columns.
    """
    mean_cols = [c for c in df.columns if '_mean' in c and c != dapi_col]
    for col in mean_cols:
        norm_name = col.replace('_mean', '_dapi_norm')
        df[norm_name] = df[col] / (df[dapi_col] + 1e-6)
    return df

def plot_embryo_heatmap(df, image_id, channel_name, ax=None, title=None):
    """
    Plots a spatial heatmap for a single embryo and specific channel.
    """
    import matplotlib.pyplot as plt
    
    # Filter for the specific embryo
    embryo_df = df[df['image_id'] == image_id]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot where color (c) is determined by the intensity
    # cmap='magma' or 'viridis' are great for fluorescence
    sc = ax.scatter(
        embryo_df['center_x'], 
        embryo_df['center_y'], 
        c=embryo_df[channel_name], 
        cmap='magma', 
        s=50,             # Size of the "nucleus" dot
        edgecolor='none'
    )
    
    ax.set_aspect('equal')
    ax.set_title(title or f"{image_id} - {channel_name}")
    ax.invert_yaxis() # Match microscopy orientation (Top-Left is 0,0)
    plt.colorbar(sc, ax=ax, label='Normalized Intensity')
    
    return ax

def map_values_to_labels(labels, df, column_name):
    """
    Creates a new 3D array where each label's pixels are replaced 
    by the value in df[column_name] for that specific label.
    """
    # Create a float array to hold the intensity values
    value_map = np.zeros(labels.shape, dtype=np.float32)
    
    # Create a dictionary for O(1) lookup: {label_id: value}
    # We use 'nucleus_id' because that's what's in your CSV
    lookup = dict(zip(df['nucleus_id'], df[column_name]))
    
    # Iterate through the unique labels present in this specific image
    unique_ids = np.unique(labels)
    for lid in unique_ids:
        if lid == 0: continue # Skip background
        
        val = lookup.get(lid, 0)
        value_map[labels == lid] = val
        
    return value_map

def calculate_patterning_score(df_sub, channel):
    """
    Calculates the 'Polarization Index' (0.0 to 1.0).
    0.0 = Uniform/Symmetric distribution
    1.0 = Maximum possible asymmetry (all signal at the edge)
    """
    if df_sub.empty or channel not in df_sub.columns:
        return 0
    
    # 1. Get weights (Intensity)
    weights = df_sub[channel].values + 1e-6
    
    # 2. Geometric Center (The physical middle)
    gc_x = df_sub['center_x'].mean()
    gc_y = df_sub['center_y'].mean()
    
    # 3. Weighted Center (The 'Center of Brightness')
    wc_x = np.sum(df_sub['center_x'] * weights) / np.sum(weights)
    wc_y = np.sum(df_sub['center_y'] * weights) / np.sum(weights)
    
    # 4. Calculate Raw Offset (Distance in pixels)
    raw_offset = np.sqrt((wc_x - gc_x)**2 + (wc_y - gc_y)**2)
    
    # 5. Calculate Average Radius of the Embryo
    # Distance of every nucleus from the geometric center
    distances = np.sqrt((df_sub['center_x'] - gc_x)**2 + (df_sub['center_y'] - gc_y)**2)
    avg_radius = distances.max() # Use the furthest nucleus as the boundary
    
    if avg_radius == 0: return 0
    
    # 6. Final Normalized Score
    # This represents how far the 'Center of Mass' has shifted as a % of radius
    polarization_index = raw_offset / avg_radius
    
    return polarization_index

def update_master_study_log(stat_df, dataset_name, project_root="."):
    """
    Appends high-level metadata to a master log file in the project root.
    """
    from datetime import datetime
    
    # Ensure the log stays in the main project folder
    master_log_path = Path(project_root) / "master_study_log.csv"
    
    # Calculate high-level stats for this batch
    summary_data = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'dataset_id': dataset_name,
        'n_total': len(stat_df),
        'n_control': len(stat_df[stat_df['group'] == 'Control']),
        'n_treated': len(stat_df[stat_df['group'] == 'Treated']),
        'avg_pearson_r': stat_df['pearson_r'].mean(),
        'avg_gata3_y_slope': stat_df['gata3_y_slope'].mean(),
        'avg_polarization': stat_df['pattern_score'].mean()
    }
    
    new_entry = pd.DataFrame([summary_data])

    if not master_log_path.exists():
        new_entry.to_csv(master_log_path, index=False)
        print(f"🆕 Created NEW Master Study Log: {master_log_path}")
    else:
        new_entry.to_csv(master_log_path, mode='a', header=False, index=False)
        print(f"📖 Appended stats to Master Study Log.")

