from __future__ import annotations
from pathlib import Path
from datetime import datetime
import os
import yaml
import tifffile as tiff
import numpy as np
import pandas as pd
import json


def load_config(config_path='configs/config.yaml'):
    base_path = Path(__file__).parent.parent
    full_path = base_path / config_path
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)


def get_image_paths(directory, extension=".tif"):
    """Finds all files with a specific extension in a given directory."""
    path = Path(directory)
    return sorted(list(path.glob(f"*{extension}")))


def get_masks(directory):
    """Finds all .tif files in the provided directory."""
    path = Path(directory)
    return sorted(list(path.glob("*.tif")))


def process_image_loading(file_path):
    """Loads image, prints stats, and generates a MIP."""
    img = tiff.imread(file_path)
    identifier = Path(file_path).stem
    print(f"--- Processing: {identifier} ---")
    print(f"Shape: {img.shape} | Dtype: {img.dtype}")
    mip = np.max(img, axis=0) if img.ndim >= 3 else img
    return img, mip, identifier


def get_voxel_size_from_json(json_path):
    z_default, xy_default = 0.8, 0.1226
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                data = data[0]
            v_size = data.get('voxel_size_um', {})
            z_um = v_size.get('depth', z_default)
            xy_um = v_size.get('width', xy_default)
            return [z_um, xy_um, xy_um]
    except Exception:
        return [z_default, xy_default, xy_default]


def load_rotation_log(log_path):
    """Reads rotation_log.csv and returns a dict of {identifier: angle}."""
    try:
        df = pd.read_csv(log_path)
        df['identifier'] = df['identifier'].astype(str).str.replace('.tif', '', regex=False)
        return dict(zip(df['identifier'], df['rotation_angle']))
    except Exception as e:
        print(f" ERROR: Could not read rotation log at {log_path}: {e}")
        return {}


def log_rotation(output_path, identifier, angle):
    """Saves the rotation angle for a specific image to rotation_log.csv."""
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, "rotation_log.csv")
    new_data = pd.DataFrame({'identifier': [identifier], 'rotation_angle': [angle]})
    if not os.path.exists(log_file):
        new_data.to_csv(log_file, index=False)
    else:
        new_data.to_csv(log_file, mode='a', header=False, index=False)
    print(f"Logged {angle}° for {identifier} in rotation_log.csv")


def update_master_study_log(stat_df, dataset_name, project_root="."):
    """Appends high-level metadata to a master log file in the project root."""
    master_log_path = Path(project_root) / "master_study_log.csv"
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
        print(f"Created NEW Master Study Log: {master_log_path}")
    else:
        new_entry.to_csv(master_log_path, mode='a', header=False, index=False)
        print(f"Appended stats to Master Study Log.")
