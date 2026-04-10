from __future__ import annotations
import numpy as np
import pandas as pd
from skimage.measure import regionprops


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
        d_idx = channels_dict.get('dapi', 0)
        d_crop = img_stack[z0:z1, d_idx, y0:y1, x0:x1]
        if d_crop.shape != mask.shape:
            continue
        dapi_mean = np.mean(d_crop[mask])
        stats['dapi_mean'] = dapi_mean
        for name, idx in channels_dict.items():
            if name == 'dapi':
                continue
            ch_crop = img_stack[z0:z1, idx, y0:y1, x0:x1]
            if ch_crop.shape != mask.shape:
                continue
            mean_val = np.mean(ch_crop[mask])
            stats[f'{name}_mean'] = mean_val
            stats[f'{name}_dapi_norm'] = mean_val / (dapi_mean + 1e-6)
        results.append(stats)
    return results


def normalize_by_dapi(df, dapi_col='dapi_mean'):
    """Creates _dapi_norm columns for all _mean columns (except dapi)."""
    mean_cols = [c for c in df.columns if '_mean' in c and c != dapi_col]
    for col in mean_cols:
        norm_name = col.replace('_mean', '_dapi_norm')
        df[norm_name] = df[col] / (df[dapi_col] + 1e-6)
    return df


def map_values_to_labels(labels, df, column_name):
    """
    Creates a new 3D array where each label's pixels are replaced
    by the value in df[column_name] for that specific label.
    """
    value_map = np.zeros(labels.shape, dtype=np.float32)
    lookup = dict(zip(df['nucleus_id'], df[column_name]))
    unique_ids = np.unique(labels)
    for lid in unique_ids:
        if lid == 0:
            continue
        val = lookup.get(lid, 0)
        value_map[labels == lid] = val
    return value_map


def calculate_patterning_score(df_sub, channel):
    """
    Calculates the Polarization Index (0.0 to 1.0).
    0.0 = uniform distribution, 1.0 = maximum asymmetry (all signal at edges).
    Measures weighted center-of-mass offset relative to geometric center.
    """
    if df_sub.empty or channel not in df_sub.columns:
        return 0
    weights = df_sub[channel].values + 1e-6
    gc_x = df_sub['center_x'].mean()
    gc_y = df_sub['center_y'].mean()
    wc_x = np.sum(df_sub['center_x'] * weights) / np.sum(weights)
    wc_y = np.sum(df_sub['center_y'] * weights) / np.sum(weights)
    raw_offset = np.sqrt((wc_x - gc_x)**2 + (wc_y - gc_y)**2)
    distances = np.sqrt((df_sub['center_x'] - gc_x)**2 + (df_sub['center_y'] - gc_y)**2)
    avg_radius = distances.max()
    if avg_radius == 0:
        return 0
    return raw_offset / avg_radius
