import yaml
import numpy as np
import pandas as pd
import btrack
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

REG_CSV      = Path(cfg['paths']['features_registered_csv'])
RAW_CSV      = Path(cfg['paths']['features_csv'])
FEATURES_CSV = REG_CSV if REG_CSV.exists() else RAW_CSV
CONFIG_JSON  = cfg['paths']['btrack_config_json']
OUT_DIR      = Path(cfg['paths']['btrack_output_dir'])
VERSION      = 'v1'

df = pd.read_csv(FEATURES_CSV)
registered = 'z_um_reg' in df.columns
print(f'Loaded {len(df)} detections across {df["t"].nunique()} timepoints '
      f'({"registered" if registered else "raw"} coords)', flush=True)

# Use registered coords for tracking if available
z_col = 'z_um_reg' if registered else 'z_um'
y_col = 'y_um_reg' if registered else 'y_um'
x_col = 'x_um_reg' if registered else 'x_um'

locs = df[['t', z_col, y_col, x_col]].values
objects = btrack.io.objects_from_array(locs, default_keys=['t', 'z', 'y', 'x'])

for obj, (_, row) in zip(objects, df.iterrows()):
    obj.properties['area_vox']       = int(row['area_vox'])
    obj.properties['area_um3']       = float(row['area_um3'])
    obj.properties['mean_intensity'] = float(row['mean_intensity'])
    obj.properties['max_intensity']  = float(row['max_intensity'])
    obj.properties['label_id']       = int(row['label_id'])

print(f'Created {len(objects)} btrack objects', flush=True)

with btrack.BayesianTracker() as tracker:
    tracker.configure(str(CONFIG_JSON))
    tracker.append(objects)
    # Large volume so no nuclei are classified as border events
    # Data ranges: z=1.6-113.6, y=4.8-207.3, x=78.3-286.6 (all in µm)
    tracker.volume = ((-1e4, 1e4), (-1e4, 1e4), (-1e4, 1e4))
    tracker.track()
    tracker.optimize()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tracker.export(str(OUT_DIR / f'tracks_{VERSION}_raw.h5'), obj_type='obj_type_1')
    data, properties, graph = tracker.to_napari()
    tracks_raw = tracker.tracks

    # Build full DataFrame with properties inside context manager
    records = []
    for trk in tracks_raw:
        prop_names = list(trk.properties.keys())
        prop_arrays = [trk.properties[p] for p in prop_names]
        for t, z, y, x, *props in zip(trk.t, trk.z, trk.y, trk.x, *prop_arrays):
            row = {'track_id': trk.ID, 't': t, 'z_um': z, 'y_um': y, 'x_um': x}
            for name, val in zip(prop_names, props):
                if isinstance(val, (np.ndarray, list)):
                    val = val[0]
                row[name] = val
            records.append(row)

tracks_df = pd.DataFrame(records).sort_values(['track_id', 't']).reset_index(drop=True)

np.savez_compressed(OUT_DIR / f'tracks_{VERSION}_napari_raw.npz',
                    data=data, properties=properties, graph=graph)
tracks_df.to_csv(OUT_DIR / f'tracks_{VERSION}_raw.csv', index=False)

lengths = tracks_df.groupby('track_id')['t'].count()
print(f'\nbtrack found {len(lengths)} tracks', flush=True)
print(f'Full timecourse (101): {(lengths==101).sum()}', flush=True)
print(f'>=50 frames:           {(lengths>=50).sum()}', flush=True)
print(f'>=10 frames:           {(lengths>=10).sum()}', flush=True)
print(f'1-2 frames:            {(lengths<=2).sum()}', flush=True)
print(f'Median length:         {lengths.median():.0f}', flush=True)
print(f'\nSaved: tracks_{VERSION}_raw.csv', flush=True)
