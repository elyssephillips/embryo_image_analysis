import yaml
import numpy as np
import pandas as pd
import btrack
from btrack.constants import BayesianUpdateFeatures
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset003_icm_te_250914_stack5.yaml'  # edit to switch dataset
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

REG_CSV      = Path(cfg['paths']['features_registered_csv'])
RAW_CSV      = Path(cfg['paths']['features_csv'])
FEATURES_CSV = REG_CSV if REG_CSV.exists() else RAW_CSV
CONFIG_JSON  = cfg['paths']['btrack_config_json']
OUT_DIR      = Path(cfg['paths']['btrack_output_dir'])
N_TIMEPOINTS = cfg['microscopy']['n_timepoints']
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

# Visual (appearance) features for VISUAL tracking updates below - z-scored
# across the whole dataset first, since btrack combines features into the
# update *unnormalized* (confirmed in btrack/btypes.py: PyTrackObject.set_features
# concatenates raw values with no scaling). Without this, area_um3 (mean
# ~1085, up to 8733) would swamp mean_intensity (mean ~338, up to 990) in any
# combined distance - not because intensity doesn't matter, but purely
# because of unit/magnitude mismatch.
mean_i, std_i = df['mean_intensity'].mean(), df['mean_intensity'].std()
mean_a, std_a = df['area_um3'].mean(), df['area_um3'].std()
df['mean_intensity_z'] = (df['mean_intensity'] - mean_i) / std_i
df['area_um3_z']       = (df['area_um3'] - mean_a) / std_a

for obj, (_, row) in zip(objects, df.iterrows()):
    obj.properties['area_vox']         = int(row['area_vox'])
    obj.properties['area_um3']         = float(row['area_um3'])
    obj.properties['mean_intensity']   = float(row['mean_intensity'])
    obj.properties['max_intensity']    = float(row['max_intensity'])
    obj.properties['label_id']         = int(row['label_id'])
    obj.properties['mean_intensity_z'] = float(row['mean_intensity_z'])
    obj.properties['area_um3_z']       = float(row['area_um3_z'])

print(f'Created {len(objects)} btrack objects', flush=True)

with btrack.BayesianTracker() as tracker:
    tracker.configure(str(CONFIG_JSON))
    tracker.append(objects)
    # Large volume so no nuclei are classified as border events
    # Data ranges: z=1.6-113.6, y=4.8-207.3, x=78.3-286.6 (all in µm)
    tracker.volume = ((-1e4, 1e4), (-1e4, 1e4), (-1e4, 1e4))

    # By default btrack tracks on motion (position) alone - cell_config.json
    # has no ObjectModel/features configured (ObjectModel is for cell-state
    # HMMs like interphase/mitosis, a different thing - not appearance).
    # Adding VISUAL alongside MOTION brings in intensity/size at the
    # hypothesis-scoring stage itself, which should make ambiguous cases
    # (touching/crowded nuclei with similar motion) less symmetric for GLPK -
    # tried per the user's hypothesis that this might also make the ILP
    # easier to solve to true optimality, not just improve track quality.
    tracker.configuration.features = ['mean_intensity_z', 'area_um3_z']
    tracker.configuration.tracking_updates = [
        BayesianUpdateFeatures.MOTION, BayesianUpdateFeatures.VISUAL,
    ]

    tracker.track()
    # tracker.optimize() with no args passes options=None straight through to
    # GLPK, which overrides GLPK's own built-in tm_lim default (60s) rather
    # than falling back to it (cell_config.json has no optimizer_options
    # block either) - confirmed on this dataset: GLPK ran for >25 hours with
    # no time limit, even though the MIP objective was already within 0.1% of
    # its final value within the log's very first entries - the rest was pure
    # branch-and-bound spent proving optimality, not improving the solution.
    # tm_lim is in milliseconds; GLPK returns its best-found solution so far
    # when it expires, not an error.
    tracker.optimize(tm_lim=120000)  # 2 min cap
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
print(f'Full timecourse ({N_TIMEPOINTS}): {(lengths==N_TIMEPOINTS).sum()}', flush=True)
print(f'>=50 frames:           {(lengths>=50).sum()}', flush=True)
print(f'>=10 frames:           {(lengths>=10).sum()}', flush=True)
print(f'1-2 frames:            {(lengths<=2).sum()}', flush=True)
print(f'Median length:         {lengths.median():.0f}', flush=True)
print(f'\nSaved: tracks_{VERSION}_raw.csv', flush=True)
