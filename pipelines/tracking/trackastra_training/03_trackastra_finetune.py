"""
Fine-tune Trackastra's pretrained 3D "ctc" model on the curated windows
exported by 03_trackastra_export_ctc.py, holding one window out per fold.

Each fold trains on the other windows and validates on the held-out one;
train.py keeps the lowest-val-loss epoch as {run_dir}/model.pt + config.yaml,
loadable with Trackastra.from_folder(run_dir). Note that the held-out window
also picks which epoch is kept, so its score is mildly optimistic.

Runs the trackastra repo's scripts/train.py (cloned at TRAIN_SCRIPT, tag
matching the installed trackastra) with configs/tracking/trackastra_finetune.yaml.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.tracking_config import load_tracking_config
cfg, CONFIG_PATH = load_tracking_config()   # --config X.yaml, else TRACKING_CONFIG / default

TRAIN_DIR     = Path(cfg['paths']['trackastra_train_dir'])
FINETUNE_CFG  = REPO_ROOT / 'configs' / 'tracking' / 'trackastra_finetune.yaml'
TRAIN_SCRIPT  = Path('/mnt/md0/elysse/code/trackastra/scripts/train.py')
PRETRAINED    = Path.home() / '.local' / 'share' / 'trackastra' / 'models' / 'ctc'

# Exported window versions: folder names under TRAIN_DIR, or '{dataset}:{version}'
# for another dataset's window, exported with its own --config into its own
# trackastra_train_dir (dataset = configs/tracking/dataset003_icm_te_{dataset}.yaml).
# Other datasets' registered coords are v2, so they need --registration v2.
WINDOWS = ['linked_c3_winA', 'linked_c3_winB', 'linked_c3_winC', 'linked_c2_winD',
           'linked_c2_winE', 'linked_c4_winF', 'linked_c2_winG',
           # 2026-10-21, full z, tumbling/collapse stage, seeded from ft_reg2_final
           'stack2:linked_c1_win6_12', 'stack8:linked_c1_win15_21', 'stack11:linked_c2_win2_8',
           # 2026-10-22, full z, dense late stage, seeded from ft_reg2_final
           'stack2:linked_c1_win100_106', 'stack8:linked_c1_win100_106', 'stack11:linked_c1_win100_106']

# Windows sharing timepoints/nuclei: when one is held out, the other is left
# out of training too, or the model would be tested on nuclei it trained on.
# (D is t70-75 full z, seeded from B's t70-80 band curation.)
OVERLAPS = {
    'linked_c3_winB': ['linked_c2_winD'],
    'linked_c2_winD': ['linked_c3_winB'],
}

# Which windows to hold out, one fold each. A holdout can be several windows
# joined with '+' (e.g. both of one dataset's windows, for leave-one-embryo-out):
# all are left out of training and all validate.
HOLDOUTS = WINDOWS

# Also (or only, with HOLDOUTS = []) train the production model on every
# window. Nothing is held out, so the length is fixed instead of picked on a
# test window: FINAL_EPOCHS from the folds' typical best epoch (2026-10-01
# folds peaked at epochs 7, 1, 1, 0; the 2026-10-22 13-window folds at
# 7, 3, 3, 1, 3, 1, 2, 5, 0, 1 - 0-based, median 2.5, so 4 epochs). train.py still needs a validation set
# to save a checkpoint, so it gets the last window - which it also trains on,
# so that loss only decides which of the FINAL_EPOCHS epochs is kept.
TRAIN_FINAL = False
FINAL_EPOCHS = 4

# Train on a variant export of the training windows, e.g. '_silver20' from
# 03_trackastra_export_ctc.py's SILVER_SOURCE. Validation always uses the
# held-out window's plain curated export, so checkpoints are picked on real labels.
TRAIN_SUFFIX = ''

# Train (and validate) on registered centroids - see 03_trackastra_train_registered.py.
# Run the resulting model with 03_run_trackastra.py's REGISTER = True.
REGISTERED = True

# Swap in mesh-based shape features (mesh_feats.csv, from
# 03_trackastra_export_extras.py). Run the resulting models with
# 03_run_trackastra.py --mesh-feats.
MESH_FEATS = False

# Which registration's centroids to train on: 'v1' = registered_coords.csv
# (from the dataset's features_registered_csv, what the curated windows were
# exported with) or 'v2' = registered_coords_v2.csv (the config's
# training_registration_dir, written by 03_trackastra_export_extras.py).
# Run v2 models with 03_run_trackastra.py --registered-csv <that dir>/features_registered.csv.
REGISTRATION = 'v1'

parser = argparse.ArgumentParser()
parser.add_argument('--mesh-feats', action='store_true', default=None)
parser.add_argument('--registration', choices=['v1', 'v2'], default=None)
parser.add_argument('--holdouts', nargs='*', default=None,
                    help="window versions to hold out (one fold each; join with '+' to hold out "
                         "several together); none = no folds")
parser.add_argument('--final', action='store_true', default=None, help='also train the all-windows model')
parser.add_argument('--tag', default='', help='added to run names, e.g. xs -> ft_reg2_xs_holdout_...')
args = parser.parse_args()
if args.mesh_feats is not None: MESH_FEATS = args.mesh_feats
if args.registration is not None: REGISTRATION = args.registration
COORDS_FILE = {'v1': 'registered_coords.csv', 'v2': 'registered_coords_v2.csv'}[REGISTRATION]
if args.holdouts is not None:   HOLDOUTS = args.holdouts
if args.final is not None:      TRAIN_FINAL = args.final
assert REGISTERED or not MESH_FEATS, '--mesh-feats needs REGISTERED = True'


def win_dir(w: str) -> Path:
    if ':' not in w:
        return TRAIN_DIR / w
    dataset, version = w.split(':')
    with open(REPO_ROOT / 'configs' / 'tracking' / f'dataset003_icm_te_{dataset}.yaml') as f:
        return Path(yaml.safe_load(f)['paths']['trackastra_train_dir']) / version


def gt_dir(w: str) -> Path:
    d = win_dir(w)
    return d.with_name(f'{d.name}_GT') / 'TRA'


assert REGISTRATION == 'v2' or not any(':' in w for w in WINDOWS), \
    "other datasets' windows have v2 registered coords - train with --registration v2"
for w in WINDOWS:
    for v in {w, w + TRAIN_SUFFIX}:
        assert win_dir(v).is_dir() and gt_dir(v).is_dir(), \
            f'{v} not exported - run 03_trackastra_export_ctc.py first'
        for extra in (['mesh_feats.csv'] if MESH_FEATS else []) + [COORDS_FILE]:
            assert (gt_dir(v) / extra).exists(), \
                f'{v}: {extra} missing - run 03_trackastra_export_extras.py first'
assert (PRETRAINED / 'model.pt').exists(), f'pretrained model missing at {PRETRAINED}'


def _cachedir(inputs: list[str]) -> Path:
    # trackastra keys its dataset cache on arguments only, so a window
    # re-exported under the same name would silently reuse stale data - key
    # the cache folder on the exported track files' contents instead.
    seed = str(REGISTERED) + ('|mesh' if MESH_FEATS else '') + ('|v2' if REGISTRATION == 'v2' else '')
    h = hashlib.sha256(seed.encode())
    for w in sorted(inputs):
        for f in ('man_track.txt', COORDS_FILE, 'mesh_feats.csv' if MESH_FEATS else ''):
            if not f:
                continue
            p = gt_dir(w) / f
            h.update(p.read_bytes() if p.exists() else b'')
    return TRAIN_DIR / '.cache' / h.hexdigest()[:16]


prefix = ((('ft_reg2' if REGISTRATION == 'v2' else 'ft_reg') if REGISTERED else 'ft') + ('_mesh' if MESH_FEATS else '')
          + (f'_{args.tag}' if args.tag else ''))
env = {**os.environ, 'TRACKASTRA_MESH_FEATS': '1' if MESH_FEATS else '0', 'TRACKASTRA_COORDS_FILE': COORDS_FILE}
jobs = []  # (run name, train windows, val window, extra train.py args)
for holdout in HOLDOUTS:
    held = holdout.split('+')
    excluded = {h2 for h in held for h2 in (h, *OVERLAPS.get(h, []))}
    train = [w + TRAIN_SUFFIX for w in WINDOWS if w not in excluded]
    datasets = {h.split(':')[0] for h in held if ':' in h}
    if len(held) > 1 and len(datasets) == 1 and all(':' in h for h in held):
        name = f'{datasets.pop()}_all'               # every listed window of one dataset
    else:
        name = '+'.join(h.replace(':', '_') for h in held)
    jobs.append((f'{prefix}{TRAIN_SUFFIX}_holdout_{name}', train, held, []))
if TRAIN_FINAL:
    jobs.append((f'{prefix}{TRAIN_SUFFIX}_final_all', [w + TRAIN_SUFFIX for w in WINDOWS], [WINDOWS[-1]],
                 ['--epochs', str(FINAL_EPOCHS)]))

script = Path(__file__).resolve().parent / '03_trackastra_train_registered.py' if REGISTERED else TRAIN_SCRIPT
for name, train, val, extra in jobs:
    cmd = [
        sys.executable, str(script),
        '-c', str(FINETUNE_CFG),
        '--name', name,
        '--model', str(PRETRAINED),
        '--input_train', *[str(win_dir(w)) for w in train],
        '--input_val', *[str(win_dir(v)) for v in val],
        '--outdir', str(TRAIN_DIR / 'runs'),
        '--cachedir', str(_cachedir([*train, *val])),
        *extra,
    ]
    print(f'\n=== {name} | train on {train} | val {val}', flush=True)
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=TRAIN_SCRIPT.parent, check=True, env=env)

print(f'\nRuns in {TRAIN_DIR / "runs"} - load a run with Trackastra.from_folder(run_dir).')
