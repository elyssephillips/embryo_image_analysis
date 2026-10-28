"""
Shared dataset-config loading for pipelines/tracking scripts.

To switch dataset, edit pipelines/tracking/active_dataset.yaml (one line).

Every script resolves its config the same way — first match wins:
  1. --config PATH  (or --config=PATH)
  2. a bare *.yaml / *.yml argument  (legacy: `python 05_erk_cn_ratio.py configs/tracking/x.yaml`)
  3. the TRACKING_CONFIG environment variable
  4. pipelines/tracking/active_dataset.yaml  (what the VS Code Run button uses)

Relative paths resolve against the repo root. The config argument is removed
from sys.argv, so each script's own argparse never sees it.

To switch every script to a new dataset for a shell session:
  export TRACKING_CONFIG=configs/tracking/dataset003_icm_te_250914_stack5.yaml
"""

import json
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

ACTIVE_DATASET_FILE = REPO_ROOT / 'pipelines' / 'tracking' / 'active_dataset.yaml'


def _active_config():
    with open(ACTIVE_DATASET_FILE) as f:
        return yaml.safe_load(f)['config']


DEFAULT_CONFIG = _active_config()


def _pop_config_arg(argv):
    """Remove and return the config path from argv (in place), or None."""
    for i, arg in enumerate(argv[1:], start=1):
        if arg == '--config':
            if i + 1 >= len(argv):
                sys.exit('--config needs a path')
            path = argv[i + 1]
            del argv[i:i + 2]
            return path
        if arg.startswith('--config='):
            del argv[i]
            return arg.split('=', 1)[1]
        if not arg.startswith('-') and arg.endswith(('.yaml', '.yml')):
            del argv[i]
            return arg
    return None


def resolve_config_path(path=None):
    path = path or os.environ.get('TRACKING_CONFIG') or DEFAULT_CONFIG
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_tracking_config(path=None):
    """Return (cfg, config_path). With path=None, reads it from sys.argv / env / default."""
    config_path = resolve_config_path(path or _pop_config_arg(sys.argv))
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    print(f'Config: {config_path}  (tracks: {cfg["tracking"]["input_version"]})')
    return cfg, config_path


def tracks_csv(cfg):
    """Path to the curated tracks CSV that seeds the analysis (05, 12).

    Read from paths.tracks_dir, falling back to linked_tracks_dir for
    configs (e.g. dataset001) that predate tracks_dir.
    """
    tracks_dir = cfg['paths'].get('tracks_dir') or cfg['paths']['linked_tracks_dir']
    return Path(tracks_dir) / f'tracks_{cfg["tracking"]["input_version"]}.csv'


def crop_info_path(cfg):
    """prepare_inference.py's crop_info.json: paths.crop_info, else its default
    spot two levels above nnunet_dir (<output_dir>/predictions/<sub>/)."""
    p = cfg['paths'].get('crop_info')
    return Path(p) if p else Path(cfg['paths']['nnunet_dir']).parents[1] / 'crop_info.json'


def load_crop_info(cfg):
    """{tp_id: {first_z, last_z, original_shape, cropped_shape, ...}}.

    Stacks prepared with prepare_inference.py's UNIFORM_Z_CROP (stack2, stack8
    and later) share one first_z/depth across all timepoints and have no
    debris_shift_slices, so the per-frame offset handling in the consumers is
    a constant shift for them. Only 250914_stack5 (prepared before that) has
    a different z-crop per frame plus debris shifts
    (scripts/fix_debris_crop_offsets.py).
    """
    with open(crop_info_path(cfg)) as f:
        return json.load(f)


def implantation_onset(cfg):
    """biology.implantation_onset_t: splits pre/post-implantation windows and
    is the reference frame (± 2) for per-track ICM distance. Default 30
    (dataset001's onset)."""
    return int((cfg.get('biology') or {}).get('implantation_onset_t', 30))


def analysis_windows(cfg):
    """Per-track summary windows for 06-08, from the optional `analysis` block.

    onset       implantation_onset_t - pre is t_start <= t < onset
    during_end  last frame of the during-implantation window (default t_end)
    end_frames  'end' = the movie's last end_frames frames (default 20)
    min_frames  windows with fewer frames than this give NaN (default 3)
    """
    a = cfg.get('analysis') or {}
    during_end = a.get('during_end_t')
    return {
        'onset':      implantation_onset(cfg),
        'during_end': int(cfg['tracking']['t_end'] if during_end is None else during_end),
        'end_frames': int(a.get('end_frames', 20)),
        'min_frames': int(a.get('min_window_frames', 3)),
    }
