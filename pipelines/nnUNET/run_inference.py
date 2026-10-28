"""
Run Dataset003 nnUNet inference for one stack → 3-class predictions that feed
00_preprocess_segmentation_dataset003.py.

Two steps, toggled in the CONFIG block below:

  1. Prep - prepare_inference.py with this config's
     nnunet_inference paths. Crops every timepoint (Cam_long and Cam_short) to
     the same z range (UNIFORM_Z_CROP: union of all frames' signal +
     SAFETY_SLICES), writes imagesTs/, imagesTs_gpuN/, cam_short_cropped/ and
     crop_info.json under nnunet_inference.output_dir. A uniform crop means
     voxel index z is the same physical depth in every frame - 250914_stack5
     had per-frame crops and needed first_z padding workarounds downstream.
  2. Predict - one nnUNetv2_predict per GPU in nnunet_inference.gpus, each on
     its imagesTs_gpuN chunk, all writing to paths.nnunet_dir (safe: the
     timepoint chunks never overlap). Logs go to predict_gpuN.log next to the
     data. Blocks until every GPU finishes.

Everything runs in the nnunet_inference.conda_env env: launched from any other
interpreter (e.g. napari_env), the script re-execs itself there via `conda run`.

See inference_commands.md for model details.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.tracking_config import load_tracking_config

# Which stack: pipelines/tracking/active_dataset.yaml (VS Code Run), or --config X.yaml
cfg, CONFIG_PATH = load_tracking_config()

# =============================================================================
#  CONFIG
# =============================================================================

RUN_PREP    = True   # crop + write nnUNet inputs (re-running overwrites them)
RUN_PREDICT = True   # nnUNetv2_predict on every GPU in nnunet_inference.gpus

INF       = cfg['nnunet_inference']
OUT_DIR   = Path(INF['output_dir'])
PRED_DIR  = Path(cfg['paths']['nnunet_dir'])
CONDA_ENV = INF['conda_env']
GPUS      = INF['gpus']

PREP_SCRIPT = Path(__file__).resolve().parent / 'prepare_inference.py'

# Re-launch in the nnUNet env if started from another interpreter (napari_env etc.)
if Path(sys.prefix).name != CONDA_ENV:
    print(f'Re-launching in conda env {CONDA_ENV!r} (was {sys.prefix})', flush=True)
    # pass the resolved config on explicitly, so the relaunch can't pick up a different one
    os.execvp('conda', ['conda', 'run', '--no-capture-output', '-n', CONDA_ENV,
                        'python', str(Path(__file__).resolve()), '--config', str(CONFIG_PATH), *sys.argv[1:]])

ENV_BIN = Path(sys.executable).parent

# =============================================================================

if RUN_PREP:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Prep: {INF["source_long_dir"]} -> {OUT_DIR}')
    with open(OUT_DIR / 'prepare_inference.log', 'w') as log:
        subprocess.run(
            [sys.executable, PREP_SCRIPT,
             '--long-dir', INF['source_long_dir'],
             '--short-dir', INF['source_short_dir'],
             '--output-dir', OUT_DIR,
             '--n-gpu-splits', str(len(GPUS))],
            stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    print(f'  done - see {OUT_DIR / "prepare_inference.log"}')

if RUN_PREDICT:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    procs = []
    for split_idx, gpu in enumerate(GPUS):
        in_dir = OUT_DIR / f'imagesTs_gpu{split_idx}' if len(GPUS) > 1 else OUT_DIR / 'imagesTs'
        log_path = OUT_DIR / f'predict_gpu{gpu}.log'
        cmd = [ENV_BIN / 'nnUNetv2_predict',
               '-d', INF['dataset'],
               '-i', in_dir,
               '-o', PRED_DIR,
               '-f', *map(str, INF['folds']),
               '-tr', INF['trainer'],
               '-c', INF['configuration'],
               '-p', INF['plans']]
        print(f'GPU {gpu}: {in_dir.name} -> {PRED_DIR}  (log: {log_path.name})')
        log = open(log_path, 'w')
        procs.append((gpu, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}), log))

    failed = []
    for gpu, proc, log in procs:
        proc.wait()
        log.close()
        if proc.returncode != 0:
            failed.append(gpu)
    n_pred = len(list(PRED_DIR.glob('*.tif')))
    print(f'{n_pred} predictions in {PRED_DIR}')
    if failed:
        raise RuntimeError(f'nnUNetv2_predict failed on GPU(s) {failed} - check predict_gpuN.log')
