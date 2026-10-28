"""
Run the post-curation analysis chain on one dataset config, in order.

Every step reads tracks_{input_version}.csv (05, 06) or the outputs of earlier
steps from output_dir, so the only thing to change between curation rounds is
tracking.input_version in the config.

Default steps:
  05 ERK C/N → 06 track table + per-track summary → 07 track plots → 08 comparisons
Left out: 09 (napari viewer) and 10 (figure snapshots for chosen tracks) -
run them by hand. Old analyses live in archive/.

Usage
-----
  python pipelines/tracking/run_analysis.py                          # default config
  python pipelines/tracking/run_analysis.py --config configs/tracking/x.yaml
  python pipelines/tracking/run_analysis.py --steps 06 07            # just these
  python pipelines/tracking/run_analysis.py --from 06                # resume at 06
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.tracking_config import DEFAULT_CONFIG, resolve_config_path

CORE_STEPS = ['05', '06', '07', '08']

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--config', default=os.environ.get('TRACKING_CONFIG', DEFAULT_CONFIG))
parser.add_argument('--steps', nargs='+', default=CORE_STEPS,
                    help='script number prefixes to run, in the order given')
parser.add_argument('--from', dest='start', default=None,
                    help='skip steps before this one')
args = parser.parse_args()

config_path = resolve_config_path(args.config)
steps = args.steps
if args.start:
    steps = steps[steps.index(args.start):]


def script_for(step):
    matches = sorted(HERE.glob(f'{step}_*.py'))
    if len(matches) != 1:
        sys.exit(f'step {step}: expected one {step}_*.py, found {[m.name for m in matches]}')
    return matches[0]


scripts = [script_for(s) for s in steps]
print(f'Config: {config_path}')
print('Steps:  ' + ' → '.join(s.stem for s in scripts))

for script in scripts:
    print(f'\n{"═" * 78}\n▶ {script.name}\n{"═" * 78}', flush=True)
    result = subprocess.run([sys.executable, str(script), '--config', str(config_path)],
                            cwd=REPO_ROOT)
    if result.returncode != 0:
        sys.exit(f'\n✗ {script.name} failed (exit {result.returncode}). '
                 f'Fix it, then resume with: --from {script.name[:2]}')

print(f'\n✓ Done: {len(scripts)} steps')
