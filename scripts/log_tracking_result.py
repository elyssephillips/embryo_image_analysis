"""
Interactive prompt to append a new experiment entry to TRACKING_LOG.md.
Run with: conda run -n btrack2 python3 scripts/log_tracking_result.py
"""

import datetime
import yaml
import pandas as pd
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
LOG_PATH    = REPO_ROOT / 'TRACKING_LOG.md'

with open(REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation.yaml') as f:
    _cfg = yaml.safe_load(f)
LINKED_DIR = Path(_cfg['paths']['linked_tracks_dir'])


def get_track_stats(csv_path: Path, n_timepoints: int = 101) -> str:
    try:
        df = pd.read_csv(csv_path)
        lengths = df.groupby('track_id')['t'].count()
        return (f"{len(lengths)} tracks | "
                f"{(lengths == n_timepoints).sum()} full | "
                f"{(lengths >= 50).sum()} ≥50 | "
                f"median {lengths.median():.0f}")
    except Exception as e:
        return f"(could not read: {e})"


def prompt(label: str, default: str = '') -> str:
    suffix = f' [{default}]' if default else ''
    val = input(f'{label}{suffix}: ').strip()
    return val if val else default


def main():
    today = datetime.date.today().strftime('%Y-%m-%d')

    print('\n=== Add tracking experiment to TRACKING_LOG.md ===\n')

    version  = prompt('Version label (e.g. v4_shape, linked_c24)')
    script   = prompt('Script run (e.g. link_tracks.py, run_tracking_visual.py)')
    params   = prompt('Key params changed (e.g. MAX_COST=2.0, W_NEIGHBOR=1.5)')
    out_file = prompt('Output CSV filename (e.g. tracks_linked_c24.csv)')

    # Auto-compute stats if file exists
    csv_path = LINKED_DIR / out_file
    if out_file and csv_path.exists():
        auto_stats = get_track_stats(csv_path)
        print(f'  Auto-computed: {auto_stats}')
        results = prompt('Results (press Enter to use auto-computed)', default=auto_stats)
    else:
        results = prompt('Results (e.g. 254 tracks | 6 full | 35 ≥50 | median 11)')

    notes = prompt('Notes / observations (what worked, what failed, next idea)')

    entry = f"""
### {today} | {version}
**Script:** `{script}`
**Key params:** {params}
**Results:** {results}
**Output files:** `{out_file}`
**Notes:** {notes}

---"""

    marker = '<!-- New entries go above this line -->'
    text   = LOG_PATH.read_text()

    if marker in text:
        text = text.replace(marker, entry + '\n' + marker)
    else:
        text = text + entry + '\n'

    LOG_PATH.write_text(text)
    print(f'\nEntry added to {LOG_PATH}')


if __name__ == '__main__':
    main()
