"""
Gap-close Trackastra output with the custom linker's stitch_fragments
(03_link_tracks.py): joins a track end to a track start 1..MAX_GAP frames
later using the same spatial/volume/intensity/neighbour cost and thresholds.

Trackastra's errors are mostly missing links (a nucleus left unlinked to its
next-frame self) rather than swaps, so this targets its main failure mode.
Stitching runs in registered coordinates (features_registered.csv), joined
on (t, label_id).

Output: trackastra/tracks_trackastra_{version}_stitched.csv - scored by
03_score_tracks_vs_curated.py alongside the unstitched version.
"""

import importlib
import sys
import yaml
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'pipelines' / 'tracking'))
linker = importlib.import_module('03_link_tracks')

CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset003_icm_te_250914_stack5.yaml'  # edit to switch dataset
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

TRACKASTRA_DIR = Path(cfg['paths']['trackastra_output_dir'])

# 03_run_trackastra.py VERSIONs to stitch.
VERSIONS = [
    *[f'ft_reg7_holdout_{w}' for w in ['linked_c3_winA', 'linked_c3_winB', 'linked_c3_winC', 'linked_c2_winD',
                                       'linked_c2_winE', 'linked_c4_winF', 'linked_c2_winG']],
    'ft_reg7_final',   # production model trained on all windows
]

MAX_GAP  = 3                        # frames; same as 03_link_tracks.py's default
MAX_DIST = linker.STITCH_MAX_DIST   # um in registered space

FEAT_COLS = ['t', 'label_id', 'z_um_reg', 'y_um_reg', 'x_um_reg', 'area_um3', 'mean_intensity']


def _summary(df: pd.DataFrame) -> str:
    L = df.groupby('track_id')['t'].count()
    return f'{len(L)} tracks, {(L >= 50).sum()} >=50 frames, {(L >= 10).sum()} >=10, median {L.median():.0f}'


def main():
    feat = pd.read_csv(cfg['paths']['features_registered_csv'], usecols=FEAT_COLS)
    for version in VERSIONS:
        src = TRACKASTRA_DIR / f'tracks_trackastra_{version}.csv'
        tracks = pd.read_csv(src)
        df = tracks.merge(feat, on=['t', 'label_id'], how='left', validate='one_to_one')
        assert df['z_um_reg'].notna().all(), f'{version}: rows missing from features_registered.csv'

        print(f'\n{version}\n  before: {_summary(df)}', flush=True)
        stitched = linker.stitch_fragments(df, max_gap=MAX_GAP, max_dist=MAX_DIST)
        assert not stitched.duplicated(['track_id', 't']).any(), f'{version}: stitching produced overlapping tracks'
        stitched['track_id'] = stitched['track_id'].rank(method='dense').astype(int)
        print(f'  after:  {_summary(stitched)}', flush=True)

        out = TRACKASTRA_DIR / f'tracks_trackastra_{version}_stitched.csv'
        stitched[tracks.columns].sort_values(['track_id', 't']).to_csv(out, index=False)
        print(f'  saved {out.name}', flush=True)


if __name__ == '__main__':
    main()
