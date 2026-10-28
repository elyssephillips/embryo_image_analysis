"""
Score tracker outputs against hand-curated windows (04_curate_tracks_window.ipynb),
by frame-to-frame links between segmented nuclei, matched on (t, label_id).

Per curated window and tracker:
  link recall      - fraction of curated t->t+1 links the tracker also makes
  succ correct/wrong/missing - for curated nuclei that have a curated
                     successor: tracker links to the same nucleus / a
                     different one / none
  false links      - tracker links a curated nucleus onward where curation
                     says its track ends (only counted when the target is
                     also a curated nucleus - see below)
  perfect tracks   - curated tracks (>=2 frames) whose every link the
                     tracker reproduces

Only curated nuclei are judged: curation covers just the window's z-band, so
a link to a nucleus outside it can't be called right or wrong and is ignored.
Rows without label_id (btrack gap-fill dummies) are not nuclei and break
links. Caveat: the windows were curated starting from btrack output, so any
btrack error the curator didn't catch counts as correct for btrack.
"""

import sys
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.tracking_config import load_tracking_config
cfg, CONFIG_PATH = load_tracking_config()   # --config X.yaml, else TRACKING_CONFIG / default

TRACKS_ROOT = Path(cfg['paths']['btrack_output_dir']).parent

# Curated ground truth per window, relative to results/tracks/.
CURATED = {
    'A': 'btrack/tracks_linked_c3_winA.csv',
    'B': 'btrack/tracks_linked_c3_winB.csv',
    'C': 'btrack/tracks_linked_c3_winC.csv',
    # Full z, t70-75; seeded from B's band curation + btrack elsewhere, so
    # btrack's head start here is larger than in A-C.
    'D': 'btrack/tracks_linked_c2_winD.csv',
    # Full z, seeded from stitched trackastra - so trackastra, not btrack,
    # gets the head start on these three.
    'E': 'trackastra/tracks_linked_c2_winE.csv',
    'F': 'trackastra/tracks_linked_c4_winF.csv',
    'G': 'trackastra/tracks_linked_c2_winG.csv',
}

# Windows sharing nuclei: a model trained on one counts as trained on the other.
OVERLAPS = {'B': {'D'}, 'D': {'B'}}

# Tracker outputs to score, relative to results/tracks/. Optional third item:
# windows this tracker was trained on (fine-tuned models) - reported as
# 'trained' rather than scored as if held out.
CANDIDATES = [
    ('btrack',              'btrack/tracks_v1_raw.csv'),
    ('custom linker',       'linked/tracks_linked.csv'),
    ('trackastra zero-shot', 'trackastra/tracks_trackastra_v1.csv'),
    ('trackastra zero-shot registered', 'trackastra/tracks_trackastra_reg_v1.csv'),
]
# Fine-tuned runs from 03_run_trackastra.py are picked up automatically:
# trackastra/tracks_trackastra_ft{variant}_holdout_<...>win{W}.csv, trained on the other windows.


def _nodes_and_links(df: pd.DataFrame):
    d = df.dropna(subset=['label_id']).astype({'label_id': int, 't': int})
    d = d.sort_values(['track_id', 't'])
    nxt = d.groupby('track_id')[['t', 'label_id']].shift(-1)
    consecutive = nxt['t'] == d['t'] + 1
    succ = dict(zip(zip(d.loc[consecutive, 't'], d.loc[consecutive, 'label_id']),
                    nxt.loc[consecutive, 'label_id'].astype(int)))
    return d, succ


def score(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    gt, gt_succ = _nodes_and_links(gt_df)
    t0, t1 = gt['t'].min(), gt['t'].max()
    _, pred_succ = _nodes_and_links(pred_df[pred_df['t'].between(t0, t1)])

    gt_nodes = set(zip(gt['t'], gt['label_id']))
    correct = wrong = missing = false_link = 0
    for node in gt_nodes:
        if node[0] == t1:
            continue
        g, p = gt_succ.get(node), pred_succ.get(node)
        if g is not None:
            if p == g:
                correct += 1
            elif p is None:
                missing += 1
            else:
                wrong += 1
        elif p is not None and (node[0] + 1, p) in gt_nodes:
            false_link += 1

    perfect = total = 0
    for _, trk in gt.groupby('track_id'):
        links = [(t, l) for t, l in zip(trk['t'], trk['label_id']) if (t, l) in gt_succ]
        if not links:
            continue
        total += 1
        perfect += all(pred_succ.get(n) == gt_succ[n] for n in links)

    n = correct + wrong + missing
    return {
        'links': n,
        'recall': correct / n if n else float('nan'),
        'correct': correct, 'wrong': wrong, 'missing': missing,
        'false_links': false_link,
        'perfect_tracks': f'{perfect}/{total}',
    }


def main():
    gts = {w: pd.read_csv(TRACKS_ROOT / p) for w, p in CURATED.items()}
    candidates = [(name, TRACKS_ROOT / p, set()) for name, p in CANDIDATES]
    for w in CURATED:
        for p in sorted((TRACKS_ROOT / 'trackastra').glob(f'tracks_trackastra_ft*_holdout_*win{w}*.csv')):
            variant = p.stem.removeprefix('tracks_trackastra_').split('_holdout_')[0]
            if p.stem.endswith('_stitched'):
                variant += '+stitch'
            # Training used every window except the holdout and its overlaps
            # (03_trackastra_finetune.py's OVERLAPS), so those are scoreable.
            candidates.append((f'trackastra {variant} (held out {w})', p,
                               set(CURATED) - {w} - OVERLAPS.get(w, set())))

    rows = []
    for name, path, trained_on in candidates:
        if not path.exists():
            print(f'skip {name}: {path} not found', file=sys.stderr)
            continue
        pred = pd.read_csv(path)
        for w, gt in gts.items():
            if w in trained_on:
                rows.append({'tracker': name, 'window': w, 'recall': 'trained'})
                continue
            rows.append({'tracker': name, 'window': w, **score(gt, pred)})

    table = pd.DataFrame(rows)
    with pd.option_context('display.width', 200, 'display.max_columns', 20, 'display.float_format', '{:.3f}'.format):
        print(table.to_string(index=False))


if __name__ == '__main__':
    main()
