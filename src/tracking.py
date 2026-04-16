from __future__ import annotations

import numpy as np
import pandas as pd


def apply_corrections(df: pd.DataFrame, corrections: list[dict]) -> pd.DataFrame:
    """Apply a list of manual track corrections to a tracks DataFrame.

    All masks are computed against the original state of the DataFrame, so
    corrections are independent of each other regardless of order.

    Correction types
    ----------------
    swap      — swap two track IDs from timepoint t_from onward
                {'type': 'swap', 't_from': int, 'id_a': int, 'id_b': int}
    reassign  — reassign all rows of from_id at t >= t onward to to_id
                {'type': 'reassign', 't': int, 'from_id': int, 'to_id': int}
    break     — split a track at t_break; rows from t_break onward get a new ID
                {'type': 'break', 'track_id': int, 't_break': int}
    """
    df = df.copy()
    original = df.copy()
    next_id = int(df['track_id'].max()) + 1
    updates: list[tuple[pd.Series, int]] = []

    for corr in corrections:
        ctype = corr['type']
        if ctype == 'swap':
            t0, a, b = corr['t_from'], corr['id_a'], corr['id_b']
            updates.append(((original['track_id'] == a) & (original['t'] >= t0), b))
            updates.append(((original['track_id'] == b) & (original['t'] >= t0), a))
        elif ctype == 'reassign':
            t, fid, tid = corr['t'], corr['from_id'], corr['to_id']
            updates.append(((original['track_id'] == fid) & (original['t'] >= t), tid))
        elif ctype == 'break':
            tid, tb = corr['track_id'], corr['t_break']
            updates.append(((original['track_id'] == tid) & (original['t'] >= tb), next_id))
            next_id += 1
        else:
            raise ValueError(f"Unknown correction type: {ctype!r}")

    for mask, new_val in updates:
        df.loc[mask, 'track_id'] = new_val

    return df.sort_values(['track_id', 't']).reset_index(drop=True)


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of (track_id, t) pairs that appear more than once.

    An empty result means the tracks are clean.
    """
    counts = df.groupby(['track_id', 't']).size()
    return counts[counts > 1].reset_index(name='count')


def build_track_label_stack(label_stack: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Return a uint32 array shaped like label_stack with values = track_id + 1.

    Each nucleus's voxels are assigned its track_id + 1 so that track_id=0
    remains visible and background stays 0.

    Parameters
    ----------
    label_stack : (T, Z, Y, X) integer array of instance label IDs
    df          : tracks DataFrame with columns [t, label_id, track_id]
    """
    remap: dict[int, dict[int, int]] = {}
    for row in df.itertuples(index=False):
        t = int(row.t)
        remap.setdefault(t, {})[int(row.label_id)] = int(row.track_id)

    out = np.zeros(label_stack.shape, dtype=np.uint32)
    for t in range(label_stack.shape[0]):
        if t not in remap:
            continue
        frame = label_stack[t]
        mapping = remap[t]
        for lbl_id, trk_id in mapping.items():
            out[t][frame == lbl_id] = trk_id + 1
    return out
