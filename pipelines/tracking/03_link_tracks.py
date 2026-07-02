"""
Custom frame-by-frame nucleus linker using Hungarian algorithm with a
combined cost of: position + volume + intensity + neighborhood geometry.

Replaces the btrack Kalman filter step. Gaps (missing detections at a
given frame) are left as fragment breaks and handled by stitch_tracks.py.

Neighborhood geometry fingerprint:
  For each nucleus, compute vectors to its K nearest neighbours in
  registered space. For a candidate link A(t) → B(t+1), run a K×K
  Hungarian match between A's and B's neighbour vectors. The mean
  residual after optimal matching measures how similar the local
  spatial arrangement is — a strong identity signal that requires no
  prior track assignments.

Output:
  tracks_linked.csv  — track_id, t, registered coords, original image
                       coords, and all nucleus features
"""

import sys
import yaml
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.log import log_run

CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation.yaml'
with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

FEATURES_CSV     = Path(_cfg['paths']['features_registered_csv'])
RAW_FEATURES_CSV = Path(_cfg['paths']['features_csv'])
OUT_DIR          = Path(_cfg['paths']['linked_tracks_dir'])

# =============================================================================
#  CONFIG
# =============================================================================

# Number of nearest neighbours used for geometry fingerprint.
# More neighbours → more distinctive fingerprint but slower and noisier
# at frames with many missing detections.
K_NEIGHBORS = 5

# Maximum spatial distance (µm, registered space) to consider linking.
# Should be comfortably below inter-nuclear spacing (~17.7 µm).
MAX_SPATIAL_DIST = 16.0

# Maximum combined cost to accept a link. Links above this are rejected
# (the nucleus starts a new fragment). With all weights = 1.0 and costs
# normalised to ~[0, 1] each, this allows moderate deviation across all
# four features simultaneously.
MAX_COST = 2.5

# Feature weights in the combined cost.
W_SPATIAL  = 1.0   # spatial distance cost
W_VOL      = 0.5   # volume fractional change
W_INT      = 0.5   # intensity fractional change
W_NEIGHBOR = 1.0   # neighbourhood geometry residual

# Normalisation scale for neighbour geometry residual (µm).
# Set to roughly the median inter-nuclear neighbour distance (~23 µm for
# this dataset). True-link residuals (median ~2.5 µm) will cost ~0.12,
# while wrong-pairing residuals (~23 µm) will cost ~1.0.
NEIGHBOR_SCALE = 20.0

# Stitching uses a larger spatial distance than frame-by-frame linking
# since it bridges detection gaps across multiple frames, including at
# hard transitions. Neighbour consistency compensates for the looser
# spatial threshold.
STITCH_MAX_DIST = 22.0

# =============================================================================


def get_neighbor_vectors(centroids, idx, k):
    """
    Return (k, 3) array of vectors from centroids[idx] to its k nearest
    neighbours (excluding itself). If fewer than k points exist, returns
    all available neighbours.
    """
    pos = centroids[idx]
    others = np.delete(centroids, idx, axis=0)
    if len(others) == 0:
        return np.zeros((0, 3))
    dists = np.linalg.norm(others - pos, axis=1)
    k_actual = min(k, len(others))
    nn_idx = np.argpartition(dists, k_actual - 1)[:k_actual]
    nn_idx = nn_idx[np.argsort(dists[nn_idx])]  # sort by distance
    return others[nn_idx] - pos


def neighbor_geometry_cost(vecs_a, vecs_b):
    """
    Minimum mean residual after optimal bipartite matching of two sets
    of neighbour vectors. Returns a value in µm.

    If either set is empty or too small, returns a high penalty.
    """
    ka, kb = len(vecs_a), len(vecs_b)
    if ka == 0 or kb == 0:
        return NEIGHBOR_SCALE * 2  # high penalty if no neighbours

    k = min(ka, kb)
    va = vecs_a[:k]
    vb = vecs_b[:k]

    # k×k distance matrix between neighbour vectors
    dist = np.sqrt(((va[:, None] - vb[None, :]) ** 2).sum(axis=2))  # (k, k)
    row_ind, col_ind = linear_sum_assignment(dist)
    return dist[row_ind, col_ind].mean()


def precompute_neighbor_vecs(centroids, k):
    """
    For all nuclei in a frame, compute K-neighbour vectors.
    Returns list of (k_actual, 3) arrays.
    """
    return [get_neighbor_vectors(centroids, i, k) for i in range(len(centroids))]


def build_cost_matrix(frame_t, frame_t1, k=K_NEIGHBORS,
                      max_dist=MAX_SPATIAL_DIST,
                      w_spatial=W_SPATIAL, w_vol=W_VOL,
                      w_int=W_INT, w_neighbor=W_NEIGHBOR,
                      neighbor_scale=NEIGHBOR_SCALE):
    """
    Build the N_t × N_{t+1} cost matrix for linking frame t to t+1.

    frame_t, frame_t1: DataFrames with columns
        z_um_reg, y_um_reg, x_um_reg, area_um3, mean_intensity
    """
    INF = 1e9

    coords_t  = frame_t [['z_um_reg', 'y_um_reg', 'x_um_reg']].values
    coords_t1 = frame_t1[['z_um_reg', 'y_um_reg', 'x_um_reg']].values
    vols_t    = frame_t ['area_um3'].values
    vols_t1   = frame_t1['area_um3'].values
    ints_t    = frame_t ['mean_intensity'].values
    ints_t1   = frame_t1['mean_intensity'].values

    N, M = len(coords_t), len(coords_t1)

    # Pre-compute neighbour vectors for all nuclei in both frames
    nvecs_t  = precompute_neighbor_vecs(coords_t,  k)
    nvecs_t1 = precompute_neighbor_vecs(coords_t1, k)

    cost = np.full((N, M), INF)

    for i in range(N):
        for j in range(M):
            dist = np.linalg.norm(coords_t[i] - coords_t1[j])
            if dist > max_dist:
                continue

            c_spatial = w_spatial * dist / max_dist

            vol_ratio = abs(vols_t[i] - vols_t1[j]) / max(vols_t[i], 1e-6)
            c_vol = w_vol * vol_ratio

            int_ratio = abs(ints_t[i] - ints_t1[j]) / max(ints_t[i], 1e-6)
            c_int = w_int * int_ratio

            nb_res = neighbor_geometry_cost(nvecs_t[i], nvecs_t1[j])
            c_neighbor = w_neighbor * nb_res / neighbor_scale

            cost[i, j] = c_spatial + c_vol + c_int + c_neighbor

    return cost


def link_frames(df, k=K_NEIGHBORS, max_dist=MAX_SPATIAL_DIST,
                max_cost=MAX_COST):
    """
    Link detections frame-by-frame using Hungarian assignment.

    Returns DataFrame with 'track_id' column assigned.
    """
    timepoints = sorted(df['t'].unique())
    n_frames   = len(timepoints)

    # Initialise: each nucleus at t=0 gets its own track ID
    frame0 = df[df['t'] == timepoints[0]].copy().reset_index(drop=True)
    next_track_id = 0

    # track_id for each nucleus in current frame (index-aligned to frame df)
    current_ids = list(range(len(frame0)))
    next_track_id = len(frame0)

    records = []
    for row_i, row in frame0.iterrows():
        r = row.to_dict()
        r['track_id'] = current_ids[row_i]
        records.append(r)

    n_links    = 0
    n_breaks   = 0
    n_unmatched_new = 0

    for fi in range(n_frames - 1):
        t      = timepoints[fi]
        t1     = timepoints[fi + 1]
        frame_t  = df[df['t'] == t ].reset_index(drop=True)
        frame_t1 = df[df['t'] == t1].reset_index(drop=True)

        N, M = len(frame_t), len(frame_t1)

        if N == 0 or M == 0:
            # Assign new IDs to all nuclei in t+1
            new_ids = list(range(next_track_id, next_track_id + M))
            next_track_id += M
            n_unmatched_new += M
            for row_j, row in frame_t1.iterrows():
                r = row.to_dict()
                r['track_id'] = new_ids[row_j]
                records.append(r)
            current_ids = new_ids
            continue

        cost = build_cost_matrix(frame_t, frame_t1, k=k, max_dist=max_dist)

        row_ind, col_ind = linear_sum_assignment(cost)

        # Determine accepted links (below MAX_COST)
        t1_assigned = {}  # j → track_id from t
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < max_cost:
                t1_assigned[c] = current_ids[r]
                n_links += 1
            else:
                n_breaks += 1

        # Assign IDs for t+1
        new_ids = []
        for j in range(M):
            if j in t1_assigned:
                new_ids.append(t1_assigned[j])
            else:
                new_ids.append(next_track_id)
                next_track_id += 1
                n_unmatched_new += 1

        for row_j, row in frame_t1.iterrows():
            r = row.to_dict()
            r['track_id'] = new_ids[row_j]
            records.append(r)

        current_ids = new_ids

        if (fi + 1) % 10 == 0 or fi == 0:
            print(f'  t={t}→{t1}: {len(t1_assigned)}/{M} linked', flush=True)

    print(f'\nTotal links accepted: {n_links}')
    print(f'Links rejected (cost > {max_cost}): {n_breaks}')
    print(f'New track fragments started: {n_unmatched_new}')

    return pd.DataFrame(records).sort_values(['track_id', 't']).reset_index(drop=True)


def stitch_fragments(frags_df, max_gap=3, max_dist=MAX_SPATIAL_DIST,
                     w_spatial=W_SPATIAL, w_vol=W_VOL, w_int=W_INT,
                     w_neighbor=W_NEIGHBOR, neighbor_scale=NEIGHBOR_SCALE,
                     cost_thresh=MAX_COST):
    """
    Reconnect track fragments across gaps using the same combined cost,
    now also incorporating neighbourhood consistency via track assignments.

    For gap-closing, neighbour consistency is computed using the track IDs
    already assigned: neighbours of the end of fragment A are identified by
    track_id, and checked against neighbours at the start of fragment B.
    """
    ends, starts = {}, {}
    for tid, grp in frags_df.groupby('track_id'):
        grp = grp.sort_values('t')
        ends[tid]   = grp.iloc[-1]
        starts[tid] = grp.iloc[0]

    end_ids   = list(ends.keys())
    start_ids = list(starts.keys())
    INF = 1e9
    cost = np.full((len(end_ids), len(start_ids)), INF)

    # Build a lookup: (t, z_um_reg, y_um_reg, x_um_reg) → track_id for
    # neighbourhood consistency check at the stitching stage
    # Use registered coords rounded to avoid float issues
    coord_lookup = {}
    for tid, grp in frags_df.groupby('track_id'):
        for _, row in grp.iterrows():
            t = int(row['t'])
            if t not in coord_lookup:
                coord_lookup[t] = {'coords': [], 'track_ids': []}
            coord_lookup[t]['coords'].append(
                [row['z_um_reg'], row['y_um_reg'], row['x_um_reg']])
            coord_lookup[t]['track_ids'].append(tid)
    for t in coord_lookup:
        coord_lookup[t]['coords'] = np.array(coord_lookup[t]['coords'])

    def get_neighbor_track_ids(t, pos, k):
        """Return set of track IDs of K nearest nuclei at timepoint t."""
        if t not in coord_lookup:
            return set()
        coords = coord_lookup[t]['coords']
        tids   = coord_lookup[t]['track_ids']
        dists  = np.linalg.norm(coords - pos, axis=1)
        k_act  = min(k, len(dists) - 1)
        if k_act <= 0:
            return set()
        nn = np.argpartition(dists, k_act)[:k_act + 1]
        nn = nn[np.argsort(dists[nn])]
        # Exclude self (dist ≈ 0)
        nn_filtered = [i for i in nn if dists[i] > 0.01][:k_act]
        return set(tids[i] for i in nn_filtered)

    for i, eid in enumerate(end_ids):
        e = ends[eid]
        e_pos = np.array([e['z_um_reg'], e['y_um_reg'], e['x_um_reg']])
        e_nbrs = get_neighbor_track_ids(int(e['t']), e_pos, K_NEIGHBORS)

        for j, sid in enumerate(start_ids):
            if sid == eid:
                continue
            s = starts[sid]
            dt = int(s['t']) - int(e['t'])
            if dt <= 0 or dt > max_gap:
                continue

            s_pos = np.array([s['z_um_reg'], s['y_um_reg'], s['x_um_reg']])
            spatial = np.linalg.norm(e_pos - s_pos)
            if spatial > max_dist:
                continue

            c_spatial = w_spatial * spatial / max_dist
            c_vol = w_vol * abs(s['area_um3'] - e['area_um3']) / max(e['area_um3'], 1e-6)
            c_int = w_int * abs(s['mean_intensity'] - e['mean_intensity']) / max(e['mean_intensity'], 1e-6)

            # Neighbourhood consistency: fraction of shared neighbour tracks
            s_nbrs = get_neighbor_track_ids(int(s['t']), s_pos, K_NEIGHBORS)
            if e_nbrs and s_nbrs:
                shared = len(e_nbrs & s_nbrs) / max(len(e_nbrs), len(s_nbrs))
                c_neighbor = w_neighbor * (1.0 - shared)  # 0 = identical neighbours
            else:
                c_neighbor = w_neighbor * 0.5  # unknown, moderate penalty

            cost[i, j] = c_spatial + c_vol + c_int + c_neighbor

    row_ind, col_ind = linear_sum_assignment(cost)
    merges = {}
    n_stitched = 0
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < cost_thresh:
            merges[start_ids[c]] = end_ids[r]
            n_stitched += 1

    def resolve_root(fid):
        visited = set()
        while fid in merges and fid not in visited:
            visited.add(fid)
            fid = merges[fid]
        return fid

    result = frags_df.copy()
    result['track_id'] = result['track_id'].map(resolve_root)

    n_before = frags_df['track_id'].nunique()
    n_after  = result['track_id'].nunique()
    print(f'Stitched {n_before} fragments → {n_after} tracks ({n_stitched} merges)')
    return result


def main():
    feat_csv = FEATURES_CSV if FEATURES_CSV.exists() else RAW_FEATURES_CSV
    df = pd.read_csv(feat_csv)
    registered = 'z_um_reg' in df.columns
    print(f'Loaded {len(df)} detections across {df["t"].nunique()} timepoints '
          f'({"registered" if registered else "raw"} coords)')

    if not registered:
        print('WARNING: running without registration. '
              'Run register_centroids.py first for best results.')
        df['z_um_reg'] = df['z_um']
        df['y_um_reg'] = df['y_um']
        df['x_um_reg'] = df['x_um']

    # --- Frame-by-frame linking ---
    print('\nLinking frames...')
    linked_df = link_frames(df)

    lengths = linked_df.groupby('track_id')['t'].count()
    print(f'\nAfter frame-by-frame linking:')
    print(f'  Tracks:            {len(lengths)}')
    print(f'  Full timecourse:   {(lengths == df["t"].nunique()).sum()}')
    print(f'  >= 50 frames:      {(lengths >= 50).sum()}')
    print(f'  1-2 frames:        {(lengths <= 2).sum()}')
    print(f'  Median length:     {lengths.median():.0f}')

    # --- Stitching ---
    print('\nStitching fragments...')
    stitched_df = stitch_fragments(linked_df, max_dist=STITCH_MAX_DIST)

    lengths2 = stitched_df.groupby('track_id')['t'].count()
    print(f'\nAfter stitching:')
    print(f'  Tracks:            {len(lengths2)}')
    print(f'  Full timecourse:   {(lengths2 == df["t"].nunique()).sum()}')
    print(f'  >= 50 frames:      {(lengths2 >= 50).sum()}')
    print(f'  >= 10 frames:      {(lengths2 >= 10).sum()}')
    print(f'  Median length:     {lengths2.median():.0f}')

    # --- Join original image coords ---
    raw_df = pd.read_csv(RAW_FEATURES_CSV)[
        ['t', 'label_id', 'z_um', 'y_um', 'x_um']].rename(
        columns={'z_um': 'z_um_orig', 'y_um': 'y_um_orig', 'x_um': 'x_um_orig'})

    stitched_df['label_id'] = stitched_df['label_id'].astype(int)
    stitched_df = stitched_df.merge(raw_df, on=['t', 'label_id'], how='left')

    # Fill any unmatched rows (shouldn't happen) with registered coords
    for col, fallback in [('z_um_orig', 'z_um_reg'),
                           ('y_um_orig', 'y_um_reg'),
                           ('x_um_orig', 'x_um_reg')]:
        stitched_df[col] = stitched_df[col].fillna(stitched_df[fallback])

    # --- Save ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'tracks_linked.csv'
    stitched_df.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')

    log_run("tracking", CONFIG_PATH.stem, "03_link_tracks.py",
            output_path=str(out_path), detail="detailed",
            data_path=str(Path(_cfg['paths']['features_registered_csv']).parent))


if __name__ == '__main__':
    main()
