from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def compute_kinematics(
    df: pd.DataFrame,
    frame_interval_min: float,
    z_col: str = 'z_um_orig',
    y_col: str = 'y_um_orig',
    x_col: str = 'x_um_orig',
) -> pd.DataFrame:
    """Add per-step kinematic columns to a tracks DataFrame.

    Assumes df has columns [track_id, t, <z_col>, <y_col>, <x_col>].
    Rows are sorted by (track_id, t) before computing differences.

    Added columns
    -------------
    dz_um, dy_um, dx_um   : signed frame-to-frame displacement components (µm)
    step_um               : scalar step size = ‖(dz, dy, dx)‖ (µm)
    speed_um_per_min      : step_um / frame_interval_min
    cumpath_um            : cumulative path length from first observation of track
    net_disp_um           : distance from track origin at current timepoint

    The first row of each track has NaN for displacement-derived columns
    (no previous frame to difference against).
    """
    df = df.sort_values(['track_id', 't']).copy()

    grp = df.groupby('track_id', sort=False)

    df['dz_um'] = grp[z_col].diff()
    df['dy_um'] = grp[y_col].diff()
    df['dx_um'] = grp[x_col].diff()

    df['step_um'] = np.sqrt(df['dz_um']**2 + df['dy_um']**2 + df['dx_um']**2)
    df['speed_um_per_min'] = df['step_um'] / frame_interval_min

    df['cumpath_um'] = grp['step_um'].cumsum()

    # Net displacement from each track's first observed position
    origin = grp[[z_col, y_col, x_col]].transform('first')
    df['net_disp_um'] = np.sqrt(
        (df[z_col] - origin[z_col])**2 +
        (df[y_col] - origin[y_col])**2 +
        (df[x_col] - origin[x_col])**2
    )

    return df.reset_index(drop=True)


def summarize_tracks(kin_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-track summary statistics from a kinematics DataFrame.

    Returns one row per track_id with:
      n_frames           : number of timepoints in track
      t_start, t_end     : first and last observed timepoint
      total_path_um      : sum of all step sizes (cumulative path length)
      net_disp_um        : displacement from first to last position
      straightness       : net_disp / total_path  (1=straight, 0=random walk)
                           NaN for single-point tracks
      mean_speed_um_per_min
      max_speed_um_per_min
      median_speed_um_per_min
    """
    records = []
    for tid, grp in kin_df.groupby('track_id'):
        grp = grp.sort_values('t')
        steps = grp['step_um'].dropna()
        total_path = steps.sum()
        net_disp   = grp['net_disp_um'].iloc[-1]

        records.append({
            'track_id':              tid,
            'n_frames':              len(grp),
            't_start':               int(grp['t'].min()),
            't_end':                 int(grp['t'].max()),
            'total_path_um':         total_path,
            'net_disp_um':           net_disp,
            'straightness':          net_disp / total_path if total_path > 0 else np.nan,
            'mean_speed_um_per_min': grp['speed_um_per_min'].dropna().mean(),
            'max_speed_um_per_min':  grp['speed_um_per_min'].dropna().max(),
            'median_speed_um_per_min': grp['speed_um_per_min'].dropna().median(),
        })

    return pd.DataFrame(records).sort_values('track_id').reset_index(drop=True)


def compute_msd(
    kin_df: pd.DataFrame,
    z_col: str = 'z_um_orig',
    y_col: str = 'y_um_orig',
    x_col: str = 'x_um_orig',
    max_lag_fraction: float = 0.5,
) -> pd.DataFrame:
    """Compute mean squared displacement at each lag for every track.

    For each lag τ, MSD(τ) = mean over all valid pairs of |r(t+τ) - r(t)|².
    Only lags up to max_lag_fraction * track_length are computed to keep
    statistics reliable (few pairs at long lags inflates variance).

    Returns one row per (track_id, lag) with:
      msd_um2     : 3D MSD in µm²
      msd_xy_um2  : MSD in Y-X plane only (avoids Z anisotropy artefacts)
      msd_z_um2   : MSD along Z only
      n_pairs     : number of displacement pairs averaged
    """
    records = []
    for tid, grp in kin_df.groupby('track_id'):
        grp = grp.sort_values('t')
        pos = grp[[z_col, y_col, x_col]].values  # (N, 3)
        N = len(pos)
        max_lag = max(1, int(N * max_lag_fraction))

        for lag in range(1, max_lag + 1):
            disp = pos[lag:] - pos[:-lag]          # (N-lag, 3)
            sq   = np.sum(disp ** 2, axis=1)
            records.append({
                'track_id':   tid,
                'lag':        lag,
                'msd_um2':    sq.mean(),
                'msd_xy_um2': np.sum(disp[:, 1:] ** 2, axis=1).mean(),
                'msd_z_um2':  (disp[:, 0] ** 2).mean(),
                'n_pairs':    len(disp),
            })

    return pd.DataFrame(records).sort_values(['track_id', 'lag']).reset_index(drop=True)


def fit_msd_alpha(
    msd_df: pd.DataFrame,
    frame_interval_min: float,
    min_lags: int = 4,
) -> pd.DataFrame:
    """Fit MSD(τ) = 6D·τ^α per track in log-log space.

    α classifies the motion mode:
      α ≈ 1  : normal diffusion (Brownian)
      α < 1  : sub-diffusion / confined
      α > 1  : super-diffusion / directed  (α ≈ 2 = purely ballistic)

    D_eff is the effective diffusion coefficient in µm²/min, estimated
    from the intercept assuming MSD(τ) = 6D·τ^α (3D form).

    Returns one row per track_id with alpha, D_eff_um2_per_min, and n_lags.
    Tracks with fewer than min_lags lag points are skipped.
    """
    records = []
    for tid, grp in msd_df.groupby('track_id'):
        grp = grp.sort_values('lag')
        if len(grp) < min_lags:
            continue

        # Convert lag (frames) to real time (min) for physically meaningful D
        tau   = grp['lag'].values * frame_interval_min
        msd   = grp['msd_um2'].values

        log_tau = np.log(tau)
        log_msd = np.log(msd)

        alpha, intercept = np.polyfit(log_tau, log_msd, 1)
        # MSD = 6D * tau^alpha  =>  intercept = log(6D)
        D_eff = np.exp(intercept) / 6.0

        records.append({
            'track_id':           tid,
            'alpha':              alpha,
            'D_eff_um2_per_min':  D_eff,
            'n_lags':             len(grp),
        })

    return pd.DataFrame(records).sort_values('track_id').reset_index(drop=True)


def compute_local_flow(
    kin_df: pd.DataFrame,
    k_neighbors: int,
    frame_interval_min: float,
    z_col: str = 'z_um_orig',
    y_col: str = 'y_um_orig',
    x_col: str = 'x_um_orig',
) -> pd.DataFrame:
    """Add local (collective) and autonomous flow columns to a kinematics DataFrame.

    At each timepoint, for each nucleus the local flow is the mean velocity
    of its k nearest spatial neighbors (in 3D Z-Y-X, excluding itself).
    The autonomous motion is the nucleus's own velocity minus the local flow.

    Nuclei with no valid velocity (first frame of track) are passed through
    with NaN for the new columns.

    Added columns
    -------------
    local_vy_um, local_vx_um  : mean neighbor displacement (µm/frame)
    local_speed_um_per_min    : magnitude of local flow (µm/min)
    rel_vy_um, rel_vx_um      : autonomous displacement = own - local (µm/frame)
    rel_speed_um_per_min      : magnitude of autonomous motion (µm/min)
    """
    out_frames = []

    for t, frame in kin_df.groupby('t'):
        frame = frame.copy()
        valid = frame.dropna(subset=['dy_um', 'dx_um'])

        if len(valid) < k_neighbors + 1:
            for col in ['local_vy_um', 'local_vx_um', 'local_speed_um_per_min',
                        'rel_vy_um', 'rel_vx_um', 'rel_speed_um_per_min']:
                frame[col] = np.nan
            out_frames.append(frame)
            continue

        positions  = valid[[z_col, y_col, x_col]].values  # (M, 3) — 3D neighbor search
        velocities = valid[['dy_um', 'dx_um']].values     # (M, 2) — Y-X velocity only

        tree = cKDTree(positions)
        _, neighbor_idx = tree.query(positions, k=k_neighbors + 1)
        neighbor_idx = neighbor_idx[:, 1:]   # drop self (index 0)

        local_vy = velocities[neighbor_idx, 0].mean(axis=1)
        local_vx = velocities[neighbor_idx, 1].mean(axis=1)

        valid['local_vy_um'] = local_vy
        valid['local_vx_um'] = local_vx
        valid['local_speed_um_per_min'] = (
            np.sqrt(local_vy**2 + local_vx**2) / frame_interval_min
        )
        valid['rel_vy_um'] = valid['dy_um'] - local_vy
        valid['rel_vx_um'] = valid['dx_um'] - local_vx
        valid['rel_speed_um_per_min'] = (
            np.sqrt(valid['rel_vy_um']**2 + valid['rel_vx_um']**2) / frame_interval_min
        )

        # Flow alignment: cosine similarity between own velocity and local flow
        # +1 = moving perfectly with field, -1 = moving against field
        own_spd   = np.sqrt(valid['dy_um'].values**2 + valid['dx_um'].values**2)
        loc_spd   = np.sqrt(local_vy**2 + local_vx**2)
        dot       = valid['dy_um'].values * local_vy + valid['dx_um'].values * local_vx
        denom     = own_spd * loc_spd
        valid['flow_alignment'] = np.where(denom > 0, dot / denom, np.nan)

        # Merge back — rows with NaN velocity get NaN for new cols
        new_cols = ['local_vy_um', 'local_vx_um', 'local_speed_um_per_min',
                    'rel_vy_um', 'rel_vx_um', 'rel_speed_um_per_min', 'flow_alignment']
        frame = frame.drop(
            columns=[c for c in new_cols if c in frame.columns],
            errors='ignore',
        )
        frame = frame.merge(
            valid[['track_id'] + new_cols],
            on='track_id', how='left',
        )
        out_frames.append(frame)

    return (
        pd.concat(out_frames)
        .sort_values(['track_id', 't'])
        .reset_index(drop=True)
    )
