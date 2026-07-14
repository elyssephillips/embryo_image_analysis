"""
Script 40: ERK C/N ratio overlaid on raw H2B nuclear channel, cropped around and
highlighting two tracks, for a few pre-implantation timepoints.

For each timepoint: max-Z projects the raw H2B channel (inverted grayscale, white
background) and the per-nucleus ERK C/N ratio (blue-to-red colormap, alpha-blended
over nuclei only), rotates the whole frame in-plane so the ICM point sits directly
below the embryo centroid in every snapshot (consistent orientation for comparison),
crops to the whole embryo, and outlines each highlighted track's nucleus with a
distinct color (legend labeled by ICM distance, not track_id).

Usage
-----
  conda run -n napari_env python3 pipelines/tracking/40_highlight_tracks_cn_overlay.py --tracks 48 71 --timepoints 16 22 29

  Or set TRACK_IDS / TIMEPOINTS below and just hit Run in VS Code — no CLI args needed.
"""

TRACK_IDS  = [48, 71]     # <-- tracks to highlight
TIMEPOINTS = [16, 22, 29]  # <-- pre-implantation timepoints to snapshot

import argparse
import sys
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import tifffile
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

REPO_ROOT   = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

RAW_DIR      = Path(cfg['paths']['raw_dir'])
RAW_GLOB     = cfg['paths']['raw_glob']
LABEL_DIR    = Path(cfg['paths']['label_dir'])
LABEL_GLOB   = cfg['paths']['label_glob']
OUT_DIR      = Path(cfg['paths']['output_dir'])
VERSION      = cfg['tracking']['input_version']
N_T          = cfg['microscopy']['n_timepoints']
VX_Z, VX_Y, VX_X = cfg['microscopy']['voxel_size_zyx']
SCALE_BAR_UM = 20

CROP_PAD_UM  = 20   # padding around the whole embryo (union of all nuclei)
# Kept off the blue/red family so outlines never blend into the ERK C/N colormap
TRACK_COLORS = {0: '#984ea3', 1: '#2ca02c', 2: '#ff7f00', 3: '#33a02c'}

parser = argparse.ArgumentParser()
parser.add_argument('--tracks', type=int, nargs='+', default=None, help='overrides TRACK_IDS above')
parser.add_argument('--timepoints', type=int, nargs='+', default=None, help='overrides TIMEPOINTS above')
args, _unknown = parser.parse_known_args()
if args.tracks is not None:
    TRACK_IDS = args.tracks
if args.timepoints is not None:
    TIMEPOINTS = args.timepoints

SNAP_DIR = OUT_DIR / 'snapshots'
SNAP_DIR.mkdir(parents=True, exist_ok=True)

# ── load ERK C/N + kinematics + ICM distance ────────────────────────────────────

erk = pd.read_csv(OUT_DIR / f'erk_cn_ratio_{VERSION}.csv')
erk['label_id'] = erk['label_id'].astype('Int64')

vstats = pd.read_csv(OUT_DIR / f'volume_track_stats_{VERSION}.csv')[['track_id', 'icm_dist_um']]
icm_lookup = vstats.set_index('track_id')['icm_dist_um']

ICM_Z_UM, ICM_Y_UM, ICM_X_UM = cfg['biology']['icm_centroid_t30_zyx']
ICM_ROW_PX = ICM_Y_UM / VX_Y
ICM_COL_PX = ICM_X_UM / VX_X

label_files = sorted(LABEL_DIR.glob(LABEL_GLOB))[:N_T]
raw_files   = sorted(RAW_DIR.glob(RAW_GLOB))[:N_T]
assert len(label_files) == N_T and len(raw_files) == N_T

for t in TIMEPOINTS:
    if not (0 <= t < N_T):
        sys.exit(f'Timepoint {t} out of range [0, {N_T - 1}]')


def _cn_lut_for_t(t):
    grp = erk[(erk['t'] == t) & erk['label_id'].notna() & erk['erk_cn_ratio'].notna()]
    if grp.empty:
        return np.array([], dtype=np.float32)
    max_lid = int(grp['label_id'].max())
    lut = np.full(max_lid + 1, np.nan, dtype=np.float32)
    lut[grp['label_id'].astype(int).values] = grp['erk_cn_ratio'].astype(np.float32).values
    return lut


def _map_lut_to_frame(labels, lut):
    out = np.full(labels.shape, np.nan, dtype=np.float32)
    if len(lut) == 0:
        return out
    flat_lbl = labels.ravel()
    flat_out = out.ravel()
    nonbg = flat_lbl > 0
    if nonbg.any():
        ids = flat_lbl[nonbg]
        in_range = ids < len(lut)
        flat_out[np.where(nonbg)[0][in_range]] = lut[ids[in_range]]
    return out


def _embryo_crop_box_px_from_mask(mask2d, pad_um):
    """Bounding box of the whole embryo (union of all labeled nuclei), not just
    the highlighted tracks, so the full embryo is visible and not cut off."""
    rows = np.where(np.any(mask2d, axis=1))[0]
    cols = np.where(np.any(mask2d, axis=0))[0]
    y0, y1 = rows[0], rows[-1]
    x0, x1 = cols[0], cols[-1]
    pad_px_x = pad_um / VX_X
    pad_px_y = pad_um / VX_Y
    x0 = int(max(x0 - pad_px_x, 0))
    x1 = int(min(x1 + pad_px_x, mask2d.shape[1]))
    y0 = int(max(y0 - pad_px_y, 0))
    y1 = int(min(y1 + pad_px_y, mask2d.shape[0]))
    return x0, x1, y0, y1


def _rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _rotate_point(point_rc, theta_rot, center_rc):
    """Forward-rotate a (row, col) point by theta_rot about center_rc."""
    d = np.array(point_rc) - np.array(center_rc)
    return tuple(np.array(center_rc) + _rotation_matrix(theta_rot) @ d)


def _rotate_image(arr, theta_rot, center_rc, order, cval):
    """Rotate a 2D array by theta_rot about center_rc (content moves the same
    way _rotate_point moves a coordinate — i.e. forward rotation of content)."""
    m_inv = _rotation_matrix(-theta_rot)
    c = np.array(center_rc)
    offset = c - m_inv @ c
    return ndi.affine_transform(arr, matrix=m_inv, offset=offset, order=order,
                                 cval=cval, mode='constant')


def _rotation_to_point_down(center_rc, target_rc):
    """Angle that rotates the vector center->target to point straight down
    (positive row direction, zero column offset)."""
    dr, dc = target_rc[0] - center_rc[0], target_rc[1] - center_rc[1]
    theta_current = np.arctan2(dc, dr)
    return -theta_current


def _add_scalebar(ax, shape, um_per_px, scale_um):
    h, w = shape
    bar_px = scale_um / um_per_px
    pad_x, pad_y = w * 0.04, h * 0.06
    bar_h = max(h * 0.02, 3)
    x0, y0 = pad_x, h - pad_y - bar_h
    ax.add_patch(plt.Rectangle((x0, y0), bar_px, bar_h, color='black', clip_on=False))
    ax.text(x0 + bar_px / 2, y0 - bar_h, f'{scale_um} µm', ha='center', va='bottom',
            color='black', fontsize=13, fontweight='bold')


# ── pass 1: build crops + gather global color scale ────────────────────────────

frames = {}
all_cn_vals = []

for t in TIMEPOINTS:
    labels = tifffile.imread(str(label_files[t]))
    h2b    = tifffile.imread(str(raw_files[t]))

    cn_vol  = _map_lut_to_frame(labels, _cn_lut_for_t(t))
    cn_proj = np.nanmax(cn_vol, axis=0)
    h2b_proj = h2b.max(axis=0).astype(np.float32)
    embryo_mask = np.any(labels > 0, axis=0)

    # ── rotate in-plane so the ICM point is always straight below the embryo
    # centroid — a consistent orientation for comparing tracks across snapshots
    rows_nz, cols_nz = np.nonzero(embryo_mask)
    center_rc = (rows_nz.mean(), cols_nz.mean())
    theta_rot = _rotation_to_point_down(center_rc, (ICM_ROW_PX, ICM_COL_PX))

    h2b_rot    = _rotate_image(h2b_proj, theta_rot, center_rc, order=1, cval=0.0)
    cn_rot     = _rotate_image(cn_proj, theta_rot, center_rc, order=0, cval=np.nan)
    embryo_rot = _rotate_image(embryo_mask.astype(np.float32), theta_rot, center_rc,
                                order=0, cval=0.0) > 0.5
    icm_rc_rot = _rotate_point((ICM_ROW_PX, ICM_COL_PX), theta_rot, center_rc)

    x0, x1, y0, y1 = _embryo_crop_box_px_from_mask(embryo_rot, CROP_PAD_UM)
    cn_crop  = cn_rot[y0:y1, x0:x1]
    h2b_crop = h2b_rot[y0:y1, x0:x1]
    icm_xy_crop = (icm_rc_rot[1] - x0, icm_rc_rot[0] - y0)

    track_masks = {}
    track_cn = {}
    for track_id in TRACK_IDS:
        row = erk[(erk['t'] == t) & (erk['track_id'] == track_id)]
        if row.empty:
            continue
        lid = int(row['label_id'].iloc[0])
        mask2d = np.any(labels == lid, axis=0).astype(np.float32)
        mask_rot = _rotate_image(mask2d, theta_rot, center_rc, order=0, cval=0.0) > 0.5
        track_masks[track_id] = mask_rot[y0:y1, x0:x1]
        track_cn[track_id] = row['erk_cn_ratio'].iloc[0]

    frames[t] = dict(h2b=h2b_crop, cn=cn_crop, masks=track_masks, track_cn=track_cn,
                      icm_xy=icm_xy_crop)
    finite = cn_crop[np.isfinite(cn_crop)]
    if len(finite) > 0:
        all_cn_vals.append(finite)

all_cn_vals = np.concatenate(all_cn_vals) if all_cn_vals else np.array([0.0, 1.0])
vmin, vmax = np.percentile(all_cn_vals, (2, 98))
print(f'ERK C/N color scale: [{vmin:.3f}, {vmax:.3f}]')

# ── pass 2: composite + render ───────────────────────────────────────────────

cmap = plt.cm.get_cmap('coolwarm')

icm_dists = {tid: icm_lookup.get(tid, np.nan) for tid in TRACK_IDS}

for t in TIMEPOINTS:
    h2b_crop = frames[t]['h2b']
    cn_crop  = frames[t]['cn']
    masks    = frames[t]['masks']
    track_cn = frames[t]['track_cn']

    finite = h2b_crop[np.isfinite(h2b_crop)]
    bg_lo, bg_hi = np.percentile(finite, (1, 99.5))
    bg = np.clip((h2b_crop - bg_lo) / (bg_hi - bg_lo + 1e-9), 0, 1)
    bg = 1 - bg   # invert: bright nuclei -> dark, background -> white
    rgb = np.stack([bg] * 3, axis=-1)

    valid = np.isfinite(cn_crop)
    norm = np.clip((cn_crop - vmin) / (vmax - vmin + 1e-9), 0, 1)
    overlay = cmap(norm)[..., :3]
    alpha = np.where(valid, 0.85, 0.0)[..., None]
    comp = rgb * (1 - alpha) + overlay * alpha

    # Force each highlighted track's own ERK C/N value inside its outline, so an
    # overlapping neighbor's (higher) value can't steal the pixels in the max
    # projection and hide the highlighted nucleus's true color.
    for track_id, mask in masks.items():
        val = track_cn.get(track_id)
        if val is None or pd.isna(val) or not mask.any():
            continue
        own_norm = np.clip((val - vmin) / (vmax - vmin + 1e-9), 0, 1)
        own_color = np.array(cmap(own_norm)[:3])
        comp[mask] = rgb[mask] * (1 - 0.85) + own_color * 0.85

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(comp, interpolation='nearest')
    ax.axis('off')

    legend_handles = []
    for i, track_id in enumerate(TRACK_IDS):
        color = TRACK_COLORS.get(i, '#ffffff')
        mask = masks.get(track_id)
        if mask is not None and mask.any():
            ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=2.8)
        dist = icm_dists.get(track_id)
        icm_label = f'{dist:.0f} µm from ICM' if not pd.isna(dist) else 'ICM dist: n/a'
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=2.2, label=icm_label))

    ax.legend(handles=legend_handles, loc='lower right', bbox_to_anchor=(1.0, 1.04),
              fontsize=9, framealpha=0.75, labelcolor='black', facecolor='white',
              edgecolor='black', ncol=len(legend_handles))

    _add_scalebar(ax, h2b_crop.shape, VX_X, SCALE_BAR_UM)
    ax.text(0.02, 1.02, f't={t}  ({t * cfg["tracking"]["frame_interval_min"]} min)',
            transform=ax.transAxes, ha='left', va='bottom', color='black',
            fontsize=12, fontweight='bold')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    cb = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.04, pad=0.03, shrink=0.7)
    cb.set_label('ERK C/N ratio', fontsize=9, color='black')
    cb.ax.tick_params(labelsize=8, labelcolor='black')

    out_path = SNAP_DIR / f'highlight_t{t:03d}_{"_".join(str(i) for i in TRACK_IDS)}_{VERSION}.png'
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out_path}')
