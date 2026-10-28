"""
One colour per track for every napari view of tracks (the 04 curation
notebooks, curate_pairs.py, 01_view_meshes.py's 'label' colouring): Glasbey
- colorcet's glasbey_light, 256 colours chosen to be as distinct from each
other as possible, all light enough to read on napari's black background -
indexed by id mod 256. A track gets the same colour in its label, mesh,
point and tail layers, and in every viewer. Nuclei in no track are grey.
"""

import numpy as np
import colorcet as cc
from matplotlib.colors import to_rgba
from napari.utils.colormaps import Colormap, DirectLabelColormap

PALETTE = np.array([to_rgba(c) for c in cc.glasbey_light])
UNTRACKED = (0.6, 0.6, 0.6, 1.0)

# Piecewise-constant over [0, 1]: glasbey_values(ids) -> PALETTE[id mod 256], for
# layers coloured through a colormap (tracks tails - napari passes a
# colormaps_dict property through unscaled - and surfaces with contrast 0-1).
GLASBEY_CMAP = Colormap(PALETTE, name='glasbey_tracks', interpolation='zero',
                        controls=np.linspace(0, 1, len(PALETTE) + 1))


def glasbey_values(ids):
    """Scalar per id for GLASBEY_CMAP."""
    return (np.asarray(ids, dtype=np.int64) % len(PALETTE) + 0.5) / len(PALETTE)


def track_rgba(track_ids):
    """(N, 4) RGBA per id; NaN (not in a track) -> grey."""
    ids = np.asarray(track_ids, dtype=float)
    out = np.tile(UNTRACKED, (len(ids), 1))
    ok = ~np.isnan(ids)
    out[ok] = PALETTE[ids[ok].astype(np.int64) % len(PALETTE)]
    return out


def label_colormap(track_ids, offset=1):
    """Labels-layer colormap for a volume whose voxels are track_id + offset
    (the 04 notebooks' track_labels); 0 and unlisted values are transparent."""
    ids = np.unique(np.asarray(track_ids, dtype=float))
    ids = ids[~np.isnan(ids)].astype(np.int64)
    colors = {None: (0.0, 0.0, 0.0, 0.0), 0: (0.0, 0.0, 0.0, 0.0)}
    colors.update(zip((ids + offset).tolist(), map(tuple, track_rgba(ids))))
    return DirectLabelColormap(color_dict=colors)


def tracks_style(track_ids):
    """add_tracks keyword arguments colouring tails by track_id (one per row of
    the tracks data, i.e. its first column)."""
    return dict(properties={'glasbey': glasbey_values(track_ids)}, color_by='glasbey',
                colormaps_dict={'glasbey': GLASBEY_CMAP})


def set_tracks(layer, data):
    """Replace a tracks layer's data, keeping the Glasbey colouring."""
    style = tracks_style(data[:, 0])
    layer.data = data
    layer.properties = style['properties']
    layer.colormaps_dict = style['colormaps_dict']
    layer.color_by = style['color_by']
