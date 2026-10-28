"""
Pair-view track curation: one transition at a time, light enough for a laptop.

Shows two frames side by side - "from" frame (left) and "to" frame t (right) -
and only keeps a handful of frames in memory, loading others on demand and
preloading the next one in the background. Click a nucleus on the left, then
its match on the right, and that single link is set. The left frame defaults
to t-1 but can be switched to t-2 / t-3 to link across a missing nucleus.

Reads and writes the same tracks CSVs as 04_curate_tracks*.ipynb
(track_id, t, label_id, ... ), so saves feed straight into the notebooks,
03_trackastra_export_ctc.py, etc.

Mouse / keys (napari viewer):
  click left frame     select the "from" nucleus (red ring)
  click right frame    with a "from" nucleus selected: link them
                       (replaces that nucleus's old link onward, and the
                       right nucleus's old incoming link); otherwise select it
  b                    break the selected nucleus's link (onward for left,
                       incoming for right)
  n / p                next / previous transition (t+1 / t-1)
  1 / 2 / 3            compare t with t-1 / t-2 / t-3
  u                    undo last edit
  s                    save to OUTPUT_CSV
  Escape               clear selection
Click-drag still pans; only a click without dragging selects.

Self-contained: needs only this file, the data files listed in SETTINGS, and
a python env with  napari pyqt tifffile pandas scikit-image  (+ colorcet for
the Glasbey track colours shared with the 04 notebooks - without it, tracks
fall back to golden-ratio hues).
Run:  python curate_pairs.py
"""

import colorsys
import glob
import json
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops_table

try:   # Glasbey, indexed by track id mod 256 - same colours as src/track_colors.py
    import colorcet
    from matplotlib.colors import to_rgba
    GLASBEY = [to_rgba(c) for c in colorcet.glasbey_light]
except ImportError:
    GLASBEY = None

# =============================================================================
#  SETTINGS - point these at the data on this machine
# =============================================================================

# Dataset folder. On the Mac, e.g. Path('~/data/250914_stack5'). Everything
# below is inside it by default - edit any line directly if your copy differs.
DATA_DIR = Path('/mnt/md0/elysse/nnUNet/inference/Dataset003_icm_te/250914_stack5')

LABEL_DIR        = DATA_DIR / 'labels'                       # *_instances_reclassified.tif, one per t
RAW_DIR          = DATA_DIR / 'imagesTs' / 'tifs'            # Dataset003_*_0000.tif, one per t
CROP_INFO_JSON   = DATA_DIR / 'crop_info.json'               # z-crop offsets (per-frame on stack5, uniform on later stacks)
FEATURES_CSV     = DATA_DIR / 'results' / 'features' / 'features.csv'
FEATURES_REG_CSV = DATA_DIR / 'results' / 'features' / 'features_registered.csv'

# Tracks to load, and where to save. Saving never overwrites INPUT_CSV, and
# the app won't start if OUTPUT_CSV already exists - bump the version each pass.
INPUT_CSV  = DATA_DIR / 'results' / 'tracks' / 'trackastra' / 'tracks_linked_c3.csv'
OUTPUT_CSV = DATA_DIR / 'results' / 'tracks' / 'trackastra' / 'tracks_linked_c4.csv'

LABEL_GLOB = '*_instances_reclassified.tif'
RAW_GLOB   = 'Dataset003_*_0000.tif'
VX_Z, VX_Y, VX_X = 2.0, 0.208, 0.208   # um per voxel (z, y, x)
N_T = 121                              # timepoints

START_T      = 1     # first "to" frame shown
CACHE_SIZE   = 3     # frames kept in memory (~0.4 GB each): the pair shown + the next, preloading
MAX_LOOKBACK = 3     # how far back the left frame can go

# =============================================================================

LABEL_DIR, RAW_DIR, CROP_INFO_JSON, FEATURES_CSV, FEATURES_REG_CSV, INPUT_CSV, OUTPUT_CSV = (
    Path(p).expanduser() for p in
    (LABEL_DIR, RAW_DIR, CROP_INFO_JSON, FEATURES_CSV, FEATURES_REG_CSV, INPUT_CSV, OUTPUT_CSV))
assert INPUT_CSV != OUTPUT_CSV, 'OUTPUT_CSV must differ from INPUT_CSV'

with open(CROP_INFO_JSON) as f:
    CROP_INFO = json.load(f)
Z_FULL, Y_FULL, X_FULL = next(iter({tuple(v['original_shape']) for v in CROP_INFO.values()}))

LABEL_FILES = sorted(glob.glob(str(LABEL_DIR / LABEL_GLOB)))
RAW_FILES   = sorted(glob.glob(str(RAW_DIR / RAW_GLOB)))
assert len(LABEL_FILES) >= N_T and len(RAW_FILES) >= N_T, \
    f'found {len(LABEL_FILES)} label / {len(RAW_FILES)} raw files for {N_T} timepoints - check LABEL_DIR / RAW_DIR'


# =============================================================================
#  Frames: loaded on demand at their saved (z-cropped) size
# =============================================================================

class Frame:
    def __init__(self, raw, labels, z0, centroids):
        self.raw = raw              # (z, Y, X) as saved on disk - not padded
        self.labels = labels        # instance labels, same shape
        self.z0 = z0                # slice of the full stack where this crop starts
        self.centroids = centroids  # label_id -> (z, y, x) voxel, z in full-stack slices


def _load_frame(t: int) -> Frame:
    lf, rf = LABEL_FILES[t], RAW_FILES[t]
    tp = f'{t:05d}'
    assert tp in Path(lf).name and tp in Path(rf).name, f't={t}: unexpected files {lf}, {rf}'
    # first_z is where what's on disk starts in the full stack - differs per
    # frame on stack5 (prepare_inference.py's old per-frame crop), constant on
    # later stacks (UNIFORM_Z_CROP). Frames are shifted there
    # in the viewer instead of padded, so t and t+1 line up at about half the memory.
    z0 = CROP_INFO[tp]['first_z']
    labels = tifffile.imread(lf)
    raw = tifffile.imread(rf)
    props = regionprops_table(labels, properties=('label', 'centroid'))
    centroids = {int(l): (z + z0, y, x) for l, z, y, x in
                 zip(props['label'], props['centroid-0'], props['centroid-1'], props['centroid-2'])}
    return Frame(raw, labels, z0, centroids)


class FrameStore:
    """Small LRU cache of frames with background preloading."""

    def __init__(self, size: int):
        self.size = size
        self._cache = OrderedDict()
        self._pending = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=2)

    def _future(self, t):
        with self._lock:
            if t in self._cache:
                self._cache.move_to_end(t)
                return None
            if t not in self._pending:
                self._pending[t] = self._pool.submit(_load_frame, t)
            return self._pending[t]

    def get(self, t: int) -> Frame:
        fut = self._future(t)
        if fut is not None:
            frame = fut.result()
            with self._lock:
                self._pending.pop(t, None)
                self._cache[t] = frame
                while len(self._cache) > self.size:
                    self._cache.popitem(last=False)
        with self._lock:
            return self._cache[t]

    def prefetch(self, t: int):
        if 0 <= t < N_T:
            self._future(t)

    def is_ready(self, t: int) -> bool:
        with self._lock:
            return t in self._cache or (t in self._pending and self._pending[t].done())


# =============================================================================
#  Track graph: nuclei (t, label_id) linked forward; tracks = chains
# =============================================================================

class TrackGraph:
    def __init__(self, df: pd.DataFrame):
        df = df.dropna(subset=['label_id']).astype({'label_id': int, 't': int, 'track_id': int})
        assert not df.duplicated(['t', 'label_id']).any(), 'a nucleus is in more than one track'
        assert not df.duplicated(['track_id', 't']).any(), 'a track has two nuclei at one t'
        self.columns = list(df.columns)
        self.rows = {(t, l): row for (t, l), row in zip(zip(df['t'], df['label_id']), df.to_dict('records'))}
        self.track = {n: r['track_id'] for n, r in self.rows.items()}
        self.succ = {}
        for _, g in df.sort_values('t').groupby('track_id'):
            nodes = list(zip(g['t'], g['label_id']))
            self.succ.update(zip(nodes, nodes[1:]))
        self.history = []
        self.dirty = False

    @property
    def pred(self):
        return {v: u for u, v in self.succ.items()}

    def add_node(self, node, row: dict):
        if node not in self.rows:
            self.rows[node] = row
            self.track[node] = max(self.track.values(), default=-1) + 1

    def _snapshot(self):
        self.history.append((dict(self.succ), dict(self.track)))

    def link(self, a, b):
        assert a[0] < b[0], 'links go forward in time'
        if self.succ.get(a) == b:
            return False
        self._snapshot()
        self.succ.pop(a, None)
        old_pred = self.pred.get(b)
        if old_pred is not None:
            self.succ.pop(old_pred)
        self.succ[a] = b
        self._relabel()
        return True

    def break_after(self, a):
        if a not in self.succ:
            return False
        self._snapshot()
        self.succ.pop(a)
        self._relabel()
        return True

    def break_before(self, b):
        a = self.pred.get(b)
        return self.break_after(a) if a is not None else False

    def undo(self):
        if not self.history:
            return False
        self.succ, self.track = self.history.pop()
        self.dirty = bool(self.history)
        return True

    def _relabel(self):
        # Each chain keeps the track id its first nucleus had, so an edit only
        # renumbers the piece that was split off or joined.
        pred = self.pred
        used, new_track = set(), {}
        next_id = max(self.track.values(), default=-1) + 1
        for start in sorted(n for n in self.rows if n not in pred):
            tid = self.track[start]
            if tid in used:
                tid, next_id = next_id, next_id + 1
            used.add(tid)
            n = start
            while True:
                new_track[n] = tid
                if n not in self.succ:
                    break
                n = self.succ[n]
        self.track = new_track
        self.dirty = True

    def to_dataframe(self) -> pd.DataFrame:
        recs = [{**self.rows[n], 'track_id': self.track[n]} for n in self.rows]
        return pd.DataFrame(recs)[self.columns].sort_values(['track_id', 't']).reset_index(drop=True)


def _feature_rows():
    """Rows for nuclei not yet in any track, in the tracks CSV's column names."""
    reg = pd.read_csv(FEATURES_REG_CSV, usecols=['t', 'label_id', 'z_um_reg', 'y_um_reg', 'x_um_reg'])
    raw = pd.read_csv(FEATURES_CSV, usecols=['t', 'label_id', 'z_um', 'y_um', 'x_um'])
    raw = raw.rename(columns={'z_um': 'z_um_orig', 'y_um': 'y_um_orig', 'x_um': 'x_um_orig'})
    feat = reg.merge(raw, on=['t', 'label_id'])
    return {(int(r['t']), int(r['label_id'])): r for r in feat.to_dict('records')}


# =============================================================================
#  Viewer
# =============================================================================

def main():
    import napari
    from napari.qt.threading import thread_worker
    from napari.utils.colormaps import DirectLabelColormap
    from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

    if OUTPUT_CSV.exists():
        raise SystemExit(f'{OUTPUT_CSV} already exists - set OUTPUT_CSV to a new version '
                         f'(saves within one session overwrite it; a new session never does).')
    graph = TrackGraph(pd.read_csv(INPUT_CSV))
    features = _feature_rows()
    store = FrameStore(CACHE_SIZE)
    scale = (VX_Z, VX_Y, VX_X)
    right_shift = X_FULL * VX_X + 20.0   # um between the two frames
    state = {'t': max(1, START_T), 'back': 1, 'sel': None, 'frames': {}, 'busy': False}

    viewer = napari.Viewer(title='Pair curation')
    placeholder = np.zeros((1, 1, 1), dtype=np.uint16)   # replaced by real frames on load
    layers = {}
    for side, dx in (('L', 0.0), ('R', right_shift)):
        tr = (0.0, 0.0, dx)
        layers[f'raw{side}'] = viewer.add_image(placeholder, name=f'raw {side}', scale=scale, translate=tr, colormap='gray')
        layers[f'trk{side}'] = viewer.add_labels(placeholder, name=f'tracks {side}', scale=scale, translate=tr, opacity=0.35)
        layers[f'pts{side}'] = viewer.add_points(
            np.empty((0, 3)), name=f'ids {side}', scale=scale, translate=tr, size=6,
            face_color='transparent', border_color='transparent', out_of_slice_display=True,
            features={'track': np.empty(0, dtype=int)},
            text={'string': '{track}', 'size': 9, 'color': 'white'})
    sel_layer = viewer.add_points(np.empty((0, 3)), name='selected', scale=scale, size=14,
                                  face_color='transparent', border_color='red', border_width=0.15,
                                  out_of_slice_display=True)
    # Display-only layers: no painting/point editing, and keep an image layer
    # active so napari's per-layer shortcuts (1-6, p, ...) can't take over our keys.
    for layer in viewer.layers:
        layer.editable = False
    viewer.layers.selection.active = layers['rawR']

    # --- dock widget ---
    panel = QWidget()
    lay = QVBoxLayout(panel)
    row = QHBoxLayout()
    row.addWidget(QLabel('to frame t'))
    t_spin = QSpinBox()
    t_spin.setRange(1, N_T - 1)
    t_spin.setValue(state['t'])
    row.addWidget(t_spin)
    row.addWidget(QLabel('from'))
    back_box = QComboBox()
    back_box.addItems([f't-{k}' for k in range(1, MAX_LOOKBACK + 1)])
    row.addWidget(back_box)
    lay.addLayout(row)
    btns = QHBoxLayout()
    for text, key in (('◀ prev (p)', 'p'), ('next ▶ (n)', 'n'), ('break (b)', 'b'), ('undo (u)', 'u'), ('save (s)', 's')):
        b = QPushButton(text)
        b.clicked.connect(lambda _=False, k=key: actions[k]())
        btns.addWidget(b)
    lay.addLayout(btns)
    status = QLabel('')
    status.setWordWrap(True)
    lay.addWidget(status)
    lay.addStretch()
    viewer.window.add_dock_widget(panel, name='pair curation', area='right')

    def left_t():
        return state['t'] - state['back']

    def say(msg=''):
        unsaved = '  [unsaved edits]' if graph.dirty else ''
        sel = ''
        if state['sel'] is not None:
            side, node = state['sel']
            nxt = graph.succ.get(node)
            prv = graph.pred.get(node)
            sel = (f"\nselected ({'left' if side == 'L' else 'right'}): t={node[0]} label {node[1]}, "
                   f"track {graph.track.get(node, '-')}"
                   + (f", linked to t={nxt[0]} label {nxt[1]}" if side == 'L' and nxt else '')
                   + (f", linked from t={prv[0]} label {prv[1]}" if side == 'R' and prv else ''))
        status.setText(f"from t={left_t()}  ->  to t={state['t']}{unsaved}{sel}\n{msg}")

    def track_color(tid):
        # Same track -> same colour in both frames and in the 04 notebooks.
        if GLASBEY is not None:
            return GLASBEY[tid % len(GLASBEY)]
        r, g, b = colorsys.hsv_to_rgb((tid * 0.618033988749895) % 1.0, 0.75, 1.0)
        return (r, g, b, 1.0)

    def track_colormap(t, frame):
        # Colour the segmentation labels by track directly - no extra
        # track-relabelled copy of the volume. Untracked nuclei show grey.
        colors = {None: (0.6, 0.6, 0.6, 1.0), 0: (0.0, 0.0, 0.0, 0.0)}
        for l in frame.centroids:
            tid = graph.track.get((t, l))
            if tid is not None:
                colors[l] = track_color(tid)
        return DirectLabelColormap(color_dict=colors)

    def draw_side(side, t, frame):
        layers[f'trk{side}'].colormap = track_colormap(t, frame)
        nodes = [(l, c) for l, c in frame.centroids.items() if (t, l) in graph.track]
        layers[f'pts{side}'].data = np.array([c for _, c in nodes]).reshape(-1, 3)
        layers[f'pts{side}'].features = {'track': np.array([graph.track[(t, l)] for l, _ in nodes], dtype=int)}

    def draw_selection():
        if state['sel'] is None:
            sel_layer.data = np.empty((0, 3))
            return
        side, (t, l) = state['sel']
        frame = state['frames'].get(side)
        if frame is None or l not in frame.centroids:
            return
        z, y, x = frame.centroids[l]
        sel_layer.data = np.array([[z, y, x + (right_shift / VX_X if side == 'R' else 0.0)]])

    def redraw_tracks():
        for side, t in (('L', left_t()), ('R', state['t'])):
            if side in state['frames']:
                draw_side(side, t, state['frames'][side])
        draw_selection()

    @thread_worker
    def _load_pair(tl, tr):
        return store.get(tl), store.get(tr)

    def show_pair():
        if state['busy']:
            return
        tl, tr = left_t(), state['t']
        state['busy'] = True
        say('loading...' if not (store.is_ready(tl) and store.is_ready(tr)) else '')

        def done(frames):
            fl, fr = frames
            state['frames'] = {'L': fl, 'R': fr}
            if 'clim' not in state:
                sample = fr.raw[:, ::4, ::4]
                state['clim'] = (float(np.percentile(sample, 1)), float(np.percentile(sample, 99.8)))
            for side, f in (('L', fl), ('R', fr)):
                offset = (f.z0 * VX_Z, 0.0, right_shift if side == 'R' else 0.0)
                for kind, data in (('raw', f.raw), ('trk', f.labels)):
                    layers[f'{kind}{side}'].data = data
                    layers[f'{kind}{side}'].translate = offset
                layers[f'raw{side}'].contrast_limits = state['clim']
            redraw_tracks()
            state['busy'] = False
            say()
            store.prefetch(tr + 1)
            if (tl, tr) != (left_t(), state['t']):   # moved while loading
                show_pair()

        def failed(e):
            state['busy'] = False
            say(f'load failed: {e}')

        w = _load_pair(tl, tr)
        w.returned.connect(done)
        w.errored.connect(failed)
        w.start()

    def go(t=None, back=None):
        if t is not None:
            state['t'] = int(np.clip(t, 1, N_T - 1))
        if back is not None:
            state['back'] = back
        state['back'] = min(state['back'], state['t'])
        state['sel'] = None
        for widget, value in ((t_spin, state['t']), (back_box, state['back'] - 1)):
            widget.blockSignals(True)
            (widget.setValue if widget is t_spin else widget.setCurrentIndex)(value)
            widget.blockSignals(False)
        show_pair()

    def node_at(position):
        z, y, x = position
        side = 'R' if x >= right_shift else 'L'
        frame = state['frames'].get(side)
        if frame is None:
            return None
        xv = (x - (right_shift if side == 'R' else 0.0)) / VX_X
        zi = int(round(z / VX_Z)) - frame.z0   # full-stack slice -> this frame's crop
        yi, xi = int(round(y / VX_Y)), int(round(xv))
        nz, ny, nx = frame.labels.shape
        if not (0 <= zi < nz and 0 <= yi < ny and 0 <= xi < nx):
            return None
        label = int(frame.labels[zi, yi, xi])
        if label == 0:
            return None
        t = left_t() if side == 'L' else state['t']
        return side, (t, label)

    def ensure_node(node):
        if node not in graph.rows and node in features:
            graph.add_node(node, {**{c: np.nan for c in graph.columns}, **{
                k: v for k, v in features[node].items() if k in graph.columns}})
        return node in graph.rows

    def handle_click(position):
        if state['busy']:
            return
        hit = node_at(position)
        if hit is None:
            return say('clicked background - click inside a nucleus')
        side, node = hit
        if not ensure_node(node):
            return say(f'label {node[1]} at t={node[0]} not in features table')
        if side == 'R' and state['sel'] is not None and state['sel'][0] == 'L':
            a = state['sel'][1]
            changed = graph.link(a, node)
            state['sel'] = None
            redraw_tracks()
            return say(f'linked t={a[0]} label {a[1]} -> t={node[0]} label {node[1]} (track {graph.track[node]})'
                       if changed else 'already linked')
        state['sel'] = (side, node)
        draw_selection()
        say('now click its match on the right' if side == 'L' else '')

    def on_mouse(viewer, event):
        if event.button != 1 or event.modifiers:
            return
        start = np.array(event.position)
        moved = False
        yield
        while event.type == 'mouse_move':
            moved = moved or np.abs(np.array(event.position) - start).max() > 1.0
            yield
        if not moved:
            handle_click(start)

    viewer.mouse_drag_callbacks.append(on_mouse)

    def do_break():
        if state['sel'] is None:
            return say('select a nucleus first')
        side, node = state['sel']
        changed = graph.break_after(node) if side == 'L' else graph.break_before(node)
        state['sel'] = None
        redraw_tracks()
        say('link broken' if changed else 'nothing to break')

    def do_undo():
        say('undone' if graph.undo() else 'nothing to undo')
        redraw_tracks()

    def do_save():
        df = graph.to_dataframe()
        assert not df.duplicated(['track_id', 't']).any() and not df.duplicated(['t', 'label_id']).any()
        df.to_csv(OUTPUT_CSV, index=False)
        graph.dirty = False
        say(f'saved {OUTPUT_CSV.name} ({df.track_id.nunique()} tracks, {len(df)} nuclei)')

    def clear_sel():
        state['sel'] = None
        draw_selection()
        say()

    actions = {
        'n': lambda: go(t=state['t'] + 1), 'p': lambda: go(t=state['t'] - 1),
        'b': do_break, 'u': do_undo, 's': do_save, 'Escape': clear_sel,
        '1': lambda: go(back=1), '2': lambda: go(back=2), '3': lambda: go(back=3),
    }
    for key, fn in actions.items():
        viewer.bind_key(key, lambda v, fn=fn: fn(), overwrite=True)
    t_spin.valueChanged.connect(lambda v: go(t=v))
    back_box.currentIndexChanged.connect(lambda i: go(back=i + 1))

    viewer.dims.ndisplay = 2
    viewer.dims.set_point(0, Z_FULL * VX_Z / 2)
    go()
    napari.run()


if __name__ == '__main__':
    main()
