"""Interactive crop-bounds inspector for live-imaging (timecourse) data.

Same idea as pipelines/IF/inspect_crop_bounds.py but for live_timecourse
datasets: shows each stack's first and last timepoint MIP (per channel, 2
rows) with the current auto-crop box overlaid, since embryo position can
drift across a long movie and a box that looks right at t=0 may miss the
embryo by the last frame. Drag to draw an XY box; press Enter to accept, S
to skip, Q to quit. For stacks with multiple embryos in one field of view,
draw one box per embryo (A to add each) so each gets its own cropped
TIFF/timepoint-folder instead of being auto-cropped together as a single
blob. Accepted overrides are saved next to the config, named by the config's
own live_timecourse.crop_overrides_file (falling back to crop_overrides.yaml
if unset) so multiple dataset configs sharing a folder don't collide on one
shared overrides file. Same format as the fixed-IF pipeline; convert_h5_timecourse_to_tiff.py
picks the same file up automatically.

Uses each stack's scope-generated mip/*.max.z.tiff previews when available
(much faster than re-streaming full h5 volumes just to look at them),
falling back to computing the MIP directly from the h5 file otherwise.

To run: edit CONFIG_PATH / STACKS below if needed, then click VS Code's "Run
Python File" button (or Ctrl+F5) — no terminal command needed. Needs a
display (X11/VNC), since it opens a matplotlib window.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collections import Counter

import h5py
import numpy as np
import tifffile
from skimage import io as skio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RectangleSelector
import yaml

from src.conversion import (
    autocrop_bounds_from_timepoints,
    build_live_stack_groups,
    find_h5_files_sorted,
    get_config_value,
    load_yaml_config,
    _effective_zyx_shape,
    _get_h5_dataset,
    _read_zslice,
)

# ============================== EDIT THESE ==============================
CONFIG_PATH = PROJECT_ROOT / "configs" / "other live images" / "260721_e45c_fgf_oct4_snap.yaml"
STACKS = None   # e.g. ["stack_0", "stack_9-fgf "] to restrict, or None for all
# ==========================================================================

OVERRIDES_FILE = "crop_overrides.yaml"


# ---------------------------------------------------------------------------
# Overrides file helpers
# ---------------------------------------------------------------------------

def load_overrides(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def save_overrides(path: Path, overrides: dict) -> None:
    with open(path, "w") as f:
        yaml.dump(overrides, f, default_flow_style=False, sort_keys=True)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# MIP helpers — prefer the scope's own precomputed mip/*.max.z.* preview,
# fall back to streaming the h5 volume only if that's missing.
# ---------------------------------------------------------------------------

def _find_precomputed_mip(h5_path: Path) -> Path | None:
    name = h5_path.name
    stem = h5_path.stem
    for suffix in (".lux.h5", ".h5", ".hdf5"):
        if name.lower().endswith(suffix):
            stem = name[: -len(suffix)]
            break
    mip_dir = h5_path.parent / "mip"
    if not mip_dir.is_dir():
        return None
    for ext in ("tiff", "tif", "jpg", "jpeg"):
        candidate = mip_dir / f"{stem}.max.z.{ext}"
        if candidate.exists():
            return candidate
    return None


def _compute_mip_from_h5(h5_file: Path, dataset_path: str | None) -> np.ndarray:
    with h5py.File(h5_file, "r") as f:
        ds = _get_h5_dataset(f, dataset_path)
        nz, ny, nx = _effective_zyx_shape(ds)
        mip = np.zeros((ny, nx), dtype=np.float32)
        for z in range(nz):
            sl = np.asarray(_read_zslice(ds, z), dtype=np.float32)
            np.maximum(mip, sl, out=mip)
    return mip


def get_mip(h5_file: Path, dataset_path: str | None) -> np.ndarray:
    precomputed = _find_precomputed_mip(h5_file)
    if precomputed is not None:
        img = tifffile.imread(str(precomputed)) if precomputed.suffix.lower() in (".tif", ".tiff") \
            else skio.imread(str(precomputed))
        return np.asarray(img, dtype=np.float32)
    return _compute_mip_from_h5(h5_file, dataset_path)


# ---------------------------------------------------------------------------
# Interactive inspector for one stack
# ---------------------------------------------------------------------------

def inspect_stack(
    stack_id: str,
    channel_labels: list,
    t0_mips: list,
    t_last_mips: list,
    t0_idx: int,
    t_last_idx: int,
    auto_bounds,           # (z0, z1, y0, y1, x0, x1) or None
    existing_override,     # str | list[str] | None
) -> tuple:
    """Show t=0 (top row) + t=last (bottom row) MIPs per channel; let the
    user draw one or more crop rectangles, mirrored across every subplot.

    Controls:
      Drag     – draw a crop box (lime)
      A        – add the current box to the list (turns cyan); draw another
      Enter    – accept all added boxes (or the current one if none added,
                 or the auto bounds if nothing was drawn at all)
      S        – skip this stack (no change saved)
      Q / Esc  – quit the whole session

    Returns (action, crops_list) where action is 'accept', 'skip', or 'quit'
    and crops_list is a list of (y0, y1, x0, x1) tuples, or None (use auto).
    """
    n = len(channel_labels)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10), squeeze=False)

    if hasattr(fig.canvas, "manager") and hasattr(fig.canvas.manager, "key_press_handler_id"):
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

    source = f"override: {existing_override}" if existing_override else "auto-crop"
    fig.suptitle(
        f"{stack_id}  [{source}]   top row: t={t0_idx}   bottom row: t={t_last_idx}\n"
        "Drag to draw crop  |  A = add box  |  E/Enter = save  |  S = skip  |  Q = quit",
        fontsize=9,
    )

    all_axes = list(axes.flat)  # row-major: [t0 ch0, t0 ch1, ..., tlast ch0, tlast ch1, ...]
    all_mips = list(t0_mips) + list(t_last_mips)
    row_labels = [f"{lbl} (t={t0_idx})" for lbl in channel_labels] + \
                 [f"{lbl} (t={t_last_idx})" for lbl in channel_labels]

    for ax, mip, label in zip(all_axes, all_mips, row_labels):
        p1, p99 = np.percentile(mip, (1, 99))
        ax.imshow(mip, cmap="gray", vmin=p1, vmax=p99, origin="upper")
        ax.set_title(label, fontsize=9)
        ax.axis("off")

        if auto_bounds is not None:
            z0, z1, y0, y1, x0, x1 = auto_bounds
            ax.add_patch(mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=1.5, edgecolor="yellow", facecolor="none",
            ))

    axes[0, 0].legend(
        handles=[
            mpatches.Patch(edgecolor="yellow", facecolor="none", label="current bounds"),
            mpatches.Patch(edgecolor="lime",   facecolor="none", label="drawing"),
            mpatches.Patch(edgecolor="cyan",   facecolor="none", label="added"),
        ],
        loc="lower right", fontsize=7, framealpha=0.6,
    )

    current_rects = [None] * len(all_axes)
    state = {"new_yx": None}
    added_crops = []

    def _update_current_rects(y0n, y1n, x0n, x1n):
        for i, ax in enumerate(all_axes):
            if current_rects[i] is not None:
                current_rects[i].remove()
            r = mpatches.Rectangle(
                (x0n, y0n), x1n - x0n, y1n - y0n,
                linewidth=1.5, edgecolor="lime", facecolor="none",
            )
            ax.add_patch(r)
            current_rects[i] = r
        fig.canvas.draw_idle()

    def on_select(eclick, erelease):
        x0n = int(round(min(eclick.xdata, erelease.xdata)))
        x1n = int(round(max(eclick.xdata, erelease.xdata)))
        y0n = int(round(min(eclick.ydata, erelease.ydata)))
        y1n = int(round(max(eclick.ydata, erelease.ydata)))
        state["new_yx"] = (y0n, y1n, x0n, x1n)
        print(f"  Drawn: y={y0n}:{y1n}  x={x0n}:{x1n}  (A to add, Enter to accept)", flush=True)
        _update_current_rects(y0n, y1n, x0n, x1n)

    # Attach the selector to the top-left axis only; mirrored to all axes above.
    selector = RectangleSelector(   # noqa: F841 (kept alive by reference)
        axes[0, 0], on_select,
        useblit=True, button=[1],
        minspanx=5, minspany=5,
        spancoords="pixels", interactive=True,
    )

    result = {"action": None}

    def _add_current():
        yx = state["new_yx"]
        if yx is None:
            print("  Nothing drawn to add.", flush=True)
            return
        y0n, y1n, x0n, x1n = yx
        added_crops.append(yx)
        for i, ax in enumerate(all_axes):
            if current_rects[i] is not None:
                current_rects[i].remove()
                current_rects[i] = None
            ax.add_patch(mpatches.Rectangle(
                (x0n, y0n), x1n - x0n, y1n - y0n,
                linewidth=1.5, edgecolor="cyan", facecolor="none",
            ))
        state["new_yx"] = None
        fig.canvas.draw_idle()
        print(f"  Added crop {len(added_crops)}: y={y0n}:{y1n}  x={x0n}:{x1n}  — draw another or press Enter", flush=True)

    def on_key(event):
        key = event.key
        if key in ("a", "A"):
            _add_current()
        elif key in ("enter", "e", "E"):
            result["action"] = "accept"
            plt.close(fig)
        elif key in ("s", "S"):
            result["action"] = "skip"
            plt.close(fig)
        elif key in ("q", "Q", "escape"):
            result["action"] = "quit"
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show(block=True)

    action = result.get("action", "skip")

    if action == "accept":
        if state["new_yx"] is not None:
            added_crops.append(state["new_yx"])
        crops_list = added_crops if added_crops else None
    else:
        crops_list = None

    return action, crops_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config_path = CONFIG_PATH
    if not config_path.exists():
        print(f"Config not found: {config_path} — edit CONFIG_PATH at the top of this script.")
        sys.exit(1)

    config = load_yaml_config(config_path)
    live_cfg = config.get("live_timecourse") or {}

    def _cv(keys, default=None):
        return get_config_value(live_cfg, keys if isinstance(keys, list) else [keys], default)

    root_dir = Path(_cv("root_dir"))
    dataset_path = _cv("dataset_path")
    auto_crop_channel      = _cv("auto_crop_channel")
    auto_crop_threshold    = _cv("auto_crop_threshold", 0) or 0
    auto_crop_threshold_pc = _cv("auto_crop_threshold_percentile")
    auto_crop_blur_sigma   = _cv("auto_crop_blur_sigma", 0)
    pad                    = _cv("pad", 0)
    n_tp_for_crop          = _cv("auto_crop_n_timepoints")  # None -> first+last only
    skip_stacks            = set(_cv("skip_stacks") or [])

    channel_name_by_index = {
        c["index"]: c.get("name")
        for c in (get_config_value(config, ["microscopy", "channels"]) or [])
        if isinstance(c, dict) and "index" in c
    }

    # Scoped to this dataset via live_timecourse.crop_overrides_file so multiple
    # dataset configs sharing the same folder don't collide on one shared file.
    overrides_filename = _cv("crop_overrides_file", OVERRIDES_FILE)
    overrides_path = config_path.parent / overrides_filename
    overrides = load_overrides(overrides_path)

    groups = build_live_stack_groups(root_dir)
    if not groups:
        print(f"No stack_*_channel_* folders found under {root_dir}. Check CONFIG_PATH's live_timecourse.root_dir.")
        sys.exit(1)

    stack_ids = sorted(groups.keys())
    if STACKS:
        missing = [s for s in STACKS if s not in groups]
        if missing:
            print(f"Stack(s) not found: {missing}")
            sys.exit(1)
        stack_ids = [s for s in stack_ids if s in STACKS]
    if skip_stacks:
        stack_ids = [s for s in stack_ids if s not in skip_stacks]

    # Auto-skip stacks that already have a saved override, unless the caller
    # explicitly named them in STACKS (which means "review this one again").
    if not STACKS:
        already_done = [s for s in stack_ids if s in overrides]
        if already_done:
            print(f"Skipping {len(already_done)} stack(s) with an existing override "
                  f"(list them in STACKS to review again): {already_done}\n")
        stack_ids = [s for s in stack_ids if s not in overrides]

    all_ch_sets = {sid: frozenset(ci for ci, _ in groups[sid]) for sid in stack_ids}
    expected_channels = Counter(all_ch_sets.values()).most_common(1)[0][0]
    bad_stacks = set()
    for sid in stack_ids:
        ch_set = all_ch_sets[sid]
        missing_ch = sorted(expected_channels - ch_set)
        extra_ch = sorted(ch_set - expected_channels)
        if missing_ch or extra_ch:
            bad_stacks.add(sid)
            msg = f"  WARNING: {sid}"
            if missing_ch:
                msg += f" — missing channels {missing_ch}"
            if extra_ch:
                msg += f" — unexpected extra channels {extra_ch}"
            print(msg)
    if bad_stacks:
        print(f"\n{len(bad_stacks)} stack(s) have channel mismatches and will be skipped.\n")

    print(f"Found {len(stack_ids)} stack(s) ({len(bad_stacks)} skipped). Overrides already saved: {sorted(overrides)}\n")
    print("Controls: drag to draw crop box | A = add box | E/Enter = save | S = skip | Q = quit\n")

    for i, stack_id in enumerate(stack_ids, 1):
        if stack_id in bad_stacks:
            print(f"[{i}/{len(stack_ids)}] {stack_id} — SKIPPED (channel mismatch)")
            continue

        items_sorted = sorted(groups[stack_id], key=lambda x: x[0])
        channel_indices = [ci for ci, _ in items_sorted]
        channel_timepoint_files = [find_h5_files_sorted(folder) for _, folder in items_sorted]

        tp_counts = [len(f) for f in channel_timepoint_files]
        if len(set(tp_counts)) > 1:
            print(f"  WARNING: unequal timepoint counts across channels {tp_counts}; using minimum.")
        n_tp = min(tp_counts)
        channel_timepoint_files = [files[:n_tp] for files in channel_timepoint_files]
        t0_idx, t_last_idx = 0, n_tp - 1

        channel_labels = [channel_name_by_index.get(ci) or f"ch{ci}" for ci in channel_indices]

        tp_indices = (
            [int(x) for x in np.linspace(0, n_tp - 1, n_tp_for_crop, dtype=int)]
            if n_tp_for_crop is not None else [t0_idx, t_last_idx]
        )
        try:
            print(f"[{i}/{len(stack_ids)}] {stack_id} — computing auto-crop bounds...")
            auto_bounds = autocrop_bounds_from_timepoints(
                channel_timepoint_files, tp_indices, dataset_path,
                pad, auto_crop_threshold, auto_crop_threshold_pc, auto_crop_blur_sigma,
                auto_crop_channel,
            )
            z0, z1, y0, y1, x0, x1 = auto_bounds
            print(f"  Auto: z={z0}:{z1}  y={y0}:{y1}  x={x0}:{x1}")
        except Exception as e:
            print(f"  Auto-crop failed ({e}) — showing full field.")
            with h5py.File(channel_timepoint_files[0][0], "r") as f:
                nz, ny, nx = _effective_zyx_shape(_get_h5_dataset(f, dataset_path))
            auto_bounds = (0, nz, 0, ny, 0, nx)

        print("  Loading t=0 and t=last MIPs (using precomputed mip/ previews where available)...", flush=True)
        t0_mips = [get_mip(files[t0_idx], dataset_path) for files in channel_timepoint_files]
        t_last_mips = [get_mip(files[t_last_idx], dataset_path) for files in channel_timepoint_files]

        stack_entry = overrides.get(stack_id, {})
        existing = stack_entry.get("crops") or stack_entry.get("crop")
        if existing:
            print(f"  Existing override: {existing}")

        action, crops_list = inspect_stack(
            stack_id, channel_labels, t0_mips, t_last_mips, t0_idx, t_last_idx,
            auto_bounds, existing,
        )

        if action == "quit":
            print("Quitting early.")
            break
        elif action == "accept":
            z0, z1 = auto_bounds[0], auto_bounds[1]
            if crops_list is not None and len(crops_list) > 1:
                crop_strs = [f"{z0}:{z1}:{y0n}:{y1n}:{x0n}:{x1n}" for y0n, y1n, x0n, x1n in crops_list]
                overrides[stack_id] = {"crops": crop_strs}
                print(f"  {len(crop_strs)} crops saved: {crop_strs}")
            elif crops_list is not None and len(crops_list) == 1:
                y0n, y1n, x0n, x1n = crops_list[0]
                crop_str = f"{z0}:{z1}:{y0n}:{y1n}:{x0n}:{x1n}"
                overrides[stack_id] = {"crop": crop_str}
                print(f"  Override saved: {crop_str}")
            else:
                crop_str = f"{z0}:{z1}:{y0}:{y1}:{x0}:{x1}"
                overrides[stack_id] = {"crop": crop_str}
                print(f"  Auto bounds locked in: {crop_str}")
            save_overrides(overrides_path, overrides)
        else:
            print("  Skipped.")

    print("\nDone.")


if __name__ == "__main__":
    main()
