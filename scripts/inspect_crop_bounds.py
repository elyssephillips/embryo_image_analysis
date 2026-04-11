"""Interactive crop-bounds inspector.

Shows the max-Z MIP for each stack with the current auto-crop box overlaid.
Drag a rectangle to redefine the XY crop; press Enter to accept, S to skip,
Q to quit.  Accepted overrides are saved to crop_overrides.yaml next to the
config, and convert_h5_channels_to_tiff.py will use them automatically.

Usage:
    python scripts/inspect_crop_bounds.py
    python scripts/inspect_crop_bounds.py --config configs/IF/config.yaml
    python scripts/inspect_crop_bounds.py --stack stack_3   # single stack
    python scripts/inspect_crop_bounds.py --stack stack_3 stack_7
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RectangleSelector
import yaml

from src.conversion import (
    build_stack_groups,
    compute_autocrop_bounds_streaming,
    find_h5_file,
    get_config_value,
    get_h5_conversion_config,
    load_yaml_config,
    _effective_zyx_shape,
    _get_h5_dataset,
    _read_zslice,
)

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
# MIP helper (full field, no cropping)
# ---------------------------------------------------------------------------

def compute_mip_full(h5_file: Path, dataset_path: str | None) -> np.ndarray:
    with h5py.File(h5_file, "r") as f:
        ds = _get_h5_dataset(f, dataset_path)
        nz, ny, nx = _effective_zyx_shape(ds)
        mip = np.zeros((ny, nx), dtype=np.float32)
        for z in range(nz):
            sl = np.asarray(_read_zslice(ds, z), dtype=np.float32)
            np.maximum(mip, sl, out=mip)
    return mip


# ---------------------------------------------------------------------------
# Interactive inspector for one stack
# ---------------------------------------------------------------------------

def inspect_stack(
    stack_id: str,
    channel_h5_files: list,
    channel_names: list,
    dataset_path: str | None,
    auto_bounds,           # (z0, z1, y0, y1, x0, x1) or None
    existing_override,     # str | list[str] | None
) -> tuple:
    """Show MIP + current bounds; let user draw one or more crop rectangles.

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
    n = len(channel_h5_files)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    # Disconnect matplotlib's default key handler so 's' doesn't open a save
    # dialog and other shortcuts don't interfere with our controls.
    if hasattr(fig.canvas, "manager") and hasattr(fig.canvas.manager, "key_press_handler_id"):
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

    source = f"override: {existing_override}" if existing_override else "auto-crop"
    fig.suptitle(
        f"{stack_id}  [{source}]\n"
        "Drag to draw crop  |  A = add box  |  Enter = accept  |  S = skip  |  Q = quit",
        fontsize=9,
    )

    print(f"  Computing MIPs...", flush=True)
    mips = [compute_mip_full(h, dataset_path) for h in channel_h5_files]

    current_rects = [None] * n   # lime: the box currently being drawn
    state = {"new_yx": None}     # (y0, y1, x0, x1) of the current drawn box
    added_crops = []             # finalized list of (y0, y1, x0, x1) added via A

    for i, (mip, ax) in enumerate(zip(mips, axes[0])):
        p1, p99 = np.percentile(mip, (1, 99))
        ax.imshow(mip, cmap="gray", vmin=p1, vmax=p99, origin="upper")
        label = channel_names[i] if i < len(channel_names) else f"ch{i}"
        ax.set_title(label, fontsize=9)
        ax.axis("off")

        # Draw current bounds in yellow
        if auto_bounds is not None:
            z0, z1, y0, y1, x0, x1 = auto_bounds
            ax.add_patch(mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=1.5, edgecolor="yellow", facecolor="none",
                label="current",
            ))

    # Small legend in the first axis
    axes[0, 0].legend(
        handles=[
            mpatches.Patch(edgecolor="yellow", facecolor="none", label="current bounds"),
            mpatches.Patch(edgecolor="lime",   facecolor="none", label="drawing"),
            mpatches.Patch(edgecolor="cyan",   facecolor="none", label="added"),
        ],
        loc="lower right", fontsize=7, framealpha=0.6,
    )

    def _update_current_rects(y0n, y1n, x0n, x1n):
        for i, ax in enumerate(axes[0]):
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

    # Attach selector to first axis only; the callback mirrors to all axes
    selector = RectangleSelector(   # noqa: F841 (kept alive by reference)
        axes[0, 0], on_select,
        useblit=True, button=[1],
        minspanx=5, minspany=5,
        spancoords="pixels", interactive=True,
    )

    result = {"action": None}

    def _add_current():
        """Commit the currently-drawn (lime) box to the added (cyan) list."""
        yx = state["new_yx"]
        if yx is None:
            print("  Nothing drawn to add.", flush=True)
            return
        y0n, y1n, x0n, x1n = yx
        added_crops.append(yx)
        # Repaint the lime rect as cyan to indicate it has been committed
        for i, ax in enumerate(axes[0]):
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
        elif key in ("enter", " "):
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

    # Build the crops_list to return
    if action == "accept":
        if added_crops:
            crops_list = added_crops          # explicitly added via A
        elif state["new_yx"] is not None:
            crops_list = [state["new_yx"]]    # single drawn box, Enter pressed directly
        else:
            crops_list = None                 # nothing drawn → accept auto bounds
    else:
        crops_list = None

    return action, crops_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive crop-bounds inspector.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--stack", nargs="+", default=None,
                        help="Inspect only these stack IDs (e.g. --stack stack_3 stack_7)")
    args = parser.parse_args()

    # Locate config
    config_path = args.config
    if config_path is None:
        for candidate in [Path("configs/config.yaml"), Path("config.yaml")]:
            if candidate.exists():
                config_path = candidate
                break
    if config_path is None:
        found = list(Path("configs").glob("**/config.yaml")) if Path("configs").exists() else []
        if len(found) == 1:
            config_path = found[0]
        elif len(found) > 1:
            print("Multiple config.yaml files found. Use --config to specify one.")
            sys.exit(1)

    config = load_yaml_config(config_path) if config_path else {}
    h5_config = get_h5_conversion_config(config)

    root_dir = Path(get_config_value(h5_config, ["root_dir"]))
    dataset_path = get_config_value(h5_config, ["dataset_path"])
    channel_names = get_config_value(config, ["microscopy", "channel_names"]) or []
    auto_crop_channel       = get_config_value(h5_config, ["auto_crop_channel"], None)
    auto_crop_threshold     = get_config_value(h5_config, ["auto_crop_threshold"], 0) or 0
    auto_crop_threshold_pct = get_config_value(h5_config, ["auto_crop_threshold_percentile"], None)
    auto_crop_blur_sigma    = get_config_value(h5_config, ["auto_crop_blur_sigma"], 0)
    pad                     = get_config_value(h5_config, ["pad"], 0)

    overrides_path = (config_path.parent / OVERRIDES_FILE) if config_path else Path(OVERRIDES_FILE)
    overrides = load_overrides(overrides_path)

    groups    = build_stack_groups(root_dir)
    stack_ids = sorted(groups.keys())
    if args.stack:
        missing = [s for s in args.stack if s not in groups]
        if missing:
            print(f"Stack(s) not found: {missing}")
            sys.exit(1)
        stack_ids = [s for s in stack_ids if s in args.stack]

    # Determine expected channel set from the most common set across all stacks,
    # then flag any that deviate (same logic as convert_h5_channels_to_tiff.py).
    from collections import Counter
    all_channel_sets = {sid: frozenset(ci for ci, _ in groups[sid]) for sid in stack_ids}
    expected_channels = Counter(all_channel_sets.values()).most_common(1)[0][0]
    bad_stacks = set()
    for sid in stack_ids:
        ch_set = all_channel_sets[sid]
        missing_ch = sorted(expected_channels - ch_set)
        extra_ch   = sorted(ch_set - expected_channels)
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

    print(f"Found {len(stack_ids)} stack(s) ({len(bad_stacks)} skipped).  Overrides already saved: {sorted(overrides)}\n")
    print("Controls: drag left-click to draw crop box | A = add box | Enter = accept | S = skip | Q = quit\n")

    for i, stack_id in enumerate(stack_ids, 1):
        if stack_id in bad_stacks:
            print(f"[{i}/{len(stack_ids)}] {stack_id} — SKIPPED (channel mismatch)")
            continue

        items_sorted   = sorted(groups[stack_id], key=lambda x: x[0])
        channel_h5_files = [find_h5_file(f) for _, f in items_sorted]

        # Compute auto-crop bounds (same logic as the main script)
        crop_files = (
            [channel_h5_files[auto_crop_channel]]
            if auto_crop_channel is not None
            else channel_h5_files
        )
        try:
            print(f"[{i}/{len(stack_ids)}] {stack_id} — computing auto-crop bounds...")
            auto_bounds = compute_autocrop_bounds_streaming(
                crop_files, dataset_path, pad=pad,
                threshold=auto_crop_threshold,
                threshold_percentile=auto_crop_threshold_pct,
                blur_sigma=auto_crop_blur_sigma,
            )
            z0, z1, y0, y1, x0, x1 = auto_bounds
            print(f"  Auto: z={z0}:{z1}  y={y0}:{y1}  x={x0}:{x1}")
        except Exception as e:
            print(f"  Auto-crop failed ({e}) — showing full field.")
            with h5py.File(channel_h5_files[0], "r") as f:
                nz, ny, nx = _effective_zyx_shape(_get_h5_dataset(f, dataset_path))
            auto_bounds = (0, nz, 0, ny, 0, nx)

        stack_entry = overrides.get(stack_id, {})
        existing = stack_entry.get("crops") or stack_entry.get("crop")
        if existing:
            print(f"  Existing override: {existing}")

        action, crops_list = inspect_stack(
            stack_id, channel_h5_files, channel_names, dataset_path,
            auto_bounds, existing,
        )

        if action == "quit":
            print("Quitting early.")
            break
        elif action == "accept":
            z0, z1 = auto_bounds[0], auto_bounds[1]
            if crops_list is not None and len(crops_list) > 1:
                # Multiple embryos — save as a list
                crop_strs = [f"{z0}:{z1}:{y0n}:{y1n}:{x0n}:{x1n}" for y0n, y1n, x0n, x1n in crops_list]
                overrides[stack_id] = {"crops": crop_strs}
                print(f"  {len(crop_strs)} crops saved: {crop_strs}")
            elif crops_list is not None and len(crops_list) == 1:
                y0n, y1n, x0n, x1n = crops_list[0]
                crop_str = f"{z0}:{z1}:{y0n}:{y1n}:{x0n}:{x1n}"
                overrides[stack_id] = {"crop": crop_str}
                print(f"  Override saved: {crop_str}")
            else:
                # Nothing drawn — lock in the auto bounds
                crop_str = f"{z0}:{z1}:{y0}:{y1}:{x0}:{x1}"
                overrides[stack_id] = {"crop": crop_str}
                print(f"  Auto bounds locked in: {crop_str}")
            save_overrides(overrides_path, overrides)
        else:
            print(f"  Skipped.")

    print("\nDone.")


if __name__ == "__main__":
    main()
