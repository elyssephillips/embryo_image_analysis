"""
Generates a gridded, tick-labeled t=0 MIP image (both channels) for every
embryo folder in the config, to look at before filling in
configs/other live images/icm_membrane_vector_overrides.yaml.

For each embryo: look at analysis/icm_membrane_vector/<embryo>/seed_pick_t0.png
and decide either
  skip: true              (no real ICM signal -- membrane-only, per yaml notes)
  icm_seed_yx_t0: [y, x]   (approximate pixel location of the real ICM cluster)

Only regenerates images for embryos not yet resolved in the overrides file
(skip: true, or a non-null icm_seed_yx_t0) -- set FORCE_REGENERATE_ALL=True to
redo all of them (e.g. after changing the grid/contrast settings).

Edit the CONFIG section below, then run (VS Code Run button -- no CLI args).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import yaml

from src.conversion import get_config_value, load_yaml_config
from icm_membrane_vector_3d import CHANNEL_MEMBRANE, CHANNEL_ICM

# ============================== EDIT THESE ==============================
CONFIG_PATH = PROJECT_ROOT / "configs" / "other live images" / "260721_e45c_fgf_oct4_snap_2.yaml"
OVERRIDES_PATH = PROJECT_ROOT / "configs" / "other live images" / "icm_membrane_vector_overrides.yaml"
FORCE_REGENERATE_ALL = False
GRID_SPACING_PX = 100
# ==========================================================================


def is_resolved(entry):
    if not entry:
        return False
    return bool(entry.get("skip")) or entry.get("icm_seed_yx_t0") is not None


def main():
    config = load_yaml_config(CONFIG_PATH) if CONFIG_PATH.exists() else {}
    live_cfg = config.get("live_timecourse") or {}
    output_dir = Path(get_config_value(live_cfg, ["output_dir"]) or ".")
    analysis_dir = Path(get_config_value(config, ["paths", "output_dir"]) or (PROJECT_ROOT / "analysis"))

    overrides = {}
    if OVERRIDES_PATH.exists():
        overrides = yaml.safe_load(OVERRIDES_PATH.read_text()) or {}

    embryo_folders = sorted(p.name for p in output_dir.iterdir() if p.is_dir())
    print(f"Found {len(embryo_folders)} embryo folders in {output_dir}")

    # Make sure every embryo has a placeholder entry in the overrides file,
    # without clobbering ones already filled in.
    changed = False
    for name in embryo_folders:
        if name not in overrides:
            overrides[name] = {"skip": None, "icm_seed_yx_t0": None}
            changed = True
    if changed:
        OVERRIDES_PATH.write_text(yaml.dump(overrides, sort_keys=True, default_flow_style=None))
        print(f"Added placeholder entries to {OVERRIDES_PATH}")

    n_generated = 0
    n_skipped_existing = 0
    for name in embryo_folders:
        if not FORCE_REGENERATE_ALL and is_resolved(overrides.get(name)):
            n_skipped_existing += 1
            continue

        stack_path = output_dir / name
        tiff_files = sorted(stack_path.glob("t*.tif"))
        if not tiff_files:
            print(f"  {name}: no t*.tif files, skipping")
            continue

        arr = tifffile.imread(str(tiff_files[0]))  # (C, Z, Y, X)
        membrane = arr[CHANNEL_MEMBRANE].astype(np.float32)
        icm = arr[CHANNEL_ICM].astype(np.float32)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=130)
        for ax, img, title in [(axes[0], membrane.max(axis=0), "membrane (ch0)"),
                                (axes[1], icm.max(axis=0), "ICM/oct4 (ch1)")]:
            p_lo, p_hi = np.percentile(img, (1, 99.7))
            ax.imshow(np.clip((img - p_lo) / (p_hi - p_lo + 1e-9), 0, 1), cmap="gray")
            ax.set_title(f"{title}  shape={img.shape}")
            ax.set_xticks(np.arange(0, img.shape[1], GRID_SPACING_PX))
            ax.set_yticks(np.arange(0, img.shape[0], GRID_SPACING_PX))
            ax.grid(color="red", alpha=0.3, linewidth=0.5)
            ax.tick_params(labelsize=7)

        fig.suptitle(f"{name}  t=0  (axes = pixel y,x; gridlines every {GRID_SPACING_PX}px)")
        fig.tight_layout()

        out_dir = analysis_dir / "icm_membrane_vector" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "seed_pick_t0.png"
        fig.savefig(out_path, facecolor="white")
        plt.close(fig)
        print(f"  {name}: saved {out_path}")
        n_generated += 1

    print(f"\n{n_generated} image(s) generated, {n_skipped_existing} already resolved in "
          f"{OVERRIDES_PATH} (skipped).")
    print(f"Fill in 'skip' and/or 'icm_seed_yx_t0' for each unresolved embryo in {OVERRIDES_PATH}, "
          f"then run batch_icm_membrane_vector.py.")


if __name__ == "__main__":
    main()
