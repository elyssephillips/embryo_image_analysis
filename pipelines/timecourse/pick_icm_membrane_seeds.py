"""
Interactive seed-picker for icm_membrane_vector_overrides.yaml -- replaces
opening each embryo's seed_pick_t0.png by hand and typing pixel coordinates.

Shows both channels' t=0 MIP (membrane left, ICM right) for one embryo folder
at a time, gridded the same way as generate_seed_pick_images.py's static
PNGs. Click on the ICM panel to place the ICM seed (required); click on the
membrane panel to place an explicit membrane seed (optional -- leave unset to
default to frame-center, but do set it whenever a neighbor embryo is visible
in the crop, since that default landing on the wrong embryo is exactly what
caused several embryos to silently track a neighbor's blob in the first
batch run -- see icm_membrane_vector_3d.py's module docstring).

Controls:
  Click membrane panel (left)  -- set/move the membrane seed marker (cyan)
  Click ICM panel (right)      -- set/move the ICM seed marker (yellow star)
  Enter   -- accept: save this embryo (requires an ICM seed) and advance
  S       -- skip this embryo (membrane-only / no real ICM signal) and advance
  C       -- clear both markers for the current embryo, try again
  B       -- go back to the previous embryo (its existing entry is loaded so
             you can review/redo it)
  Q       -- save progress and quit (each accept/skip is already written to
             disk immediately, so quitting never loses picked embryos)

Only visits embryos not yet resolved in the overrides file (skip=true, or a
non-null icm_seed_yx_t0) -- set FORCE_REVISIT_ALL=True to cycle through every
embryo regardless, e.g. to re-check ones you already did.

Edit the CONFIG section below, then run (VS Code Run button -- no CLI args).
Needs a real display (X11/VNC) since it opens a matplotlib window.
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
FORCE_REVISIT_ALL = False
GRID_SPACING_PX = 100
# ==========================================================================


def is_resolved(entry):
    if not entry:
        return False
    return bool(entry.get("skip")) or entry.get("icm_seed_yx_t0") is not None


class SeedPicker:
    def __init__(self, names, output_dir, overrides_path, overrides):
        self.names = names
        self.output_dir = output_dir
        self.overrides_path = overrides_path
        self.overrides = overrides
        self.idx = 0

        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 7.5), dpi=120)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.mem_point = None   # (y, x) or None
        self.icm_point = None

        self.load_embryo(self.names[self.idx])
        plt.show()

    # -- data / drawing -----------------------------------------------
    def load_embryo(self, name):
        stack_path = self.output_dir / name
        tiff_files = sorted(stack_path.glob("t*.tif"))
        if not tiff_files:
            print(f"  {name}: no t*.tif files, skipping")
            self.advance()
            return

        arr = tifffile.imread(str(tiff_files[0]))  # (C, Z, Y, X)
        self.membrane_mip = arr[CHANNEL_MEMBRANE].astype(np.float32).max(axis=0)
        self.icm_mip = arr[CHANNEL_ICM].astype(np.float32).max(axis=0)
        self.current_name = name

        existing = self.overrides.get(name) or {}
        self.mem_point = tuple(existing["membrane_seed_yx_t0"]) if existing.get("membrane_seed_yx_t0") else None
        icm = existing.get("icm_seed_yx_t0")
        self.icm_point = tuple(icm) if icm else None

        self.draw()

    def draw(self):
        for ax, img, title in [(self.axes[0], self.membrane_mip, "membrane (click = membrane seed)"),
                                (self.axes[1], self.icm_mip, "ICM/oct4 (click = ICM seed)")]:
            ax.clear()
            p_lo, p_hi = np.percentile(img, (1, 99.7))
            ax.imshow(np.clip((img - p_lo) / (p_hi - p_lo + 1e-9), 0, 1), cmap="gray")
            ax.set_title(f"{title}  shape={img.shape}")
            ax.set_xticks(np.arange(0, img.shape[1], GRID_SPACING_PX))
            ax.set_yticks(np.arange(0, img.shape[0], GRID_SPACING_PX))
            ax.grid(color="red", alpha=0.3, linewidth=0.5)
            ax.tick_params(labelsize=7)

        if self.mem_point is not None:
            self.axes[0].scatter(self.mem_point[1], self.mem_point[0], c="cyan", marker="o",
                                  s=100, edgecolor="black", zorder=5)
        if self.icm_point is not None:
            self.axes[1].scatter(self.icm_point[1], self.icm_point[0], c="yellow", marker="*",
                                  s=220, edgecolor="black", zorder=5)

        n_done = sum(1 for n in self.names if is_resolved(self.overrides.get(n)))
        self.fig.suptitle(
            f"[{self.idx + 1}/{len(self.names)}]  {self.current_name}   "
            f"({n_done} resolved so far)\n"
            "Enter=save&next   S=skip (no ICM signal)   C=clear   B=back   Q=quit",
            fontsize=11,
        )
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    # -- events ----------------------------------------------------------
    def on_click(self, event):
        if event.inaxes is self.axes[0]:
            self.mem_point = (round(float(event.ydata), 1), round(float(event.xdata), 1))
        elif event.inaxes is self.axes[1]:
            self.icm_point = (round(float(event.ydata), 1), round(float(event.xdata), 1))
        else:
            return
        self.draw()

    def on_key(self, event):
        if event.key == "enter":
            if self.icm_point is None:
                print(f"  {self.current_name}: no ICM seed placed yet -- click the ICM panel first, "
                      "or press S to skip this embryo.")
                return
            self.overrides[self.current_name] = {
                "skip": False,
                "icm_seed_yx_t0": [self.icm_point[0], self.icm_point[1]],
                "membrane_seed_yx_t0": [self.mem_point[0], self.mem_point[1]] if self.mem_point else None,
            }
            self.save_overrides()
            print(f"  {self.current_name}: saved icm={self.icm_point} membrane={self.mem_point}")
            self.advance()
        elif event.key == "s":
            self.overrides[self.current_name] = {"skip": True, "icm_seed_yx_t0": None}
            self.save_overrides()
            print(f"  {self.current_name}: marked skip=true")
            self.advance()
        elif event.key == "c":
            self.mem_point = None
            self.icm_point = None
            self.draw()
        elif event.key == "b":
            self.idx = max(self.idx - 2, -1)
            self.advance()
        elif event.key == "q":
            print(f"\nQuitting. Progress saved to {self.overrides_path}.")
            plt.close(self.fig)

    def advance(self):
        self.idx += 1
        if self.idx >= len(self.names):
            print(f"\nAll embryos visited. Saved to {self.overrides_path}.")
            plt.close(self.fig)
            return
        self.load_embryo(self.names[self.idx])

    def save_overrides(self):
        self.overrides_path.write_text(yaml.dump(self.overrides, sort_keys=True, default_flow_style=None))


def main():
    config = load_yaml_config(CONFIG_PATH) if CONFIG_PATH.exists() else {}
    live_cfg = config.get("live_timecourse") or {}
    output_dir = Path(get_config_value(live_cfg, ["output_dir"]) or ".")

    overrides = {}
    if OVERRIDES_PATH.exists():
        overrides = yaml.safe_load(OVERRIDES_PATH.read_text()) or {}

    embryo_folders = sorted(p.name for p in output_dir.iterdir() if p.is_dir())
    for name in embryo_folders:
        if name not in overrides:
            overrides[name] = {"skip": None, "icm_seed_yx_t0": None}

    names = embryo_folders if FORCE_REVISIT_ALL else [
        n for n in embryo_folders if not is_resolved(overrides.get(n))
    ]
    if not names:
        print("Nothing to do -- every embryo is already resolved (set FORCE_REVISIT_ALL=True to redo all).")
        return

    print(f"{len(names)} embryo(s) to review out of {len(embryo_folders)} total.")
    SeedPicker(names, output_dir, OVERRIDES_PATH, overrides)


if __name__ == "__main__":
    main()
