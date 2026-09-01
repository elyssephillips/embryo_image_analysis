"""
Runs icm_membrane_vector_3d.py's process_embryo() across every embryo folder
listed in configs/other live images/icm_membrane_vector_overrides.yaml.

Run generate_seed_pick_images.py first (creates that overrides file with a
placeholder entry per embryo), fill in each entry by eye, then run this.

Each entry in the overrides file looks like:
  stack_10_embryo1:
    skip: false
    icm_seed_yx_t0: [800.0, 650.0]
    membrane_seed_yx_t0: null   # optional; null = default to frame center at t=0

Entries are skipped (with a reason printed) if: skip is true, icm_seed_yx_t0
is still null/unset, or an error is raised while processing (so one bad
embryo doesn't stop the batch) -- these get collected into a summary at the
end rather than silently disappearing.

Edit the CONFIG section below, then run (VS Code Run button -- no CLI args).
"""
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import yaml

from src.conversion import get_config_value, load_yaml_config
from icm_membrane_vector_3d import process_embryo, QC_N_TIMEPOINTS

# ============================== EDIT THESE ==============================
CONFIG_PATH = PROJECT_ROOT / "configs" / "other live images" / "260721_e45c_fgf_oct4_snap_2.yaml"
OVERRIDES_PATH = PROJECT_ROOT / "configs" / "other live images" / "icm_membrane_vector_overrides.yaml"
QC_N_TIMEPOINTS_BATCH = QC_N_TIMEPOINTS  # override here if you want fewer/more QC panels per embryo
# ==========================================================================


def main():
    config = load_yaml_config(CONFIG_PATH) if CONFIG_PATH.exists() else {}
    live_cfg = config.get("live_timecourse") or {}
    output_dir = Path(get_config_value(live_cfg, ["output_dir"]) or ".")
    voxel_zyx = get_config_value(config, ["microscopy", "voxel_size_zyx_um"]) or [1.0, 1.0, 1.0]
    voxel_zyx = np.array(voxel_zyx, dtype=np.float64)
    analysis_dir = Path(get_config_value(config, ["paths", "output_dir"]) or (PROJECT_ROOT / "analysis"))

    if not OVERRIDES_PATH.exists():
        raise FileNotFoundError(
            f"{OVERRIDES_PATH} not found -- run generate_seed_pick_images.py first."
        )
    overrides = yaml.safe_load(OVERRIDES_PATH.read_text()) or {}

    processed, skipped, failed = [], [], []
    for name in sorted(overrides):
        entry = overrides[name] or {}
        if entry.get("skip"):
            skipped.append((name, "marked skip=true"))
            continue
        icm_seed = entry.get("icm_seed_yx_t0")
        if icm_seed is None:
            skipped.append((name, "icm_seed_yx_t0 not set yet"))
            continue

        mem_seed = entry.get("membrane_seed_yx_t0")
        print(f"\n=== {name} ===")
        try:
            process_embryo(
                name, tuple(icm_seed), output_dir, analysis_dir, voxel_zyx,
                membrane_seed_yx_t0=tuple(mem_seed) if mem_seed is not None else None,
                qc_n_timepoints=QC_N_TIMEPOINTS_BATCH,
            )
            processed.append(name)
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed.append((name, str(e)))

    print(f"\n\n=== Batch summary ===")
    print(f"Processed: {len(processed)}")
    for name in processed:
        print(f"  ok      {name}")
    print(f"Skipped: {len(skipped)}")
    for name, reason in skipped:
        print(f"  skip    {name}  ({reason})")
    print(f"Failed: {len(failed)}")
    for name, err in failed:
        print(f"  FAILED  {name}  ({err})")


if __name__ == "__main__":
    main()
