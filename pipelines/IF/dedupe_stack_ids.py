"""One-off fix for the 20260730_c_meki_gata3_nmmiia_cdx2 dataset: several wells
reuse the same stack id (e.g. 18 different wells are all "stack_0-ctrl"), which
breaks crop_overrides.yaml lookups since overrides are keyed by stack id alone.

Renames the channel folders inside each well's raw/ dir so every stack id is
unique dataset-wide: wells whose stack id is already unique are left alone,
duplicates get a fresh number (continuing after the highest existing stack
number) assigned in chronological order (well folder name = acquisition
timestamp), while keeping any descriptive suffix (e.g. "-ctrl") intact.

To run: edit the settings block below if needed, then click VS Code's
"Run Python File" button (or Ctrl+F5) — no terminal command needed.
Runs with DRY_RUN = True first to show the plan before touching anything.
"""
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.conversion import detect_h5_layout

# ============================== EDIT THESE ==============================
ROOT_DIR = Path("/mnt/md1/elysse/20260730_c_meki_gata3_nmmiia_cdx2")
DRY_RUN = False   # True = only print the rename plan. Flip to False to execute.
# ==========================================================================

STACK_ID_RE = re.compile(r"^stack_(\d+)(.*)$")


def main():
    mode, info = detect_h5_layout(ROOT_DIR)
    if mode != "wells":
        raise ValueError(f"Expected a multi-well layout under {ROOT_DIR}, got {mode!r}.")

    wells = info["wells"]
    bad = {w: sids for w, sids in wells.items() if len(sids) != 1}
    if bad:
        raise ValueError(f"Expected exactly one stack per well; found others: {bad}")

    well_stack = {w: sids[0] for w, sids in wells.items()}
    counts = Counter(well_stack.values())
    dup_ids = {sid for sid, c in counts.items() if c > 1}

    max_num = -1
    for sid in counts:
        m = STACK_ID_RE.match(sid)
        if m:
            max_num = max(max_num, int(m.group(1)))
    next_num = max_num + 1

    plan = []  # (well_name, old_stack_id, new_stack_id)
    for well_name in sorted(well_stack):
        old_id = well_stack[well_name]
        if old_id not in dup_ids:
            continue
        suffix = (STACK_ID_RE.match(old_id) or re.match(r"^()(.*)$", old_id)).group(2)
        new_id = f"stack_{next_num}{suffix}"
        next_num += 1
        plan.append((well_name, old_id, new_id))

    if not plan:
        print("No duplicate stack ids found — nothing to do.")
        return

    print(f"{len(plan)} well(s) will be renamed (new ids stack_{max_num + 1}..stack_{next_num - 1}):\n")
    for well_name, old_id, new_id in plan:
        print(f"  {well_name}: {old_id!r} -> {new_id!r}")

    if DRY_RUN:
        print("\nDRY_RUN is True — nothing renamed. Set DRY_RUN = False at the top of this file to execute.")
        return

    print()
    for well_name, old_id, new_id in plan:
        raw_dir = ROOT_DIR / well_name / "raw"
        folders = [f for f in raw_dir.iterdir() if f.is_dir() and f.name.startswith(f"{old_id}_channel_")]
        for folder in sorted(folders):
            new_name = new_id + folder.name[len(old_id):]
            dest = folder.parent / new_name
            if dest.exists():
                raise FileExistsError(f"Rename target already exists: {dest}")
            folder.rename(dest)
            print(f"  {folder} -> {dest}")

    print(f"\nDone. Renamed {len(plan)} well(s).")


if __name__ == "__main__":
    main()
