"""Single entry point to convert an IF dataset's raw HDF5 into combined TIFF stacks.

Reads h5_conversion.root_dir from configs/IF/config.yaml (the same active config
every other IF pipeline script uses — no path to edit here) and auto-detects
which layout it is:

  flat    - root_dir directly contains stack_*_channel_* folders (one acquisition)
            -> runs convert_h5_channels_to_tiff.py once.
  wells   - root_dir contains one subfolder per well, each with its own raw/
            folder of stack_*_channel_* folders (fixed samples imaged well-by-well)
            -> runs convert_h5_channels_to_tiff.py once per well, all writing into
            the same raw_data_dir. If two wells share a stack id (e.g. two separate
            acquisitions both produced "stack_7-meki"), the colliding output is
            renamed with the well's timestamp appended rather than silently
            overwritten.
  timecourse - looks like live-imaging data (multiple h5 files per channel folder,
            one per timepoint). Not a fixed-IF layout; this script stops and points
            you at convert_h5_timecourse_to_tiff.py instead.

Before converting, it also checks configs/IF/crop_overrides.yaml and flags any
stack that doesn't have a manual crop override yet — those will fall back to
plain auto_crop (one bounding box for the whole stack). If a flagged stack
actually has multiple embryos in one field of view, run
pipelines/IF/inspect_crop_bounds.py first so each embryo gets split into its
own TIFF, then re-run this script.

Pipeline order: new_if_config.py -> fill in config.yaml TODOs ->
inspect_crop_bounds.py -> convert_h5_to_tiff.py (this script).

After conversion, logs to logs/conversions.md (raw path, output path, stack count)
and logs/IF.md (status update) using the project's existing src/log.py conventions.

To run: click VS Code's "Run Python File" button (or Ctrl+F5) — no terminal
command or edits needed unless you want DRY_RUN or WELLS below.
"""
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.conversion import detect_h5_layout, get_config_value, get_h5_conversion_config, load_yaml_config
from src.io import get_storage_note, get_config_notes, get_config_n_conditions, summarize_config_metadata
from src.log import log_conversion, log_run, sync_notes

CONFIG_PATH = PROJECT_ROOT / "configs" / "IF" / "config.yaml"
CROP_OVERRIDES_PATH = CONFIG_PATH.parent / "crop_overrides.yaml"

# ============================== EDIT THESE (optional) ==============================
WELLS = None           # e.g. ["2026-06-04_121950", "2026-06-04_122408"] to restrict, or None for all
DRY_RUN = False          # True = just report the detected layout and wells/stacks, no files written.
                        # Flip to False once the dry-run output looks right.
# =====================================================================================


def _convert_one(root_dir: Path, output_dir: Path) -> None:
    subprocess.run(
        [sys.executable, "scripts/convert_h5_channels_to_tiff.py",
         str(root_dir), str(output_dir), "--config", str(CONFIG_PATH)],
        check=True,
        cwd=PROJECT_ROOT,
    )


def _convert_wells(wells_root: Path, well_names: list[str], output_dir: Path) -> None:
    for i, well_name in enumerate(well_names, 1):
        well_dir = wells_root / well_name
        print(f"\n===== Well [{i}/{len(well_names)}]: {well_name} =====")
        tmp_out = output_dir / f"_tmp_{well_name}"
        if tmp_out.exists():
            shutil.rmtree(tmp_out)

        _convert_one(well_dir / "raw", tmp_out)

        for tif in sorted(tmp_out.glob("*.tif")):
            dest = output_dir / tif.name
            if dest.exists():
                dest_renamed = output_dir / f"{tif.stem}_{well_name}{tif.suffix}"
                print(
                    f"  WARNING: {tif.name} already exists in {output_dir} from a previous well "
                    f"(duplicate stack id across acquisitions). Saving this one as "
                    f"{dest_renamed.name} instead — check both files."
                )
                shutil.move(str(tif), str(dest_renamed))
            else:
                shutil.move(str(tif), str(dest))

        shutil.rmtree(tmp_out)


def main():
    config = load_yaml_config(CONFIG_PATH)
    h5_config = get_h5_conversion_config(config)

    h5_root = get_config_value(h5_config, ["root_dir"])
    if h5_root is None:
        raise ValueError(f"h5_conversion.root_dir not set in {CONFIG_PATH}")
    h5_root = Path(h5_root)

    output_dir = get_config_value(h5_config, ["output_dir"]) or get_config_value(config, ["raw_data_dir"])
    if output_dir is None:
        raise ValueError(f"raw_data_dir (or h5_conversion.output_dir) not set in {CONFIG_PATH}")
    output_dir = Path(output_dir)

    mode, info = detect_h5_layout(h5_root)
    print(f"Detected layout: {mode!r} under {h5_root}")

    if mode == "unknown":
        raise ValueError(f"No stack_*_channel_* folders found under {h5_root} (flat or nested). Check folder naming.")

    if mode == "timecourse":
        print(
            "This looks like live-imaging data (multiple h5 files per channel folder, "
            "one per timepoint), not a fixed-sample IF layout. Use "
            "pipelines/timecourse/convert_h5_timecourse_to_tiff.py instead — this script is for fixed IF only."
        )
        return

    if mode == "flat":
        print(f"  stacks: {info['stack_ids']}")
    else:
        well_names = sorted(info["wells"]) if WELLS is None else [w for w in sorted(info["wells"]) if w in WELLS]
        for w in well_names:
            print(f"  {w}: {info['wells'][w]}")
        stack_ids_seen = {}
        for w in well_names:
            for sid in info["wells"][w]:
                stack_ids_seen.setdefault(sid, []).append(w)
        collisions = {sid: ws for sid, ws in stack_ids_seen.items() if len(ws) > 1}
        if collisions:
            print("\nWARNING: these stack ids appear in more than one well and will be disambiguated on write:")
            for sid, ws in collisions.items():
                print(f"  {sid}: {ws}")

    all_stack_ids = info["stack_ids"] if mode == "flat" else sorted({sid for w in info["wells"].values() for sid in w})
    overrides = {}
    if CROP_OVERRIDES_PATH.exists():
        with open(CROP_OVERRIDES_PATH) as f:
            overrides = yaml.safe_load(f) or {}
    no_override = [sid for sid in all_stack_ids if sid not in overrides]
    if no_override:
        print(f"\n{len(no_override)}/{len(all_stack_ids)} stack(s) have no crop override yet — they'll use plain "
              f"auto_crop (one bounding box covering all signal in the stack). If any of these contain multiple "
              f"embryos, run pipelines/IF/inspect_crop_bounds.py first so each embryo gets its own cropped TIFF:")
        for sid in no_override:
            print(f"  {sid}")

    if DRY_RUN:
        print("\nDRY_RUN is True — nothing was converted. Set DRY_RUN = False at the top of this file to actually convert.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "flat":
        n_stacks = len(info["stack_ids"])
        _convert_one(h5_root, output_dir)
    else:
        well_names = sorted(info["wells"]) if WELLS is None else [w for w in sorted(info["wells"]) if w in WELLS]
        n_stacks = sum(len(info["wells"][w]) for w in well_names)
        _convert_wells(h5_root, well_names, output_dir)

    print(f"\nDone. Combined TIFFs written to {output_dir}")

    dataset_id = config.get("datasets") or output_dir.parts[-2]
    log_conversion(dataset_id, raw_path=str(h5_root), output_path=str(output_dir), n_stacks=n_stacks)
    log_run("IF", dataset_id, "convert_h5_to_tiff.py",
            output_path=str(output_dir), data_path=str(h5_root), detail="light",
            storage=get_storage_note(CONFIG_PATH), n_conditions=get_config_n_conditions(CONFIG_PATH),
            **summarize_config_metadata(config))
    sync_notes("IF", dataset_id, get_config_notes(CONFIG_PATH))


if __name__ == "__main__":
    main()
