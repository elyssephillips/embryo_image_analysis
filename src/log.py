"""Lightweight run logger for embryo image analysis pipelines.

Appends timestamped entries to logs/<pipeline>.md, organized by dataset.
Each log file has a dataset index table at the top for quick scanning.
H5 conversions are tracked separately in logs/conversions.md.

Usage:
    from src.log import log_run, log_conversion
    log_run("tracking", "dataset001_implantation", "03_link_tracks.py",
            output_path=str(OUT_DIR), detail="detailed")
    log_conversion("20260519_mtmg_fgf_e45", raw_path=str(root_dir),
                   output_path=str(output_dir), n_stacks=7,
                   acquisition_date="2026-05-19", condition="mT/mG FGF E4.5")
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR  = REPO_ROOT / "logs"

PIPELINE_FILES = {
    "preprocessing": "preprocessing.md",
    "tracking":      "tracking.md",
    "IF":            "IF.md",
    "nnunet":        "nnunet.md",
}

_IDX_END  = "<!-- index-end -->"
_DS_START = "<!-- ds:{} -->"
_DS_END   = "<!-- ds-end:{} -->"
_IDX_ROW  = "<!-- row:{} -->"

_CONV_END = "<!-- conv-end -->"
_CONV_ROW = "<!-- conv:{} -->"


def _prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"  {label}{suffix}: ").strip()
        return val if val else default
    except EOFError:
        return default


def _log_path(pipeline: str) -> Path:
    fname = PIPELINE_FILES.get(pipeline)
    if not fname:
        raise ValueError(f"Unknown pipeline '{pipeline}'. Options: {list(PIPELINE_FILES)}")
    return LOGS_DIR / fname


def _init_log(path: Path, pipeline: str) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    title = {"preprocessing": "Preprocessing", "tracking": "Tracking",
             "IF": "IF", "nnunet": "nnUNet"}.get(pipeline, pipeline)
    path.write_text(
        f"# {title} Log\n\n"
        "## Dataset Index\n\n"
        "| Dataset | Description | N / Conditions | Data path | Last updated | Status |\n"
        "|---|---|---|---|---|---|\n"
        f"{_IDX_END}\n\n"
        "---\n"
    )


def _is_new_dataset(text: str, dataset_id: str) -> bool:
    return _DS_START.format(dataset_id) not in text


def _add_dataset_section(text: str, dataset_id: str,
                          description: str, n_conditions: str, data_path: str) -> str:
    section = (
        f"\n{_DS_START.format(dataset_id)}\n"
        f"## {dataset_id}\n\n"
        f"**Description:** {description}  \n"
        f"**N / Conditions:** {n_conditions}  \n"
        f"**Data path:** {data_path}\n\n"
        f"{_DS_END.format(dataset_id)}\n"
    )
    return text + section


def _update_index_row(text: str, dataset_id: str, description: str,
                       n_conditions: str, data_path: str, status: str) -> str:
    today = datetime.date.today().isoformat()
    marker = _IDX_ROW.format(dataset_id)
    short_path = data_path if len(data_path) < 50 else "..." + data_path[-47:]
    new_row = (f"| {dataset_id} | {description} | {n_conditions} | "
               f"`{short_path}` | {today} | {status} | {marker}")
    if marker in text:
        return "".join(
            new_row + "\n" if marker in line else line
            for line in text.splitlines(keepends=True)
        )
    return text.replace(_IDX_END, new_row + "\n" + _IDX_END)


def _insert_entry(text: str, dataset_id: str, entry: str) -> str:
    ds_end = _DS_END.format(dataset_id)
    if ds_end not in text:
        return text + "\n" + entry
    return text.replace(ds_end, entry + "\n" + ds_end)


def log_run(
    pipeline: str,
    dataset_id: str,
    script: str,
    output_path: str = "",
    detail: str = "light",
    data_path: str = "",
) -> None:
    """Append a run entry to logs/<pipeline>.md.

    First-time datasets always trigger the detailed prompt regardless of *detail*,
    so the dataset index gets populated from the start.
    Ctrl+C at any prompt skips logging gracefully.

    Args:
        pipeline:    'preprocessing' | 'tracking' | 'IF' | 'nnunet'
        dataset_id:  Short identifier, e.g. '20260519_mtmg_fgf_e45'
        script:      Script filename, e.g. 'convert_h5_timecourse_to_tiff.py'
        output_path: Where outputs were written (pre-filled in prompt).
        detail:      'light' (just notes) or 'detailed' (full structured prompt).
        data_path:   Root data directory for this dataset (shown in index table).
    """
    if not sys.stdin.isatty():
        return

    path = _log_path(pipeline)
    if not path.exists():
        _init_log(path, pipeline)

    text = path.read_text()
    now  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    is_new           = _is_new_dataset(text, dataset_id)
    effective_detail = "detailed" if is_new else detail

    print(f"\n[log] Logging to {path.name}  (Ctrl+C to skip)")

    try:
        if effective_detail == "detailed":
            if is_new:
                print(f"[log] First entry for '{dataset_id}' — a few quick questions:")
            description  = _prompt("Description (e.g. 'mT/mG FGF E4.5 live imaging')")
            n_conditions = _prompt("N / conditions (e.g. '7 stacks: 4 ctrl+FGF, 3 FGF+ctrl')")
            if not data_path:
                data_path = _prompt("Data root path", default=data_path)
            what_done   = _prompt("What was done")
            key_info    = _prompt("Key params / findings (Enter to skip)")
            output_path = _prompt("Output location", default=output_path)
            what_next   = _prompt("What's next")
            status      = _prompt("Status (e.g. 'converting', 'tracking', 'analysis done')",
                                   default="in progress")

            parts = [f"### {now} | {script}",
                     f"**Output:** {output_path}",
                     f"**Done:** {what_done}"]
            if key_info:
                parts.append(f"**Params/findings:** {key_info}")
            parts += [f"**Next:** {what_next}", ""]

        else:  # light
            notes  = _prompt("Notes for this run? (Enter to skip)")
            status = _prompt("Status", default="in progress")
            description = n_conditions = ""

            parts = [f"### {now} | {script}"]
            if output_path:
                parts.append(f"**Output:** {output_path}")
            if notes:
                parts.append(f"**Notes:** {notes}")
            parts.append("")

    except KeyboardInterrupt:
        print("\n[log] Skipped.")
        return

    entry = "\n".join(parts)

    if is_new:
        text = _add_dataset_section(text, dataset_id, description, n_conditions, data_path)
    text = _update_index_row(text, dataset_id, description, n_conditions, data_path, status)
    text = _insert_entry(text, dataset_id, entry)
    path.write_text(text)
    print(f"[log] Saved → {path}")


def _init_conversions(path: Path) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    path.write_text(
        "# Dataset Conversion Registry\n\n"
        "Tracks raw H5 acquisitions → TIFF output. "
        "Updated automatically by `convert_h5_timecourse_to_tiff.py`.\n\n"
        "| Dataset | Acq. date | Condition | N stacks | Raw path | TIFF path | Converted | Notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"{_CONV_END}\n"
    )


def log_conversion(
    dataset_id: str,
    raw_path: str,
    output_path: str,
    n_stacks: int,
    acquisition_date: str = "",
    condition: str = "",
) -> None:
    """Append or update a row in logs/conversions.md.

    Prompts for a one-liner note (Enter to skip). All other fields are
    auto-populated from the script context. Re-running updates the existing row.
    """
    path = LOGS_DIR / "conversions.md"
    if not path.exists():
        _init_conversions(path)

    text  = path.read_text()
    today = datetime.date.today().isoformat()

    notes = ""
    if sys.stdin.isatty():
        print(f"\n[log] Conversion registry  (Ctrl+C to skip)")
        try:
            notes = _prompt("Notes (Enter to skip)")
        except KeyboardInterrupt:
            print("\n[log] Skipped.")
            return

    marker    = _CONV_ROW.format(dataset_id)
    short_raw = raw_path if len(raw_path) < 45 else "..." + raw_path[-42:]
    short_out = output_path if len(output_path) < 45 else "..." + output_path[-42:]
    new_row   = (f"| {dataset_id} | {acquisition_date} | {condition} | {n_stacks} | "
                 f"`{short_raw}` | `{short_out}` | {today} | {notes} | {marker}")

    if marker in text:
        text = "".join(
            new_row + "\n" if marker in line else line
            for line in text.splitlines(keepends=True)
        )
    else:
        text = text.replace(_CONV_END, new_row + "\n" + _CONV_END)

    path.write_text(text)
    print(f"[log] Conversion logged → {path}")
