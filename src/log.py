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
import re
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
                          description: str, n_conditions: str, data_path: str,
                          storage: str = "", channels: str = "", voxel_size: str = "",
                          exclusions: str = "") -> str:
    fields = [("Description", description), ("N / Conditions", n_conditions), ("Data path", data_path)]
    for label, value in [("Storage", storage), ("Channels", channels),
                          ("Voxel size (zyx, µm)", voxel_size), ("Exclusions", exclusions)]:
        if value:
            fields.append((label, value))
    body = "  \n".join(f"**{label}:** {value}" for label, value in fields) + "\n\n"
    section = (
        f"\n{_DS_START.format(dataset_id)}\n"
        f"## {dataset_id}\n\n"
        f"{body}"
        f"{_DS_END.format(dataset_id)}\n"
    )
    return text + section


def _set_field_line(text: str, dataset_id: str, label: str, value: str) -> str:
    """Replace the value of an existing '**{label}:**' line within dataset_id's section.

    No-op if the dataset section, or that field line within it, doesn't exist.
    Preserves the trailing markdown line-break spaces ("  ") if the line had them.
    """
    ds_start = _DS_START.format(dataset_id)
    ds_end = _DS_END.format(dataset_id)
    start_idx = text.find(ds_start)
    end_idx = text.find(ds_end)
    if start_idx == -1 or end_idx == -1:
        return text

    section = text[start_idx:end_idx]
    pattern = re.compile(rf"(\*\*{re.escape(label)}:\*\*) *[^\n]*?( *)\n")
    new_section, n = pattern.subn(
        lambda m: f"{m.group(1)} {value}{m.group(2)}\n", section, count=1
    )
    if n == 0:
        return text
    return text[:start_idx] + new_section + text[end_idx:]


# Canonical header-field order for dataset sections. Drives where _upsert_field_line
# inserts a field that isn't present yet: right after the last of these fields
# (in this order) that IS already in the section.
_FIELD_ORDER = ["Description", "N / Conditions", "Data path", "Storage",
                "Channels", "Voxel size (zyx, µm)", "Exclusions"]


def _upsert_field_line(text: str, dataset_id: str, label: str, value: str) -> str:
    """Update the '**{label}:**' line within dataset_id's section, or insert it if absent.

    New lines are inserted right after the last _FIELD_ORDER field already present
    (falling back to right after the '## dataset_id' heading if none are). No-op
    if the dataset section itself doesn't exist.
    """
    ds_start = _DS_START.format(dataset_id)
    ds_end = _DS_END.format(dataset_id)
    start_idx = text.find(ds_start)
    end_idx = text.find(ds_end)
    if start_idx == -1 or end_idx == -1:
        return text

    section = text[start_idx:end_idx]
    if re.search(rf"\*\*{re.escape(label)}:\*\*", section):
        return _set_field_line(text, dataset_id, label, value)

    new_line = f"**{label}:** {value}\n"
    field_idx = _FIELD_ORDER.index(label) if label in _FIELD_ORDER else len(_FIELD_ORDER)
    for anchor in reversed(_FIELD_ORDER[:field_idx]):
        anchor_pattern = re.compile(rf"(\*\*{re.escape(anchor)}:\*\* *[^\n]*?) *\n")
        m = anchor_pattern.search(section)
        if m:
            line_text = m.group(1)
            section = section[:m.start()] + f"{line_text}  \n{new_line}" + section[m.end():]
            return text[:start_idx] + section + text[end_idx:]

    heading_pattern = re.compile(rf"(## {re.escape(dataset_id)}\n\n)")
    section, n = heading_pattern.subn(lambda m: m.group(1) + new_line, section, count=1)
    if n == 0:
        return text
    return text[:start_idx] + section + text[end_idx:]


def sync_dataset_fields(pipeline: str, dataset_id: str, storage: str = "", data_path: str = "",
                         n_conditions: str = "", channels: str = "", voxel_size: str = "",
                         exclusions: str = "") -> None:
    """Quietly sync mechanical config-derived fields in logs/<pipeline>.md. No prompts.

    Meant to run on every pipeline script invocation (not just ones that call
    log_run), so the log reflects the active config as soon as it's edited —
    e.g. a '#storage:'/'#n_conditions:' comment, raw_data_dir after repointing at
    a new drive, or microscopy/exclusions settings (see src.io.summarize_config_metadata,
    src.io.get_storage_note, src.io.get_config_n_conditions). Only fields passed in
    non-empty are touched; each is a 1:1 mirror of a config value, so overwriting is
    safe (unlike Description, which is curated by hand). n_conditions/data_path also
    update the matching index-table cells, leaving Description/Status untouched.
    No-op if dataset_id is empty, the log file doesn't exist yet, or the dataset
    has no section yet (that gets created by the first log_run).
    """
    if not dataset_id:
        return
    path = _log_path(pipeline)
    if not path.exists():
        return
    text = path.read_text()
    if _is_new_dataset(text, dataset_id):
        return  # no section yet -- don't fabricate one or an index row outside log_run
    new_text = text
    fields = [
        ("Storage", storage),
        ("Data path", data_path),
        ("N / Conditions", n_conditions),
        ("Channels", channels),
        ("Voxel size (zyx, µm)", voxel_size),
        ("Exclusions", exclusions),
    ]
    for label, value in fields:
        if value:
            new_text = _upsert_field_line(new_text, dataset_id, label, value)
    if n_conditions or data_path:
        new_text = _update_index_row(new_text, dataset_id, n_conditions=n_conditions, data_path=data_path)
    if new_text != text:
        path.write_text(new_text)


def sync_notes(pipeline: str, dataset_id: str, note: str) -> None:
    """Append a dated note entry to a dataset's section if `note` is new. No prompts.

    Meant to mirror a config's '#notes:' comment (see src.io.get_config_notes) into
    a running history, unlike the mechanical fields in sync_dataset_fields which
    overwrite in place. Compares against the most recently logged '**Note:**' line
    for this dataset and no-ops if it's unchanged, so re-running a script with the
    same config note doesn't spam a new entry every time. No-op if note/dataset_id
    is empty, the log file doesn't exist yet, or the dataset has no section yet.
    """
    if not note or not dataset_id:
        return
    path = _log_path(pipeline)
    if not path.exists():
        return
    text = path.read_text()
    ds_start = _DS_START.format(dataset_id)
    ds_end = _DS_END.format(dataset_id)
    start_idx = text.find(ds_start)
    end_idx = text.find(ds_end)
    if start_idx == -1 or end_idx == -1:
        return

    section = text[start_idx:end_idx]
    existing_notes = re.findall(r"\*\*Note:\*\* (.*)", section)
    if existing_notes and existing_notes[-1].strip() == note.strip():
        return  # unchanged since last sync -- don't re-log

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"### {now} | note\n**Note:** {note}\n"
    new_text = _insert_entry(text, dataset_id, entry)
    path.write_text(new_text)


def _parse_index_row(text: str, dataset_id: str) -> dict:
    """Return dataset_id's current index-row cell values, or {} if it has no row yet."""
    marker = _IDX_ROW.format(dataset_id)
    for line in text.splitlines():
        if marker in line:
            body = line.split(marker)[0].strip().strip("|")
            cells = [c.strip() for c in body.split("|")]
            if len(cells) >= 6:
                return {
                    "description": cells[1],
                    "n_conditions": cells[2],
                    "data_path": cells[3].strip("`"),
                    "status": cells[5],
                }
    return {}


def _update_index_row(text: str, dataset_id: str, description: str = "",
                       n_conditions: str = "", data_path: str = "", status: str = "") -> str:
    """Upsert dataset_id's index row. Any arg left '' keeps that cell's existing
    value (so a quiet partial sync can't blow away Description/Status it wasn't
    given), except Last-updated, which always bumps to today."""
    existing = _parse_index_row(text, dataset_id)
    description  = description  or existing.get("description", "")
    n_conditions = n_conditions or existing.get("n_conditions", "")
    data_path    = data_path    or existing.get("data_path", "")
    status       = status       or existing.get("status", "in progress")

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
    storage: str = "",
    n_conditions: str = "",
    channels: str = "",
    voxel_size: str = "",
    exclusions: str = "",
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
        n_conditions: Used as the detailed-prompt default (still overridable by
                     typing over it); in light mode it's synced as-is with no
                     prompt, same as storage/channels/voxel_size/exclusions.
        storage, channels, voxel_size, exclusions:
                     Mechanical config-derived fields (see src.io.get_storage_note
                     and src.io.summarize_config_metadata). Written into the dataset
                     section on first run and kept in sync on later runs when
                     non-empty; each left untouched if omitted.
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
            n_conditions = _prompt("N / conditions (e.g. '7 stacks: 4 ctrl+FGF, 3 FGF+ctrl')",
                                    default=n_conditions)
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
            description = ""

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
        text = _add_dataset_section(text, dataset_id, description, n_conditions, data_path,
                                     storage=storage, channels=channels, voxel_size=voxel_size,
                                     exclusions=exclusions)
    else:
        for label, value in [("Data path", data_path), ("Storage", storage),
                              ("N / Conditions", n_conditions), ("Channels", channels),
                              ("Voxel size (zyx, µm)", voxel_size), ("Exclusions", exclusions)]:
            if value:
                text = _upsert_field_line(text, dataset_id, label, value)
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
