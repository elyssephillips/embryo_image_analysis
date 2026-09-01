"""Combine 04_plot_intensities.py outputs across multiple IF datasets into one
side-by-side comparison plot + master CSVs.

Point DATASET_IDS at whichever datasets you want compared together -- these
are the same ids used in the `datasets:` field of configs/IF/config*.yaml and
in logs/IF.md's section headers (e.g. "20260114_fgf_cdx2_ppmlc_gata3").

Each dataset must already have had 03_extract_intensities.py (for
nuclear_intensities_raw.csv) and 04_plot_intensities.py (for
vertical_patterning_summary.csv, incl. pattern_score) run. Datasets are looked
up by scanning every configs/IF/config*.yaml for a matching `datasets:` field
and reading its output_dir -- so this doesn't care which physical drive/path a
dataset currently lives under, same as everything else in this repo.

Set COMPARISON_NAME to label this particular combination (e.g. "fgf_only",
"fgf_vs_meki") -- it names the output folder so switching which experiment
type/subset you're comparing doesn't overwrite a previous comparison's output.
"""
import sys
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
from scipy.stats import ttest_ind

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import remove_intensity_outliers

CONFIG_DIR = PROJECT_ROOT / "configs" / "IF"

# ============================== EDIT THESE ==============================
DATASET_IDS = [
    "20260218_fgf_cdx2_ppmlc_gata3",
    "20260129_fgf_cdx2_ppmlc_gata3",
    "20260304_fgf_cdx2_ppmlc_gata3",
]  # dataset ids to combine -- see the Dataset Index table in logs/IF.md for
   # the full list of what's available and what stage each one is at

COMPARISON_NAME = "fgf_te_patterning"  # names the output folder -- change this
                                        # whenever you switch which datasets
                                        # you're comparing, so runs don't clobber

OUTPUT_DIR = None  # None = "/mnt/md1/elysse/combined_analysis/<COMPARISON_NAME>"

STANDARDIZE_BY_SIZE = False  # True (default): current behavior -- each embryo's
                             # Y-position is rescaled to [0, 1] (top nucleus=0,
                             # bottom nucleus=1), which erases absolute size
                             # differences between embryos. Set False to instead
                             # keep the top nucleus at 0 but leave the natural
                             # spread in real microns -- lets size-driven
                             # patterning differences show up instead of being
                             # normalized away. Always writes to a separate
                             # "standardized"/"unstandardized" subfolder so
                             # toggling this never overwrites the other mode's output.

Y_BIN_WIDTH_UM = 10  # bin width for the vertical-profile lineplot when
                      # STANDARDIZE_BY_SIZE is False (ignored otherwise, which
                      # always bins at 0.1 of the normalized [0,1] range)
# ==========================================================================


def assign_group(name):
    """Checks if the filename starts with 'c' (handles c_, coo, etc.)"""
    n = str(name).lower().strip()
    return 'Control' if n.startswith('c') else 'Treated'


def resolve_dataset_output_dirs(dataset_ids):
    """Maps each dataset id to its output_dir + xy voxel size (microns/px) by
    scanning configs/IF/config*.yaml for a matching `datasets:` field
    (config.yaml included). Returns {dataset_id: {"output_dir": Path, "xy_um":
    float | None}} for whatever was found; prints a warning listing every
    dataset id actually available if any requested id is missing. xy_um is
    None if a config doesn't have microscopy.voxel_size_zyx set.
    """
    by_id = {}
    for config_path in sorted(CONFIG_DIR.glob("config*.yaml")):
        try:
            config = yaml.safe_load(config_path.read_text()) or {}
        except yaml.YAMLError as e:
            print(f"WARNING: couldn't parse {config_path.name}, skipping it: {e}")
            continue
        dataset_id = config.get("datasets")
        output_dir = config.get("output_dir")
        if dataset_id and output_dir and dataset_id not in by_id:
            voxel_size = (config.get("microscopy") or {}).get("voxel_size_zyx")
            xy_um = voxel_size[1] if voxel_size and len(voxel_size) > 1 else None
            by_id[dataset_id] = {"output_dir": Path(output_dir), "xy_um": xy_um, "_source": config_path.name}

    resolved = {}
    missing = []
    for dataset_id in dataset_ids:
        if dataset_id in by_id:
            resolved[dataset_id] = by_id[dataset_id]
        else:
            missing.append(dataset_id)

    if missing:
        available = "\n".join(f"  - {k}  (from {v['_source']})" for k, v in sorted(by_id.items()))
        print(f"WARNING: no config found for: {', '.join(missing)}")
        print(f"Dataset ids available across configs/IF/config*.yaml:\n{available}")

    return resolved


def aggregate_project_data():
    if not DATASET_IDS:
        print("Set DATASET_IDS at the top of this file to the datasets you want combined.")
        return

    dataset_dirs = resolve_dataset_output_dirs(DATASET_IDS)
    if not dataset_dirs:
        print("None of DATASET_IDS resolved to a config. Nothing to combine.")
        return

    base_output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else Path(f"/mnt/md1/elysse/combined_analysis/{COMPARISON_NAME}")
    output_dir = base_output_dir / ("standardized" if STANDARDIZE_BY_SIZE else "unstandardized")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_files, raw_files = {}, {}
    for dataset_id, info in dataset_dirs.items():
        out_dir = info["output_dir"]
        s = out_dir / "vertical_patterning_summary.csv"
        r = out_dir / "nuclear_intensities_raw.csv"
        missing = [name for name, p in [("summary", s), ("raw", r)] if not p.exists()]
        if missing:
            print(f"WARNING: skipping '{dataset_id}' -- missing {' & '.join(missing)} CSV in {out_dir} "
                  "(run 03_extract_intensities.py / 04_plot_intensities.py for it first).")
            continue
        summary_files[dataset_id] = s
        raw_files[dataset_id] = r

    if not summary_files:
        print("No dataset had both required CSVs. Nothing to combine.")
        return

    print(f"Combining {len(summary_files)} dataset(s) into '{COMPARISON_NAME}': {', '.join(summary_files)}")

    # 1. COMBINE SUMMARY DATA
    print("Processing Summary Data...")
    mega_summary = pd.concat(
        [pd.read_csv(f).assign(batch_id=dataset_id) for dataset_id, f in summary_files.items()],
        ignore_index=True
    )
    mega_summary['group'] = mega_summary['image_id'].apply(assign_group)

    # 2. COMBINE RAW DATA
    all_nuclei = []
    print(f"Sorting {len(raw_files)} batches by image prefix...")

    for dataset_id, f in raw_files.items():
        df = pd.read_csv(f)
        df.columns = df.columns.str.lower()

        # Mapping handles column name variations across batches
        rename_logic = {
            'c0_mean': 'dapi_mean', 'dapi_mean': 'dapi_mean',
            'c1_mean': 'gata3_mean', 'gfp_mean': 'gata3_mean',
            'c3_mean': 'cdx2_mean', 'cy5_mean': 'cdx2_mean'
        }
        df.rename(columns=rename_logic, inplace=True, errors='ignore')
        df['group'] = df['image_id'].apply(assign_group)

        c_n = len(df[df['group'] == 'Control'])
        t_n = len(df[df['group'] == 'Treated'])
        print(f"   {dataset_id} | Found: {c_n} Ctrl / {t_n} Trtd")

        if 'gata3_mean' in df.columns and 'dapi_mean' in df.columns:
            df['GATA3_norm'] = df['gata3_mean'] / (df['dapi_mean'] + 1e-6)
            df['CDX2_norm'] = df['cdx2_mean'] / (df['dapi_mean'] + 1e-6)

            if STANDARDIZE_BY_SIZE:
                def norm_y(g):
                    return (g['center_y'] - g['center_y'].min()) / (g['center_y'].max() - g['center_y'].min() + 1e-6)
                df['y_rel'] = df.groupby('image_id', group_keys=False).apply(norm_y)
            else:
                xy_um = dataset_dirs[dataset_id]["xy_um"]
                if xy_um is None:
                    print(f"   WARNING: '{dataset_id}' has no microscopy.voxel_size_zyx in its config -- "
                          "using raw pixels instead of microns for this dataset's Y-position.")
                    xy_um = 1.0

                def top_zeroed_y(g):
                    return (g['center_y'] - g['center_y'].min()) * xy_um
                df['y_rel'] = df.groupby('image_id', group_keys=False).apply(top_zeroed_y)

            df['batch_id'] = dataset_id
            all_nuclei.append(df)
        else:
            print(f"   WARNING: '{dataset_id}' is missing gata3_mean/dapi_mean after renaming -- "
                  "excluded from the nuclei-level (vertical profile) plots.")

    if not all_nuclei:
        print("No dataset had usable GATA3/DAPI columns. Nothing to combine.")
        return

    mega_nuclei = pd.concat(all_nuclei, ignore_index=True)

    # 3. CLEANING & BINNING
    print("Cleaning up outliers...")
    mega_nuclei = remove_intensity_outliers(mega_nuclei, 'GATA3_norm')
    mega_nuclei = remove_intensity_outliers(mega_nuclei, 'CDX2_norm')
    if STANDARDIZE_BY_SIZE:
        mega_nuclei['y_bin'] = mega_nuclei['y_rel'].round(1)
    else:
        mega_nuclei['y_bin'] = (mega_nuclei['y_rel'] / Y_BIN_WIDTH_UM).round() * Y_BIN_WIDTH_UM

    # 4. METADATA & STATS
    n_ctrl_emb = mega_summary[mega_summary['group'] == 'Control']['image_id'].nunique()
    n_trtd_emb = mega_summary[mega_summary['group'] == 'Treated']['image_id'].nunique()
    n_ctrl_nuc = len(mega_nuclei[mega_nuclei['group'] == 'Control'])
    n_trtd_nuc = len(mega_nuclei[mega_nuclei['group'] == 'Treated'])

    ctrl_label = f"Control ({n_ctrl_emb} emb / {n_ctrl_nuc} nuc)"
    trtd_label = f"Treated ({n_trtd_emb} emb / {n_trtd_nuc} nuc)"
    grad_palette = {ctrl_label: '#1f77b4', trtd_label: '#ff7f0e'}
    mega_nuclei['legend_group'] = mega_nuclei['group'].map({'Control': ctrl_label, 'Treated': trtd_label})

    ctrl_r = mega_summary[mega_summary['group'] == 'Control']['pearson_r']
    trtd_r = mega_summary[mega_summary['group'] == 'Treated']['pearson_r']
    _, p_val_r = ttest_ind(ctrl_r, trtd_r, nan_policy='omit')

    ctrl_p = mega_summary[mega_summary['group'] == 'Control']['pattern_score']
    trtd_p = mega_summary[mega_summary['group'] == 'Treated']['pattern_score']
    _, p_val_p = ttest_ind(ctrl_p, trtd_p, nan_policy='omit')

    # 5. EXPORT
    mega_nuclei.to_csv(output_dir / "master_nuclei_data_cleaned.csv", index=False)

    stats_df = pd.DataFrame({
        'Metric': ['Pearson r (Mean)', 'Patterning Score (Mean)', 'Embryos (N)', 'Nuclei (N)'],
        'Control': [ctrl_r.mean(), ctrl_p.mean(), n_ctrl_emb, n_ctrl_nuc],
        'Treated': [trtd_r.mean(), trtd_p.mean(), n_trtd_emb, n_trtd_nuc],
        'P-Value': [p_val_r, p_val_p, None, None]
    })
    stats_df.to_csv(output_dir / "stats_summary.csv", index=False)

    # 6. MASTER PLOT (1x4 subplots)
    fig, axes = plt.subplots(1, 4, figsize=(28, 8), gridspec_kw={'width_ratios': [1, 1, 2, 2]})
    unique_batches = mega_summary['batch_id'].unique()
    batch_map = dict(zip(unique_batches, sns.color_palette("husl", len(unique_batches))))

    panel_titles = {
        'pearson_r': 'Pearson R',
        'pattern_score': 'GATA3 Polarization Index',  # weighted-intensity center vs. geometric center, size-normalized
    }
    for i, col in enumerate(['pearson_r', 'pattern_score']):
        sns.boxplot(data=mega_summary, x='group', y=col, hue='group',
                    palette={'Control': '#1f77b4', 'Treated': '#ff7f0e'},
                    ax=axes[i], showfliers=False, legend=False, width=0.6)
        sns.stripplot(data=mega_summary, x='group', y=col, hue='batch_id',
                      palette=batch_map, ax=axes[i], dodge=True, alpha=0.6, size=5,
                      edgecolor='gray', linewidth=0.5)
        c_vals = mega_summary[mega_summary['group'] == 'Control'][col]
        t_vals = mega_summary[mega_summary['group'] == 'Treated'][col]
        _, p = ttest_ind(c_vals, t_vals, nan_policy='omit')
        axes[i].set_title(f"{panel_titles[col]}\n(p = {p:.2e})")
        if i == 0:
            axes[0].legend(title="Dataset", bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=8)

    for i, (marker, title) in enumerate([('GATA3_norm', 'GATA3'), ('CDX2_norm', 'CDX2')]):
        ax = axes[i+2]
        sns.lineplot(data=mega_nuclei, x='y_bin', y=marker, hue='legend_group',
                     palette=grad_palette, ax=ax, lw=4.0, errorbar=('ci', 95), n_boot=1000)
        ax.set_title(f"{title} Vertical Profile")
        if STANDARDIZE_BY_SIZE:
            ax.set_xlim(0, 1.0)
            ax.set_xlabel("Relative Y (0=Top, 1=Bottom)")
        else:
            ax.set_xlim(left=0)
            ax.set_xlabel("Distance from top nucleus (µm)")
        ax.legend(title="Key", loc='best', fontsize=9)

    mode_label = "size-standardized" if STANDARDIZE_BY_SIZE else "NOT size-standardized"
    plt.suptitle(f"{COMPARISON_NAME} ({mode_label})", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "summary_allbatches_bin10.png", dpi=300, bbox_inches='tight')
    print(f"\nSUCCESS! Combined '{COMPARISON_NAME}' [{mode_label}] "
          f"({len(summary_files)} dataset(s)) saved to: {output_dir}")


if __name__ == "__main__":
    aggregate_project_data()
