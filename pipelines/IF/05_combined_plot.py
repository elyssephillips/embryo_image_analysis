import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.io import load_config, get_storage_note, get_config_notes, get_config_n_conditions, summarize_config_metadata
from src.log import sync_dataset_fields, sync_notes
from src.analysis import normalize_by_dapi, remove_intensity_outliers

# ============================== EDIT THESE ==============================
STANDARDIZE_BY_SIZE = False  # True (default): each embryo's Y-position is
                             # rescaled to [0, 1] (top nucleus=0, bottom
                             # nucleus=1), which erases absolute size
                             # differences between embryos. Set False to
                             # instead keep the top nucleus at 0 but leave the
                             # natural spread in real microns -- lets
                             # size-driven patterning differences show up
                             # instead of being normalized away. Always writes
                             # to a separate "standardized"/"unstandardized"
                             # subfolder so toggling this never overwrites the
                             # other mode's output.
# ==========================================================================


def plot_combined_vertical_gradient():
    config = load_config()
    _dataset_id = config.get("datasets", "")
    sync_dataset_fields("IF", _dataset_id, storage=get_storage_note(),
                         data_path=config.get("raw_data_dir", ""),
                         n_conditions=get_config_n_conditions(), **summarize_config_metadata(config))
    sync_notes("IF", _dataset_id, get_config_notes())
    output_dir = Path(config['output_dir'])

    data_path = output_dir / "nuclear_intensities_raw.csv"
    if not data_path.exists():
        print(f"Error: CSV not found at {data_path}")
        return

    df_raw = pd.read_csv(data_path)

    # QC exclusions
    exclude_dict = config.get('exclusions', {})
    exclude_ids = list(exclude_dict.keys())
    df = df_raw[~df_raw['image_id'].isin(exclude_ids)].copy()
    print(f"QC: Excluded {len(exclude_ids)} embryos from combined plot.")

    # Rename & normalize
    keys = list(config['microscopy']['channels'].keys())
    names = config['microscopy']['channel_names']
    df.rename(columns={f"{k}_mean": f"{n}_mean" for k, n in zip(keys, names)}, inplace=True)
    df = normalize_by_dapi(df, dapi_col='dapi_mean')
    df['group'] = df['image_id'].apply(lambda x: 'Control' if x.startswith('c') else 'Treated')

    # Y-position per embryo, top nucleus = 0
    if STANDARDIZE_BY_SIZE:
        def normalize_y(group):
            y = group['center_y']
            if y.max() == y.min():
                return 0.5
            return (y - y.min()) / (y.max() - y.min())
        df['y_normalized'] = df.groupby('image_id', group_keys=False).apply(normalize_y)
    else:
        xy_um = config['microscopy']['voxel_size_zyx'][1]

        def top_zeroed_y(group):
            return (group['center_y'] - group['center_y'].min()) * xy_um
        df['y_normalized'] = df.groupby('image_id', group_keys=False).apply(top_zeroed_y)

    # Drop per-nucleus intensity outliers (computed after y-normalization so a
    # dropped nucleus can't shift another nucleus's top/bottom reference frame)
    df = remove_intensity_outliers(df, 'GATA3_dapi_norm')

    # Plot
    g = sns.lmplot(
        data=df,
        x='y_normalized',
        y='GATA3_dapi_norm',
        hue='group',
        palette={'Control': '#1f77b4', 'Treated': '#ff7f0e'},
        scatter_kws={'alpha': 0.15, 's': 10},
        line_kws={'lw': 3},
        height=7, aspect=1.4,
        legend=True
    )

    mode_label = "size-standardized" if STANDARDIZE_BY_SIZE else "NOT size-standardized"
    plt.title(f"GATA3 Patterning: Top-to-Bottom Gradient (QC Filtered, {mode_label})", fontsize=16)
    if STANDARDIZE_BY_SIZE:
        plt.xlabel("Relative Vertical Position (0 = Top, 1 = Bottom)", fontsize=14)
    else:
        plt.xlabel("Distance from top nucleus (µm)", fontsize=14)
    plt.ylabel("GATA3 Intensity (DAPI Normalized)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.4)

    plot_dir = output_dir / ("standardized" if STANDARDIZE_BY_SIZE else "unstandardized")
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = plot_dir / "combined_gata3_vertical_gradient_CLEAN.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Clean vertical gradient plot [{mode_label}] saved to {output_path}")


if __name__ == '__main__':
    plot_combined_vertical_gradient()
