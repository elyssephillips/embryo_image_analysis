import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import seaborn as sns
import tifffile as tiff
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr, linregress, ttest_ind
from src.io import load_config, update_master_study_log, get_storage_note, get_config_notes, get_config_n_conditions, summarize_config_metadata
from src.analysis import normalize_by_dapi, map_values_to_labels, calculate_patterning_score
from src.log import log_run, sync_notes


def run_full_analysis():
    # 1. SETUP & LOAD
    config = load_config()
    output_dir = Path(config['output_dir'])
    data_path = output_dir / "nuclear_intensities_raw.csv"

    if not data_path.exists():
        print(f"Error: CSV not found at {data_path}. Run Script 02 first!")
        return

    df_raw = pd.read_csv(data_path)

    # --- QC EXCLUSION ---
    exclude_dict = config.get('exclusions', {})
    exclude_ids = list(exclude_dict.keys())

    found_ids = df_raw['image_id'].unique()
    for eid in exclude_ids:
        if eid not in found_ids:
            print(f"Warning: Excluded ID '{eid}' not found in dataset. Check for typos!")

    df = df_raw[~df_raw['image_id'].isin(exclude_ids)].copy()
    removed_count = len(found_ids) - len(df['image_id'].unique())
    print(f"QC: Excluded {removed_count} embryos. {len(df['image_id'].unique())} remaining.")

    qc_report_path = output_dir / "qc_exclusion_report.txt"
    with open(qc_report_path, 'w') as f:
        f.write(f"QC Exclusion Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("-" * 50 + "\n")
        for eid, reason in exclude_dict.items():
            f.write(f"ID: {eid} | Reason: {reason}\n")

    # 2. RENAME & NORMALIZE
    keys = list(config['microscopy']['channels'].keys())
    names = config['microscopy']['channel_names']
    rename_map = {f"{k}_mean": f"{n}_mean" for k, n in zip(keys, names)}
    df.rename(columns=rename_map, inplace=True)
    df = normalize_by_dapi(df, dapi_col='dapi_mean')
    df['group'] = df['image_id'].apply(lambda x: 'Control' if x.startswith('c') else 'Treated')

    # 3. GLOBAL SCALES
    xy_um = config['microscopy']['voxel_size_zyx'][1]

    # Zeroed per embryo (top nucleus = 0), not just the image crop's top edge --
    # doesn't change slopes/correlations (shift-invariant), only makes the
    # "Top=0" axis label below actually exact instead of crop-margin-dependent.
    def top_zeroed_y_um(group):
        return (group['center_y'] - group['center_y'].min()) * xy_um
    df['center_y_um'] = df.groupby('image_id', group_keys=False).apply(top_zeroed_y_um)

    g_max = df['GATA3_dapi_norm'].quantile(0.98)
    c_max = df['CDX2_dapi_norm'].quantile(0.98)

    # 4. ANALYSIS LOOP
    stats_list = []
    plot_dir = output_dir / "spatial_reports_Y_axis"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for eid in df['image_id'].unique():
        sub = df[df['image_id'] == eid]

        r_val, _ = pearsonr(sub['GATA3_dapi_norm'], sub['CDX2_dapi_norm'])
        g_slope, _, g_r, g_p, _ = linregress(sub['center_y_um'], sub['GATA3_dapi_norm'])
        c_slope, _, c_r, c_p, _ = linregress(sub['center_y_um'], sub['CDX2_dapi_norm'])

        stats_list.append({
            'image_id': eid,
            'group': sub['group'].iloc[0],
            'pearson_r': r_val,
            'gata3_y_slope': g_slope,
            'cdx2_y_slope': c_slope,
            'pattern_score': calculate_patterning_score(sub, 'GATA3_dapi_norm')
        })

        # --- SPATIAL MAPPING ---
        mask_path = output_dir / f"{eid}_eroded_seg.tif"
        if not mask_path.exists():
            continue

        labels = tiff.imread(mask_path)
        g_mip = np.max(map_values_to_labels(labels, sub, 'GATA3_dapi_norm'), axis=0)
        c_mip = np.max(map_values_to_labels(labels, sub, 'CDX2_dapi_norm'), axis=0)

        # Custom single-color colormaps: white (low) → saturated color (high)
        h, w = g_mip.shape
        gn, cn = np.clip(g_mip/g_max, 0, 1), np.clip(c_mip/c_max, 0, 1)
        g_masked = np.ma.masked_where(g_mip == 0, g_mip)
        c_masked = np.ma.masked_where(c_mip == 0, c_mip)

        cmap_g = LinearSegmentedColormap.from_list('white_magenta', ['white', '#FFD0EE', '#FF66CC'])
        cmap_c = LinearSegmentedColormap.from_list('white_green', ['white', '#C8F0C8', '#44BB44'])
        cmap_g.set_bad('white'); cmap_c.set_bad('white')

        # Merge: multiplicative blend using global normalization
        nucleus_mask = (g_mip == 0) & (c_mip == 0)
        g_rgb = cmap_g(gn)[..., :3]
        c_rgb = cmap_c(cn)[..., :3]
        g_rgb[g_mip == 0] = [1, 1, 1]
        c_rgb[c_mip == 0] = [1, 1, 1]
        overlay = g_rgb * c_rgb
        overlay[nucleus_mask] = [1, 1, 1]

        # Per-embryo plot
        fig = plt.figure(figsize=(20, 12), facecolor='white')
        gs = fig.add_gridspec(2, 3)
        ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
        ax4, ax5 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

        extent_um = [0, w * xy_um, h * xy_um, 0]
        scalebar_um = 50
        fp = fm.FontProperties(size=8)

        ax1.imshow(g_masked, cmap=cmap_g, vmin=0, vmax=g_max, extent=extent_um)
        ax1.set_facecolor('white'); ax1.set_title("GATA3 Intensity Map")
        ax1.set_xlabel("X (µm)"); ax1.set_ylabel("Y (µm, Top=0)")
        ax1.add_artist(AnchoredSizeBar(ax1.transData, scalebar_um, f'{scalebar_um} µm',
                        'lower right', pad=0.4, color='black', frameon=False,
                        size_vertical=h * xy_um * 0.015, fontproperties=fp))

        ax2.imshow(c_masked, cmap=cmap_c, vmin=0, vmax=c_max, extent=extent_um)
        ax2.set_facecolor('white'); ax2.set_title("CDX2 Intensity Map")
        ax2.set_xlabel("X (µm)"); ax2.set_ylabel("Y (µm, Top=0)")
        ax2.add_artist(AnchoredSizeBar(ax2.transData, scalebar_um, f'{scalebar_um} µm',
                        'lower right', pad=0.4, color='black', frameon=False,
                        size_vertical=h * xy_um * 0.015, fontproperties=fp))

        ax3.imshow(overlay, extent=extent_um); ax3.set_facecolor('white')
        ax3.set_title("MERGE (Magenta=GATA3 / Green=CDX2 / Black=overlap)")
        ax3.add_artist(AnchoredSizeBar(ax3.transData, scalebar_um, f'{scalebar_um} µm',
                        'lower right', pad=0.4, color='black', frameon=False,
                        size_vertical=h * xy_um * 0.015, fontproperties=fp))
        ax3.axis('off')

        sns.regplot(data=sub, x='center_y_um', y='GATA3_dapi_norm', ax=ax4,
                    scatter_kws={'alpha': 0.2, 'color': '#FF66CC'}, line_kws={'color': '#AA0066'})
        ax4.set_title("GATA3 vs Y-Position"); ax4.set_xlabel("Y-position (µm, Top=0)")
        g_p_str = f"{g_p:.3f}" if g_p >= 0.001 else f"{g_p:.2e}"
        ax4.annotate(f"y = {g_slope:.4f}x\nR² = {g_r**2:.3f},  p = {g_p_str}",
                     xy=(0.05, 0.95), xycoords='axes fraction', va='top', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFD0EE', alpha=0.7))

        sns.regplot(data=sub, x='center_y_um', y='CDX2_dapi_norm', ax=ax5,
                    scatter_kws={'alpha': 0.2, 'color': '#44BB44'}, line_kws={'color': '#1E7A1E'})
        ax5.set_title("CDX2 vs Y-Position"); ax5.set_xlabel("Y-position (µm, Top=0)")
        c_p_str = f"{c_p:.3f}" if c_p >= 0.001 else f"{c_p:.2e}"
        ax5.annotate(f"y = {c_slope:.4f}x\nR² = {c_r**2:.3f},  p = {c_p_str}",
                     xy=(0.05, 0.95), xycoords='axes fraction', va='top', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#C8F0C8', alpha=0.7))

        plt.suptitle(f"Embryo Vertical Patterning: {eid}", fontsize=18)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{eid}_vertical_report.png", dpi=300, facecolor='white')
        plt.close()

    # 5. SAVE SUMMARY DATA
    stat_df = pd.DataFrame(stats_list)
    csv_path = output_dir / "vertical_patterning_summary.csv"
    stat_df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to: {csv_path}")

    dataset_name = output_dir.parent.name
    dataset_description = config.get('name', dataset_name)
    config_file = "configs/config.yaml"
    analysis_version = config.get('analysis_version', "")
    notes = config.get('description', "")

    update_master_study_log(
        stat_df,
        dataset_name,
        project_name="IF",
        pipeline_name="IF",
        dataset_description=dataset_description,
        config_file=config_file,
        analysis_version=analysis_version,
        notes=notes,
    )

    # 6. GROUP STATS & BOXPLOTS
    controls = stat_df[stat_df['group'] == 'Control']
    treated = stat_df[stat_df['group'] == 'Treated']

    def get_stars(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "ns"

    plt.figure(figsize=(22, 6), facecolor='white')

    ax_p = plt.subplot(1, 3, 1)
    sns.boxplot(data=stat_df, x='group', y='pearson_r', palette='Set2', showfliers=False)
    sns.stripplot(data=stat_df, x='group', y='pearson_r', color='black', alpha=0.6)
    _, p_corr = ttest_ind(controls['pearson_r'], treated['pearson_r'])
    y_max = stat_df['pearson_r'].max()
    p_rng = stat_df['pearson_r'].max() - stat_df['pearson_r'].min()
    p_off = max(p_rng * 0.08, 0.02)
    ax_p.plot([0, 0, 1, 1], [y_max + p_off, y_max + p_off*1.5, y_max + p_off*1.5, y_max + p_off], lw=1.5, c='k')
    ax_p.text(0.5, y_max + p_off*1.6, get_stars(p_corr), ha='center', fontweight='bold')
    plt.title(f"Co-Expression (p={p_corr:.4f})")

    ax_s = plt.subplot(1, 3, 2)
    sns.boxplot(data=stat_df, x='group', y='gata3_y_slope', palette='Set2', showfliers=False)
    sns.stripplot(data=stat_df, x='group', y='gata3_y_slope', color='black', alpha=0.6)
    _, p_slope = ttest_ind(controls['gata3_y_slope'], treated['gata3_y_slope'])
    y_max_s = stat_df['gata3_y_slope'].max()
    s_rng = stat_df['gata3_y_slope'].max() - stat_df['gata3_y_slope'].min()
    s_off = max(s_rng * 0.08, abs(y_max_s) * 0.05)
    ax_s.plot([0, 0, 1, 1], [y_max_s + s_off, y_max_s + s_off*1.5, y_max_s + s_off*1.5, y_max_s + s_off], lw=1.5, c='k')
    ax_s.text(0.5, y_max_s + s_off*1.6, get_stars(p_slope), ha='center', fontweight='bold')
    plt.title(f"GATA3 Gradient Strength (p={p_slope:.4f})")

    ax_pat = plt.subplot(1, 3, 3)
    sns.boxplot(data=stat_df, x='group', y='pattern_score', palette='Set2', showfliers=False)
    sns.stripplot(data=stat_df, x='group', y='pattern_score', color='black', alpha=0.6)
    _, p_pattern = ttest_ind(controls['pattern_score'], treated['pattern_score'])
    y_max_pat = stat_df['pattern_score'].max()
    pat_rng = stat_df['pattern_score'].max() - stat_df['pattern_score'].min()
    pat_off = max(pat_rng * 0.08, 0.02)
    ax_pat.plot([0, 0, 1, 1], [y_max_pat + pat_off, y_max_pat + pat_off*1.5, y_max_pat + pat_off*1.5, y_max_pat + pat_off], lw=1.5, c='k')
    ax_pat.text(0.5, y_max_pat + pat_off*1.6, get_stars(p_pattern), ha='center', fontweight='bold')
    plt.title(f"GATA3 Polarization Index (p={p_pattern:.4f})")

    plt.tight_layout()
    plt.savefig(output_dir / "vertical_group_comparison_with_stats.png", dpi=300, facecolor='white')
    print(f"Comparison plot saved. Pearson P={p_corr:.5f}, Slope P={p_slope:.5f}, Pattern P={p_pattern:.5f}")


if __name__ == "__main__":
    run_full_analysis()
    config = load_config()
    dataset_id = Path(config['output_dir']).parts[-2]
    log_run("IF", dataset_id, "04_plot_intensities.py",
            output_path=config['output_dir'], detail="detailed",
            data_path=config.get('raw_data_dir', config['output_dir']),
            storage=get_storage_note(), n_conditions=get_config_n_conditions(),
            **summarize_config_metadata(config))
    sync_notes("IF", dataset_id, get_config_notes())
