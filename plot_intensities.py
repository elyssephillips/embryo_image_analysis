import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile as tiff
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr, linregress, ttest_ind
from src.utils import (load_config, normalize_by_dapi, map_values_to_labels, 
                       calculate_patterning_score, update_master_study_log)


def run_full_analysis():
    # 1. SETUP & LOAD
    config = load_config()
    output_dir = Path(config['output_dir'])
    data_path = output_dir / "nuclear_intensities_raw.csv"
    
    if not data_path.exists():
        print(f"❌ Error: CSV not found at {data_path}. Run Script 02 first!")
        return

    df_raw = pd.read_csv(data_path)
    
    # --- QC EXCLUSION LOGIC ---
    exclude_dict = config.get('exclusions', {})
    exclude_ids = list(exclude_dict.keys())
    
    # Check for typos in config vs CSV
    found_ids = df_raw['image_id'].unique()
    for eid in exclude_ids:
        if eid not in found_ids:
            print(f"⚠️ Warning: Excluded ID '{eid}' not found in dataset. Check for typos!")

    # Filter the dataframe
    df = df_raw[~df_raw['image_id'].isin(exclude_ids)].copy()
    removed_count = len(found_ids) - len(df['image_id'].unique())
    print(f"🚫 QC: Excluded {removed_count} embryos. {len(df['image_id'].unique())} remaining.")

    # Save a record of exclusions
    qc_report_path = output_dir / "qc_exclusion_report.txt"
    with open(qc_report_path, 'w') as f:
        f.write(f"QC Exclusion Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("-" * 50 + "\n")
        for eid, reason in exclude_dict.items():
            f.write(f"ID: {eid} | Reason: {reason}\n")
    # --------------------------

    # 2. RENAME & NORMALIZE
    keys = list(config['microscopy']['channels'].keys())
    names = config['microscopy']['channel_names']
    rename_map = {f"{k}_mean": f"{n}_mean" for k, n in zip(keys, names)}
    df.rename(columns=rename_map, inplace=True)
    df = normalize_by_dapi(df, dapi_col='dapi_mean')
    df['group'] = df['image_id'].apply(lambda x: 'Control' if x.startswith('c') else 'Treated')

    # 3. GLOBAL SCALES
    g_max = df['GATA3_dapi_norm'].quantile(0.98)
    c_max = df['CDX2_dapi_norm'].quantile(0.98)

    # 4. ANALYSIS LOOP
    stats_list = []
    plot_dir = output_dir / "spatial_reports_Y_axis"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for eid in df['image_id'].unique():
        sub = df[df['image_id'] == eid]
        
        r_val, _ = pearsonr(sub['GATA3_dapi_norm'], sub['CDX2_dapi_norm'])
        g_slope, _, _, _, _ = linregress(sub['center_y'], sub['GATA3_dapi_norm'])
        c_slope, _, _, _, _ = linregress(sub['center_y'], sub['CDX2_dapi_norm'])
        
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
        if not mask_path.exists(): continue
        
        labels = tiff.imread(mask_path)
        g_mip = np.max(map_values_to_labels(labels, sub, 'GATA3_dapi_norm'), axis=0)
        c_mip = np.max(map_values_to_labels(labels, sub, 'CDX2_dapi_norm'), axis=0)

        # Create RGB Overlay
        h, w = g_mip.shape
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        gn, cn = np.clip(g_mip/g_max, 0, 1), np.clip(c_mip/c_max, 0, 1)
        overlay[..., 0], overlay[..., 1], overlay[..., 2] = cn, gn, np.clip(gn+cn, 0, 1)

        # --- PLOTTING PER EMBRYO ---
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3) 
        ax1, ax2, ax3 = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[0,2])
        ax4, ax5 = fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])

        ax1.imshow(g_mip, cmap='magma', vmin=0, vmax=g_max); ax1.set_title("GATA3 Intensity Map")
        ax2.imshow(c_mip, cmap='viridis', vmin=0, vmax=c_max); ax2.set_title("CDX2 Intensity Map")
        ax3.imshow(overlay); ax3.set_title("MERGE (Cyan:GATA3 / Magenta:CDX2)"); ax3.axis('off')

        sns.regplot(data=sub, x='center_y', y='GATA3_dapi_norm', ax=ax4, scatter_kws={'alpha':0.2, 'color':'cyan'}, line_kws={'color':'red'})
        ax4.set_title("GATA3 vs Y-Position"); ax4.set_xlabel("Y-coordinate (Top=0)")
        
        sns.regplot(data=sub, x='center_y', y='CDX2_dapi_norm', ax=ax5, scatter_kws={'alpha':0.2, 'color':'magenta'}, line_kws={'color':'red'})
        ax5.set_title("CDX2 vs Y-Position"); ax5.set_xlabel("Y-coordinate (Top=0)")

        plt.suptitle(f"Embryo Vertical Patterning: {eid}", fontsize=18)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{eid}_vertical_report.png", dpi=300)
        plt.close()

    # 5. SAVE SUMMARY DATA
    stat_df = pd.DataFrame(stats_list)
    csv_path = output_dir / "vertical_patterning_summary.csv"
    stat_df.to_csv(csv_path, index=False)
    print(f"📄 Summary CSV saved to: {csv_path}")

    # CALL THE UTILS LOGGING FUNCTION
    # We use .parent.name to get the folder name which should correspond to the dataset name
    dataset_name = output_dir.parent.name 
    update_master_study_log(stat_df, dataset_name)

    # 6. STATS & BOXPLOTS
    controls = stat_df[stat_df['group'] == 'Control']
    treated = stat_df[stat_df['group'] == 'Treated']
    
    def get_stars(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "ns"

    plt.figure(figsize=(15, 6))
    
    # Plot 1: Pearson r
    ax_p = plt.subplot(1, 2, 1)
    sns.boxplot(data=stat_df, x='group', y='pearson_r', palette='Set2', showfliers=False)
    sns.stripplot(data=stat_df, x='group', y='pearson_r', color='black', alpha=0.6)
    _, p_corr = ttest_ind(controls['pearson_r'], treated['pearson_r'])
    y_max = stat_df['pearson_r'].max()
    ax_p.plot([0, 0, 1, 1], [y_max+0.1, y_max+0.15, y_max+0.15, y_max+0.1], lw=1.5, c='k')
    ax_p.text(0.5, y_max+0.16, get_stars(p_corr), ha='center', fontweight='bold')
    plt.title(f"Co-Expression+ (p={p_corr:.4f})")

    # Plot 2: Y-Slope
    ax_s = plt.subplot(1, 2, 2)
    sns.boxplot(data=stat_df, x='group', y='gata3_y_slope', palette='Set2', showfliers=False)
    sns.stripplot(data=stat_df, x='group', y='gata3_y_slope', color='black', alpha=0.6)
    _, p_slope = ttest_ind(controls['gata3_y_slope'], treated['gata3_y_slope'])
    y_max_s = stat_df['gata3_y_slope'].max()
    ax_s.plot([0, 0, 1, 1], [y_max_s+0.001, y_max_s+0.0015, y_max_s+0.0015, y_max_s+0.001], lw=1.5, c='k')
    ax_s.text(0.5, y_max_s+0.0016, get_stars(p_slope), ha='center', fontweight='bold')
    plt.title(f"GATA3 Gradient Strength (p={p_slope:.4f})")

    plt.tight_layout()
    plt.savefig(output_dir / "vertical_group_comparison_with_stats.png", dpi=300)
    print(f"📈 Comparison plot saved. Pearson P={p_corr:.5f}, Slope P={p_slope:.5f}")

if __name__ == "__main__":
    run_full_analysis()