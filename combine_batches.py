import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_ind

def remove_intensity_outliers(df, column):
    """Removes nuclei that are extreme statistical outliers based on 1.5x IQR."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    removed_count = len(df) - len(filtered_df)
    if removed_count > 0:
        print(f"   🧹 Scrubbed {removed_count} outliers from {column}")
    return filtered_df

def assign_group(name):
    """Checks if the filename starts with 'c' (handles c_, coo, etc.)"""
    n = str(name).lower().strip()
    if n.startswith('c'):
        return 'Control'
    return 'Treated'

def aggregate_project_data():
    # --- CONFIGURATION ---
    BASE_DATA_PATH = "/Users/elyssephillips/Desktop/FGF IF data" 
    root = Path(BASE_DATA_PATH)
    
    if not root.exists():
        print(f"❌ Error: Path '{BASE_DATA_PATH}' not found.")
        return

    summary_files = list(root.glob("**/analysis/vertical_patterning_summary.csv"))
    raw_files = list(root.glob("**/analysis/nuclear_intensities_raw.csv"))
    
    if not summary_files or not raw_files:
        print("❌ Missing files. Ensure Script 03 was run in all subfolders.")
        return

    # 1. COMBINE SUMMARY DATA (Panels 1 & 2)
    print("📊 Processing Summary Data...")
    mega_summary = pd.concat([pd.read_csv(f).assign(batch_id=f.parent.parent.name) for f in summary_files], ignore_index=True)
    mega_summary['group'] = mega_summary['image_id'].apply(assign_group)

    # 2. COMBINE RAW DATA (Panels 3 & 4)
    all_nuclei = []
    print(f"🔎 AUDIT: Sorting {len(raw_files)} batches by image prefix...")

    for f in raw_files:
        batch_id = f.parent.parent.name
        df = pd.read_csv(f)
        df.columns = df.columns.str.lower()
        
        # Mapping: c0=DAPI, c1=GATA3(GFP), c3=CDX2(Cy5)
        rename_logic = {
            'c0_mean': 'dapi_mean', 'dapi_mean': 'dapi_mean',
            'c1_mean': 'gata3_mean', 'gfp_mean': 'gata3_mean',
            'c3_mean': 'cdx2_mean', 'cy5_mean': 'cdx2_mean'
        }
        df.rename(columns=rename_logic, inplace=True, errors='ignore')
        
        df['group'] = df['image_id'].apply(assign_group)
        
        # Audit Print
        c_n = len(df[df['group'] == 'Control'])
        t_n = len(df[df['group'] == 'Treated'])
        print(f"   📂 {batch_id[:30]}... | Found: {c_n} Ctrl / {t_n} Trtd")

        if 'gata3_mean' in df.columns and 'dapi_mean' in df.columns:
            df['GATA3_norm'] = df['gata3_mean'] / (df['dapi_mean'] + 1e-6)
            df['CDX2_norm'] = df['cdx2_mean'] / (df['dapi_mean'] + 1e-6)
            
            def norm_y(g):
                return (g['center_y'] - g['center_y'].min()) / (g['center_y'].max() - g['center_y'].min() + 1e-6)
            
            df['y_rel'] = df.groupby('image_id', group_keys=False).apply(norm_y)
            df['batch_id'] = batch_id 
            all_nuclei.append(df)

    mega_nuclei = pd.concat(all_nuclei, ignore_index=True)

    # 3. CLEANING & BINNING
    print("✨ Cleaning up outliers...")
    mega_nuclei = remove_intensity_outliers(mega_nuclei, 'GATA3_norm')
    mega_nuclei = remove_intensity_outliers(mega_nuclei, 'CDX2_norm')
    mega_nuclei['y_bin'] = mega_nuclei['y_rel'].round(1)

    # 4. METADATA & STATS CALCULATION
    n_ctrl_emb = mega_summary[mega_summary['group'] == 'Control']['image_id'].nunique()
    n_trtd_emb = mega_summary[mega_summary['group'] == 'Treated']['image_id'].nunique()
    n_ctrl_nuc = len(mega_nuclei[mega_nuclei['group'] == 'Control'])
    n_trtd_nuc = len(mega_nuclei[mega_nuclei['group'] == 'Treated'])

    ctrl_label = f"Control ({n_ctrl_emb} emb / {n_ctrl_nuc} nuc)"
    trtd_label = f"Treated ({n_trtd_emb} emb / {n_trtd_nuc} nuc)"
    grad_palette = {ctrl_label: '#1f77b4', trtd_label: '#ff7f0e'}
    mega_nuclei['legend_group'] = mega_nuclei['group'].map({'Control': ctrl_label, 'Treated': trtd_label})

    # Stats for Export
    ctrl_r = mega_summary[mega_summary['group'] == 'Control']['pearson_r']
    trtd_r = mega_summary[mega_summary['group'] == 'Treated']['pearson_r']
    _, p_val_r = ttest_ind(ctrl_r, trtd_r, nan_policy='omit')
    
    ctrl_p = mega_summary[mega_summary['group'] == 'Control']['pattern_score']
    trtd_p = mega_summary[mega_summary['group'] == 'Treated']['pattern_score']
    _, p_val_p = ttest_ind(ctrl_p, trtd_p, nan_policy='omit')

    # 5. EXPORT FILES
    mega_nuclei.to_csv(root / "master_nuclei_data_cleaned.csv", index=False)
    
    stats_df = pd.DataFrame({
        'Metric': ['Pearson r (Mean)', 'Patterning Score (Mean)', 'Embryos (N)', 'Nuclei (N)'],
        'Control': [ctrl_r.mean(), ctrl_p.mean(), n_ctrl_emb, n_ctrl_nuc],
        'Treated': [trtd_r.mean(), trtd_p.mean(), n_trtd_emb, n_trtd_nuc],
        'P-Value': [p_val_r, p_val_p, None, None]
    })
    stats_df.to_csv(root / "stats_summary.csv", index=False)

    # 6. MASTER PLOT
    fig, axes = plt.subplots(1, 4, figsize=(28, 8), gridspec_kw={'width_ratios': [1, 1, 2, 2]})
    unique_batches = mega_summary['batch_id'].unique()
    batch_map = dict(zip(unique_batches, sns.color_palette("husl", len(unique_batches))))

    # Panels 1 & 2
    for i, col in enumerate(['pearson_r', 'pattern_score']):
        sns.boxplot(data=mega_summary, x='group', y=col, hue='group', palette={'Control': '#1f77b4', 'Treated': '#ff7f0e'}, ax=axes[i], showfliers=False, legend=False, width=0.6)
        sns.stripplot(data=mega_summary, x='group', y=col, hue='batch_id', palette=batch_map, ax=axes[i], dodge=True, alpha=0.6, size=5, edgecolor='gray', linewidth=0.5)
        
        c_vals = mega_summary[mega_summary['group'] == 'Control'][col]
        t_vals = mega_summary[mega_summary['group'] == 'Treated'][col]
        _, p = ttest_ind(c_vals, t_vals, nan_policy='omit')
        axes[i].set_title(f"{col.replace('_', ' ').title()}\n(p = {p:.2e})")
        if i == 0:
            axes[0].legend(title="Batch ID", bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=8)

    # Panels 3 & 4
    for i, (marker, title) in enumerate([('GATA3_norm', 'GATA3'), ('CDX2_norm', 'CDX2')]):
        ax = axes[i+2]
        sns.lineplot(data=mega_nuclei, x='y_bin', y=marker, hue='legend_group', palette=grad_palette, ax=ax, lw=4.0, errorbar=('ci', 95), n_boot=1000)
        ax.set_title(f"{title} Vertical Profile")
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Relative Y (0=Top, 1=Bottom)")
        ax.legend(title="Key", loc='best', fontsize=9)

    plt.tight_layout()
    plt.savefig(root / "summary_allbatches_bin10.png", dpi=300, bbox_inches='tight')
    print(f"\n✨ SUCCESS! All files saved to: {root}")

if __name__ == "__main__":
    aggregate_project_data()