import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.utils import load_config, normalize_by_dapi

def plot_combined_vertical_gradient():
    config = load_config()
    output_dir = Path(config['output_dir'])
    
    # 1. Load Data
    data_path = output_dir / "nuclear_intensities_raw.csv"
    if not data_path.exists():
        print(f"❌ Error: CSV not found at {data_path}")
        return
        
    df_raw = pd.read_csv(data_path)

    # --- ADD QC EXCLUSION LOGIC HERE ---
    exclude_dict = config.get('exclusions', {})
    exclude_ids = list(exclude_dict.keys())
    
    # Filter the dataframe so the combined plot is "Clean"
    df = df_raw[~df_raw['image_id'].isin(exclude_ids)].copy()
    print(f"🚫 QC: Excluded {len(exclude_ids)} embryos from combined plot.")
    # -----------------------------------

    # 2. Standard Cleanup
    keys = list(config['microscopy']['channels'].keys())
    names = config['microscopy']['channel_names']
    df.rename(columns={f"{k}_mean": f"{n}_mean" for k, n in zip(keys, names)}, inplace=True)
    df = normalize_by_dapi(df, dapi_col='dapi_mean')
    
    # Fix: Ensure group identification is consistent
    df['group'] = df['image_id'].apply(lambda x: 'Control' if x.startswith('c') else 'Treated')

    # 3. NORMALIZE Y-DISTANCE PER EMBRYO
    def normalize_y(group):
        y = group['center_y']
        # Handle case where an embryo might have only 1 nucleus (unlikely but safe)
        if y.max() == y.min(): return 0.5 
        return (y - y.min()) / (y.max() - y.min())

    df['y_normalized'] = df.groupby('image_id', group_keys=False).apply(normalize_y)

    # 4. PLOT VERTICAL GRADIENT
    # Added legend=True explicitly and ensured palette matches the hue
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

    plt.title("GATA3 Patterning: Top-to-Bottom Gradient (QC Filtered)", fontsize=16)
    plt.xlabel("Relative Vertical Position (0 = Top, 1 = Bottom)", fontsize=14)
    plt.ylabel("GATA3 Intensity (DAPI Normalized)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.4)

    # Save
    output_path = output_dir / "combined_gata3_vertical_gradient_CLEAN.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Clean vertical gradient plot saved to {output_path}")

if __name__ == '__main__':
    plot_combined_vertical_gradient()