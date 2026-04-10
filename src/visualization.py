from __future__ import annotations
import matplotlib.pyplot as plt


def plot_embryo_heatmap(df, image_id, channel_name, ax=None, title=None):
    """Plots a spatial heatmap for a single embryo and specific channel."""
    embryo_df = df[df['image_id'] == image_id]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        embryo_df['center_x'],
        embryo_df['center_y'],
        c=embryo_df[channel_name],
        cmap='magma',
        s=50,
        edgecolor='none'
    )
    ax.set_aspect('equal')
    ax.set_title(title or f"{image_id} - {channel_name}")
    ax.invert_yaxis()
    plt.colorbar(sc, ax=ax, label='Normalized Intensity')
    return ax
