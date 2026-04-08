
import os
import glob
import numpy as np
import pandas as pd
import tifffile
import napari
from magicgui import magicgui
from skimage.measure import regionprops_table
from pathlib import Path
from src.utils import load_config

# --- GLOBAL LAYERS ---
viewer = None
image_layer = None
label_layer = None
points_layer = None
current_index = {"value": 0}
raw_files = []
label_files = []
output_folder = None
voxel_size = [4, 1, 1] # [Z, Y, X] from your snippet

def update_centroids():
    """Computes centroids of labels and updates the red ID points."""
    global points_layer
    if points_layer is not None and points_layer in viewer.layers:
        viewer.layers.remove(points_layer)
    
    labels = label_layer.data
    props = regionprops_table(labels, properties=["label", "centroid"])
    df = pd.DataFrame(props)

    if len(df) == 0:
        return

    # Centroids are (Z, Y, X)
    points = np.stack([df["centroid-0"], df["centroid-1"], df["centroid-2"]], axis=1)

    points_layer = viewer.add_points(
        points,
        name="ID_Labels",
        properties={"label_id": df["label"].astype(str)},
        text={"text": "{label_id}", "anchor": "center", "color": "white", "size": 10},
        size=12,
        face_color="red",
        scale=voxel_size,
        symbol="disc",
    )

def load_file(index):
    global image_layer, label_layer
    
    # ... (index checks) ...

    image = tifffile.imread(raw_files[index]) # Shape: (151, 3, 1170, 1152)
    labels = tifffile.imread(label_files[index]) # Shape: (151, 1170, 1152)

    # 1. Clear old layers
    viewer.layers.clear()

    # 2. Add Raw Image (Splitting the 3 channels)
    # channel_axis=1 tells napari that '3' is the number of channels
    image_layer = viewer.add_image(
        image, 
        name=["DAPI","GATA3","ppMLC","CDX2"], # Names for your 3 channels
        channel_axis=1, 
        scale=voxel_size,
        contrast_limits=[np.percentile(image, 1), np.percentile(image, 99)]
    )

    # 3. Add Labels (3D Volume)
    label_layer = viewer.add_labels(
        labels, 
        name="edit_labels", 
        scale=voxel_size
    )

    # 4. Force 3D Display Mode
    viewer.dims.ndisplay = 3
    
    update_centroids()

def run_segmentation_editor():
    global viewer, current_index, raw_files, label_files, output_folder
    
    # 1. LOAD PATHS FROM CONFIG
    config = load_config()
    
    # These should point to your desktop folders via the config.yaml
    raw_folder = Path(config['raw_data_dir'])
    label_folder = Path(config['segmentation_dir_raw'])
    output_folder = Path(config['segmentation_dir'])
    
    # Ensure the cleaned directory exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Use glob to find the tiffs
    raw_files = sorted(list(raw_folder.glob("*.tif")))
    label_files = sorted(list(label_folder.glob("*.tif")))

    if len(raw_files) != len(label_files):
        print(f"❌ Warning: File count mismatch! Raw: {len(raw_files)}, Masks: {len(label_files)}")

    # 2. START VIEWER
    viewer = napari.Viewer(ndisplay=3)

    # 3. DEFINE WIDGETS
    @magicgui(call_button="💾 Save and Next")
    def save_and_next():
        idx = current_index["value"]
        save_path = output_folder / os.path.basename(label_files[idx])
        
        # Save current state as uint16
        tifffile.imwrite(save_path, label_layer.data.astype(np.uint16))
        print(f"✅ Saved to cleaned folder: {save_path.name}")

        current_index["value"] += 1
        if current_index["value"] < len(raw_files):
            load_file(current_index["value"])
        else:
            print("🎉 Final embryo reached!")

    @magicgui(call_button="⬅️ Previous Embryo")
    def go_back():
        if current_index["value"] > 0:
            current_index["value"] -= 1
            load_file(current_index["value"])
        else:
            print("Already at the first embryo.")

    @magicgui(call_button="🗑️ Delete ID")
    def delete_by_id(label_id: int):
        mask = (label_layer.data == label_id)
        if not np.any(mask):
            print(f"ID {label_id} not found.")
            return
        label_layer.data[mask] = 0
        label_layer.refresh()
        print(f"Deleted {label_id}")
        update_centroids()

    @magicgui(call_button="🖌️ Pick New ID")
    def set_new_id():
        new_id = int(label_layer.data.max()) + 1
        label_layer.selected_label = new_id
        print(f"Brush set to new ID: {new_id}")

    # 4. ADD DOCK WIDGETS
    viewer.window.add_dock_widget([save_and_next, go_back], area="right", name="Navigation")
    viewer.window.add_dock_widget([delete_by_id, set_new_id], area="right", name="Editing Tools")

    # 5. INITIAL LOAD
    load_file(current_index["value"])
    napari.run()

if __name__ == "__main__":
    run_segmentation_editor()