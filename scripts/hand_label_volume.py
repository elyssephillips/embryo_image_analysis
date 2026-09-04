"""Hand-label a single 3D volume in napari (ground-truth annotation).

Standalone GUI editor for hand-labeling one volume at a time - e.g. to build
ground truth for grid-searching segmentation parameters (see
pipelines/tracking/00_preprocess_segmentation.py docstring). Same editing
tools as pipelines/IF/02_edit_segmentation.py (ID markers, delete-by-ID,
pick-new-ID, save), but standalone: no config, no dataset loop - just point
RAW_FILE/LABEL_FILE at a volume below and run.

If LABEL_FILE already exists, it's loaded so you can resume/refine an
existing annotation; otherwise editing starts from a blank (all-zero) label
volume the same shape as the raw image.

Usage
-----
  Edit RAW_FILE, LABEL_FILE, and VOXEL_SIZE_ZYX below, then:
  python scripts/hand_label_volume.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import napari
from magicgui import magicgui
from skimage.measure import regionprops_table

# =============================================================================
#  CONFIG - edit these for the volume you're labeling
# =============================================================================
RAW_FILE = Path("/Users/elysse/Desktop/training/Cam_long_00030.tif")
LABEL_FILE = Path("/Users/elysse/Desktop/training/Cam_long_00030_label.tif")

VOXEL_SIZE_ZYX = [2.0, 0.208, 0.208]  # µm per step/pixel
# =============================================================================


def update_centroids(viewer, label_layer, points_layer, voxel_size):
    """Computes centroids of labels and updates the red ID markers. Returns the new points layer."""
    if points_layer is not None and points_layer in viewer.layers:
        viewer.layers.remove(points_layer)

    labels = label_layer.data
    props = regionprops_table(labels, properties=["label", "centroid"])
    df = pd.DataFrame(props)

    if len(df) == 0:
        return None

    points = np.stack([df["centroid-0"], df["centroid-1"], df["centroid-2"]], axis=1)
    return viewer.add_points(
        points,
        name="ID_Labels",
        properties={"label_id": df["label"].astype(str)},
        text={"text": "{label_id}", "anchor": "center", "color": "white", "size": 10},
        size=12,
        face_color="red",
        scale=voxel_size,
        symbol="disc",
    )


def run_hand_labeler():
    state = {"points_layer": None}

    image = tifffile.imread(str(RAW_FILE))
    if LABEL_FILE.exists():
        labels = tifffile.imread(str(LABEL_FILE))
        if labels.shape != image.shape:
            raise ValueError(f"Existing label shape {labels.shape} != raw image shape {image.shape}")
        print(f"Loaded existing labels: {LABEL_FILE}")
    else:
        labels = np.zeros(image.shape, dtype=np.uint16)
        print("No existing label file found - starting from a blank volume.")

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(
        image,
        name=RAW_FILE.name,
        scale=VOXEL_SIZE_ZYX,
        contrast_limits=[np.percentile(image, 1), np.percentile(image, 99)],
    )
    label_layer = viewer.add_labels(labels, name="hand_labels", scale=VOXEL_SIZE_ZYX)
    viewer.dims.ndisplay = 3
    state["points_layer"] = update_centroids(viewer, label_layer, state["points_layer"], VOXEL_SIZE_ZYX)

    @magicgui(call_button="Save")
    def save():
        LABEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(LABEL_FILE), label_layer.data.astype(np.uint16))
        print(f"Saved: {LABEL_FILE}")

    @magicgui(call_button="Refresh ID Markers")
    def refresh_ids():
        state["points_layer"] = update_centroids(viewer, label_layer, state["points_layer"], VOXEL_SIZE_ZYX)

    @magicgui(call_button="Delete ID")
    def delete_by_id(label_id: int):
        mask = (label_layer.data == label_id)
        if not np.any(mask):
            print(f"ID {label_id} not found.")
            return
        label_layer.data[mask] = 0
        label_layer.refresh()
        print(f"Deleted {label_id}")
        state["points_layer"] = update_centroids(viewer, label_layer, state["points_layer"], VOXEL_SIZE_ZYX)

    @magicgui(call_button="Pick New ID")
    def set_new_id():
        new_id = int(label_layer.data.max()) + 1
        label_layer.selected_label = new_id
        print(f"Brush set to new ID: {new_id}")

    @magicgui(call_button="Find ID")
    def find_id(label_id: int):
        mask = label_layer.data == label_id
        if not np.any(mask):
            print(f"ID {label_id} not found.")
            return
        centroid = np.argwhere(mask).mean(axis=0)
        label_layer.selected_label = label_id
        label_layer.show_selected_label = True

        # 3D ray-casting can miss objects a few voxels or smaller (the ray grid
        # is coarser than the voxel) even after centering/isolating - drop into
        # a 2D single-slice view instead, which rasters per-pixel and can't miss it.
        viewer.dims.ndisplay = 2
        step = list(viewer.dims.current_step)
        step[0] = int(round(centroid[0]))
        viewer.dims.current_step = tuple(step)
        viewer.camera.center = tuple(float(c) for c in centroid[1:] * np.array(VOXEL_SIZE_ZYX[1:]))
        viewer.camera.zoom = 300
        print(f"Jumped to ID {label_id} at z={step[0]}, isolated in 2D slice view.")

    @magicgui(call_button="Show All Labels")
    def show_all():
        label_layer.show_selected_label = False

    viewer.window.add_dock_widget([save, refresh_ids], area="right", name="Save")
    viewer.window.add_dock_widget([delete_by_id, set_new_id], area="right", name="Editing Tools")
    viewer.window.add_dock_widget([find_id, show_all], area="right", name="Find")

    napari.run()


if __name__ == "__main__":
    run_hand_labeler()
