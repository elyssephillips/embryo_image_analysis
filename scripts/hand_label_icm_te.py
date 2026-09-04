"""Hand-classify existing instance labels as ICM vs TE (ground-truth annotation
for 3-class nnUNet training: background / ICM / TE).

Standalone GUI tool: loads an already hand-labeled instance volume (does NOT
edit the labels themselves - see hand_label_volume.py for that). You only
mark ICM nuclei - every instance you don't mark is treated as TE by default,
since ICM is the minority class and (per
pipelines/tracking/segmentation_notes.md) forms one compact 3D cluster, so
it's much faster to flag the cluster than to individually confirm every TE
nucleus around it. Marks save to a JSON sidecar next to LABEL_FILE, as
{label_id: "ICM"} - any label not present is TE - and reload on restart so
you can classify a volume across multiple sessions.

Because there's no per-instance TE confirmation step, this only works if you
visually sweep the *whole* embryo before moving to the next volume - an ICM
nucleus you never look at will silently default to TE downstream.

Local neighbor density (same ICM_NEIGHBOR_RADIUS_UM as
pipelines/tracking/00_preprocess_segmentation.py) is printed for context only
when you jump to or mark an ID - it does not assign the class. This is a
deliberately manual call: the geometric heuristic has a confirmed
misclassification right at the ICM/TE transition zone (see
pipelines/tracking/segmentation_notes.md), which is exactly what retraining
nnUNet on real ICM/TE identity is meant to do better than.

Instances are colored by classification (unmarked: distinct per-ID color so
nuclei stay visually separable while you look for the ICM cluster; ICM:
green) so progress is visible at a glance in the 3D view.

Usage
-----
  Edit RAW_FILE and LABEL_FILE below (LABEL_FILE must already exist - this
  tool classifies existing instances, it doesn't draw new ones), then:
  python scripts/hand_label_icm_te.py

  Orbit the 3D view, read off the ID markers on nuclei that look like ICM
  (compact, packed cluster), and type each ID into "Mark ICM". Use "Jump to
  ID" to double check a specific one, and "Unmark ICM" to undo a misclick.
"""
import colorsys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import napari
from magicgui import magicgui
from skimage.measure import regionprops_table
from scipy.spatial import cKDTree
from napari.utils.colormaps import DirectLabelColormap

# =============================================================================
#  CONFIG - edit these for the volume you're classifying
# =============================================================================
RAW_FILE = Path("/Users/elysse/Desktop/training/Cam_long_00075.tif")
LABEL_FILE = Path("/Users/elysse/Desktop/training/Cam_long_00075_label.tif")
CLASS_FILE = LABEL_FILE.with_suffix(".icm_te.json")

VOXEL_SIZE_ZYX = [2.0, 0.208, 0.208]  # µm per step/pixel

# Same radius as pipelines/tracking/00_preprocess_segmentation.py's
# ICM_NEIGHBOR_RADIUS_UM - shown as context only, never assigns anything.
ICM_NEIGHBOR_RADIUS_UM = 30.0
# =============================================================================

ICM_COLOR = "#3fae5c"


def unmarked_color(label_id):
    """Distinct per-ID color (golden-ratio hue hash) so unmarked (default-TE)
    nuclei stay visually separable from their neighbors, instead of one flat blob."""
    hue = (label_id * 0.6180339887498949) % 1.0
    return (*colorsys.hsv_to_rgb(hue, 0.55, 0.9), 1.0)


def load_assignments():
    if CLASS_FILE.exists():
        with open(CLASS_FILE) as f:
            return json.load(f)
    return {}


def save_assignments(assignments):
    with open(CLASS_FILE, "w") as f:
        json.dump(assignments, f, indent=2, sort_keys=True)


def run_classifier():
    image = tifffile.imread(str(RAW_FILE))
    if not LABEL_FILE.exists():
        raise FileNotFoundError(
            f"{LABEL_FILE} doesn't exist - this tool classifies existing "
            f"instances, it doesn't draw them (see hand_label_volume.py for that)."
        )
    labels = tifffile.imread(str(LABEL_FILE))
    if labels.shape != image.shape:
        raise ValueError(f"Label shape {labels.shape} != raw image shape {image.shape}")

    assignments = load_assignments()
    print(f"Loaded {len(assignments)} existing classifications from {CLASS_FILE.name}")

    props = regionprops_table(labels, properties=["label", "centroid"])
    df = pd.DataFrame(props).sort_values("label").reset_index(drop=True)
    ids = df["label"].tolist()
    centroids_vox = df[["centroid-0", "centroid-1", "centroid-2"]].to_numpy()
    centroids_um = centroids_vox * np.array(VOXEL_SIZE_ZYX)
    id_to_centroid_vox = dict(zip(ids, centroids_vox))

    # Neighbor density, for on-screen context only (see module docstring).
    tree = cKDTree(centroids_um)
    nbr_counts = tree.query_ball_point(centroids_um, r=ICM_NEIGHBOR_RADIUS_UM, return_length=True) - 1
    median_nbrs = max(np.median(nbr_counts), 1)
    id_to_context = {lid: (int(nc), nc / median_nbrs) for lid, nc in zip(ids, nbr_counts)}

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(
        image, name=RAW_FILE.name, scale=VOXEL_SIZE_ZYX,
        contrast_limits=[np.percentile(image, 1), np.percentile(image, 99)],
    )
    label_layer = viewer.add_labels(labels, name="instance_labels", scale=VOXEL_SIZE_ZYX)

    state = {"points_layer": None}

    def current_color(label_id):
        return ICM_COLOR if str(label_id) in assignments else unmarked_color(label_id)

    def refresh_view():
        color_dict = {lid: current_color(lid) for lid in ids}
        color_dict[None] = "#888888"  # fallback for any label not in `ids`, shouldn't occur
        label_layer.colormap = DirectLabelColormap(color_dict=color_dict)

        if state["points_layer"] is not None and state["points_layer"] in viewer.layers:
            viewer.layers.remove(state["points_layer"])
        state["points_layer"] = viewer.add_points(
            centroids_vox, name="ID_labels",
            properties={"label_id": [str(i) for i in ids]},
            text={"text": "{label_id}", "anchor": "center", "color": "white", "size": 10},
            size=12, face_color=[current_color(lid) for lid in ids],
            scale=VOXEL_SIZE_ZYX, symbol="disc",
        )

    def print_progress():
        n_icm = len(assignments)
        print(f"{n_icm}/{len(ids)} marked ICM - the rest ({len(ids) - n_icm}) default to TE")

    def jump_to(label_id):
        if label_id not in id_to_centroid_vox:
            print(f"ID {label_id} not found.")
            return
        centroid = id_to_centroid_vox[label_id]
        viewer.camera.center = tuple(float(c) for c in centroid * np.array(VOXEL_SIZE_ZYX))
        label_layer.selected_label = label_id
        nc, ratio = id_to_context[label_id]
        cls = "ICM" if str(label_id) in assignments else "TE (default)"
        print(f"ID {label_id}: {nc} neighbors within {ICM_NEIGHBOR_RADIUS_UM:.0f}um "
              f"({ratio:.2f}x this volume's median) - currently {cls}")

    @magicgui(call_button="Jump to ID")
    def jump_to_id(label_id: int):
        jump_to(label_id)

    @magicgui(call_button="Mark ICM")
    def mark_icm(label_id: int):
        if label_id not in id_to_centroid_vox:
            print(f"ID {label_id} not found.")
            return
        assignments[str(label_id)] = "ICM"
        save_assignments(assignments)
        refresh_view()
        print_progress()

    @magicgui(call_button="Unmark ICM")
    def unmark_icm(label_id: int):
        if assignments.pop(str(label_id), None) is None:
            print(f"ID {label_id} wasn't marked ICM.")
            return
        save_assignments(assignments)
        refresh_view()
        print_progress()

    viewer.window.add_dock_widget([jump_to_id], area="right", name="Navigate")
    viewer.window.add_dock_widget([mark_icm, unmark_icm], area="right", name="Classify")

    refresh_view()
    print_progress()
    napari.run()


if __name__ == "__main__":
    run_classifier()
