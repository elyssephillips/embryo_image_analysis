"""
Count transcripts inside the oocyte, using PNG exports of a follicle ROI.

Environment requirements: numpy, pandas, pyyaml, pillow, scipy, scikit-image,
and matplotlib.

Expects a folder containing two PNGs exported from Xenium Explorer for the same ROI:
  - an "annotations" layer: granulosa cell outlines only, transparent background
  - a "transcripts" layer: transcript dots only, transparent background
Both must be the same pixel dimensions (i.e. exported from the same view).

Usage:
  python 01_center_object_transcript_counts.py --input-dir /path/to/exported_pngs
  python 01_center_object_transcript_counts.py --config config.yaml
"""

from pathlib import Path

import argparse
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.morphology import disk
from skimage.measure import label, regionprops, find_contours
import matplotlib.pyplot as plt

# Fixed tuning knobs -- mostly depend on the exported pngs, not sample itself. 
ALPHA_MIN = 51  # out of 255; drops faint anti-aliased fringe pixels from color matches
CLOSING_RADIUS_UM = 3.0  # bridges small gaps in granulosa layer 1 before filling it in

DEFAULT_CONFIG = {
    "layers": {
        "annotations": "annotations_layer.png",
        "transcripts": "transcripts_layer.png",
    },
    "granulosa_colors": {
        "layer1": {"rgb": [255, 100, 100], "tolerance": 20},  # inner layer, borders the oocyte
        "layer2": {"rgb": [69, 139, 255], "tolerance": 20},   # outer layer
    },
    "transcript_color": {"green_dominance_min": 20},
    "scale_bar_um": 50,
    "px_per_um_override": None,
}


def load_config(path):
    with open(path) as f:
        user_config = yaml.safe_load(f)
    config = {**DEFAULT_CONFIG, **user_config}
    for key in ("layers", "granulosa_colors", "transcript_color"):
        if key in user_config:
            config[key] = {**DEFAULT_CONFIG[key], **user_config[key]}
    return config


def load_rgba(path):
    """Load a PNG as an (height, width, 4) array of red, green, blue, alpha."""
    return np.array(Image.open(path).convert("RGBA"))


def split_channels(rgba):
    """Split an RGBA image into separate red, green, blue, alpha arrays."""
    red = rgba[..., 0].astype(int)
    green = rgba[..., 1].astype(int)
    blue = rgba[..., 2].astype(int)
    alpha = rgba[..., 3].astype(int)
    return red, green, blue, alpha


def color_mask(rgba, target_rgb, tolerance, alpha_min=ALPHA_MIN):
    """True where a pixel is opaque enough and close enough in color to target_rgb.

    Edge pixels blend toward the
    (transparent) background instead of being the exact target color -- tolerance
    and alpha_min absorb that blending.
    """
    red, green, blue, alpha = split_channels(rgba)
    close_enough = (
        (np.abs(red - target_rgb[0]) <= tolerance)
        & (np.abs(green - target_rgb[1]) <= tolerance)
        & (np.abs(blue - target_rgb[2]) <= tolerance)
    )
    return (alpha >= alpha_min) & close_enough


def find_scale_bar(rgba, scale_bar_um, search_right_fraction=0.5):
    """Measure the exported scale bar (light-gray line, bottom-right) to get pixels-per-micron."""
    red, green, blue, alpha = split_channels(rgba)
    width = rgba.shape[1]
    is_scale_bar_pixel = (
        (alpha == 255)
        & (np.abs(red - 240) < 10) & (np.abs(green - 240) < 10) & (np.abs(blue - 240) < 10)
    )
    is_scale_bar_pixel[:, : int(width * search_right_fraction)] = False  # dodge the inset box on the left
    _, xs = np.where(is_scale_bar_pixel)
    if xs.size == 0:
        raise RuntimeError("Could not auto-detect scale bar; set px_per_um_override in config.")
    scale_bar_length_px = xs.max() - xs.min()
    return scale_bar_length_px / scale_bar_um


def fill_ring(granulosa_layer1_mask, closing_radius_px):
    """
    granulosa_layer1_mask can be a broken outline (cells with small gaps between
    them), not a filled shape. Bridge the gaps, fill the enclosed area,
    then remove the outline itself to leave just the oocyte.
    """
    closed_ring = binary_closing(granulosa_layer1_mask, structure=disk(max(1, round(closing_radius_px))))
    filled_disk = binary_fill_holes(closed_ring)
    oocyte = filled_disk & ~closed_ring

    labeled_regions = label(oocyte)
    if labeled_regions.max() == 0:
        raise RuntimeError("No enclosed region found -- granulosa layer 1 may not close fully in this crop.")
    if labeled_regions.max() > 1:
        largest_label = max(regionprops(labeled_regions), key=lambda region: region.area).label
        oocyte = labeled_regions == largest_label
    return oocyte


def find_dots(rgba, green_dominance_min):
    """Each transcript is a small green dot; return one (row, col) centroid per dot."""
    red, green, blue, alpha = split_channels(rgba)
    is_green_dot = (alpha > 0) & (green - red >= green_dominance_min) & (green - blue >= green_dominance_min)
    labeled_dots = label(is_green_dot)
    return np.array([region.centroid for region in regionprops(labeled_dots)])  # (row, col) i.e. (y, x)


# ---------------------------------------------------------------------------
# Pipeline steps -- run() is just these four, called in order.
# ---------------------------------------------------------------------------

def load_images(input_dir, config):
    annotations_rgba = load_rgba(input_dir / config["layers"]["annotations"])
    transcripts_rgba = load_rgba(input_dir / config["layers"]["transcripts"])
    print(f"  loaded annotations layer: {annotations_rgba.shape[1]}x{annotations_rgba.shape[0]} px")
    print(f"  loaded transcripts layer: {transcripts_rgba.shape[1]}x{transcripts_rgba.shape[0]} px")
    if annotations_rgba.shape[:2] != transcripts_rgba.shape[:2]:
        raise ValueError("annotations and transcripts layers have different dimensions -- exports must be aligned.")
    print("  dimensions match, layers are aligned")
    return annotations_rgba, transcripts_rgba


def build_oocyte_mask(annotations_rgba, config, px_per_um):
    layer1 = config["granulosa_colors"]["layer1"]
    granulosa_layer1_mask = color_mask(annotations_rgba, layer1["rgb"], layer1["tolerance"])
    print(f"  granulosa layer 1 outline: {granulosa_layer1_mask.sum():,} px matched color {layer1['rgb']}")

    closing_radius_px = CLOSING_RADIUS_UM * px_per_um
    oocyte_mask = fill_ring(granulosa_layer1_mask, closing_radius_px)
    oocyte_area_um2 = oocyte_mask.sum() / (px_per_um ** 2)
    print(f"  oocyte mask: {oocyte_mask.sum():,} px enclosed ({oocyte_area_um2:.1f} um^2)")
    return oocyte_mask


def count_transcripts_in_oocyte(transcripts_rgba, oocyte_mask, config):
    dot_positions = find_dots(transcripts_rgba, config["transcript_color"]["green_dominance_min"])
    print(f"  transcript dots detected: {len(dot_positions)}")

    dot_rows = np.clip(dot_positions[:, 0].round().astype(int), 0, oocyte_mask.shape[0] - 1)
    dot_cols = np.clip(dot_positions[:, 1].round().astype(int), 0, oocyte_mask.shape[1] - 1)
    dot_is_in_oocyte = oocyte_mask[dot_rows, dot_cols]
    print(f"  transcript dots inside oocyte: {int(dot_is_in_oocyte.sum())}")
    return dot_positions, dot_is_in_oocyte


def save_results(sample_id, output_dir, annotations_rgba, oocyte_mask, dot_positions, dot_is_in_oocyte, px_per_um):
    oocyte_area_px = int(oocyte_mask.sum())
    oocyte_area_um2 = oocyte_area_px / (px_per_um ** 2)
    transcripts_in_oocyte = int(dot_is_in_oocyte.sum())
    density_per_um2 = transcripts_in_oocyte / oocyte_area_um2 if oocyte_area_um2 else float("nan")

    result = {
        "sample_id": sample_id,
        "transcripts_total_detected": len(dot_positions),
        "transcripts_in_oocyte": transcripts_in_oocyte,
        "oocyte_area_px": oocyte_area_px,
        "oocyte_area_um2": oocyte_area_um2,
        "transcript_density_per_um2": density_per_um2,
        "px_per_um": px_per_um,
    }
    pd.DataFrame([result]).to_csv(output_dir / f"{sample_id}_oocyte_transcript_counts.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.imshow(annotations_rgba)
    for outline in find_contours(oocyte_mask.astype(float), 0.5):
        ax.plot(outline[:, 1], outline[:, 0], color="white", linewidth=1)
    if len(dot_positions):
        outside = dot_positions[~dot_is_in_oocyte]
        inside = dot_positions[dot_is_in_oocyte]
        ax.scatter(outside[:, 1], outside[:, 0], s=4, color="gray", label="outside oocyte")
        ax.scatter(inside[:, 1], inside[:, 0], s=6, color="magenta", label="inside oocyte")
    ax.legend(loc="upper right", fontsize=8, facecolor="black", labelcolor="white")
    ax.axis("off")
    fig.savefig(output_dir / f"{sample_id}_qc_overlay.png", dpi=200, bbox_inches="tight", facecolor="black")
    plt.close(fig)

    print(f"{sample_id}: {transcripts_in_oocyte}/{len(dot_positions)} transcripts in oocyte "
          f"({oocyte_area_um2:.1f} um^2, {density_per_um2:.4f} transcripts/um^2)")


def run(input_dir, output_dir=None, config_path=None):
    config = load_config(config_path) if config_path else dict(DEFAULT_CONFIG)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("1. Loading images")
    annotations_rgba, transcripts_rgba = load_images(input_dir, config)

    print("2. Finding scale (px per micron)")
    px_per_um = config["px_per_um_override"] or find_scale_bar(transcripts_rgba, config["scale_bar_um"])
    print(f"  scale: {px_per_um:.3f} px/um")

    print("3. Building oocyte mask from granulosa layer 1")
    oocyte_mask = build_oocyte_mask(annotations_rgba, config, px_per_um)

    print("4. Finding transcript dots and checking which fall inside the oocyte")
    dot_positions, dot_is_in_oocyte = count_transcripts_in_oocyte(transcripts_rgba, oocyte_mask, config)

    print("5. Saving results")
    save_results(input_dir.name, output_dir, annotations_rgba, oocyte_mask, dot_positions, dot_is_in_oocyte, px_per_um)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", help="Folder with the exported annotations/transcripts PNGs")
    parser.add_argument("--output-dir", help="Where to write results (default: <input-dir>/analysis)")
    parser.add_argument("--config", help="Optional YAML overriding colors/thresholds (see DEFAULT_CONFIG)")
    args = parser.parse_args()

    if not args.input_dir and not args.config:
        parser.error("pass --input-dir, or --config with paths.input_dir set")

    input_dir = args.input_dir
    output_dir = args.output_dir
    if args.config and not input_dir:
        with open(args.config) as f:
            cfg_paths = yaml.safe_load(f).get("paths", {})
        input_dir = cfg_paths.get("input_dir")
        output_dir = output_dir or cfg_paths.get("output_dir")

    run(input_dir, output_dir, args.config)
