"""
Preprocess nnUNet binary output → instance-segmented, ICM-removed label tiffs.

Pipeline per timepoint:
  1. Connected components on binary mask
  2. Identify ICM cluster (largest blob, clearly an outlier) → record centroid, remove
  3. Classify remaining components as single nuclei vs merged blobs using shape
     features: solidity, number of intensity peaks, and EDT concavities (curvature proxy)
  4. Classify single nuclei as TE or ICM using distance-weighted size+intensity thresholds
  5. Split merged blobs using intensity-seeded watershed; classify resulting pieces as TE/ICM
  6. Save *_instances_reclassified.tif

Tune parameters in the CONFIG block below. All thresholds are exposed there.
"""

import glob
import yaml
import numpy as np
import tifffile
from pathlib import Path
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import gaussian

REPO_ROOT   = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# =============================================================================
#  CONFIG — tune these once you have nnUNet output to test against
# =============================================================================

# Paths
NNUNET_DIR = cfg['paths']['nnunet_dir']
RAW_DIR    = cfg['paths']['raw_dir']
OUT_DIR    = cfg['paths']['label_dir']

N_TIMEPOINTS = cfg['microscopy']['n_timepoints']

# Voxel sizes
VX_Z, VX_Y, VX_X = 2.0, 0.208, 0.208
VOX_VOL = VX_Z * VX_Y * VX_X  # µm³ per voxel

# --- ICM cluster detection ---
# The ICM cluster is the single largest connected component. It must be at
# least ICM_SIZE_RATIO times larger than the second-largest to be auto-detected.
ICM_SIZE_RATIO = 2.0

# --- Single vs merged classification ---
# A component is classified as a merged blob if it fails ANY of the three checks:
#
# 1. SOLIDITY: volume / convex_hull_volume. Single nuclei are roughly convex (~0.9+).
#    Merged nuclei have concavities at junctions → lower solidity.
SOLIDITY_THRESHOLD = 0.85   # below this → likely merged

# 2. INTENSITY PEAKS: number of local maxima in raw signal within the component.
#    Single nucleus = 1 peak. Merged = multiple peaks.
PEAK_SIGMA        = 1.5     # Gaussian smoothing sigma before peak detection (voxels)
PEAK_MIN_DISTANCE = 5       # minimum voxel distance between peaks
PEAK_MIN_FRAC     = 0.3     # peak must be >= this fraction of the component's max intensity
MAX_PEAKS_SINGLE  = 1       # components with more peaks than this are flagged as merged

# 3. EDT CONCAVITIES (curvature proxy): local minima in the distance transform
#    within the component reveal pinch points between fused nuclei.
#    We measure the "concavity depth" = (max EDT value - min interior EDT value) / max EDT.
#    A single convex nucleus has a smooth EDT with no interior dip; merged nuclei do.
EDT_CONCAVITY_THRESHOLD = 0.35  # above this → likely merged (tune against your data)

# --- TE/ICM classification (distance-weighted thresholds) ---
# For each nucleus, strictness = exp(-dist_from_ICM / DISTANCE_SCALE_UM)
# strictness=1 (close to ICM) → tight thresholds
# strictness=0 (far from ICM) → lax thresholds
# Nucleus passes as TE if BOTH hold:
#   volume_um3     >= MIN_VOL_TE_BASE + strictness * MIN_VOL_TE_STRICT_ADDON
#   mean_intensity >= MIN_INT_TE_BASE + strictness * MIN_INT_TE_STRICT_ADDON
DISTANCE_SCALE_UM        = 30.0   # µm — e-folding distance for strictness
MIN_VOL_TE_BASE          = 200.0  # µm³
MIN_VOL_TE_STRICT_ADDON  = 400.0  # µm³
MIN_INT_TE_BASE          = 100.0  # AU
MIN_INT_TE_STRICT_ADDON  = 200.0  # AU

# --- Nucleus splitting (intensity-seeded watershed) ---
SPLIT_SIGMA        = 1.5   # Gaussian smoothing sigma (voxels)
SPLIT_MIN_DISTANCE = 5     # minimum voxel distance between seeds
SPLIT_MIN_PEAK_FRAC = 0.1  # seed must be >= this fraction of blob max intensity

# Distance-transform watershed alternative — swap in split_blob() to compare
SPLIT_DT_SIGMA     = 1.0

# =============================================================================


def load_files(nnunet_dir, raw_dir, n):
    nn_files  = sorted(glob.glob(f"{nnunet_dir}*.tif"))[:n]
    raw_files = sorted(glob.glob(f"{raw_dir}Cam_long_*.tif"))[:n]
    assert len(nn_files)  == n, f"Expected {n} nnUNet files, found {len(nn_files)}"
    assert len(raw_files) == n, f"Expected {n} raw files, found {len(raw_files)}"
    return nn_files, raw_files


def find_icm_cluster(props):
    """
    Return (icm_label, icm_centroid_um) for the largest connected component.
    Warns if it is not clearly larger than the second-largest.
    """
    sizes = sorted([(p.area, p.label, p.centroid) for p in props], reverse=True)
    if len(sizes) < 2:
        raise RuntimeError("Fewer than 2 connected components — check binary mask")

    if sizes[0][0] < ICM_SIZE_RATIO * sizes[1][0]:
        print(f"  WARNING: ICM cluster not clearly distinct "
              f"(largest={sizes[0][0]*VOX_VOL:.0f} µm³, "
              f"second={sizes[1][0]*VOX_VOL:.0f} µm³). Inspect manually.")

    icm_label = sizes[0][1]
    c = sizes[0][2]
    icm_centroid_um = np.array([c[0] * VX_Z, c[1] * VX_Y, c[2] * VX_X])
    return icm_label, icm_centroid_um


def count_intensity_peaks(mask, raw_vol):
    """Count local intensity maxima within mask in smoothed raw signal."""
    raw_smooth = gaussian(raw_vol.astype(np.float32), sigma=PEAK_SIGMA)
    raw_smooth[~mask] = 0
    threshold = raw_smooth[mask].max() * PEAK_MIN_FRAC
    peaks = peak_local_max(
        raw_smooth,
        min_distance=PEAK_MIN_DISTANCE,
        threshold_abs=threshold,
        labels=mask,
    )
    return len(peaks)


def edt_concavity(mask):
    """
    Measure concavity depth from the Euclidean distance transform.
    Returns (max_edt - min_interior_edt) / max_edt.
    Interior voxels = those not on the mask boundary (eroded once).
    A perfectly convex shape has one smooth peak; merged nuclei have a dip
    between them giving a high concavity score.
    """
    dt = ndi.distance_transform_edt(mask)
    max_dt = dt.max()
    if max_dt == 0:
        return 0.0
    # Interior = voxels with EDT > 1 voxel (away from surface)
    interior = dt > 1.0
    if not interior.any():
        return 0.0
    min_interior_dt = dt[interior].min()
    return (max_dt - min_interior_dt) / max_dt


def is_merged(prop, mask, raw_vol):
    """
    Return True if the component is likely a merged multi-nucleus blob.
    Uses solidity, intensity peak count, and EDT concavity.
    Flags if ANY criterion suggests merging.
    """
    reasons = []

    # 1. Solidity
    if prop.solidity < SOLIDITY_THRESHOLD:
        reasons.append(f"solidity={prop.solidity:.3f}")

    # 2. Intensity peaks
    n_peaks = count_intensity_peaks(mask, raw_vol)
    if n_peaks > MAX_PEAKS_SINGLE:
        reasons.append(f"peaks={n_peaks}")

    # 3. EDT concavity
    concavity = edt_concavity(mask)
    if concavity > EDT_CONCAVITY_THRESHOLD:
        reasons.append(f"concavity={concavity:.3f}")

    if reasons:
        return True, reasons
    return False, []


def classify_te(props, icm_centroid_um):
    """
    Return set of labels classified as TE using distance-weighted size+intensity thresholds.
    """
    te_labels = set()
    for p in props:
        centroid_um = np.array([
            p.centroid[0] * VX_Z,
            p.centroid[1] * VX_Y,
            p.centroid[2] * VX_X,
        ])
        dist       = np.linalg.norm(centroid_um - icm_centroid_um)
        strictness = np.exp(-dist / DISTANCE_SCALE_UM)

        min_vol = MIN_VOL_TE_BASE + strictness * MIN_VOL_TE_STRICT_ADDON
        min_int = MIN_INT_TE_BASE + strictness * MIN_INT_TE_STRICT_ADDON

        if p.area * VOX_VOL >= min_vol and p.intensity_mean >= min_int:
            te_labels.add(p.label)

    return te_labels


def split_blob(blob_mask, raw_vol):
    """
    Split a merged blob using intensity-seeded watershed on the raw signal.
    Returns a label array (unique int per split nucleus, starting from 1).

    To compare distance-transform watershed, swap the seed-generation block below.
    """
    raw_smooth = gaussian(raw_vol.astype(np.float32), sigma=SPLIT_SIGMA)
    raw_smooth[~blob_mask] = 0

    # --- Intensity-seeded watershed (default) ---
    threshold = raw_smooth[blob_mask].max() * SPLIT_MIN_PEAK_FRAC
    peak_coords = peak_local_max(
        raw_smooth,
        min_distance=SPLIT_MIN_DISTANCE,
        threshold_abs=threshold,
        labels=blob_mask,
    )

    # --- Distance-transform watershed (alternative — comment out above, uncomment below) ---
    # dt = ndi.distance_transform_edt(blob_mask)
    # dt_smooth = gaussian(dt.astype(np.float32), sigma=SPLIT_DT_SIGMA)
    # peak_coords = peak_local_max(dt_smooth, min_distance=SPLIT_MIN_DISTANCE, labels=blob_mask)

    if len(peak_coords) == 0:
        return blob_mask.astype(np.int32)

    seeds = np.zeros(blob_mask.shape, dtype=np.int32)
    for i, coord in enumerate(peak_coords, start=1):
        seeds[tuple(coord)] = i
    seeds_labeled = label(seeds)

    return watershed(-raw_smooth, seeds_labeled, mask=blob_mask)


def process_timepoint(nn_path, raw_path, t):
    binary = tifffile.imread(nn_path)
    raw    = tifffile.imread(raw_path).astype(np.float32)

    assert binary.shape == raw.shape, \
        f"Shape mismatch at t={t}: binary {binary.shape}, raw {raw.shape}"

    binary = (binary > 0)

    # Step 1: Connected components
    cc    = label(binary)
    props = regionprops(cc, intensity_image=raw)

    # Step 2: Find and remove ICM cluster
    icm_label, icm_centroid_um = find_icm_cluster(props)
    print(f"  ICM centroid (µm): z={icm_centroid_um[0]:.1f}, "
          f"y={icm_centroid_um[1]:.1f}, x={icm_centroid_um[2]:.1f}")
    cc[cc == icm_label] = 0

    remaining_props = [p for p in props if p.label != icm_label]
    if not remaining_props:
        print(f"  WARNING: No nuclei remaining after ICM removal at t={t}")
        return np.zeros_like(cc, dtype=np.uint16)

    # Step 3: Classify each component as single nucleus or merged blob
    single_props = []
    blob_props   = []
    for p in remaining_props:
        mask = (cc == p.label)
        merged, reasons = is_merged(p, mask, raw)
        if merged:
            blob_props.append(p)
        else:
            single_props.append(p)

    print(f"  {len(single_props)} single nuclei, {len(blob_props)} merged blobs to split")

    # Step 4: Classify single nuclei as TE or ICM
    te_labels = classify_te(single_props, icm_centroid_um)
    print(f"  TE single: {len(te_labels)}, ICM/rejected single: "
          f"{len(single_props) - len(te_labels)}")

    # Step 5: Build output label volume with sequential instance IDs
    out        = np.zeros_like(cc, dtype=np.uint16)
    next_label = 1

    for p in single_props:
        if p.label in te_labels:
            out[cc == p.label] = next_label
            next_label += 1

    for p in blob_props:
        blob_mask  = (cc == p.label)
        split      = split_blob(blob_mask, raw)
        split_props = regionprops(split, intensity_image=raw)
        te_split   = classify_te(split_props, icm_centroid_um)
        for sp in split_props:
            if sp.label in te_split:
                out[blob_mask & (split == sp.label)] = next_label
                next_label += 1

    print(f"  → {next_label - 1} final TE instances")
    return out


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    nn_files, raw_files = load_files(NNUNET_DIR, RAW_DIR, N_TIMEPOINTS)

    for t, (nn_path, raw_path) in enumerate(zip(nn_files, raw_files)):
        stem     = Path(nn_path).stem.replace('.nii', '')
        out_path = Path(OUT_DIR) / f"{stem}_instances_reclassified.tif"
        print(f"\nt={t:03d}  {Path(nn_path).name}")

        result = process_timepoint(nn_path, raw_path, t)
        tifffile.imwrite(str(out_path), result)
        print(f"  Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
