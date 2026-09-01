"""
Preprocess nnUNet binary output → instance-segmented, ICM-removed label tiffs.

Pipeline per timepoint:
  1. Connected components on binary mask
  2. Split every component with intensity-seeded watershed. A component with
     only one real intensity peak comes back as a single unsplit piece, so
     this replaces a separate "is this merged?" classification step - one
     code path handles both single nuclei and touching clumps.
  3. Merge any resulting piece smaller than MIN_PIECE_VOL_UM3 into whichever
     neighbor it shares the most boundary with - combo (intensity+EDT) seeding
     sometimes places more seeds than there are real nuclei (confirmed on real
     output: a 5-seed component that should have been ~3 nuclei, 2 seeds
     claiming zero watershed territory and a third claiming only a thin
     325um3 sliver of a neighbor's true extent - visible as a spurious tiny
     fragment, not a real nucleus).
  4. Classify each resulting piece as ICM or TE by local neighbor density
     (nuclei within ICM_NEIGHBOR_RADIUS_UM of each other) - ICM nuclei pack
     in a compact 3D cluster, TE nuclei spread across a thin curved shell, so
     ICM nuclei have noticeably more neighbors at the same radius.
  5. Save TE-only *_instances_reclassified.tif

Splitting uses two complementary signals, since neither alone was enough:

- Intensity-peak seeding + intensity-based watershed flooding was grid-searched
  first against hand-annotated ground truth for t=60 of this dataset
  (/mnt/md0/elysse/nnUNet/training files 260427/labels/Cam_long_00060_cropped_label.tif,
  Z-offset 44 into that 91-slice label to align with this run's 47-slice crop):
  61/84 instances matched at IoU>=0.5, mean IoU 0.87 among matches - already
  better than either pretrained BlastoSPIM StarDist model (late_blastocyst or
  early_embryo) tested against the same ground truth, despite an exact
  voxel-size match to BlastoSPIM's published training spacing.
- But intensity alone under-splits low-contrast touching pairs (real nuclei
  with too little brightness difference between them to register as two
  intensity peaks) and its watershed boundary, even when the split count is
  right, tracks brightness rather than the true geometric membrane - visually
  confirmed wrong against ground truth despite "matching" it by bulk overlap.
  Adding EDT-based (shape/distance-transform) seeds recovers the low-contrast
  merges intensity misses, and flooding on a blend of intensity + EDT distance
  (FLOOD_EDT_WEIGHT) places the boundary at the true geometric pinch-point
  rather than wherever intensity happens to dip. Final result: 75/84 matched,
  mean volume error 7.3% on matches (down from ~26% with intensity-only
  flooding) - grid-searched on the same t=60 ground truth (MIN_SEED_SEP_UM was
  the dominant lever; EDT_PEAK_THRESH_FRAC barely mattered in 0.25-0.4).

An earlier version of this pipeline used a solidity/peak-count/EDT-concavity
classifier to decide which components needed splitting at all; that approach
was dropped because none of the three criteria proved reliable here (in
particular, EDT concavity is not scale-invariant under a fixed-um interior
margin, and this dataset's nuclei are small enough that it never cleanly
separated single from merged - see git history for details). Always attempting
a split and letting single-peak components come back unsplit replaced it.

ICM identification (ICM_NEIGHBOR_RADIUS_UM, ICM_NEIGHBOR_THRESHOLD): an earlier
version of this pipeline identified "the ICM" as the single largest connected
component in the binary mask. That was flatly wrong at t=60 - the largest
component (23491 um3) turned out to be 55um away from the true ICM location
(confirmed via hand-labeled reference points: 5 known-TE, 4 known-ICM nuclei,
picked in napari), and was actually a large mass of tightly-touching TE cells;
the real ICM core was the *third*-largest component, plus several smaller,
separate (non-touching) components nearby - consistent with ICM nuclei not
necessarily forming one connected blob. Local neighbor density, evaluated at
those same 9 reference points, cleanly separated them (TE: 5-9 neighbors within
25um; ICM: 11-13) with no overlap, and doesn't depend on finding "the ICM blob"
as a single connected component at all. ICM_NEIGHBOR_THRESHOLD=10 sits in that
gap. This is fit from only 9 points at one timepoint - worth re-checking against
more reference points before trusting it across the full timecourse, especially
at very early/late timepoints where ICM/TE compaction may look different.

Tune parameters in the CONFIG block below.
"""

import glob
import yaml
import numpy as np
import tifffile
from pathlib import Path
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.filters import gaussian

REPO_ROOT   = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset001_implantation_250914_stack5.yaml'  # edit to switch dataset
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# =============================================================================
#  CONFIG — tune these once you have nnUNet output to test against
# =============================================================================

# Paths
NNUNET_DIR  = cfg['paths']['nnunet_dir']
NNUNET_GLOB = cfg['paths'].get('nnunet_glob', '*.tif')
RAW_DIR     = cfg['paths']['raw_dir']
RAW_GLOB    = cfg['paths'].get('raw_glob', 'Cam_long_*.tif')
OUT_DIR     = cfg['paths']['label_dir']

N_TIMEPOINTS = cfg['microscopy']['n_timepoints']

# Voxel sizes
VX_Z, VX_Y, VX_X = 2.0, 0.208, 0.208
VOX_VOL  = VX_Z * VX_Y * VX_X  # µm³ per voxel
SPACING  = (VX_Z, VX_Y, VX_X)  # for spacing-aware distance/blur ops below

# Z is ~10x coarser than X/Y in this dataset. Anything computing "distance" or
# "blur radius" in raw voxel counts (rather than physical µm, via `sampling=`
# or a per-axis sigma/box size) will be wildly wrong - e.g. a 5-voxel isotropic
# ball is 10µm tall in Z but only ~1µm wide in X/Y. Splitting below is
# spacing-aware for this reason.

# --- Nucleus splitting (combo intensity+EDT seeding, blended watershed
#     flooding, always attempted) ---
# Intensity peak params grid-searched first: sigma in {1,1.5,2,3}um, min_distance
# in {3,4,5,6,8}um, threshold_frac in {0.05,0.1,0.15,0.2,0.3} - threshold_frac
# turned out not to matter at all in that range (identical results throughout).
SPLIT_SIGMA_UM        = 1.5   # Gaussian smoothing sigma before intensity peak detection (µm)
SPLIT_MIN_DISTANCE_UM = 3.5   # minimum physical distance between intensity seeds (µm)
SPLIT_THRESH_FRAC     = 0.1   # intensity seed must be >= this fraction of component max intensity

# EDT seeds recover low-contrast touching pairs intensity seeding misses.
# EDT_PEAK_THRESH_FRAC barely mattered in {0.25,0.3,0.35,0.4} (identical
# results); MIN_SEED_SEP_UM was the dominant lever - smaller recovers more
# real splits but rapidly adds spurious fragments (e.g. at 3.5um: 76/84
# matched but 37 excess unmatched fragments, vs at 5.0um: 75/84 matched with
# only 20 excess - a much better trade for 1 fewer match).
EDT_SIGMA_UM          = 1.0   # Gaussian smoothing sigma before EDT peak detection (µm)
EDT_PEAK_THRESH_FRAC  = 0.3   # EDT seed must be >= this fraction of component max EDT value
MIN_SEED_SEP_UM       = 5.0   # an EDT seed must be at least this far (µm) from every
                               # intensity seed to be added - avoids near-duplicate seeds

# Watershed floods on a weighted blend of intensity and EDT distance, not
# intensity alone - pure intensity flooding tracks brightness, which can
# displace the boundary from the true geometric pinch-point between two
# touching nuclei of different brightness. FLOOD_EDT_WEIGHT=0.35 was the
# best of {0, 0.35 ("blend"), 1.0 ("edt")} tested: pure EDT flooding (1.0)
# gave the highest per-match IoU (0.92) but the fewest matches (54/84,
# brittle to mask shape noise); pure intensity (0) gave the most matches
# (61/84) but worse boundaries (IoU 0.87); 0.35 kept both close to their
# respective best (0.92 IoU, 57-76/84 matched depending on seeding).
FLOOD_EDT_WEIGHT      = 0.35

# The raw (unsmoothed) EDT field has mathematically sharp ridges - equidistant
# points between two seeds form an exact flat plane (a Voronoi facet), which
# shows up as visibly wrong straight-line/right-angle cuts through otherwise
# round nuclei (confirmed on real output: t=71 label 41, full-width flat cut
# at z=36-38 with the unsmoothed field, replaced by a smooth curve at the
# same z-slices once the field is Gaussian-smoothed before flooding). Smoothing
# only the field used for FLOODING here - EDT_SIGMA_UM above is separately
# used for peak detection and needs to stay sharp for that. sigma=0.5um was
# enough to remove the visible artifact and also nudged the t=60 ground truth
# score slightly better (76/84 matched vs 75/84, volume error 7.8% vs 7.3%).
FLOOD_EDT_SIGMA_UM    = 0.5

# Combo seeding sometimes places more seeds than there are real nuclei in a
# component (confirmed on real output: watershed left 2 of 5 seeds with zero
# assigned territory, and a third with just a 325um3 sliver of what should
# have been a neighboring piece - a spurious fragment, not a real nucleus).
# Any piece smaller than this gets merged into whichever piece it shares the
# most boundary with, iterating since a merge can enable further merges.
# Grid-searched against t=60 ground truth over {0,150,...,500}: matched count
# held exactly flat at 76/84 up to 400 while excess fragments dropped from 13
# to 0 beyond real matches (95->82 total pieces) and volume error improved
# slightly (7.8%->7.4%) - free precision, no recall cost. Only started
# costing real matches at 500 (76->73), so 400 keeps real margin below that.
MIN_PIECE_VOL_UM3     = 400.0

# --- ICM/TE classification (local neighbor density, relative to each
#     timepoint's own median) ---
# ICM nuclei are packed in a compact 3D cluster; TE nuclei are spread over a
# thin curved shell, so at a fixed radius, ICM nuclei have more neighbors.
# Classification is by RATIO to that timepoint's own median neighbor count,
# not an absolute count - the embryo visibly flattens against the dish over
# this timecourse, which mechanically raises local density everywhere (not
# just in the ICM) as development proceeds. Confirmed directly: median
# neighbor count within ICM_NEIGHBOR_RADIUS_UM was 14 at t=60 but 21 at
# t=90, a ~50% shift in baseline density alone. An absolute-count threshold
# fit at t=60 (even one refit after a splitting-parameter change, see git
# history) put 68-74% of nuclei in "ICM" at t=90/t=120 - implausible, and
# traced directly to that baseline shift, not real biology. The ratio
# threshold below gives a consistent ICM fraction at both t=60 (37%) and
# t=90 (34%) and passes all 9 hand-labeled reference points (5 TE, 4 ICM,
# picked in napari at t=60 - see module docstring) with a 4x margin (lowest
# ICM ratio 1.21, threshold 1.2, highest TE ratio 0.93).
ICM_NEIGHBOR_RADIUS_UM   = 30.0  # neighbor-counting radius (µm)
ICM_NEIGHBOR_RATIO_THRESH = 1.2  # >= this many times the timepoint's median neighbor count → ICM

# =============================================================================


def anisotropic_sigma(sigma_um, spacing=SPACING):
    """Per-axis voxel sigma for skimage.filters.gaussian equivalent to a
    physical blur radius of `sigma_um`, given anisotropic voxel spacing."""
    return tuple(sigma_um / s for s in spacing)


def box_size_vox(min_distance_um, spacing=SPACING):
    """Per-axis odd box size (voxels) for a max-filter neighborhood whose
    physical half-width is `min_distance_um` on each axis."""
    return tuple(max(1, int(round(min_distance_um / s)) * 2 + 1) for s in spacing)


SPLIT_BOX_VOX = box_size_vox(SPLIT_MIN_DISTANCE_UM)

# Per-axis crop padding for padded_slice() below, large enough that a
# component processed in its local crop gives identical results to
# processing the full volume: must exceed both the Gaussian kernel radius
# (~4*sigma, scipy's truncate default) and the max-filter box half-width, in
# voxels, per axis.
PAD_VOX = tuple(
    int(np.ceil(max(
        4 * anisotropic_sigma(SPLIT_SIGMA_UM)[i],
        4 * anisotropic_sigma(EDT_SIGMA_UM)[i],
        SPLIT_BOX_VOX[i] // 2,
    ))) + 2  # safety margin
    for i in range(3)
)


def vox_dist_um(a, b, spacing=SPACING):
    """Physical distance (µm) between two voxel-index coordinates."""
    a, b = np.array(a), np.array(b)
    return np.linalg.norm((a - b) * np.array(spacing))


def load_files(nnunet_dir, raw_dir, n, nnunet_glob=NNUNET_GLOB, raw_glob=RAW_GLOB):
    nn_files  = sorted(glob.glob(f"{nnunet_dir}{nnunet_glob}"))[:n]
    raw_files = sorted(glob.glob(f"{raw_dir}{raw_glob}"))[:n]
    assert len(nn_files)  == n, f"Expected {n} nnUNet files, found {len(nn_files)}"
    assert len(raw_files) == n, f"Expected {n} raw files, found {len(raw_files)}"
    return nn_files, raw_files


def padded_slice(prop, shape, pad=PAD_VOX):
    """
    Bounding-box slice for a regionprop, padded per-axis (clipped to array
    bounds) by enough voxels that cropped results match full-volume results
    exactly - this only exists for speed, not to change behavior. Default
    pad is PAD_VOX, sized from the actual smoothing/box-filter radii above.
    """
    return tuple(
        slice(max(s.start - p, 0), min(s.stop + p, dim))
        for s, p, dim in zip(prop.slice, pad, shape)
    )


def fast_peaks(image, mask, size_vox, threshold):
    """
    Local intensity maxima within mask, via scipy's fast separable
    maximum_filter (size=, not footprint=). Passing an explicit anisotropic
    footprint array to skimage's peak_local_max forces its slow generic
    (non-separable) code path, whose cost scales with footprint volume -
    prohibitively slow once min_distance is more than a few µm on this
    dataset's finely-sampled X/Y grid (confirmed empirically: >30s for a
    single component at min_distance=5um vs <0.1s here for the same case).
    Tied-value plateaus are merged to one peak (their centroid) each.
    """
    maxf = ndi.maximum_filter(image, size=size_vox, mode='constant')
    is_peak = (image == maxf) & mask & (image >= threshold)
    lbl_peaks, n = ndi.label(is_peak)
    if n == 0:
        return []
    coms = ndi.center_of_mass(is_peak, lbl_peaks, range(1, n + 1))
    return [tuple(int(round(c)) for c in com) for com in coms]


def merge_small_pieces(split, min_vol_um3):
    """
    Merge any watershed piece smaller than min_vol_um3 into whichever
    neighboring piece it shares the most boundary surface with. Iterates
    since a merge can enlarge a piece enough to absorb another small
    neighbor, or free up a merge that was previously blocked.
    Relabels to consecutive ids afterward. Operates on a local (already
    cropped) split array, same convention as the rest of this module.
    """
    changed = True
    while changed:
        changed = False
        n = split.max()
        if n <= 1:
            break
        vols_vox = ndi.sum(np.ones_like(split), split, range(1, n + 1))
        for pid in range(1, n + 1):
            vol_vox = vols_vox[pid - 1]
            if vol_vox == 0 or vol_vox * VOX_VOL >= min_vol_um3:
                continue
            piece_mask = (split == pid)
            dilated = ndi.binary_dilation(piece_mask, iterations=2)
            neighbor_vals, neighbor_counts = np.unique(split[dilated & ~piece_mask], return_counts=True)
            keep = neighbor_vals > 0
            neighbor_vals, neighbor_counts = neighbor_vals[keep], neighbor_counts[keep]
            if len(neighbor_vals) == 0:
                continue  # isolated piece with no neighbor to merge into - leave as-is
            best_neighbor = neighbor_vals[np.argmax(neighbor_counts)]
            split[piece_mask] = best_neighbor
            changed = True
            break  # restart scan - volumes/adjacency changed
    remaining = np.unique(split)
    remaining = remaining[remaining != 0]
    relabel_map = np.zeros(split.max() + 1, dtype=np.int32)
    for i, v in enumerate(remaining, start=1):
        relabel_map[v] = i
    return relabel_map[split]


def split_component(mask, raw_vol):
    """
    Split a connected component using combo (intensity + EDT) seeding and
    blended (intensity + EDT distance) watershed flooding. A component with
    only one seed comes back unsplit (single piece), so this is always
    applied - no separate single-vs-merged classification.

    Intensity seeds alone miss low-contrast touching pairs (real nuclei too
    similar in brightness to register as two peaks); EDT seeds (local maxima
    of the distance transform) catch those via shape alone, so seeds are the
    union of both, deduplicated by MIN_SEED_SEP_UM. Flooding then descends a
    blend of both surfaces (FLOOD_EDT_WEIGHT) so the boundary lands near the
    true geometric pinch-point rather than tracking brightness alone.
    Returns a label array (unique int per piece, starting from 1).
    """
    raw_smooth = gaussian(raw_vol.astype(np.float32), sigma=anisotropic_sigma(SPLIT_SIGMA_UM))
    raw_smooth[~mask] = 0

    if raw_smooth[mask].max() <= 0:
        return mask.astype(np.int32)

    int_threshold = raw_smooth[mask].max() * SPLIT_THRESH_FRAC
    int_peaks = fast_peaks(raw_smooth, mask, SPLIT_BOX_VOX, int_threshold)

    dt = ndi.distance_transform_edt(mask, sampling=SPACING)
    dt_smooth = gaussian(dt.astype(np.float32), sigma=anisotropic_sigma(EDT_SIGMA_UM))
    dt_smooth[~mask] = 0
    dt_threshold = dt_smooth[mask].max() * EDT_PEAK_THRESH_FRAC
    dt_peaks = fast_peaks(dt_smooth, mask, SPLIT_BOX_VOX, dt_threshold)

    peak_coords = list(int_peaks)
    for p in dt_peaks:
        if all(vox_dist_um(p, q) > MIN_SEED_SEP_UM for q in peak_coords):
            peak_coords.append(p)

    if len(peak_coords) <= 1:
        return mask.astype(np.int32)

    seeds = np.zeros(mask.shape, dtype=np.int32)
    for i, coord in enumerate(peak_coords, start=1):
        seeds[coord] = i
    seeds_labeled = label(seeds)

    # Smoothed separately from `dt` above (which stays sharp for peak
    # detection) - the raw EDT field has exact flat ridges between
    # equidistant seeds that read as visibly wrong straight cuts through
    # round nuclei once used as a flooding surface. See FLOOD_EDT_SIGMA_UM.
    dt_flood = gaussian(dt.astype(np.float32), sigma=anisotropic_sigma(FLOOD_EDT_SIGMA_UM))
    dt_flood[~mask] = 0

    dt_norm  = dt_flood / (dt_flood.max() + 1e-9)
    int_norm = raw_smooth / (raw_smooth.max() + 1e-9)
    cost = -(FLOOD_EDT_WEIGHT * dt_norm + (1 - FLOOD_EDT_WEIGHT) * int_norm)

    split = watershed(cost, seeds_labeled, mask=mask)
    if MIN_PIECE_VOL_UM3 > 0:
        split = merge_small_pieces(split, MIN_PIECE_VOL_UM3)
    return split


def process_timepoint(nn_path, raw_path, t):
    binary = tifffile.imread(nn_path)
    raw    = tifffile.imread(raw_path).astype(np.float32)

    assert binary.shape == raw.shape, \
        f"Shape mismatch at t={t}: binary {binary.shape}, raw {raw.shape}"

    binary = (binary > 0)

    # Step 1: Connected components
    cc    = label(binary)
    props = regionprops(cc, intensity_image=raw)

    # Step 2: Split every component (checked on a padded local crop, not the
    # full volume - components are tiny relative to the ~1400x1400xZ frame).
    # No ICM special-casing here - ICM/TE identity is decided in step 3,
    # after all pieces exist, using their positions relative to each other.
    pieces = []  # dicts: label_id, slice, local_mask, split_mask_id, centroid_um, out_slice
    next_label = 1
    for p in props:
        sl         = padded_slice(p, cc.shape)
        local_mask = (cc[sl] == p.label)
        split      = split_component(local_mask, raw[sl])
        split_props = regionprops(split, intensity_image=raw[sl])
        offset_vox  = tuple(s.start for s in sl)
        for sp in split_props:
            centroid_um = np.array([
                (sp.centroid[0] + offset_vox[0]) * VX_Z,
                (sp.centroid[1] + offset_vox[1]) * VX_Y,
                (sp.centroid[2] + offset_vox[2]) * VX_X,
            ])
            pieces.append(dict(
                id=next_label, sl=sl, piece_mask=(split == sp.label), local_mask=local_mask,
                centroid_um=centroid_um,
            ))
            next_label += 1

    if not pieces:
        print(f"  WARNING: No nuclei found at t={t}")
        return np.zeros_like(cc, dtype=np.uint16)

    # Step 3: Classify ICM vs TE by local neighbor density, relative to this
    # timepoint's own median (not an absolute count - see CONFIG comment on
    # ICM_NEIGHBOR_RATIO_THRESH for why: the embryo's overall compactness
    # changes over the timecourse, e.g. flattening against the dish, which
    # shifts the density baseline for every nucleus, not just the ICM).
    coords = np.array([pc['centroid_um'] for pc in pieces])
    tree = cKDTree(coords)
    nbr_counts = tree.query_ball_point(coords, r=ICM_NEIGHBOR_RADIUS_UM, return_length=True) - 1
    median_nbrs = max(np.median(nbr_counts), 1)
    nbr_ratio = nbr_counts / median_nbrs
    n_icm = int((nbr_ratio >= ICM_NEIGHBOR_RATIO_THRESH).sum())

    # Step 4: Build output label volume, TE only
    out        = np.zeros_like(cc, dtype=np.uint16)
    next_out_label = 1
    for pc, ratio in zip(pieces, nbr_ratio):
        if ratio >= ICM_NEIGHBOR_RATIO_THRESH:
            continue  # ICM - excluded from output
        out[pc['sl']][pc['piece_mask']] = next_out_label
        next_out_label += 1

    print(f"  {len(props)} components → {len(pieces)} split pieces (median nbrs={median_nbrs:.0f}) → "
          f"{n_icm} ICM (excluded), {next_out_label - 1} final TE instances")
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
