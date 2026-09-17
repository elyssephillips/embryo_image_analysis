"""
Preprocess Dataset003 (3-class, eroded-instance-trained) nnUNet output →
instance-segmented, ICM-removed label tiffs.

Unlike 00_preprocess_segmentation.py (Dataset001, binary output, no instance
awareness at all), this model was trained on contact-layer-separated instances
(prepare_nnunet.py's separate_touching_instances: the 1-voxel interface between
any two touching hand-labeled instances is zeroed before binarizing, on
whichever axis the contact occurs - Z, Y, or X). So instance identity should
already mostly exist in the raw prediction as disconnected components, and
ICM/TE identity is a direct network output (class 2 vs class 1) instead of the
post-hoc neighbor-density heuristic the Dataset001 pipeline needed. In
principle this collapses the old watershed/seeding/flooding reconstruction
(see segmentation_notes.md) down to:

  1. Connected components on the foreground (class 1 or 2).
  2. Per instance, majority-vote ICM vs TE from the class channel.
  3. Save TE-only *_instances_reclassified.tif (ICM excluded), matching the
     Dataset001 pipeline's output contract so downstream 01_extract_features.py
     etc. don't need to change.

In practice step 1 needed one real lever: checked directly against t=60
ground truth (see EXTRA_ERODE_UM below), the network's own contact-layer gap
alone (plain connected components, no extra erosion) only separated 54/84
instances correctly - confirmed visually as real residual merges, some in Z,
some in XY, not just eyeballing noise. Eroding the foreground a bit further
before taking connected components, then growing each resulting piece back
out to its true (pre-erosion) extent with skimage.segmentation.expand_labels
(clipped to the original foreground so growth can't invent volume beyond what
the network actually predicted as nuclei), recovered most of the gap: 76/84
at a 1.5um erosion. expand_labels assigns each background voxel to its
*nearest* label within the given distance - unlike dilation, two components a
few voxels apart don't get re-fused, so this works as a real split, not just
a blur. Diminishing returns past 1.5um and a real cost to thin instances at
some larger radius not yet found - see CONFIG comment for the exact numbers
and re-check against more than one timepoint before trusting this further.

No raw-intensity step is needed here at all (no watershed) - the previous
pipeline's intensity/EDT seeding existed only to reconstruct instance identity
that the binary Dataset001 model never had; Dataset003 already encodes it.

Tune parameters in the CONFIG block below.
"""

import glob
import yaml
import numpy as np
import tifffile
from pathlib import Path
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.segmentation import expand_labels

REPO_ROOT   = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / 'configs' / 'tracking' / 'dataset003_icm_te_250914_stack5.yaml'  # edit to switch dataset
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# =============================================================================
#  CONFIG
# =============================================================================

PRED_DIR  = cfg['paths']['nnunet_dir']
PRED_GLOB = cfg['paths'].get('nnunet_glob', '*.tif')
RAW_DIR   = cfg['paths']['raw_dir']
RAW_GLOB  = cfg['paths'].get('raw_glob', '*.tif')
OUT_DIR   = cfg['paths']['label_dir']

N_TIMEPOINTS = cfg['microscopy']['n_timepoints']

VX_Z, VX_Y, VX_X = cfg['microscopy']['voxel_size_zyx']
VOX_VOL  = VX_Z * VX_Y * VX_X  # µm³ per voxel
SPACING  = (VX_Z, VX_Y, VX_X)

# --- Instancing ---
# Checked directly against the same t=60 hand-annotated ground truth used to
# tune the old Dataset001 pipeline (same physical stack, so it applies here
# too - 84 instances, /mnt/md0/elysse/nnUNet/training files 260427/labels/
# Cam_long_00060_cropped_label.tif, Z-offset 44 into its 91-slice label to
# align with this crop's 47 slices): plain connected-components on the raw
# prediction (erode=0, i.e. trusting the network's own trained-in
# contact-layer gap alone) only matched 54/84 at IoU>=0.5 - confirms the
# visually-observed residual Z/XY merges are real, not just eyeballing noise,
# and that the network's own gap under-separates on its own. A small extra
# erosion before connected components recovers most of the gap: 0.5um jumped
# to 73/84, 1.5um reached 76/84 (mean IoU 0.831) - matching the old pipeline's
# best result despite doing far less work. Diminishing returns past 1.5um
# (2.0um: still 76/84, slightly lower mean IoU) - so 1.5um is the current
# default, not 0. Only a one-timepoint check so far (see segmentation_notes.md
# for the validation-habit rationale) - worth re-checking against another
# annotated timepoint before treating this as final.
EXTRA_ERODE_UM = 1.5   # spacing-aware (physical µm) erosion radius before CC, 0 to disable

# How far to grow each eroded piece back out toward its true boundary.
# Should be >= EXTRA_ERODE_UM (plus a small margin) so erosion doesn't
# permanently cost real volume - expand_labels won't grow past EXPAND_UM or
# past the original foreground mask (see clip step below), so setting this
# generously high costs nothing extra as long as EXTRA_ERODE_UM > 0. 3.0 and
# 4.0 gave identical results in the t=60 check above, so this isn't sensitive.
EXPAND_UM = 3.0

# --- ICM/TE classification ---
# An instance is only called ICM if at least this fraction of its voxels are
# network class-2; anything else (any real TE presence) is called TE. This is
# deliberately not a 50/50 majority vote: dropping a real TE nucleus by
# misclassifying it ICM silently removes a real cell from every downstream
# analysis, while wrongly keeping a real ICM nucleus in the TE output is just
# one noisy point - the two error types are not equally costly, so the
# threshold shouldn't treat them as if they were. Checked directly against
# the t=60 ICM/TE ground truth (/mnt/md0/elysse/nnUNet/training files 260904/
# te labels/Cam_long_00060_cropped_label.icm_te.json): every matched instance
# at that timepoint was either 100% or 0% class-2 (no partially-mixed
# instances at all), so majority-vote and this stricter rule score identically
# there (0/60 TE misclassified, 0/21 ICM misclassified either way) - this
# timepoint can't demonstrate a benefit, only that the stricter rule costs
# nothing on it. The real payoff is expected on instances where the erosion/
# expand_labels step in instance_from_prediction pulls in a few voxels of a
# neighboring instance's class near an ICM/TE boundary - re-check this
# threshold once a timepoint with real mixed-class instances is found.
ICM_MIN_FRAC = 1.0

# Any final instance smaller than this is almost certainly a stray fragment
# (a nick left by erosion, or noise in the raw prediction), not a real
# nucleus - drop it rather than counting it as a spurious tiny "cell".
MIN_INSTANCE_VOL_UM3 = 100.0

# QC only: flag instances more than this many times the timepoint's median
# instance volume as likely still-merged, so visual spot-checking (napari) can
# be targeted at the worst offenders instead of scrolling every timepoint.
MERGE_FLAG_VOLUME_RATIO = 2.5

# =============================================================================


def instance_from_prediction(pred_fg, erode_um, spacing=SPACING):
    """
    Connected-components instancing with optional extra erosion + expand-back.
    pred_fg: boolean foreground mask (prediction > 0) for one timepoint.
    Returns an int32 label array, same shape as pred_fg.

    Erosion is done by thresholding the (spacing-aware) interior distance
    transform rather than a structuring element sized in voxels - Z is ~10x
    coarser than X/Y here (2.0 vs 0.208 um/voxel, see 00_preprocess_segmentation.py's
    module docstring for the anisotropic bugs a voxel-count footprint caused
    previously), so a voxel-radius ball would erode ~10x too far in Z for the
    same physical distance in X/Y. distance_transform_edt(sampling=spacing)
    gives true physical distance to the nearest background voxel on every
    axis, so thresholding it by erode_um is correct regardless of anisotropy.
    """
    if erode_um > 0:
        interior_dist = ndi.distance_transform_edt(pred_fg, sampling=spacing)
        seed_mask = interior_dist > erode_um
    else:
        seed_mask = pred_fg

    cc = label(seed_mask)
    if cc.max() == 0:
        return cc.astype(np.int32)

    if erode_um > 0:
        expanded = expand_labels(cc, distance=EXPAND_UM, spacing=spacing)
        expanded[~pred_fg] = 0  # never invent volume beyond the network's own prediction
        return expanded.astype(np.int32)

    return cc.astype(np.int32)


def classify_and_filter(instances, pred, min_vol_um3=MIN_INSTANCE_VOL_UM3, icm_min_frac=ICM_MIN_FRAC):
    """
    Per instance: ICM (class 2) vs TE (class 1) from the original prediction -
    ICM only if >= icm_min_frac of the instance's voxels are class-2, TE
    otherwise (see ICM_MIN_FRAC for why this isn't a 50/50 majority vote) -
    and drop anything under min_vol_um3. Returns (te_only_relabeled,
    all_relabeled, class_map, n_dropped, frac_icm_map) where class_map maps
    the all_relabeled ids to 'ICM'/'TE' and frac_icm_map to the raw class-2
    voxel fraction (for QC - see flag_ambiguous_classification).
    """
    ids = np.unique(instances)
    ids = ids[ids != 0]
    if len(ids) == 0:
        empty = np.zeros_like(instances, dtype=np.uint16)
        return empty, empty, {}, 0, {}

    vols_vox   = ndi.sum(np.ones_like(instances), instances, ids)
    n_icm_vox  = ndi.sum(pred == 2, instances, ids)
    frac_icm   = n_icm_vox / vols_vox

    keep_mask = (vols_vox * VOX_VOL) >= min_vol_um3

    all_out = np.zeros_like(instances, dtype=np.uint16)
    te_out  = np.zeros_like(instances, dtype=np.uint16)
    class_map, frac_icm_map = {}, {}
    next_all_id, next_te_id = 1, 1
    for iid, frac in zip(ids[keep_mask], frac_icm[keep_mask]):
        is_icm = frac >= icm_min_frac
        mask = (instances == iid)
        all_out[mask] = next_all_id
        class_map[next_all_id] = 'ICM' if is_icm else 'TE'
        frac_icm_map[next_all_id] = float(frac)
        next_all_id += 1
        if not is_icm:
            te_out[mask] = next_te_id
            next_te_id += 1

    n_dropped = int((~keep_mask).sum())
    return te_out, all_out, class_map, n_dropped, frac_icm_map


def flag_ambiguous_classification(all_labels, raw, class_map, frac_icm_map, lo=0.05, hi=0.95):
    """
    QC only: instances whose class-2 voxel fraction falls strictly between lo
    and hi are the ones the ICM_MIN_FRAC threshold actually has to make a call
    on (see that CONFIG comment - the t=60 check found none, so this is
    speculative until a timepoint with real mixed instances turns one up).
    For each, also reports mean raw intensity and elongation (major/minor
    axis ratio from the inertia tensor: for a solid ellipsoid with semi-axes
    a>=b>=c, a^2 ~ lam1+lam2-lam3 and c^2 ~ lam2+lam3-lam1 where lam1>=lam2>=lam3
    are skimage's inertia_tensor_eigvals, returned in *decreasing* order - a
    plain elongation = major/minor axis ratio (1.0 = sphere)
    01_extract_features.py assumed *increasing* order instead, which silently
    inverts its elongation column (spherical nuclei read >1, elongated ones
    read <1) - not fixed here, flag for the user) as candidate secondary
    signals - not applied as a rule anywhere yet, since neither has been
    checked against ground truth the way ICM_MIN_FRAC and EXTRA_ERODE_UM were.
    Print/inspect these first; only promote one to an actual override once it
    demonstrably separates misclassified cases on a real ambiguous example.
    """
    ambiguous = [iid for iid, f in frac_icm_map.items() if lo < f < hi]
    if not ambiguous:
        return []
    props = {p.label: p for p in regionprops(all_labels, intensity_image=raw)}
    rows = []
    for iid in ambiguous:
        p = props.get(iid)
        if p is None:
            continue
        lam = np.array(p.inertia_tensor_eigvals)  # skimage: decreasing order (lam1>=lam2>=lam3)
        a2 = max(lam[0] + lam[1] - lam[2], 0.0)  # ~ longest semi-axis^2
        c2 = max(lam[1] + lam[2] - lam[0], 0.0)  # ~ shortest semi-axis^2
        elongation = np.sqrt(a2 / c2) if c2 > 0 else float('nan')
        rows.append(dict(
            id=iid, called=class_map[iid], frac_icm=frac_icm_map[iid],
            mean_intensity=float(p.intensity_mean), elongation=float(elongation),
        ))
    return rows


def flag_likely_merges(all_labels, ratio_thresh=MERGE_FLAG_VOLUME_RATIO):
    """QC helper: instance ids more than ratio_thresh x the timepoint's own
    median instance volume - candidates for a residual (still-merged) nucleus,
    to target visual spot-checking rather than scanning every timepoint."""
    ids = np.unique(all_labels)
    ids = ids[ids != 0]
    if len(ids) < 2:
        return []
    vols = ndi.sum(np.ones_like(all_labels), all_labels, ids)
    median_vol = np.median(vols)
    if median_vol <= 0:
        return []
    flagged = ids[vols >= ratio_thresh * median_vol]
    return sorted(int(i) for i in flagged)


def process_timepoint(pred_path, raw_path, t):
    pred = tifffile.imread(pred_path)
    raw  = tifffile.imread(raw_path).astype(np.float32)
    pred_fg = pred > 0

    instances = instance_from_prediction(pred_fg, EXTRA_ERODE_UM)
    n_components = int(instances.max())

    te_out, all_out, class_map, n_dropped, frac_icm_map = classify_and_filter(instances, pred)
    n_icm = sum(1 for c in class_map.values() if c == 'ICM')
    n_te  = sum(1 for c in class_map.values() if c == 'TE')

    flagged = flag_likely_merges(all_out)
    ambiguous = flag_ambiguous_classification(all_out, raw, class_map, frac_icm_map)

    print(f"  {n_components} components (erode={EXTRA_ERODE_UM}um) → "
          f"{n_dropped} dropped (<{MIN_INSTANCE_VOL_UM3}um3) → "
          f"{n_icm} ICM (excluded), {n_te} final TE instances"
          + (f"  [flagged possible merges: {flagged}]" if flagged else ""))
    if ambiguous:
        print(f"  [ambiguous classification, ICM_MIN_FRAC={ICM_MIN_FRAC} - id/called/frac_icm/mean_intensity/elongation]")
        for r in ambiguous:
            print(f"    id={r['id']} called={r['called']} frac_icm={r['frac_icm']:.2f} "
                  f"mean_intensity={r['mean_intensity']:.1f} elongation={r['elongation']:.2f}")

    return te_out


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    pred_files = sorted(glob.glob(f"{PRED_DIR}{PRED_GLOB}"))[:N_TIMEPOINTS]
    raw_files  = sorted(glob.glob(f"{RAW_DIR}{RAW_GLOB}"))[:N_TIMEPOINTS]
    assert len(pred_files) == N_TIMEPOINTS, \
        f"Expected {N_TIMEPOINTS} prediction files, found {len(pred_files)}"
    assert len(raw_files) == N_TIMEPOINTS, \
        f"Expected {N_TIMEPOINTS} raw files, found {len(raw_files)}"

    for t, (pred_path, raw_path) in enumerate(zip(pred_files, raw_files)):
        stem     = Path(pred_path).stem.replace('.nii', '')
        out_path = Path(OUT_DIR) / f"{stem}_instances_reclassified.tif"
        print(f"\nt={t:03d}  {Path(pred_path).name}")

        result = process_timepoint(pred_path, raw_path, t)
        tifffile.imwrite(str(out_path), result)
        print(f"  Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
