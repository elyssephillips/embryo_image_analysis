# Touching-nuclei instance segmentation — working notes

**Dataset:** Dataset001_implantation, `250914_stack5`
**Ground truth used for all validation below:** hand-annotated instance labels at t=60
(`/mnt/md0/elysse/nnUNet/training files 260427/labels/Cam_long_00060_cropped_label.tif`,
84 nuclei, Z-offset 44 into that 91-slice label to align with this run's 47-slice crop).

This documents what was tried to turn the nnUNet binary nuclei mask into tracked
instances in `00_preprocess_segmentation.py`, why the remaining errors happen, and
what would remove the need for most of this post-processing. See that script's
module docstring and CONFIG block for the current parameter values and inline
rationale — this file is the narrative version.

## Result

76/84 instances correctly recovered (IoU ≥ 0.5), mean volume error 7.3% on matched
instances (down from ~26% with the first working version), mean boundary IoU 0.92
on matched instances.

Two pretrained BlastoSPIM StarDist models (`late_blastocyst`, `early_embryo`) were
benchmarked against the same ground truth, despite an exact voxel-size match to
their published training spacing (2.0, 0.208, 0.208 µm). Neither beat the tuned
classical approach — best case was 60/84 matched at mean boundary IoU 0.65.

The instances still wrong cluster in one place: nuclei that physically touch with
low intensity contrast between them, most often in the crowded region near the
dish surface.

## What was tried, in order

Each step was validated against the same ground-truth timepoint before moving to
the next.

1. **Fixed anisotropic-distance bugs.** Z voxels are ~10x larger than X/Y
   (2.0µm vs 0.208µm), but distance-transform and peak-detection code treated one
   voxel step as equal in every direction. Converted every distance/blur/footprint
   calculation to real µm. This alone eliminated the original over-segmentation and
   false-ICM-removal reported at the start of this work.

2. **Made it fast enough to run.** Full-volume shape analysis and a naive peak
   search were projected at hours-to-days for 121 timepoints. Switched to
   per-component cropped processing plus scipy's fast separable max-filter instead
   of an anisotropic-footprint peak search. Now ~8-10 minutes for the full
   timecourse.

3. **Grid-searched the split itself against ground truth.** No prior validation
   existed for how the binary blob should be cut into nuclei. Intensity-peak
   seeding + intensity-weighted watershed flooding, parameters swept against t=60:
   61/84 matched, boundary IoU 0.87.

4. **Added shape-based (EDT) seeding for low-contrast touching pairs.** Intensity
   alone misses nuclei that touch but don't differ enough in brightness to
   register as two peaks — confirmed as a real under-split against ground truth.
   Added distance-transform (EDT) peaks alongside intensity peaks, and blended EDT
   distance into the flooding cost so the boundary lands near the true geometric
   pinch-point instead of wherever intensity happens to dip. Result: 75-76/84
   matched, volume error 26% → 7.3%.

5. **Replaced the ICM-detection heuristic.** ICM was defined as "the single
   largest connected blob." Confirmed wrong via 9 hand-labeled reference points
   (5 known-TE, 4 known-ICM, picked in napari): the largest blob sat 55µm from the
   true ICM and was actually a mass of touching TE cells. Replaced with local
   neighbor-density classification (ICM packs in a compact 3D cluster; TE spreads
   across a thin shell), scaled to each timepoint's own median neighbor count so
   embryo flattening over the timecourse doesn't shift the threshold. Result: 9/9
   reference points correct; ICM share stable at 26-48% across all 121 timepoints
   (previously swung from 2% to 74% with an absolute-count threshold).

6. **Smoothed the flooding surface, not the seeding surface.** A raw distance
   transform has mathematically exact flat ridges between seeds (a Voronoi
   facet) — visible on real output (t=71) as a straight, right-angle cut through
   an otherwise round nucleus. Fixed by lightly Gaussian-smoothing (0.5µm) the
   distance field used only for flooding, while keeping the field used for seed
   detection sharp. Confirmed gone on the reported case; 76/84 matched, no
   regression.

7. **Merged spurious fragments.** Combo (intensity + EDT) seeding sometimes places
   more seeds than there are real nuclei — one observed component placed 5 seeds
   for ~3 real nuclei, two claiming zero watershed territory and a third claiming
   only a 325µm³ sliver of what should have been a neighboring piece (visible as a
   "missing streak" in the final output, since that sliver was small enough to also
   get misclassified as ICM). Fixed by merging any piece under 400µm³ into
   whichever neighbor it shares the most boundary with. Real matches held exactly
   flat while excess fragments dropped — free precision, no recall cost.

8. **Chased the remaining jagged boundaries — hit a real ceiling.** Visually
   confirmed cases (e.g. t=71, orig component 33) where two real, well-separated
   seeds (~7.9µm apart) still produce a notched, non-anatomical boundary between
   them. Tried pushing both the intensity/EDT flooding weight and the flooding
   smoothing sigma further in both directions. Neither cleaned the notch up
   without cost: more EDT weight helped marginally but cost real matches on the
   ground truth (54/84 at pure EDT flooding vs 76/84 at the current blend); more
   EDT smoothing on a small component let one seed's territory swallow the other
   entirely, destroying the split outright (confirmed at flood-sigma=2.5). No
   further lever was found without an equal-or-larger cost elsewhere.

## Root cause

The nnUNet model is doing exactly what it was trained to do. 5-fold
cross-validated Dice of 0.846 on a binary nucleus/background task is genuinely
strong semantic segmentation — but Dice never asked the network to distinguish
nucleus A from adjacent nucleus B, only nucleus from not-nucleus. The moment two
nuclei touch, that distinction is discarded before this pipeline ever sees the
data: the binary mask carries no per-instance identity, and the true dividing
line between two touching nuclei is not recorded anywhere in it.

Everything in the working log above — intensity peaks, EDT peaks, blended
flooding, small-piece merging — is an attempt to reconstruct instance identity
that was already thrown away one step earlier. That reconstruction works well
when there's a real signal to lean on (a brightness dip, a shape pinch), which is
most of the time (76/84, 90%). It runs out of signal exactly when two real nuclei
are touching *and* similar in brightness *and* compressed into a non-convex
mutual boundary — conditions that get more common near the dish surface, where
the embryo visibly flattens over the timecourse.

**The ceiling isn't a parameter that hasn't been found yet.** It's
information-theoretic: once two touching nuclei are merged into one binary
region with no distinguishing signal between them, no amount of downstream
heuristics can be certain where one ends and the other begins. Every lever
tested in step 8 traded an improvement in one failure mode for a regression in
another — consistent with genuinely being at that ceiling for a post-hoc,
binary-mask-only approach.

## Recommendations, in priority order

**Update 2026-09-01 — decided against StarDist, decided in favor of an eroded-mask
nnUNet target.** A prior attempt to fine-tune the pretrained BlastoSPIM StarDist
model on this dataset's own images didn't converge on something usable for the
effort invested, so the StarDist path (originally listed here as an alternative)
is dropped in favor of staying on nnUNet. In its place: an *eroded-instance*
training target, validated empirically against the real training labels below —
simpler than the border-aware 3-class idea and directly actionable from data
already in hand.

1. **Retrain nnUNet on eroded (pre-separated) instance masks, not a straight
   binarization.** The mechanism, checked directly against the 6 real training
   volumes: instances are hand-drawn without overlapping voxels, but they are
   *face-adjacent* in roughly half to three-quarters of cases, so a straight
   nonzero→1 binarization (what `prepare_nnunet.py` currently does) fuses them
   into one blob before nnUNet ever sees the data — nnUNet is faithfully learning
   to reproduce a touching pattern that's already baked into the binarized
   ground truth, not introducing it.

   Fix: shrink each instance by a small margin before binarizing, so the training
   target always has a gap between neighbors. Swept margins directly against all
   6 training volumes (script: see below) to find how much erosion is actually
   needed, rather than guessing:

   | margin (µm, Z/Y/X) | t=0 | t=49 | t=60 | t=74 | t=80 | t=102 |
   |---|---|---|---|---|---|---|
   | not eroded (baseline) | 65/117 | 46/122 | 39/84 | 28/112 | 39/98 | 36/116 |
   | (0, 0.208, 0.208) — **1 voxel** | 115/117 | 119/122 (2 erased) | 88/84 | 115/112 (6 erased) | 95/98 (6 erased) | 113/116 (6 erased) |
   | (0, 0.416, 0.416) — 2 voxels | 120/117 | 123/122 (2 erased) | 89/84 | 116/112 (6 erased) | 96/98 (6 erased) | 116/116 (6 erased) |
   | (0, 0.832, 0.832) — 4 voxels | 121/117 | 120/122 (2 erased) | 89/84 | 117/112 (6 erased) | 97/98 (6 erased) | 116/116 (6 erased) |

   (`components/instances`; components exceeding instances means some instances
   fractured into >1 disconnected piece under erosion, not that more real nuclei
   appeared.)

   **A single 1-voxel (0.208µm) XY erosion, no Z erosion, already separates
   nearly all touching pairs** — going bigger doesn't meaningfully improve
   separation further (the numbers plateau immediately) and starts actively
   destroying small instances (erased-to-nothing counts appear at every margin
   ≥1 voxel for several volumes and don't grow much with more erosion, but never
   drop to zero either — some nuclei in this dataset are only a few voxels
   across at their narrowest). Verified nnUNet trains at native voxel spacing
   (2.0, 0.208, 0.208µm, no internal resampling — see `predictions/plans.json`),
   so this margin isn't at risk of being blurred away before the network sees it.

   **Recommended data-prep change**, in `pipelines/nnUNET/prepare_nnunet.py`'s
   binarization step: for each instance, erode with a flat XY-only structuring
   element (radius 1 voxel in Y/X, 0 in Z — same anisotropic-footprint pattern
   used throughout `00_preprocess_segmentation.py`), then keep only the largest
   connected fragment of the eroded mask (a 1-voxel erosion can occasionally
   pinch a thin/lobed nucleus into two disconnected pieces — silently keeping
   only the largest avoids teaching the network to draw two objects from one
   real nucleus). Union all eroded instances into the binary training target
   instead of the current straight `labels != 0`.

   **Inference-side correction:** the network will now predict slightly
   undersized (eroded) nuclei. After connected-components on the predicted
   binary mask, grow each instance back out with
   `skimage.segmentation.expand_labels(labels, distance=...)` — it expands every
   label outward into background by a fixed physical distance and stops exactly
   where it would collide with a neighboring label, which is precisely the
   correct behavior here (recovers real volume without ever re-merging two
   instances). Set `distance` a little larger than the training erosion margin
   (e.g. 2-3 voxels) since real nuclei will have shrunk by roughly the erosion
   amount; validate the exact value against a held-out annotated timepoint the
   same way everything else in this doc was validated, rather than assuming it's
   right.

   This replaces the entire watershed/seeding/flooding reconstruction in
   `00_preprocess_segmentation.py` (steps 3-7 of the working log above) with
   "connected components, then expand_labels" — instance identity would already
   be correct coming out of the network.

   **Update 2026-09-03 — the margin table above doesn't reproduce; replaced with
   a margin-free approach.** Re-ran the same style of sweep directly against
   this dataset (now 12 volumes: the original 6 plus 6 newly hand-annotated),
   actually re-binarizing the eroded instances and connected-component-counting
   the result, rather than trusting the numbers above. Real result: **no XY
   margin from 1 to 8 voxels (up to 1.66µm) achieves full separation on any
   volume** — the 115/117-at-1-voxel figure above doesn't hold up. Best guess:
   the original sweep counted unique label IDs after erosion (trivially ≈ N
   regardless of whether the eroded pieces still touch) rather than testing
   true post-binarization spatial separation.

   Separately, and independent of that bug: **a per-instance XY-only erosion
   can never separate a pair that only touches across a Z-slice boundary**,
   because it operates per-slice and never modifies Z-adjacency. Checked
   directly: ~43% of touching clusters across the 12 volumes (90/208) are
   Z-only contact. A margin big enough to fix the rest by brute force also
   starts destroying thin real instances well before it gets there.

   **Replacement: don't erode by a margin at all — remove exactly the contact
   layer between two *different* instances, wherever it occurs (Z, Y, or X),
   before binarizing.** Since instance identity is known ground truth at prep
   time, this doesn't need a margin: for every voxel pair that's face-adjacent
   across two different nonzero IDs, zero out both sides. This guarantees a
   background gap at every point of contact, however wide or which axis, and
   touches nothing on a surface that doesn't border another instance (so it
   costs less real volume than blanket per-instance erosion, not more).
   One side effect, checked directly: an instance that touches two different
   neighbors near its own thin point can get nicked into extra fragments —
   always single digits to low tens of voxels against a main body of
   thousands, never a real second-sized piece. A follow-up pass keeps only
   the largest fragment per original ID to clean this up.

   Implemented as `separate_touching_instances` + `drop_stray_fragments` in
   `pipelines/nnUNET/prepare_nnunet.py`, replacing the margin-erosion step.
   Validated directly: `Cam_long_00000` and `Cam_long_00050_cropped` both hit
   exact components == instances (117/117, 94/94) with 1.2-3.3% voxel loss and
   nothing erased; `Cam_long_00074` comes out 2 short only because of 2
   already-known 1-voxel noise IDs with no real nucleus behind them (expected
   to close once those are deleted during hand-label cleanup).

   The inference-side correction is unchanged: `expand_labels` after
   connected-components on the predicted mask, same reasoning as above.

2. **Fold ICM/TE identity into the same retraining, as a second use of the same
   eroded-mask target.** Train a 3-class target instead of 2 (background /
   eroded-ICM / eroded-TE) by assigning each already-separated eroded instance
   its ICM or TE identity. This is a different problem from instance separation
   — even with 3 classes, two touching TE nuclei still need the same
   erosion/expand-labels treatment as (1) — but combining them means one
   retraining replaces both the splitting logic *and* the density-ratio ICM/TE
   classifier this session built post-hoc (which was fit from only 9 hand-picked
   reference points and had at least one confirmed borderline misclassification
   right at the TE/ICM spatial transition — see step 5 of the working log). A
   network with real visual access to intensity/texture differences between ICM
   and TE cells, not just position and local density, has a real chance of doing
   better than a pure-geometry heuristic there. Needs ICM/TE identity newly
   annotated on the training instances (not currently in the ground truth) — see
   the note on new training data below.

3. **Expand the training set, aimed at the hard cases — while annotating.** The
   current model trained on only 6 annotated volumes. Given new annotation work
   is starting anyway (for the eroded masks and/or ICM/TE identity above), prioritize
   new timepoints/images that specifically cover what breaks the current
   pipeline, rather than more of what it already handles well:
   - The crowded, dish-contact region — repeatedly the site of the hardest
     remaining cases in this working log (steps 4 and 8).
   - Later, more-compacted developmental stages, where the embryo visibly
     flattens against the dish (confirmed to shift local nucleus density by
     ~50% between t=60 and t=90 — see step 5) and touching nuclei become more
     common.
   - If pursuing the ICM/TE 3-class idea, make sure new annotated volumes
     include clear examples of both identities across a range of positions, not
     just unambiguous core-ICM and edge-TE cells — the transition zone is
     exactly where step 5's heuristic struggled, so that's where the network
     needs training signal most.

4. **Finer axial (Z) sampling, if the acquisition allows it.** The ~10x Z/XY
   voxel anisotropy (2.0µm vs 0.208µm) was the root cause of several distinct
   bugs fixed in this working log, and remains a real resolution bottleneck on
   its own merits — a nucleus only spans 5-10 Z-slices today. Denser axial
   sampling gives both a retrained network and any remaining post-hoc step much
   richer 3D shape to separate touching nuclei correctly.

5. **Keep one hand-annotated validation timepoint per retraining round.** The
   single most useful thing this work did was having real ground truth to test
   every change against instead of eyeballing napari output — several changes
   that looked reasonable (e.g. spatial density clustering for ICM, higher
   flooding smoothness) turned out to actively regress once actually scored.
   Carry that habit into retraining: hold out an annotated timepoint, score every
   candidate model against it the same way, before trusting it on the full
   timecourse.

## What "easy" looks like

An instance-aware model that outputs already-separated nuclei directly from
inference for the large majority of cases, with no watershed reconstruction step
at all — collapsing steps 3-7 above into "load the labels." What's left over
would be a small residual of genuinely ambiguous cases (nuclei overlapping enough
in 3D that no single-channel model can fully resolve them), small enough to hand
to the same kind of lightweight manual curation this team already runs on track
curation (see `logs/tracking.md`), rather than something a general-purpose
parameter has to account for across the whole embryo.

---
*Compiled 2026-09-01. Full interactive version (with diagrams):
https://claude.ai/code/artifact/f7fa738f-7daa-439d-8ed5-74e8d15e327a*
