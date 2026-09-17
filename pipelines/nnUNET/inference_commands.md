# nnUNet Inference — Dataset003_icm_te

Model: `nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres`, all 5 folds
Cross-validation Dice (mean across 12 held-out cases): nuclei (class 1) ≈ 0.77, ICM (class 2) ≈ 0.54
(ICM Dice is noisy case-to-case, 0.11-0.73 — see `pipelines/tracking/segmentation_notes.md`)

Prior binary model for reference: `Dataset001_implantation`, same trainer/config, CV Dice 0.846
(background/nucleus only, no ICM/TE split, no touching-instance separation)

---

## Step 1: Activate environment

```bash
conda activate nnunet
```

---

## Step 2: Prepare images

```bash
cd /mnt/md0/elysse/code/embryo_image_analysis
python pipelines/nnUNET/prepare_inference.py
```

Edit the CONFIG block at the top of that script (`LONG_DIR`, `SHORT_DIR`, `OUTPUT_DIR`,
`DATASET_NAME`) to point at whichever raw stack and dataset you're running.

Outputs, under `OUTPUT_DIR` (e.g. `inference/Dataset003_icm_te/250914_stack5/`):
- `imagesTs/` — all Cam_long crops for nnUNet, one folder
- `imagesTs_gpu0/`, `imagesTs_gpu1/`, ... — the same crops split into `N_GPU_SPLITS`
  contiguous chunks by timepoint, for running `nnUNetv2_predict` in parallel across GPUs
  (set `N_GPU_SPLITS = 1` in the script to skip this and just use `imagesTs/`)
- `cam_short_cropped/` — matching Cam_short crops, for downstream biosensor analysis
- `crop_info.json` — per-timepoint z offsets, for mapping predictions back to the original stack

---

## Step 3: Run inference

Two GPUs, split by timepoint range — safe to write both to the same `predictions/`
folder since the two input sets never overlap:

```bash
# Terminal 1 — GPU 0
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
  -d Dataset003_icm_te \
  -i /mnt/md0/elysse/nnUNet/inference/Dataset003_icm_te/250914_stack5/imagesTs_gpu0 \
  -o /mnt/md0/elysse/nnUNet/inference/Dataset003_icm_te/250914_stack5/predictions \
  -f 0 1 2 3 4 \
  -tr nnUNetTrainer_500epochs \
  -c 3d_fullres \
  -p nnUNetPlans

# Terminal 2 — GPU 1
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict \
  -d Dataset003_icm_te \
  -i /mnt/md0/elysse/nnUNet/inference/Dataset003_icm_te/250914_stack5/imagesTs_gpu1 \
  -o /mnt/md0/elysse/nnUNet/inference/Dataset003_icm_te/250914_stack5/predictions \
  -f 0 1 2 3 4 \
  -tr nnUNetTrainer_500epochs \
  -c 3d_fullres \
  -p nnUNetPlans
```

Single-GPU alternative (no split, just `-i .../imagesTs`): same command without
`CUDA_VISIBLE_DEVICES` and with only one `-i`/`-o` pair.

> Postprocessing (removing non-largest regions) was evaluated and **not
> recommended** for the binary Dataset001 model (hurts multi-nuclei
> predictions). Not yet re-evaluated for this 3-class eroded-instance target —
> don't assume the same conclusion holds without checking.

---

## Output

Predictions are saved as 3-class `.tif` masks (0=background, 1=nuclei, 2=ICM) in
`predictions/`, one per timepoint, cropped to the same z range as the input. Use
`crop_info.json` to map z indices back to the original full stack coordinates.

Each predicted volume still has the training-time contact-layer gap between
touching instances (see `prepare_nnunet.py`'s `separate_touching_instances`) —
run connected-components then `skimage.segmentation.expand_labels` to recover
real per-instance volume before treating regions as final instances.

### Mapping back to original z coordinates

```python
import json

with open("inference/Dataset003_icm_te/250914_stack5/crop_info.json") as f:
    crop_info = json.load(f)

# For timepoint "00003":
first_z = crop_info["00003"]["first_z"]
# original_z = predicted_z + first_z
```
