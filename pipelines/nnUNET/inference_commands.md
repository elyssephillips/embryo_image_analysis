# nnUNet Inference — Dataset001_implantation

Model: `nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres`, all 5 folds  
Cross-validation Dice: 0.846

---

## Step 1: Activate environment

```bash
conda activate nnunet
```

---

## Step 2: Prepare images

```bash
cd /mnt/md0/elysse/nnUNet
python prepare_inference.py
```

Outputs:
- `inference/Dataset001_implantation/imagesTs/` — Cam_long crops for nnUNet
- `inference/Dataset001_implantation/cam_short_cropped/` — matching Cam_short crops
- `inference/Dataset001_implantation/crop_info.json` — per-timepoint z offsets

---

## Step 3: Run inference

```bash
nnUNetv2_predict \
  -d Dataset001_implantation \
  -i /mnt/md0/elysse/nnUNet/inference/Dataset001_implantation/imagesTs \
  -o /mnt/md0/elysse/nnUNet/inference/Dataset001_implantation/predictions \
  -f 0 1 2 3 4 \
  -tr nnUNetTrainer_500epochs \
  -c 3d_fullres \
  -p nnUNetPlans
```

> Postprocessing was evaluated and **not recommended** for this dataset
> (removing non-largest regions hurts multi-nuclei predictions).

---

## Output

Predictions are saved as binary `.tif` masks in `predictions/`, one per timepoint,
cropped to the same z range as the input. Use `crop_info.json` to map
z indices back to the original full stack coordinates.

### Mapping back to original z coordinates

```python
import json

with open("inference/Dataset001_implantation/crop_info.json") as f:
    crop_info = json.load(f)

# For timepoint "00003":
first_z = crop_info["00003"]["first_z"]
# original_z = predicted_z + first_z
```
