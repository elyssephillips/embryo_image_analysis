# nnUNet Setup Notes — Dataset001_implantation

## 1. Bash Environment Variables

nnUNet does not take folder paths as command-line arguments. Instead it reads three environment variables at runtime to know where to look for data and where to write outputs. These are set in `~/.bashrc` so they load automatically every time a terminal is opened.

To view or edit:
```bash
nano ~/.bashrc
```

The three required variables:
```bash
export nnUNet_raw="/mnt/md0/elysse/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/mnt/md0/elysse/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/mnt/md0/elysse/nnUNet/nnUNet_results"
```

After making any changes to `~/.bashrc`, reload it in the current terminal:
```bash
source ~/.bashrc
```

**What each variable points to:**
- `nnUNet_raw` — raw input data in nnUNet format (images + labels + dataset.json)
- `nnUNet_preprocessed` — where nnUNet writes resampled/normalised data after planning
- `nnUNet_results` — where trained model weights and logs are saved

---

## 2. Data Preparation — prepare_nnunet.py

Script location: `/mnt/md0/elysse/training/prepare_nnunet.py`

Source data:
- Labels: `/mnt/md0/elysse/training/labels/` — instance-segmented nuclei (uint16, values = instance IDs)
- Raw images: `/mnt/md0/elysse/training/raw/` — single-channel fluorescence (uint16)

Output: `/mnt/md0/elysse/nnUNet/nnUNet_raw/Dataset001_implantation/`

### What the script does

**For each label/raw pair:**

1. **Finds the first non-empty label slice** — each z-stack has a large number of empty slices at the start (the embryo only appears in the latter portion of the stack). The script finds the first slice where any nucleus is labelled.

2. **Crops both the raw image and label from that slice onwards** — the crop is applied identically to both so they remain matched. This removes between 35–66 empty slices depending on the timepoint.

3. **Binarizes the labels** — the original labels are instance-based (each nucleus has a unique integer ID). nnUNet needs semantic labels, so all non-zero values are converted to 1 (nucleus), leaving 0 as background. Output dtype is uint8.

4. **Saves as .tif** with nnUNet naming convention:
   - Raw image: `imagesTr/Dataset001_XXXXX_0000.tif` — the `_0000` suffix is required by nnUNet to indicate channel 0
   - Label: `labelsTr/Dataset001_XXXXX.tif`
   - The `XXXXX` number matches the original filename (e.g. `00049`, `00102`) for traceability

5. **Writes spacing sidecar .json files** — tif files do not store physical voxel size metadata. nnUNet reads spacing from a sidecar `.json` alongside each tif. Spacing is in [Z, Y, X] order in µm:
   ```json
   {"spacing": [2.0, 0.208, 0.208]}
   ```
   - Image sidecar: `imagesTr/Dataset001_XXXXX.json` (note: no `_0000`, nnUNet strips the channel suffix when looking for the sidecar)
   - Label sidecar: `labelsTr/Dataset001_XXXXX.json`

6. **Writes dataset.json** — metadata file nnUNet requires at the root of the dataset folder:
   ```json
   {
       "channel_names": {"0": "nuclei"},
       "labels": {"background": 0, "nucleus": 1},
       "numTraining": 6,
       "file_ending": ".tif"
   }
   ```

To re-run the preparation script:
```bash
conda activate nnunet
python /mnt/md0/elysse/training/prepare_nnunet.py
```

---

## 3. Preprocessing

```bash
conda activate nnunet
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

`-d 1` refers to the dataset ID (the number in `Dataset001_implantation`).

### What this does

**Integrity check (`--verify_dataset_integrity`)**
Verifies that `dataset.json`, image files, label files, and sidecar jsons are all consistent before doing any heavy computation. Catches missing files, shape mismatches, or naming errors early.

**Planning**
nnUNet analyses the dataset — image shapes, voxel spacing, number of cases — and automatically determines the optimal network architecture and training configuration. This includes patch size, batch size, network depth, and number of pooling layers. The significant anisotropy in this dataset (Z=2.0µm vs XY=0.208µm, ~10:1 ratio) will influence whether it recommends a `3d_fullres`, `3d_lowres`, or `2d` configuration. Plans are saved to `nnUNet_preprocessed/Dataset001_implantation/`.

**Preprocessing**
Resamples all images to a common spacing and normalises intensities. The processed arrays are saved to `nnUNet_preprocessed/`. Training reads from here rather than the raw files, which speeds up data loading during training.

### After preprocessing — check the splits

With only 6 cases, nnUNet will auto-generate a 5-fold cross-validation split of roughly 1–2 cases per fold. Review and adjust this before training if needed:

```
nnUNet_preprocessed/Dataset001_implantation/splits_final.json
```

---

## 4. Training

nnUNet does not have an `--epochs` flag. Instead you select a trainer class with `-tr` that has the epoch count baked in. The default trainer runs 1000 epochs. Available built-in options:

`1, 5, 10, 20, 50, 100, 250, 500, 750, 1000 (default), 2000, 4000, 8000`

The basic training command:
```bash
nnUNetv2_train DATASET_ID CONFIGURATION FOLD -tr TRAINER
```

For example, fold 0 with 500 epochs:
```bash
nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainer_500epochs
```

nnUNet uses 5-fold cross-validation (folds 0–4). Each fold trains a separate model on a different train/validation split, and the final model is the ensemble of all folds.

### Running two folds in parallel across two GPUs

The most practical way to use both GPUs is to split the 5 folds across two terminals, chaining commands with `&&` so each fold starts automatically when the previous one finishes. This means you can set it running and leave it unattended.

`&&` only proceeds to the next command if the previous one completed successfully — so if a fold crashes it won't silently continue to the next.

```bash
# Terminal 1 — GPU 0, folds 0 and 1
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainer_500epochs && \
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres 1 -tr nnUNetTrainer_500epochs

# Terminal 2 — GPU 1, folds 2, 3 and 4
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 1 3d_fullres 2 -tr nnUNetTrainer_500epochs && \
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 1 3d_fullres 3 -tr nnUNetTrainer_500epochs && \
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 1 3d_fullres 4 -tr nnUNetTrainer_500epochs
```

`CUDA_VISIBLE_DEVICES` tells PyTorch which GPU to use — setting it to `0` or `1` pins that process to that specific card.

### Resuming training

If a run is interrupted or you want to extend training, add `--c` to resume from the last checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainer_500epochs --c
```

### Note on epoch count with a small dataset

With only 6 training cases, 500 epochs is a reasonable starting point. The loss curves (saved in `nnUNet_results/`) will show whether training has plateaued. You can always resume with `--c` if it hasn't converged.

---

## 5. GPU Monitoring

To monitor both GPUs in real time while training:

```bash
watch -n 1 nvidia-smi
```

This refreshes every second and shows memory usage and GPU utilisation per card. What to look for:

- **GPU utilisation consistently >80%** — good, the GPU is the bottleneck as expected
- **GPU utilisation spiking low intermittently** — data loading is the bottleneck, not the GPU
- **Memory usage** — with two RTX 5090s (32GB each), memory is unlikely to be an issue for this dataset, but worth keeping an eye on during the first epoch

This machine has:
- GPU 0: NVIDIA GeForce RTX 5090 — 32GB VRAM
- GPU 1: NVIDIA GeForce RTX 5090 — 32GB VRAM
