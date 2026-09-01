# IF Log

## Dataset Index

| Dataset | Description | N / Conditions | Data path | Last updated | Status |
|---|---|---|---|---|---|
| 20260114_fgf_cdx2_ppmlc_gata3 | FGF TE patterning IF — CDX2, GATA3, ppMLC | ? | `/Users/elyssephillips/Desktop/FGF IF data/20260114_*/` | — | archived (old Mac) | <!-- row:20260114_fgf_cdx2_ppmlc_gata3 -->
| 20260121_fgf_cdx2_ppmlc_gata3 | FGF TE patterning IF — CDX2, GATA3, ppMLC | ? | `/mnt/md1/elysse/fgf exp/20260121_fgf_cdx2_ppmlc_gata3/tifs` | 2026-08-27 | needs run | <!-- row:20260121_fgf_cdx2_ppmlc_gata3 -->
| 20260129_fgf_cdx2_ppmlc_gata3 | E4.5 c or fgf tx IF for mural markers | 4 control, 5 fgf | `...ysse/fgf exp/20260129_fgf_cdx2_ppmlc_gata3/tifs` | 2026-08-28 | analysis done | <!-- row:20260129_fgf_cdx2_ppmlc_gata3 -->
| 20260218_fgf_cdx2_ppmlc_gata3 | E4.5 control or fgf treated fixed and stained for mural markers | 2C, 4 tx | `...sse/fgf exp/20260218_fgf_cdx2_ppmlc_gata3/tiffs` | 2026-08-28 | analysis done | <!-- row:20260218_fgf_cdx2_ppmlc_gata3 -->
| 20260304_fgf_cdx2_ppmlc_gata3 | E4.5 embryos treated with control or fgf then fixed and stained for mural markers | fill in | `...sse/fgf exp/20260304_fgf_cdx2_ppmlc_gata3/tiffs` | 2026-08-28 | analysis done | <!-- row:20260304_fgf_cdx2_ppmlc_gata3 -->
| 20260416_fgf_cdx2_ppmlc_gata3 | FGF TE patterning IF — CDX2, GATA3, ppMLC | ? | `/mnt/md1/elysse/20260416_fgf_cdx2_ppmlc_gata3/` | — | in progress | <!-- row:20260416_fgf_cdx2_ppmlc_gata3 -->

| 20260604_meki_cdx2_ppmlc_gata3 | IF E3.5-4.5 meki with mural gata3 cdx2 panel | 3 C, 6 meki + 7 testing second panel | `/mnt/md1/elysse/20260604_fixed` | 2026-07-16 | conversion done | <!-- row:20260604_meki_cdx2_ppmlc_gata3 -->
| 20260730_c_meki_gata3_nmmiia_cdx2 | fixed MEKi or C embryos for mural markers | combined | `/mnt/md1/elysse/20260730_c_meki_gata3_nmmiia_cdx2` | 2026-08-04 | converting | <!-- row:20260730_c_meki_gata3_nmmiia_cdx2 -->
<!-- index-end -->

---

<!-- ds:20260827_md1_recovery -->
## 2026-08-27 — md1 replacement: re-run status

**Context:** md1 failed and was replaced with a new drive. Re-running/redoing IF analysis for basically every dataset in `configs/IF/` — some already have segmentations that just need cleaning, some need segmentation from scratch. Status per dataset below is drawn from each config's `#storage:` comment (where present) and reflects what's actually recoverable, not what the config's `raw_data_dir` path claims.

| Dataset | Storage (post-failure) | Redo needed |
|---|---|---|
| 20260114_fgf_cdx2_ppmlc_gata3 | weiner server (no DAPI channel) | never analyzed — needs first full run | not using for analysis
| 20260121_fgf_cdx2_ppmlc_gata3 | raw on md1 (`fgf exp/20260121...`) | new dataset — needs full run | not using for analysis
| 20260129_fgf_cdx2_ppmlc_gata3 | t7, md1, weiner all hold raw + segs and analyses | re-run done
| 20260218_fgf_cdx2_ppmlc_gata3 | t7, md1, weiner all hold raw + segs and analyses | re-run done
| 20260304_fgf_cdx2_ppmlc_gata3 | t7, md1, weiner all hold raw + segs and analyses | re-run done
| 20260416_fgf_cdx2_ppmlc_gata3 | raw hdf5 on md1, raw on MIC | no segmentation backup — needs segmentation from scratch | E5.5, dont use for analysis
| 20260604_meki_cdx2_ppmlc_gata3 | t7 (raw, raw segs), weiner (raw, raw segs), mac (raw, cleaned segs, analysis) | mac copy may already be usable — re-point and confirm before redoing |
| 20260730_c_meki_gata3_nmmiia_cdx2 | raw hdf5 on md0 only | needs full pipeline: h5 → tiff → segment → clean → analyze | all conditions combined, not used

<!-- ds-end:20260827_md1_recovery -->

<!-- ds:unattributed_march2026 -->
## Unattributed March 2026 runs

**Description:** Three analysis runs from 2026-03-18 and 2026-03-19, migrated from master_study_log.csv. Dataset unclear — likely 20260114, 20260218, or 20260304 batches run on old Mac. Move these entries to the correct dataset section once you know which they correspond to.  
**N / Conditions:** see individual entries below  
**Data path:** /Users/elyssephillips/Desktop/ (old Mac)

### 2026-03-18 15:32 | 04_plot_intensities.py
**Done:** IF analysis run. N=6 embryos (3 control, 3 treated).
**Params/findings:** avg_pearson_r=0.54, avg_gata3_y_slope=0.00063, avg_polarization=0.081

### 2026-03-18 15:48 | 04_plot_intensities.py
**Done:** IF analysis run. N=9 embryos (4 control, 5 treated).
**Params/findings:** avg_pearson_r=0.47, avg_gata3_y_slope=0.0011, avg_polarization=0.078

### 2026-03-19 13:18 | 04_plot_intensities.py
**Done:** IF analysis run. N=6 embryos (2 control, 4 treated).
**Params/findings:** avg_pearson_r=0.53, avg_gata3_y_slope=0.00045, avg_polarization=0.041

<!-- ds-end:unattributed_march2026 -->

<!-- ds:20260114_fgf_cdx2_ppmlc_gata3 -->
## 20260114_fgf_cdx2_ppmlc_gata3

**Description:** FGF TE patterning IF — CDX2, GATA3, ppMLC (4-channel)  
**N / Conditions:** fill in  
**Data path:** /Users/elyssephillips/Desktop/FGF IF data/20260114_fgf_cdx2_ppmlc_gata3/  
**Storage:** weiner server (raw; no DAPI channel in this dataset)

<!-- ds-end:20260114_fgf_cdx2_ppmlc_gata3 -->

<!-- ds:20260121_fgf_cdx2_ppmlc_gata3 -->
## 20260121_fgf_cdx2_ppmlc_gata3

**Description:** FGF TE patterning IF — CDX2, GATA3, ppMLC (4-channel)  
**N / Conditions:** fill in  
**Data path:** /mnt/md1/elysse/fgf exp/20260121_fgf_cdx2_ppmlc_gata3/tifs  
**Storage:** raw on weiner server not using for analysis (conditions combined)

<!-- ds-end:20260121_fgf_cdx2_ppmlc_gata3 -->

<!-- ds:20260129_fgf_cdx2_ppmlc_gata3 -->
## 20260129_fgf_cdx2_ppmlc_gata3

**Description:** FGF TE patterning IF — CDX2, GATA3, ppMLC (4-channel)  
**N / Conditions:** 4 control, 5 fgf  
**Data path:** /mnt/md1/elysse/fgf exp/20260129_fgf_cdx2_ppmlc_gata3/tifs  
**Storage:** t7: raw and cleaned segs and analys; md1: ""; weiner: ""  
**Channels:** dapi, GATA3, ppMLC, CDX2  
**Voxel size (zyx, µm):** [0.7, 0.122666664, 0.122666664]  
**Exclusions:** none

### 2026-08-27 13:10 | note
**Note:** fill in

### 2026-08-27 14:04 | 04_plot_intensities.py
**Output:** /mnt/md1/elysse/fgf exp/20260129_fgf_cdx2_ppmlc_gata3/analysis
**Done:** tx E3.5-4.5 fix E4.5
**Next:** 

<!-- ds-end:20260129_fgf_cdx2_ppmlc_gata3 -->

<!-- ds:20260218_fgf_cdx2_ppmlc_gata3 -->
## 20260218_fgf_cdx2_ppmlc_gata3

**Description:** FGF TE patterning IF — CDX2, GATA3, ppMLC (4-channel)  
**N / Conditions:** 2C, 4 tx  
**Data path:** /mnt/md1/elysse/fgf exp/20260218_fgf_cdx2_ppmlc_gata3/tiffs  
**Storage:** t7: raw, seg, cleaned seg and analysis; md0: ""; weiner: ""  
**Channels:** dapi, GATA3, ppMLC, CDX2  
**Voxel size (zyx, µm):** [1, 0.122666664, 0.122666664]  
**Exclusions:** none

### 2026-08-27 14:29 | note
**Note:** fill in

### 2026-08-27 15:04 | 04_plot_intensities.py
**Output:** /mnt/md1/elysse/fgf exp/20260218_fgf_cdx2_ppmlc_gata3/analysis
**Done:** treated at E3.5, fixed at E4.5
**Next:** 

<!-- ds-end:20260218_fgf_cdx2_ppmlc_gata3 -->

<!-- ds:20260304_fgf_cdx2_ppmlc_gata3 -->
## 20260304_fgf_cdx2_ppmlc_gata3

**Description:** FGF TE patterning IF — CDX2, GATA3, ppMLC (4-channel)  
**N / Conditions:** fill in  
**Data path:** /mnt/md1/elysse/fgf exp/20260304_fgf_cdx2_ppmlc_gata3/tiffs  
**Storage:** t7: raw and segs; md1: raw and segs; weiner: raw and segs  
**Channels:** dapi, GATA3, ppMLC, CDX2  
**Voxel size (zyx, µm):** [0.7, 0.122666664, 0.122666664]  
**Exclusions:** c002: None

### 2026-08-27 15:45 | note
**Note:** fill in

### 2026-08-27 16:39 | 04_plot_intensities.py
**Output:** /mnt/md1/elysse/fgf exp/20260304_fgf_cdx2_ppmlc_gata3/analysis
**Done:** e3.5 blastocysts treated with c or fgf til e4.5
**Next:** 

<!-- ds-end:20260304_fgf_cdx2_ppmlc_gata3 -->

<!-- ds:20260416_fgf_cdx2_ppmlc_gata3 -->
## 20260416_fgf_cdx2_ppmlc_gata3

**Description:** FGF TE patterning IF — CDX2, GATA3, ppMLC (4-channel)  
**N / Conditions:** fill in  
**Data path:** /mnt/md1/elysse/20260416_fgf_cdx2_ppmlc_gata3/  
**Storage:** raw hdf5 on md1; raw on MIC

<!-- ds-end:20260416_fgf_cdx2_ppmlc_gata3 -->

<!-- ds:20260604_meki_cdx2_ppmlc_gata3 -->
## 20260604_meki_cdx2_ppmlc_gata3

**Description:** IF E3.5-4.5 meki with mural gata3 cdx2 panel  
**N / Conditions:** 3 C, 6 meki + 7 testing second panel  
**Data path:** /mnt/md1/elysse/20260604_fixed  
**Storage:** t7 (raw, raw segs); weiner (raw, raw segs); mac (raw, cleaned segs, analysis)

### 2026-07-16 11:36 | convert_h5_to_tiff.py
**Output:** /mnt/md1/elysse/20260604_fixed/cropped
**Done:** E3.5-E4.5 c or meki in IVC1
**Next:** analyze c and meki patterning for cdx2 gata3 params

<!-- ds-end:20260604_meki_cdx2_ppmlc_gata3 -->

<!-- ds:20260730_c_meki_gata3_nmmiia_cdx2 -->
## 20260730_c_meki_gata3_nmmiia_cdx2

**Description:** fixed MEKi or C embryos for mural markers  
**N / Conditions:** combined  
**Data path:** /mnt/md1/elysse/20260730_c_meki_gata3_nmmiia_cdx2  
**Storage:** raw hdf5 on md0

### 2026-08-04 16:37 | convert_h5_to_tiff.py
**Output:** /mnt/md1/elysse/20260730_c_meki_gata3_nmmiia_cdx2/cropped
**Done:** tx E3.5-E4.5 then fixed
**Next:** 

<!-- ds-end:20260730_c_meki_gata3_nmmiia_cdx2 -->
