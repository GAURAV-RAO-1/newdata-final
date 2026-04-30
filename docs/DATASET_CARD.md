# Dataset Card: newdata-final

## Dataset Name

`newdata-final`

## Dataset Type

Detector-ready SAR ship detection benchmark with real SAR images and detector-aware accepted super-resolved synthetic samples.

## Intended Use

This dataset is intended for research on:

- SAR ship detection
- synthetic SR data augmentation
- detector-aware dataset construction
- curriculum training for object detection
- cross-dataset generalization studies

## Not Intended Use

This dataset should not be used to claim:

- full physical SAR reconstruction fidelity
- phase-level SAR/SLC validity
- radiometric calibration validity
- operational maritime surveillance readiness without further validation

## Synthetic Data Construction

Synthetic samples are created using a Real-ESRGAN x2 super-resolution branch derived from real SAR crops.

Pipeline:

    real SAR crop
    -> LR generation
    -> Real-ESRGAN x2 SR
    -> Stage A quality filtering
    -> Stage B-soft detector-aware filtering
    -> accepted SR samples
    -> combined_yolo dataset

## Splits and Counts

Accepted Stage B-soft synthetic SR counts:

| Split | Real images | Accepted SR images |
|---|---:|---:|
| train | 13455 | 10749 |
| val | 3263 | 2600 |
| test | 2675 | 2154 |
| all | 19393 | 15503 |

Combined YOLO export counts:

| Split | Images |
|---|---:|
| train | 24204 |
| val | 5863 |
| test | 4829 |

## Main Benchmark Results

| Method | Combined val mAP50-95 | HRSID test mAP50-95 | SSDD test mAP50-95 |
|---|---:|---:|---:|
| Original-only YOLOv8s | 0.935 | 0.935 | 0.894 |
| Stage B-soft + curriculum YOLOv8s | 0.966 | 0.967 | 0.954 |
| RT-DETR Stage B-soft curriculum | 0.748 | 0.770 | 0.726 |
| YOLO11s Stage B-soft curriculum | 0.949 | 0.951 | 0.916 |

## SAR Fidelity / Statistical Audit

All-split summary:

| Feature | Real mean | Accepted SR mean | Delta |
|---|---:|---:|---:|
| Mean intensity | 0.1367 | 0.1313 | -3.93% |
| Std intensity | 0.1699 | 0.1546 | -9.00% |
| Contrast p95-p5 | 0.4379 | 0.3775 | -13.77% |
| Entropy | 5.8230 | 5.3941 | -7.36% |
| Local variance 8x8 | 0.0172 | 0.0119 | -30.91% |
| ENL proxy | 0.6786 | 0.7515 | +10.75% |
| Edge energy | 0.0600 | 0.0283 | -52.81% |

Interpretation: accepted SR images are globally distribution-controlled but smoother than real SAR crops. This supports detection-oriented use, not full physical SAR validation.

## Known Limitations

- The dataset is derived from existing public SAR datasets, not new raw SAR acquisitions.
- SR samples are image-level synthetic outputs and are not claimed to be physically identical to real SAR captures.
- Full radiometric, phase-level, or SLC-level SAR validation is not provided.
- Edge energy and local variance are lower in accepted SR images, showing smoother texture than real SAR crops.
- Dataset use is subject to upstream dataset license terms.

## Dataset Access

- Archive: `release_dataset_newdata_final_v1.tar.gz`
- SHA256: `7060499f2f24cf0fa9c98f42d024e983bd275d68ec52c2412011bebccbef8dd8`
- Google Drive: https://drive.google.com/drive/folders/11uX0ZFEIOKeEqdOmo-dobozEsPh3fvtk
