# newdata-final: Detector-Aware Synthetic SAR Ship Detection Dataset

This repository contains the code, configurations, result tables, and reproducibility notes for `newdata-final`, a detector-aware synthetic SAR ship detection dataset pipeline.

## Overview

`newdata-final` constructs a detector-ready SAR ship detection benchmark by combining original real SAR ship samples with accepted super-resolved synthetic samples. Synthetic samples are generated from real SAR crops using a Real-ESRGAN x2 branch and then filtered using a two-stage acceptance process.

The strongest setting is curriculum learning:

- Stage 1: train on `real_only`
- Stage 2: continue training on `combined_yolo`

## Main Contributions

- Detector-aware synthetic SAR ship dataset construction.
- Stage A image-quality filtering and Stage B-soft detector-aware filtering.
- Original-only vs Stage A vs Stage B-soft direct vs Stage B-soft curriculum comparison.
- SAR-SmallShip-YOLO-P2 and SAR-SmallShip-YOLO-P2ECA architectural ablations.
- RT-DETR-L non-YOLO / transformer-detector validation.
- YOLO11s newer YOLO-family validation.
- YOLOv8s multi-seed stability analysis.
- Image-level SAR fidelity/statistical audit.
- Public code, reports, dataset archive reference, and checksum.

## Main Results

| Method | Detector | Combined val mAP50-95 | HRSID test mAP50-95 | SSDD test mAP50-95 |
|---|---|---:|---:|---:|
| Original-only YOLOv8s | YOLOv8s | 0.935 | 0.935 | 0.894 |
| Stage B-soft + curriculum YOLOv8s | YOLOv8s | 0.966 | 0.967 | 0.954 |
| SAR-SmallShip-YOLO-P2 curriculum | YOLOv8s + P2 | 0.968 | 0.968 | 0.946 |
| SAR-SmallShip-YOLO-P2ECA curriculum | YOLOv8s + P2 + ECA | 0.966 | 0.967 | 0.951 |
| RT-DETR Stage B-soft curriculum | RT-DETR-L | 0.748 | 0.770 | 0.726 |
| YOLO11s Stage B-soft curriculum | YOLO11s | 0.949 | 0.951 | 0.916 |

## YOLO11s Ablation

| Method | Combined val mAP50-95 | HRSID test mAP50-95 | SSDD test mAP50-95 |
|---|---:|---:|---:|
| YOLO11s original-only | 0.919 | 0.920 | 0.883 |
| YOLO11s Stage B-soft curriculum | 0.949 | 0.951 | 0.916 |
| Gain | +0.030 | +0.031 | +0.033 |

## RT-DETR Non-YOLO Validation

| Method | Combined val mAP50-95 | HRSID test mAP50-95 | SSDD test mAP50-95 |
|---|---:|---:|---:|
| RT-DETR original-only | 0.634 | 0.698 | 0.643 |
| RT-DETR Stage B-soft direct | 0.645 | 0.674 | 0.625 |
| RT-DETR Stage B-soft curriculum | 0.748 | 0.770 | 0.726 |

## Multi-seed Stability

| Method | Combined val mean | HRSID mean | SSDD mean |
|---|---:|---:|---:|
| Original-only YOLOv8s | 0.939 | 0.940 | 0.906 |
| Stage B-soft curriculum YOLOv8s | 0.966 | 0.967 | 0.940 |

## Dataset Access

The full dataset archive is stored externally because GitHub is not suitable for large dataset binaries.

- Dataset archive: `release_dataset_newdata_final_v1.tar.gz`
- SHA256: `7060499f2f24cf0fa9c98f42d024e983bd275d68ec52c2412011bebccbef8dd8`
- Google Drive: https://drive.google.com/drive/folders/11uX0ZFEIOKeEqdOmo-dobozEsPh3fvtk

See:

- [`docs/DATASET_RELEASE.md`](docs/DATASET_RELEASE.md)
- [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md)
- [`docs/REPRODUCE.md`](docs/REPRODUCE.md)

## Repository Structure

- `src/sr_realesrgan/` - Real-ESRGAN SR generation, filtering, YOLO training, and custom detector scripts.
- `src/non_yolo/` - RT-DETR training and evaluation scripts.
- `src/yolo11/` - YOLO11s ablation script.
- `src/seed_stability/` - YOLOv8s multi-seed stability script.
- `src/analysis/` - SAR fidelity/statistical audit script.
- `src/finalize/` - final artifact freezing and table-building script.
- `configs/` - custom YOLO P2 and P2ECA configs.
- `reports/` - result tables, fidelity audit, multi-seed reports, and qualitative outputs.
- `docs/` - dataset release notes, dataset card, and reproduction guide.

## Quick Setup

Install dependencies:

    pip install -r requirements.txt

## Example Evaluation

    yolo detect val model=path/to/best.pt data=path/to/dataset.yaml imgsz=128 conf=0.001 iou=0.7

## Important Interpretation

This work does not claim full physical, phase-level, or radiometric SAR reconstruction fidelity. The accepted Stage B-soft samples are detector-aware, image-level synthetic SR samples derived from real SAR crops. The dataset is positioned as a detection-oriented synthetic SAR benchmark.

## License Notice

Code is released under the license in this repository. Dataset redistribution and use are subject to the licenses and terms of the original SAR datasets from which the benchmark is derived.
