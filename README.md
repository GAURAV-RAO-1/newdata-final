# newdata-final: Detector-Aware Synthetic SAR Ship Detection Dataset

This repository contains the code, configuration files, and result tables for `newdata-final`, a detector-aware synthetic SAR ship detection dataset pipeline.

## Overview

The project builds a SAR ship detection benchmark by combining original SAR ship samples with accepted super-resolved synthetic samples. Synthetic samples are generated using a Real-ESRGAN x2 branch and filtered using a two-stage quality and detector-aware acceptance pipeline.

The strongest training setting is curriculum learning:

- Stage 1: train on `real_only`
- Stage 2: continue training on `combined_yolo`

## Main Contributions

- Detector-aware synthetic SAR ship dataset construction.
- Stage A quality filtering and Stage B-soft detector-aware filtering.
- Original-only vs Stage A vs Stage B-soft direct vs Stage B-soft curriculum comparison.
- SAR-SmallShip-YOLO-P2 and P2ECA architecture ablations.
- RT-DETR-L non-YOLO validation.
- Multi-seed stability analysis.
- Detailed precision, recall, mAP50, and mAP50-95 reporting.

## Main Results

| Method | Detector | Combined val mAP50-95 | HRSID test mAP50-95 | SSDD test mAP50-95 |
|---|---|---:|---:|---:|
| Original-only YOLOv8s | YOLOv8s | 0.935 | 0.935 | 0.894 |
| Stage B-soft + curriculum YOLOv8s | YOLOv8s | 0.966 | 0.967 | 0.954 |
| SAR-SmallShip-YOLO-P2 curriculum | YOLOv8s + P2 | 0.968 | 0.968 | 0.946 |
| SAR-SmallShip-YOLO-P2ECA curriculum | YOLOv8s + P2 + ECA | 0.966 | 0.967 | 0.951 |
| RT-DETR Stage B-soft curriculum | RT-DETR-L | 0.748 | 0.770 | 0.726 |

## RT-DETR Non-YOLO Validation

| Method | Combined val mAP50-95 | HRSID test mAP50-95 | SSDD test mAP50-95 |
|---|---:|---:|---:|
| RT-DETR original-only | 0.634 | 0.698 | 0.643 |
| RT-DETR Stage B-soft direct | 0.645 | 0.674 | 0.625 |
| RT-DETR Stage B-soft curriculum | 0.748 | 0.770 | 0.726 |

## Multi-seed Stability

Extra seeds 1 and 2 confirm stable improvement.

| Method | Combined val mean | HRSID mean | SSDD mean |
|---|---:|---:|---:|
| Original-only YOLOv8s | 0.939 | 0.940 | 0.906 |
| Stage B-soft curriculum YOLOv8s | 0.966 | 0.967 | 0.940 |

## Dataset Access

The full dataset and trained weights are stored externally because they are large binary artifacts.

- Dataset archive: `release_dataset_newdata_final_v1.tar.gz`
- SHA256: `7060499f2f24cf0fa9c98f42d024e983bd275d68ec52c2412011bebccbef8dd8`
- Google Drive: https://drive.google.com/drive/folders/11uX0ZFEIOKeEqdOmo-dobozEsPh3fvtk

## Repository Structure

- `src/sr_realesrgan/` - Real-ESRGAN SR generation, filtering, YOLO training, and custom detector scripts.
- `src/non_yolo/` - RT-DETR training and evaluation scripts.
- `src/seed_stability/` - YOLOv8s multi-seed stability script.
- `src/finalize/` - final artifact freezing and table-building script.
- `configs/` - custom YOLO P2 and P2ECA configs.
- `reports/` - final result tables, RT-DETR validation, multi-seed results, and qualitative audit outputs.

## Training Example

`yolo detect train model=yolov8s.pt data=path/to/combined_yolo/dataset.yaml imgsz=128 epochs=20 batch=16`

## Evaluation Example

`yolo detect val model=path/to/best.pt data=path/to/dataset.yaml imgsz=128 conf=0.001 iou=0.7`

## Important Interpretation

This project does not claim that synthetic data always improves every detector directly. The strongest conclusion is:

> Detector-aware Stage B-soft synthetic SR data improves SAR ship detection generalization most consistently when introduced through real-only to combined curriculum training.

## License Notice

This repository may rely on external SAR datasets. Users must respect the licenses and terms of the original datasets. If external dataset terms restrict redistribution, release only scripts, labels, manifests, and derived metadata publicly.
