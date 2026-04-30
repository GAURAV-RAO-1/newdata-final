# Reproduction Guide

This document gives the practical steps to reproduce the main `newdata-final` experiments.

## 1. Environment

Recommended Python version:

    python >= 3.10

Install dependencies:

    pip install -r requirements.txt

The original experiments were run on:

- macOS with Apple MPS for YOLOv8s, YOLO11s, multi-seed, and analysis runs
- Windows CUDA with NVIDIA RTX A5000 for RT-DETR-L runs

## 2. Dataset

Download the dataset archive:

- Google Drive: https://drive.google.com/drive/folders/11uX0ZFEIOKeEqdOmo-dobozEsPh3fvtk
- Archive: `release_dataset_newdata_final_v1.tar.gz`
- SHA256: `7060499f2f24cf0fa9c98f42d024e983bd275d68ec52c2412011bebccbef8dd8`

Verify checksum:

    shasum -a 256 release_dataset_newdata_final_v1.tar.gz

Expected:

    7060499f2f24cf0fa9c98f42d024e983bd275d68ec52c2412011bebccbef8dd8  release_dataset_newdata_final_v1.tar.gz

Extract:

    tar -xzf release_dataset_newdata_final_v1.tar.gz

Expected dataset folders:

    data/detector_yolo_exports/newdata_full_realesrgan_x2_stagebsoft_v1_dataset/
    data/detector_yolo_generalization/

## 3. Main YOLOv8s Training

The main protocol is curriculum learning:

- Stage 1: train on `real_only`
- Stage 2: continue training on `combined_yolo`

Relevant scripts:

    src/sr_realesrgan/train_yolov8s_newdata_curriculum_stage1_realonly.py
    src/sr_realesrgan/train_yolov8s_newdata_curriculum_stage2_combined.py

## 4. RT-DETR Validation

Relevant scripts:

    src/non_yolo/train_rtdetr_original_only.py
    src/non_yolo/train_rtdetr_stagebsoft_direct.py
    src/non_yolo/train_rtdetr_stagebsoft_curriculum_stage2.py
    src/non_yolo/eval_rtdetr_generalization.py

Expected report:

    reports/rtdetr_generalization_current/rtdetr_generalization_results.md

## 5. YOLO11s Ablation

Run:

    python3 src/yolo11/run_yolo11s_original_vs_stagebsoft_curriculum.py

Expected report:

    reports/yolo11s_ablation_current/yolo11s_ablation_results.md

## 6. YOLOv8s Multi-seed Stability

Run:

    python3 src/seed_stability/run_yolov8s_seed_stability_overnight.py

Expected report:

    reports/yolov8s_seed_stability_current/seed_stability_results.md

## 7. SAR Fidelity / Statistical Audit

Run:

    python3 src/analysis/build_sar_fidelity_audit.py

Expected report:

    reports/sar_fidelity_audit_current/sar_fidelity_audit.md

## 8. Main Expected Results

| Method | Combined val mAP50-95 | HRSID test mAP50-95 | SSDD test mAP50-95 |
|---|---:|---:|---:|
| Original-only YOLOv8s | 0.935 | 0.935 | 0.894 |
| Stage B-soft + curriculum YOLOv8s | 0.966 | 0.967 | 0.954 |
| RT-DETR original-only | 0.634 | 0.698 | 0.643 |
| RT-DETR Stage B-soft curriculum | 0.748 | 0.770 | 0.726 |
| YOLO11s original-only | 0.919 | 0.920 | 0.883 |
| YOLO11s Stage B-soft curriculum | 0.949 | 0.951 | 0.916 |
