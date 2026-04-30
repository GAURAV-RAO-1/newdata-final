# newdata-final: Detector-Aware Synthetic SAR Ship Detection Dataset

This repository contains the code, configuration files, and result tables for `newdata-final`, a detector-aware synthetic SAR ship detection dataset pipeline.

## Overview

The project builds a SAR ship detection benchmark by combining original SAR ship samples with accepted super-resolved synthetic samples. Synthetic samples are generated using a Real-ESRGAN x2 branch and filtered using a two-stage quality and detector-aware acceptance pipeline.

The strongest training setting is curriculum learning:

Stage 1: train on `real_only`  
Stage 2: continue training on `combined_yolo`

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

## Multi-seed Stability

Extra seeds 1 and 2 confirm stable improvement:

| Method | Combined val mean | HRSID mean | SSDD mean |
|---|---:|---:|---:|
| Original-only YOLOv8s | 0.939 | 0.940 | 0.906 |
| Stage B-soft curriculum YOLOv8s | 0.966 | 0.967 | 0.940 |

## Repository Structure

```text
src/
  sr_realesrgan/
  non_yolo/
  seed_stability/
  finalize/

configs/
  custom YOLO configs

reports/
  final result tables
  RT-DETR validation tables
  multi-seed stability results
  qualitative audit outputs
EOD
