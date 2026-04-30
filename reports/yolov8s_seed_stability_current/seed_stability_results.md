# YOLOv8s Multi-seed Stability - newdata-final

Extra seeds: 1 and 2. All evaluations use imgsz=128, conf=0.001, iou=0.7.

## Raw seed results

| Seed | Method | Training | Dataset | Precision | Recall | mAP50 | mAP50-95 |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | YOLOv8s original-only | real_only | combined_val | 0.993 | 0.990 | 0.995 | 0.941 |
| 1 | YOLOv8s original-only | real_only | hrsid_test | 0.988 | 0.986 | 0.995 | 0.940 |
| 1 | YOLOv8s original-only | real_only | ssdd_test | 0.995 | 0.985 | 0.994 | 0.910 |
| 1 | YOLOv8s Stage B-soft curriculum | real_only -> combined_yolo | combined_val | 0.993 | 0.994 | 0.995 | 0.967 |
| 1 | YOLOv8s Stage B-soft curriculum | real_only -> combined_yolo | hrsid_test | 0.994 | 0.989 | 0.995 | 0.968 |
| 1 | YOLOv8s Stage B-soft curriculum | real_only -> combined_yolo | ssdd_test | 0.996 | 0.990 | 0.995 | 0.943 |
| 2 | YOLOv8s original-only | real_only | combined_val | 0.995 | 0.990 | 0.995 | 0.936 |
| 2 | YOLOv8s original-only | real_only | hrsid_test | 0.986 | 0.991 | 0.995 | 0.939 |
| 2 | YOLOv8s original-only | real_only | ssdd_test | 0.996 | 0.985 | 0.995 | 0.902 |
| 2 | YOLOv8s Stage B-soft curriculum | real_only -> combined_yolo | combined_val | 0.993 | 0.994 | 0.995 | 0.965 |
| 2 | YOLOv8s Stage B-soft curriculum | real_only -> combined_yolo | hrsid_test | 0.994 | 0.988 | 0.995 | 0.966 |
| 2 | YOLOv8s Stage B-soft curriculum | real_only -> combined_yolo | ssdd_test | 0.984 | 0.992 | 0.995 | 0.938 |

## Mean +- std mAP50-95

| Method | Dataset | Mean mAP50-95 | Std | n |
|---|---|---:|---:|---:|
| YOLOv8s original-only | combined_val | 0.939 | 0.0030 | 2 |
| YOLOv8s original-only | hrsid_test | 0.940 | 0.0001 | 2 |
| YOLOv8s original-only | ssdd_test | 0.906 | 0.0059 | 2 |
| YOLOv8s Stage B-soft curriculum | combined_val | 0.966 | 0.0012 | 2 |
| YOLOv8s Stage B-soft curriculum | hrsid_test | 0.967 | 0.0018 | 2 |
| YOLOv8s Stage B-soft curriculum | ssdd_test | 0.940 | 0.0035 | 2 |
