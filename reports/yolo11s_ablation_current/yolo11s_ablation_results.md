# YOLO11s Ablation - newdata-final

All evaluations use `imgsz=128`, `conf=0.001`, and `iou=0.7`.

| Method | Training | Dataset | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---:|---:|---:|---:|
| YOLO11s original-only | real_only | combined_val | 0.993 | 0.988 | 0.995 | 0.919 |
| YOLO11s original-only | real_only | hrsid_test | 0.989 | 0.987 | 0.995 | 0.920 |
| YOLO11s original-only | real_only | ssdd_test | 0.992 | 0.986 | 0.994 | 0.883 |
| YOLO11s Stage B-soft curriculum | real_only -> combined_yolo | combined_val | 0.996 | 0.990 | 0.995 | 0.949 |
| YOLO11s Stage B-soft curriculum | real_only -> combined_yolo | hrsid_test | 0.988 | 0.991 | 0.995 | 0.951 |
| YOLO11s Stage B-soft curriculum | real_only -> combined_yolo | ssdd_test | 0.995 | 0.984 | 0.995 | 0.916 |
