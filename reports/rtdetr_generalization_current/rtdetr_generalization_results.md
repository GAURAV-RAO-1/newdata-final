# RT-DETR Generalization Results — newdata-final

All evaluations use `imgsz=128`, `conf=0.001`, and `iou=0.7`.

| Method | Training | Dataset | Precision | Recall | mAP50 | mAP50-95 | Inference ms/img |
|---|---|---|---:|---:|---:|---:|---:|
| RT-DETR original-only | real_only | combined_val | 0.909 | 0.902 | 0.943 | 0.634 | 2.515 |
| RT-DETR original-only | real_only | hrsid_test | 0.913 | 0.913 | 0.953 | 0.698 | 2.606 |
| RT-DETR original-only | real_only | ssdd_test | 0.942 | 0.979 | 0.979 | 0.643 | 3.184 |
| RT-DETR Stage B-soft direct | combined_yolo direct | combined_val | 0.923 | 0.943 | 0.953 | 0.645 | 2.749 |
| RT-DETR Stage B-soft direct | combined_yolo direct | hrsid_test | 0.929 | 0.939 | 0.956 | 0.674 | 2.791 |
| RT-DETR Stage B-soft direct | combined_yolo direct | ssdd_test | 0.959 | 0.982 | 0.975 | 0.625 | 2.972 |
| RT-DETR Stage B-soft curriculum | real_only -> combined_yolo | combined_val | 0.986 | 0.989 | 0.992 | 0.748 | 2.808 |
| RT-DETR Stage B-soft curriculum | real_only -> combined_yolo | hrsid_test | 0.983 | 0.985 | 0.991 | 0.770 | 2.862 |
| RT-DETR Stage B-soft curriculum | real_only -> combined_yolo | ssdd_test | 1.000 | 1.000 | 0.995 | 0.726 | 3.022 |
