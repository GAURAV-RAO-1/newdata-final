# Detailed Metrics Report — newdata-final

All evaluations use `imgsz=128`, `conf=0.001`, and `iou=0.7`.

| Method | Dataset | Precision | Recall | mAP50 | mAP50-95 | Params(M) | GFLOPs | Inference ms/img |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Original-only YOLOv8s | combined_val | 0.995 | 0.989 | 0.995 | 0.935 | 11.126 | 28.400 | 0.808 |
| Original-only YOLOv8s | hrsid_test | 0.990 | 0.988 | 0.995 | 0.935 | 11.126 | 28.400 | 0.751 |
| Original-only YOLOv8s | ssdd_test | 0.995 | 0.989 | 0.995 | 0.894 | 11.126 | 28.400 | 1.404 |
| Stage A SR YOLOv8s | combined_val | 0.992 | 0.993 | 0.995 | 0.941 | 11.126 | 28.400 | 0.655 |
| Stage A SR YOLOv8s | hrsid_test | 0.992 | 0.988 | 0.995 | 0.946 | 11.126 | 28.400 | 0.680 |
| Stage A SR YOLOv8s | ssdd_test | 0.987 | 0.989 | 0.995 | 0.906 | 11.126 | 28.400 | 1.004 |
| Stage B-soft YOLOv8s direct | combined_val | 0.995 | 0.993 | 0.995 | 0.959 | 11.126 | 28.400 | 0.653 |
| Stage B-soft YOLOv8s direct | hrsid_test | 0.990 | 0.991 | 0.995 | 0.959 | 11.126 | 28.400 | 0.698 |
| Stage B-soft YOLOv8s direct | ssdd_test | 0.992 | 0.994 | 0.995 | 0.942 | 11.126 | 28.400 | 0.995 |
| Stage B-soft + curriculum YOLOv8s | combined_val | 0.996 | 0.991 | 0.995 | 0.966 | 11.126 | 28.400 | 0.660 |
| Stage B-soft + curriculum YOLOv8s | hrsid_test | 0.991 | 0.990 | 0.995 | 0.967 | 11.126 | 28.400 | 0.709 |
| Stage B-soft + curriculum YOLOv8s | ssdd_test | 0.990 | 0.994 | 0.995 | 0.954 | 11.126 | 28.400 | 0.993 |
| SAR-SmallShip-YOLO-P2 direct | combined_val | 0.993 | 0.994 | 0.995 | 0.959 | 10.627 | 36.600 | 0.829 |
| SAR-SmallShip-YOLO-P2 direct | hrsid_test | 0.990 | 0.991 | 0.995 | 0.959 | 10.627 | 36.600 | 0.846 |
| SAR-SmallShip-YOLO-P2 direct | ssdd_test | 0.997 | 0.997 | 0.995 | 0.930 | 10.627 | 36.600 | 1.324 |
| SAR-SmallShip-YOLO-P2 curriculum | combined_val | 0.995 | 0.993 | 0.995 | 0.968 | 10.627 | 36.600 | 0.870 |
| SAR-SmallShip-YOLO-P2 curriculum | hrsid_test | 0.993 | 0.990 | 0.995 | 0.968 | 10.627 | 36.600 | 0.843 |
| SAR-SmallShip-YOLO-P2 curriculum | ssdd_test | 0.982 | 0.994 | 0.995 | 0.946 | 10.627 | 36.600 | 1.198 |
| SAR-SmallShip-YOLO-P2ECA curriculum | combined_val | 0.994 | 0.994 | 0.995 | 0.966 | 10.627 | 36.600 | 0.806 |
| SAR-SmallShip-YOLO-P2ECA curriculum | hrsid_test | 0.993 | 0.987 | 0.995 | 0.967 | 10.627 | 36.600 | 0.852 |
| SAR-SmallShip-YOLO-P2ECA curriculum | ssdd_test | 0.995 | 0.992 | 0.995 | 0.951 | 10.627 | 36.600 | 1.277 |
