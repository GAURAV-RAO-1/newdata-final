# Methodology and Experiments Draft - newdata-final

## Methodology

This work proposes a detector-aware synthetic SAR ship dataset construction pipeline. The goal is not to generate arbitrary ships from noise, but to create detector-usable super-resolved SAR ship samples derived from real SAR crops.

The pipeline first merges and organizes real SAR ship data into YOLO-format detection splits. Low-resolution inputs are then generated and super-resolved using the Real-ESRGAN x2 branch. The generated SR samples are filtered using a two-stage acceptance process. Stage A performs image-quality screening, while Stage B-soft applies detector-aware acceptance to retain synthetic samples that preserve useful ship morphology and remain beneficial for object detection.

The final accepted Stage B-soft synthetic set is combined with the original real SAR samples. The final combined dataset contains 24,204 training images, 5,863 validation images, and 4,829 test images.

## Experimental Setup

The primary detector is YOLOv8s. Additional SAR-specific architectural ablations include SAR-SmallShip-YOLO-P2 and SAR-SmallShip-YOLO-P2ECA. To verify that the proposed dataset is not YOLO-specific, RT-DETR-L is also evaluated as a non-YOLO detector.

All evaluations use image size 128, confidence threshold 0.001, and IoU threshold 0.7. The main metric is mAP50-95, supported by precision, recall, and mAP50.

## Key Results

YOLOv8s original-only achieves HRSID mAP50-95 of 0.935 and SSDD mAP50-95 of 0.894. Stage B-soft + curriculum improves these to 0.967 on HRSID and 0.954 on SSDD.

RT-DETR-L further validates the approach beyond YOLO. RT-DETR original-only achieves mAP50-95 values of 0.634 on combined validation, 0.698 on HRSID, and 0.643 on SSDD. RT-DETR Stage B-soft curriculum improves these to 0.748, 0.770, and 0.726 respectively.

## Interpretation

The results show that direct synthetic addition is not always sufficient, especially for RT-DETR. However, the real-only to combined curriculum consistently improves performance across YOLO and RT-DETR. This supports the claim that detector-aware synthetic SR data is most effective when introduced through controlled curriculum training.
