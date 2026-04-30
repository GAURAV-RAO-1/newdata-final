from __future__ import annotations
from pathlib import Path
import sys
import torch
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from custom_yolo.eca_module import ECA

tasks.ECA = ECA

WEIGHTS = ROOT / "runs/sar_smallship_yolov8s_p2eca_curriculum_stage2_combined/exp/weights/best.pt"

EVALS = [
    (
        "p2eca_combined_val",
        ROOT / "data/detector_yolo_exports/newdata_full_realesrgan_x2_stagebsoft_v1_dataset/combined_yolo/dataset.yaml"
    ),
    (
        "p2eca_hrsid_test",
        ROOT / "data/detector_yolo_generalization/original_only_hrsid_test/dataset.yaml"
    ),
    (
        "p2eca_ssdd_test",
        ROOT / "data/detector_yolo_generalization/original_only_ssdd_test/dataset.yaml"
    ),
]

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}")

    print(f"Using device: {device}")
    print(f"Weights: {WEIGHTS}")

    model = YOLO(str(WEIGHTS))

    for name, data_yaml in EVALS:
        print(f"\n=== Evaluating: {name} ===")
        print(f"Data: {data_yaml}")

        model.val(
            data=str(data_yaml),
            imgsz=128,
            device=device,
            conf=0.001,
            iou=0.7,
            project=str(ROOT / "runs/sar_smallship_yolov8s_p2eca_eval"),
            name=name,
            workers=0,
            verbose=True
        )

if __name__ == "__main__":
    main()
