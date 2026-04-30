from __future__ import annotations
from pathlib import Path
from ultralytics import YOLO
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = ROOT / "data/detector_yolo_exports/newdata_full_realesrgan_x2_stagebsoft_v1_dataset/combined_yolo/dataset.yaml"
INIT_WEIGHTS = ROOT / "runs/yolov8s_newdata_curriculum_stage1_realonly/exp/weights/best.pt"
PROJECT_DIR = ROOT / "runs/yolov8s_newdata_curriculum_stage2_combined"

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Init weights: {INIT_WEIGHTS}")

    model = YOLO(str(INIT_WEIGHTS))

    model.train(
        data=str(DATA_YAML),
        imgsz=128,
        epochs=20,
        batch=16,
        device=device,
        workers=0,
        project=str(PROJECT_DIR),
        name="exp",
        pretrained=False,
        patience=8,
        save=True,
        verbose=True
    )

if __name__ == "__main__":
    main()