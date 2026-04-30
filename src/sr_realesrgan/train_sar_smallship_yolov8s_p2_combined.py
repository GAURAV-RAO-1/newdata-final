from __future__ import annotations
from pathlib import Path
from ultralytics import YOLO
import torch

ROOT = Path(__file__).resolve().parents[2]

MODEL_YAML = ROOT / "configs/sar_smallship_yolov8s_p2.yaml"
DATA_YAML = ROOT / "data/detector_yolo_exports/newdata_full_realesrgan_x2_stagebsoft_v1_dataset/combined_yolo/dataset.yaml"
PROJECT_DIR = ROOT / "runs/sar_smallship_yolov8s_p2_combined"

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model YAML: {MODEL_YAML}")
    print(f"Dataset YAML: {DATA_YAML}")

    # Build custom P2 model and partially load YOLOv8s pretrained weights
    model = YOLO(str(MODEL_YAML)).load("yolov8s.pt")

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
        patience=5,
        save=True,
        verbose=True
    )

if __name__ == "__main__":
    main()