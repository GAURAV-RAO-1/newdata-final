from __future__ import annotations
from pathlib import Path
from ultralytics import YOLO
import torch

ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = ROOT / "data/detector_yolo_exports/srgan_x2_raw_combined/combined_yolo/dataset.yaml"
PROJECT_DIR = ROOT / "runs/yolov8s_srgan_x2_raw_combined"

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Dataset YAML: {DATA_YAML}")

    model = YOLO("yolov8s.pt")

    model.train(
        data=str(DATA_YAML),
        imgsz=128,
        epochs=20,
        batch=16,
        device=device,
        workers=0,
        project=str(PROJECT_DIR),
        name="exp",
        pretrained=True,
        patience=5,
        save=True,
        verbose=True
    )

if __name__ == "__main__":
    main()
