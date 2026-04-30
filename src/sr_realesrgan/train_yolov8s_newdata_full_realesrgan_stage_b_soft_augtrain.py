from __future__ import annotations
from pathlib import Path
from ultralytics import YOLO
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = ROOT / "data/detector_yolo_exports/newdata_full_realesrgan_x2_stagebsoft_v1_augtrain/combined_yolo/dataset.yaml"
PROJECT_DIR = ROOT / "runs/yolov8s_newdata_full_realesrgan_stage_b_soft_augtrain"

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

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