from __future__ import annotations
from pathlib import Path
import torch

try:
    from ultralytics import RTDETR
    ModelClass = RTDETR
except Exception:
    from ultralytics import YOLO
    ModelClass = YOLO

ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = ROOT / "data/detector_yolo_exports/newdata_full_realesrgan_x2_stagebsoft_v1_dataset/real_only/dataset.yaml"
PROJECT_DIR = ROOT / "runs/rtdetr_original_only"

def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Dataset YAML: {DATA_YAML}")

    model = ModelClass("rtdetr-l.pt")

    model.train(
        data=str(DATA_YAML),
        imgsz=128,
        epochs=20,
        batch=24,
        device=device,
        workers=0,
        project=str(PROJECT_DIR),
        name="exp",
        patience=5,
        save=True,
        deterministic=False,
        verbose=True
    )

if __name__ == "__main__":
    main()
