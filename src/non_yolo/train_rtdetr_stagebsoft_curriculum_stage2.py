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

INIT_WEIGHTS = Path(r"C:/Users/johnm/projects/newdata/transfer_rtdetr_newdata/runs/rtdetr_original_only/exp-2/weights/best.pt")
DATA_YAML = ROOT / "data/detector_yolo_exports/newdata_full_realesrgan_x2_stagebsoft_v1_dataset/combined_yolo/dataset.yaml"
PROJECT_DIR = ROOT / "runs/rtdetr_stagebsoft_curriculum_stage2_combined"

def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Init weights: {INIT_WEIGHTS}")
    print(f"Dataset YAML: {DATA_YAML}")

    if not INIT_WEIGHTS.exists():
        raise FileNotFoundError(f"Original-only RT-DETR weights not found: {INIT_WEIGHTS}")

    model = ModelClass(str(INIT_WEIGHTS))

    model.train(
        data=str(DATA_YAML),
        imgsz=128,
        epochs=20,
        batch=24,
        device=device,
        workers=0,
        project=str(PROJECT_DIR),
        name="exp",
        patience=8,
        save=True,
        deterministic=False,
        verbose=True
    )

if __name__ == "__main__":
    main()

