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

MODEL_YAML = ROOT / "configs/sar_smallship_yolov8s_p2eca.yaml"
DATA_YAML = ROOT / "data/detector_yolo_exports/newdata_full_realesrgan_x2_stagebsoft_v1_dataset/real_only/dataset.yaml"
PROJECT_DIR = ROOT / "runs/sar_smallship_yolov8s_p2eca_curriculum_stage1_realonly"

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Model YAML: {MODEL_YAML}")
    print(f"Dataset YAML: {DATA_YAML}")

    model = YOLO(str(MODEL_YAML)).load("yolov8s.pt")

    model.train(
        data=str(DATA_YAML),
        imgsz=128,
        epochs=15,
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
