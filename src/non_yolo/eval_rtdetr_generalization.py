from __future__ import annotations
from pathlib import Path
import csv
import json
import torch

try:
    from ultralytics import RTDETR
    ModelClass = RTDETR
except Exception:
    from ultralytics import YOLO
    ModelClass = YOLO

ROOT = Path(__file__).resolve().parents[2]
DEVICE = 0 if torch.cuda.is_available() else "cpu"

OUT_DIR = ROOT / "reports/rtdetr_generalization_current"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "combined_val": ROOT / "data/detector_yolo_exports/newdata_full_realesrgan_x2_stagebsoft_v1_dataset/combined_yolo/dataset.yaml",
    "hrsid_test": ROOT / "data/detector_yolo_generalization/original_only_hrsid_test/dataset.yaml",
    "ssdd_test": ROOT / "data/detector_yolo_generalization/original_only_ssdd_test/dataset.yaml",
}

METHODS = [
    {
        "method": "RT-DETR original-only",
        "weights": ROOT / "runs/rtdetr_original_only/exp-2/weights/best.pt",
        "training": "real_only"
    },
    {
        "method": "RT-DETR Stage B-soft direct",
        "weights": ROOT / "runs/rtdetr_stagebsoft_direct/exp/weights/best.pt",
        "training": "combined_yolo direct"
    },
    {
        "method": "RT-DETR Stage B-soft curriculum",
        "weights": ROOT / "runs/rtdetr_stagebsoft_curriculum_stage2_combined/exp/weights/best.pt",
        "training": "real_only -> combined_yolo"
    },
]

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def fmt(x):
    return "—" if x is None else f"{x:.3f}"

def main():
    print(f"Using device: {DEVICE}")

    rows = []

    for method in METHODS:
        weights = method["weights"]

        if not weights.exists():
            print(f"SKIP missing weights: {method['method']} -> {weights}")
            continue

        print(f"\n==============================")
        print(f"METHOD: {method['method']}")
        print(f"WEIGHTS: {weights}")
        print(f"==============================")

        model = ModelClass(str(weights))

        for dataset_name, data_yaml in DATASETS.items():
            print(f"\n--- Evaluating {method['method']} on {dataset_name} ---")

            metrics = model.val(
                data=str(data_yaml),
                imgsz=128,
                device=DEVICE,
                conf=0.001,
                iou=0.7,
                workers=0,
                project=str(ROOT / "runs/rtdetr_generalization_eval"),
                name=f"{method['method'].replace(' ', '_').replace('-', '_')}_{dataset_name}",
                exist_ok=True,
                verbose=True
            )

            speed = getattr(metrics, "speed", {}) or {}

            rows.append({
                "method": method["method"],
                "training": method["training"],
                "dataset": dataset_name,
                "precision": safe_float(metrics.box.mp),
                "recall": safe_float(metrics.box.mr),
                "map50": safe_float(metrics.box.map50),
                "map50_95": safe_float(metrics.box.map),
                "preprocess_ms": safe_float(speed.get("preprocess")),
                "inference_ms": safe_float(speed.get("inference")),
                "postprocess_ms": safe_float(speed.get("postprocess")),
                "weights": str(weights),
                "data_yaml": str(data_yaml)
            })

    json_path = OUT_DIR / "rtdetr_generalization_results.json"
    csv_path = OUT_DIR / "rtdetr_generalization_results.csv"
    md_path = OUT_DIR / "rtdetr_generalization_results.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# RT-DETR Generalization Results — newdata-final\n\n")
        f.write("All evaluations use `imgsz=128`, `conf=0.001`, and `iou=0.7`.\n\n")
        f.write("| Method | Training | Dataset | Precision | Recall | mAP50 | mAP50-95 | Inference ms/img |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|\n")

        for r in rows:
            f.write(
                f"| {r['method']} "
                f"| {r['training']} "
                f"| {r['dataset']} "
                f"| {fmt(r['precision'])} "
                f"| {fmt(r['recall'])} "
                f"| {fmt(r['map50'])} "
                f"| {fmt(r['map50_95'])} "
                f"| {fmt(r['inference_ms'])} |\n"
            )

    print("\nDone.")
    print(md_path)
    print(csv_path)
    print(json_path)

if __name__ == "__main__":
    main()
