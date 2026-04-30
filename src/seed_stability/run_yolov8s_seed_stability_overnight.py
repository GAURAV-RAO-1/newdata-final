from __future__ import annotations

from pathlib import Path
import csv
import json
import time
import torch
from ultralytics import YOLO

ROOT = Path("/Users/gaurav/newdata")

REAL_DATA = ROOT / "data/detector_yolo_exports/newdata_full_realesrgan_x2_stagebsoft_v1_dataset/real_only/dataset.yaml"
COMBINED_DATA = ROOT / "data/detector_yolo_exports/newdata_full_realesrgan_x2_stagebsoft_v1_dataset/combined_yolo/dataset.yaml"
HRSID_DATA = ROOT / "data/detector_yolo_generalization/original_only_hrsid_test/dataset.yaml"
SSDD_DATA = ROOT / "data/detector_yolo_generalization/original_only_ssdd_test/dataset.yaml"

REPORT_DIR = ROOT / "reports/yolov8s_seed_stability_current"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ROOT = ROOT / "runs/yolov8s_seed_stability"
RUN_ROOT.mkdir(parents=True, exist_ok=True)

SEEDS = [1, 2]

if torch.cuda.is_available():
    DEVICE = 0
    WORKERS = 4
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    WORKERS = 0
else:
    DEVICE = "cpu"
    WORKERS = 0

IMG_SIZE = 128
BATCH = 16
EPOCHS_ORIGINAL = 20
EPOCHS_CURRICULUM = 20

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def run_val(weights: Path, data: Path, name: str):
    print(f"\n--- Evaluating {name} ---")
    print("Weights:", weights)
    print("Data:", data)

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        imgsz=IMG_SIZE,
        device=DEVICE,
        conf=0.001,
        iou=0.7,
        workers=WORKERS,
        project=str(RUN_ROOT / "eval"),
        name=name,
        exist_ok=True,
        verbose=True,
    )

    return {
        "precision": safe_float(metrics.box.mp),
        "recall": safe_float(metrics.box.mr),
        "map50": safe_float(metrics.box.map50),
        "map50_95": safe_float(metrics.box.map),
    }

def train_original(seed: int) -> Path:
    project = RUN_ROOT / f"seed{seed}_original_only"
    best = project / "exp/weights/best.pt"

    if best.exists():
        print(f"\nSKIP original-only seed {seed}; best.pt already exists:")
        print(best)
        return best

    print(f"\n==============================")
    print(f"TRAINING: YOLOv8s original-only | seed={seed}")
    print(f"==============================")

    model = YOLO("yolov8s.pt")
    model.train(
        data=str(REAL_DATA),
        imgsz=IMG_SIZE,
        epochs=EPOCHS_ORIGINAL,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(project),
        name="exp",
        seed=seed,
        deterministic=True,
        patience=5,
        save=True,
        plots=True,
        verbose=True,
    )

    if not best.exists():
        raise FileNotFoundError(f"Missing original best.pt after training: {best}")

    return best

def train_curriculum(seed: int, original_best: Path) -> Path:
    project = RUN_ROOT / f"seed{seed}_stagebsoft_curriculum"
    best = project / "exp/weights/best.pt"

    if best.exists():
        print(f"\nSKIP curriculum seed {seed}; best.pt already exists:")
        print(best)
        return best

    print(f"\n==============================")
    print(f"TRAINING: YOLOv8s Stage B-soft curriculum | seed={seed}")
    print(f"Init weights: {original_best}")
    print(f"==============================")

    model = YOLO(str(original_best))
    model.train(
        data=str(COMBINED_DATA),
        imgsz=IMG_SIZE,
        epochs=EPOCHS_CURRICULUM,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(project),
        name="exp",
        seed=seed,
        deterministic=True,
        patience=8,
        save=True,
        plots=True,
        verbose=True,
    )

    if not best.exists():
        raise FileNotFoundError(f"Missing curriculum best.pt after training: {best}")

    return best

def save_reports(rows):
    json_path = REPORT_DIR / "seed_stability_results.json"
    csv_path = REPORT_DIR / "seed_stability_results.csv"
    md_path = REPORT_DIR / "seed_stability_results.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    fieldnames = [
        "seed", "method", "training", "dataset",
        "precision", "recall", "map50", "map50_95", "weights"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    groups = {}
    for r in rows:
        key = (r["method"], r["dataset"])
        groups.setdefault(key, []).append(r["map50_95"])

    summary = []
    for (method, dataset), vals in groups.items():
        vals = [v for v in vals if v is not None]
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = var ** 0.5
        else:
            std = 0.0

        summary.append({
            "method": method,
            "dataset": dataset,
            "mean": mean,
            "std": std,
            "n": len(vals),
        })

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# YOLOv8s Multi-seed Stability - newdata-final\n\n")
        f.write("Extra seeds: 1 and 2. All evaluations use imgsz=128, conf=0.001, iou=0.7.\n\n")

        f.write("## Raw seed results\n\n")
        f.write("| Seed | Method | Training | Dataset | Precision | Recall | mAP50 | mAP50-95 |\n")
        f.write("|---:|---|---|---|---:|---:|---:|---:|\n")

        for r in rows:
            f.write(
                f"| {r['seed']} | {r['method']} | {r['training']} | {r['dataset']} | "
                f"{r['precision']:.3f} | {r['recall']:.3f} | {r['map50']:.3f} | {r['map50_95']:.3f} |\n"
            )

        f.write("\n## Mean +- std mAP50-95\n\n")
        f.write("| Method | Dataset | Mean mAP50-95 | Std | n |\n")
        f.write("|---|---|---:|---:|---:|\n")

        for s in summary:
            f.write(
                f"| {s['method']} | {s['dataset']} | "
                f"{s['mean']:.3f} | {s['std']:.4f} | {s['n']} |\n"
            )

    print("\nSaved:")
    print(json_path)
    print(csv_path)
    print(md_path)

def main():
    start = time.time()

    print("Using device:", DEVICE)
    print("Workers:", WORKERS)
    print("Batch:", BATCH)
    print("Seeds:", SEEDS)

    rows = []

    for seed in SEEDS:
        original_best = train_original(seed)

        for dataset_name, data in [
            ("combined_val", COMBINED_DATA),
            ("hrsid_test", HRSID_DATA),
            ("ssdd_test", SSDD_DATA),
        ]:
            m = run_val(original_best, data, f"seed{seed}_original_only_{dataset_name}")
            rows.append({
                "seed": seed,
                "method": "YOLOv8s original-only",
                "training": "real_only",
                "dataset": dataset_name,
                **m,
                "weights": str(original_best),
            })
            save_reports(rows)

        curriculum_best = train_curriculum(seed, original_best)

        for dataset_name, data in [
            ("combined_val", COMBINED_DATA),
            ("hrsid_test", HRSID_DATA),
            ("ssdd_test", SSDD_DATA),
        ]:
            m = run_val(curriculum_best, data, f"seed{seed}_stagebsoft_curriculum_{dataset_name}")
            rows.append({
                "seed": seed,
                "method": "YOLOv8s Stage B-soft curriculum",
                "training": "real_only -> combined_yolo",
                "dataset": dataset_name,
                **m,
                "weights": str(curriculum_best),
            })
            save_reports(rows)

    elapsed = (time.time() - start) / 3600
    print(f"\nDONE. Total hours: {elapsed:.2f}")
    print(REPORT_DIR / "seed_stability_results.md")

if __name__ == "__main__":
    main()
