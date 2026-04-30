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

RUN_ROOT = ROOT / "runs/yolo11s_ablation"
REPORT_DIR = ROOT / "reports/yolo11s_ablation_current"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

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
SEED = 0


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def train_original():
    project = RUN_ROOT / "yolo11s_original_only"
    best = project / "exp/weights/best.pt"

    if best.exists():
        print("SKIP original-only; best.pt exists:", best)
        return best

    print("\n==============================")
    print("TRAINING: YOLO11s original-only")
    print("==============================")

    model = YOLO("yolo11s.pt")
    model.train(
        data=str(REAL_DATA),
        imgsz=IMG_SIZE,
        epochs=EPOCHS_ORIGINAL,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(project),
        name="exp",
        seed=SEED,
        deterministic=True,
        patience=5,
        save=True,
        plots=True,
        verbose=True,
    )

    if not best.exists():
        raise FileNotFoundError(best)

    return best


def train_curriculum(original_best):
    project = RUN_ROOT / "yolo11s_stagebsoft_curriculum"
    best = project / "exp/weights/best.pt"

    if best.exists():
        print("SKIP curriculum; best.pt exists:", best)
        return best

    print("\n==============================")
    print("TRAINING: YOLO11s Stage B-soft curriculum")
    print("==============================")

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
        seed=SEED,
        deterministic=True,
        patience=8,
        save=True,
        plots=True,
        verbose=True,
    )

    if not best.exists():
        raise FileNotFoundError(best)

    return best


def evaluate(weights, method, training):
    rows = []

    for dataset_name, data in [
        ("combined_val", COMBINED_DATA),
        ("hrsid_test", HRSID_DATA),
        ("ssdd_test", SSDD_DATA),
    ]:
        print(f"\n--- Evaluating {method} on {dataset_name} ---")

        model = YOLO(str(weights))
        metrics = model.val(
            data=str(data),
            imgsz=IMG_SIZE,
            device=DEVICE,
            conf=0.001,
            iou=0.7,
            workers=WORKERS,
            project=str(RUN_ROOT / "eval"),
            name=f"{method.replace(' ', '_')}_{dataset_name}",
            exist_ok=True,
            verbose=True,
        )

        rows.append({
            "method": method,
            "training": training,
            "dataset": dataset_name,
            "precision": safe_float(metrics.box.mp),
            "recall": safe_float(metrics.box.mr),
            "map50": safe_float(metrics.box.map50),
            "map50_95": safe_float(metrics.box.map),
            "weights": str(weights),
        })

    return rows


def save_report(rows):
    json_path = REPORT_DIR / "yolo11s_ablation_results.json"
    csv_path = REPORT_DIR / "yolo11s_ablation_results.csv"
    md_path = REPORT_DIR / "yolo11s_ablation_results.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    fieldnames = [
        "method", "training", "dataset",
        "precision", "recall", "map50", "map50_95", "weights"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# YOLO11s Ablation - newdata-final\n\n")
        f.write("All evaluations use `imgsz=128`, `conf=0.001`, and `iou=0.7`.\n\n")
        f.write("| Method | Training | Dataset | Precision | Recall | mAP50 | mAP50-95 |\n")
        f.write("|---|---|---|---:|---:|---:|---:|\n")

        for r in rows:
            f.write(
                f"| {r['method']} | {r['training']} | {r['dataset']} | "
                f"{r['precision']:.3f} | {r['recall']:.3f} | "
                f"{r['map50']:.3f} | {r['map50_95']:.3f} |\n"
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

    rows = []

    original_best = train_original()
    rows.extend(evaluate(
        original_best,
        "YOLO11s original-only",
        "real_only"
    ))
    save_report(rows)

    curriculum_best = train_curriculum(original_best)
    rows.extend(evaluate(
        curriculum_best,
        "YOLO11s Stage B-soft curriculum",
        "real_only -> combined_yolo"
    ))
    save_report(rows)

    elapsed = (time.time() - start) / 3600
    print(f"\nDONE. Total hours: {elapsed:.2f}")
    print(REPORT_DIR / "yolo11s_ablation_results.md")


if __name__ == "__main__":
    main()
