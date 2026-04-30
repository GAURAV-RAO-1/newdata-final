from pathlib import Path
from datetime import datetime
import csv, json, shutil, random

ROOT = Path.cwd()
TS = datetime.now().strftime("%Y%m%d_%H%M%S")

OUT = ROOT / "reports" / "final_newdata_artifacts_current"
OUT.mkdir(parents=True, exist_ok=True)

FREEZE = ROOT / "reports" / f"frozen_rtdetr_generalization_{TS}"
FREEZE.mkdir(parents=True, exist_ok=True)

def copy_if_exists(src, dst):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        print("Missing, skipped:", src)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    print("Copied:", src)

# -----------------------------
# 1. Freeze RT-DETR results
# -----------------------------
copy_if_exists(ROOT / "reports/rtdetr_generalization_current", FREEZE / "rtdetr_generalization_current")
copy_if_exists(ROOT / "reports/rtdetr_windows_logs", FREEZE / "rtdetr_windows_logs")
copy_if_exists(ROOT / "src/non_yolo", FREEZE / "src_non_yolo")

copy_if_exists(ROOT / "runs/rtdetr_original_only/exp-2/weights/best.pt", FREEZE / "weights/rtdetr_original_only_best.pt")
copy_if_exists(ROOT / "runs/rtdetr_stagebsoft_direct/exp/weights/best.pt", FREEZE / "weights/rtdetr_stagebsoft_direct_best.pt")
copy_if_exists(ROOT / "runs/rtdetr_stagebsoft_curriculum_stage2_combined/exp/weights/best.pt", FREEZE / "weights/rtdetr_stagebsoft_curriculum_best.pt")

# -----------------------------
# 2. Final master table
# -----------------------------
rows = [
    ["Original-only YOLOv8s", "YOLOv8s", "real_only", "0.935", "0.935", "0.894", "Real-only YOLO baseline."],
    ["Stage A SR YOLOv8s", "YOLOv8s", "Original + Stage A accepted SR", "0.941", "0.946", "0.906", "Basic SR filtering gives small gains."],
    ["Stage B-soft YOLOv8s direct", "YOLOv8s", "combined_yolo direct", "0.959", "0.959", "0.942", "Direct Stage B-soft synthetic addition."],
    ["Stage B-soft + curriculum YOLOv8s", "YOLOv8s", "real_only -> combined_yolo", "0.966", "0.967", "0.954", "Best YOLOv8s cross-dataset robustness."],
    ["SAR-SmallShip-YOLO-P2 direct", "YOLOv8s + P2", "combined_yolo direct", "0.959", "0.959", "0.930", "P2 direct architectural ablation."],
    ["SAR-SmallShip-YOLO-P2 curriculum", "YOLOv8s + P2", "real_only -> combined_yolo", "0.968", "0.968", "0.946", "Best combined/HRSID result."],
    ["SAR-SmallShip-YOLO-P2ECA curriculum", "YOLOv8s + P2 + ECA", "real_only -> combined_yolo", "0.966", "0.967", "0.951", "ECA improves SSDD over P2."],
    ["RT-DETR original-only", "RT-DETR-L", "real_only", "0.634", "0.698", "0.643", "Non-YOLO real-only baseline."],
    ["RT-DETR Stage B-soft direct", "RT-DETR-L", "combined_yolo direct", "0.645", "0.674", "0.625", "Direct synthetic addition is mixed for RT-DETR."],
    ["RT-DETR Stage B-soft curriculum", "RT-DETR-L", "real_only -> combined_yolo", "0.748", "0.770", "0.726", "Non-YOLO validation confirms curriculum gains."],
]

csv_path = OUT / "final_master_results_table.csv"
md_path = OUT / "final_master_results_table.md"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Method", "Detector", "Training", "Combined val mAP50-95", "HRSID test mAP50-95", "SSDD test mAP50-95", "Note"])
    w.writerows(rows)

with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Final Master Results Table - newdata-final\n\n")
    f.write("| Method | Detector | Training | Combined val mAP50-95 | HRSID test mAP50-95 | SSDD test mAP50-95 | Note |\n")
    f.write("|---|---|---|---:|---:|---:|---|\n")
    for r in rows:
        f.write("| " + " | ".join(r) + " |\n")

# -----------------------------
# 3. Qualitative figure
# -----------------------------
qual_dir = OUT / "qualitative_figures"
qual_dir.mkdir(parents=True, exist_ok=True)

try:
    from PIL import Image, ImageDraw

    img_root = ROOT / "data/detector_yolo_exports/newdata_full_realesrgan_x2_stagebsoft_v1_dataset/combined_yolo/images"
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    imgs = []
    for split in ["train", "val", "test"]:
        d = img_root / split
        if d.exists():
            for p in d.iterdir():
                low = p.name.lower()
                if p.suffix.lower() in exts and ("sr_realesrgan" in low or low.startswith("sr_")):
                    imgs.append(p)

    random.seed(7)
    random.shuffle(imgs)
    imgs = imgs[:16]

    thumb = 150
    title_h = 40
    label_h = 20
    cols = 4
    rows_n = 4

    canvas = Image.new("RGB", (cols * thumb, title_h + rows_n * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "Accepted Stage B-soft Synthetic SR Samples", fill="black")

    for i, p in enumerate(imgs):
        img = Image.open(p).convert("RGB")
        img.thumbnail((thumb, thumb))
        tile = Image.new("RGB", (thumb, thumb), "white")
        tile.paste(img, ((thumb - img.width)//2, (thumb - img.height)//2))

        x = (i % cols) * thumb
        y = title_h + (i // cols) * (thumb + label_h)
        canvas.paste(tile, (x, y))
        draw.text((x + 4, y + thumb + 2), p.stem[-12:], fill="black")

    fig_path = qual_dir / "accepted_stagebsoft_synthetic_samples.jpg"
    canvas.save(fig_path, quality=95)

    summary = {
        "synthetic_images_found_for_sampling": len(imgs),
        "figure": str(fig_path),
        "note": "These are accepted Stage B-soft synthetic SR samples from the final combined_yolo export."
    }
    (qual_dir / "qualitative_figure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

except Exception as e:
    (qual_dir / "qualitative_figure_error.txt").write_text(str(e), encoding="utf-8")
    print("Qualitative figure failed:", e)

# -----------------------------
# 4. Methodology + experiments draft
# -----------------------------
draft = OUT / "methodology_experiments_draft.md"

draft.write_text("""# Methodology and Experiments Draft - newdata-final

## Methodology

This work proposes a detector-aware synthetic SAR ship dataset construction pipeline. The goal is not to generate arbitrary ships from noise, but to create detector-usable super-resolved SAR ship samples derived from real SAR crops.

The pipeline first merges and organizes real SAR ship data into YOLO-format detection splits. Low-resolution inputs are then generated and super-resolved using the Real-ESRGAN x2 branch. The generated SR samples are filtered using a two-stage acceptance process. Stage A performs image-quality screening, while Stage B-soft applies detector-aware acceptance to retain synthetic samples that preserve useful ship morphology and remain beneficial for object detection.

The final accepted Stage B-soft synthetic set is combined with the original real SAR samples. The final combined dataset contains 24,204 training images, 5,863 validation images, and 4,829 test images.

## Experimental Setup

The primary detector is YOLOv8s. Additional SAR-specific architectural ablations include SAR-SmallShip-YOLO-P2 and SAR-SmallShip-YOLO-P2ECA. To verify that the proposed dataset is not YOLO-specific, RT-DETR-L is also evaluated as a non-YOLO detector.

All evaluations use image size 128, confidence threshold 0.001, and IoU threshold 0.7. The main metric is mAP50-95, supported by precision, recall, and mAP50.

## Key Results

YOLOv8s original-only achieves HRSID mAP50-95 of 0.935 and SSDD mAP50-95 of 0.894. Stage B-soft + curriculum improves these to 0.967 on HRSID and 0.954 on SSDD.

RT-DETR-L further validates the approach beyond YOLO. RT-DETR original-only achieves mAP50-95 values of 0.634 on combined validation, 0.698 on HRSID, and 0.643 on SSDD. RT-DETR Stage B-soft curriculum improves these to 0.748, 0.770, and 0.726 respectively.

## Interpretation

The results show that direct synthetic addition is not always sufficient, especially for RT-DETR. However, the real-only to combined curriculum consistently improves performance across YOLO and RT-DETR. This supports the claim that detector-aware synthetic SR data is most effective when introduced through controlled curriculum training.
""", encoding="utf-8")

print("\nDONE")
print("Freeze folder:", FREEZE)
print("Final master table:", md_path)
print("Qualitative figures:", qual_dir)
print("Draft:", draft)
