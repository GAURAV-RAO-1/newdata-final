from __future__ import annotations
from pathlib import Path
import json

import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

DETECTOR_WEIGHTS = ROOT / "runs/yolo_original_only/exp4/weights/best.pt"

VAL_STAGE_A = ROOT / "data/crops_sr_realesrgan/manifests/quality_filter_realesrgan_x2_val_results.json"
HR_MANIFEST = ROOT / "data/crops_hr/crops_hr_manifest.json"

HR_DIR = ROOT / "data/crops_hr/images"
LR_DIR = ROOT / "data/crops_lr/images"
SR_DIR = ROOT / "data/crops_sr_realesrgan/images/x2"

OUT_JSON = ROOT / "reports/sr_realesrgan/x2/realesrgan_stage_b_soft_sweep_val.json"

IMG_SIZE = 128
CONF_THRESH = 0.001
IOU_THRESH = 0.7

# sweep these first
CONF_MARGINS = [0.00, 0.005, 0.01, 0.02, 0.03]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def compute_iou(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    ix1 = max(x11, x21)
    iy1 = max(y11, y21)
    ix2 = min(x12, x22)
    iy2 = min(y12, y22)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area1 = max(0.0, x12 - x11) * max(0.0, y12 - y11)
    area2 = max(0.0, x22 - x21) * max(0.0, y22 - y21)
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def run_detector_on_pil(model: YOLO, img: Image.Image):
    results = model.predict(
        source=np.array(img),
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        verbose=False,
        device="mps",
    )
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None

    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))

    return {
        "box_xyxy": boxes[best_idx].tolist(),
        "conf": float(confs[best_idx]),
    }


def main():
    stage_a = load_json(VAL_STAGE_A)
    hr_data = load_json(HR_MANIFEST)
    hr_by_ann = {rec["annotation_id"]: rec for rec in hr_data["crops"]}

    model = YOLO(str(DETECTOR_WEIGHTS))

    candidates = [r for r in stage_a if r["accepted"]]
    print(f"Stage A accepted val candidates: {len(candidates)}")

    # compute detector stats once
    cached = []
    for idx, rec in enumerate(candidates, start=1):
        ann_id = rec["annotation_id"]
        if ann_id not in hr_by_ann:
            continue

        hr_rec = hr_by_ann[ann_id]
        hr_file = hr_rec["crop_file_name"]
        lr_file = rec["lr_file_name"]
        sr_file = rec["sr_file_name"]

        hr_path = HR_DIR / "val" / hr_file
        lr_path = LR_DIR / "val" / lr_file
        sr_path = SR_DIR / "val" / sr_file

        if not hr_path.exists() or not lr_path.exists() or not sr_path.exists():
            continue

        hr_img = load_rgb(hr_path)
        lr_img = load_rgb(lr_path)
        sr_img = load_rgb(sr_path)

        hr_w, hr_h = hr_img.size
        bicubic_img = lr_img.resize((hr_w, hr_h), Image.BICUBIC)
        if sr_img.size != (hr_w, hr_h):
            sr_img = sr_img.resize((hr_w, hr_h), Image.BICUBIC)

        gt_box = xywh_to_xyxy(hr_rec["crop_bbox_xywh"])

        det_b = run_detector_on_pil(model, bicubic_img)
        det_s = run_detector_on_pil(model, sr_img)

        conf_b = det_b["conf"] if det_b is not None else 0.0
        conf_s = det_s["conf"] if det_s is not None else 0.0
        iou_b = compute_iou(gt_box, det_b["box_xyxy"]) if det_b is not None else 0.0
        iou_s = compute_iou(gt_box, det_s["box_xyxy"]) if det_s is not None else 0.0

        cached.append({
            "annotation_id": ann_id,
            "lr_file_name": lr_file,
            "sr_file_name": sr_file,
            "hr_file_name": hr_file,
            "source_dataset": rec["source_dataset"],
            "conf_bicubic": conf_b,
            "conf_sr": conf_s,
            "iou_bicubic": iou_b,
            "iou_sr": iou_s,
        })

        if idx % 100 == 0:
            print(f"Processed {idx} val candidates...")

    sweep_rows = []
    for conf_margin in CONF_MARGINS:
        accepted = [
            r for r in cached
            if (r["iou_sr"] >= r["iou_bicubic"]) and (r["conf_sr"] >= r["conf_bicubic"] - conf_margin)
        ]
        row = {
            "conf_margin": conf_margin,
            "num_candidates": len(cached),
            "num_accepted": len(accepted),
            "acceptance_rate": len(accepted) / len(cached) if cached else 0.0,
        }
        sweep_rows.append(row)
        print(row)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_candidates": len(cached),
                "rows": sweep_rows,
                "cached_records": cached,
            },
            f,
            indent=2,
        )

    print(f"\nSaved sweep results to: {OUT_JSON}")


if __name__ == "__main__":
    main()