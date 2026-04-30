from __future__ import annotations
from pathlib import Path
import json

import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

DETECTOR_WEIGHTS = ROOT / "runs/yolo_original_only/exp4/weights/best.pt"

VAL_STAGE_A = ROOT / "data/crops_sr_realesrgan/manifests/quality_filter_realesrgan_x2_val_results.json"
TEST_STAGE_A = ROOT / "data/crops_sr_realesrgan/manifests/quality_filter_realesrgan_x2_test_results.json"

HR_MANIFEST = ROOT / "data/crops_hr/crops_hr_manifest.json"
HR_DIR = ROOT / "data/crops_hr/images"
LR_DIR = ROOT / "data/crops_lr/images"
SR_DIR = ROOT / "data/crops_sr_realesrgan/images/x2"

OUT_VAL = ROOT / "data/crops_sr_realesrgan/manifests/stage_b_realesrgan_x2_val_results.json"
OUT_TEST = ROOT / "data/crops_sr_realesrgan/manifests/stage_b_realesrgan_x2_test_results.json"
OUT_SUMMARY = ROOT / "reports/sr_realesrgan/x2/stage_b_val_test_summary.json"

IMG_SIZE = 128
CONF_THRESH = 0.001
IOU_THRESH = 0.7


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
        device="mps"
    )
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None

    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()

    best_idx = int(np.argmax(confs))
    return {
        "box_xyxy": boxes[best_idx].tolist(),
        "conf": float(confs[best_idx])
    }


def process_records(records, split_name, hr_by_ann, model):
    results = []
    accepted_count = 0

    candidates = [r for r in records if r["accepted"]]

    for idx, rec in enumerate(candidates, start=1):
        ann_id = rec["annotation_id"]
        if ann_id not in hr_by_ann:
            continue

        hr_rec = hr_by_ann[ann_id]
        hr_file = hr_rec["crop_file_name"]
        lr_file = rec["lr_file_name"]
        sr_file = rec["sr_file_name"]

        hr_path = HR_DIR / split_name / hr_file
        lr_path = LR_DIR / split_name / lr_file
        sr_path = SR_DIR / split_name / sr_file

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

        accepted_stage_b = (conf_s >= conf_b) and (iou_s >= iou_b)
        if accepted_stage_b:
            accepted_count += 1

        results.append({
            "annotation_id": ann_id,
            "scale": 2,
            "split": split_name,
            "lr_file_name": lr_file,
            "sr_file_name": sr_file,
            "hr_file_name": hr_file,
            "source_dataset": rec["source_dataset"],
            "conf_bicubic": conf_b,
            "conf_sr": conf_s,
            "iou_bicubic": iou_b,
            "iou_sr": iou_s,
            "accepted_stage_b": accepted_stage_b
        })

        if idx % 100 == 0:
            print(f"{split_name}: processed {idx} samples...")

    return results, len(candidates), accepted_count


def main():
    with open(VAL_STAGE_A, "r", encoding="utf-8") as f:
        val_stage_a = json.load(f)
    with open(TEST_STAGE_A, "r", encoding="utf-8") as f:
        test_stage_a = json.load(f)
    with open(HR_MANIFEST, "r", encoding="utf-8") as f:
        hr_data = json.load(f)

    hr_by_ann = {rec["annotation_id"]: rec for rec in hr_data["crops"]}
    model = YOLO(str(DETECTOR_WEIGHTS))

    val_results, val_total, val_accepted = process_records(val_stage_a, "val", hr_by_ann, model)
    test_results, test_total, test_accepted = process_records(test_stage_a, "test", hr_by_ann, model)

    OUT_VAL.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_VAL, "w", encoding="utf-8") as f:
        json.dump(val_results, f, indent=2)
    with open(OUT_TEST, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    summary = {
        "val_total_stage_a_candidates": val_total,
        "val_accepted_stage_b": val_accepted,
        "test_total_stage_a_candidates": test_total,
        "test_accepted_stage_b": test_accepted,
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Finished.")
    print(json.dumps(summary, indent=2))
    print(f"Saved val Stage B results to: {OUT_VAL}")
    print(f"Saved test Stage B results to: {OUT_TEST}")
    print(f"Saved summary to: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()