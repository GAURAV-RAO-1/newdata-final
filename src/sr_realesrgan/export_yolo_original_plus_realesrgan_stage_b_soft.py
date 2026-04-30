from __future__ import annotations
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]

HR_MANIFEST = ROOT / "data/crops_hr/crops_hr_manifest.json"
HR_DIR = ROOT / "data/crops_hr/images"
SR_DIR = ROOT / "data/crops_sr_realesrgan/images/x2"

ACC_TRAIN = ROOT / "data/crops_sr_realesrgan/manifests/accepted_realesrgan_stage_b_soft_train_manifest.json"
ACC_VAL = ROOT / "data/crops_sr_realesrgan/manifests/accepted_realesrgan_stage_b_soft_val_manifest.json"
ACC_TEST = ROOT / "data/crops_sr_realesrgan/manifests/accepted_realesrgan_stage_b_soft_test_manifest.json"

OUT_ROOT = ROOT / "data/detector_yolo/original_plus_realesrgan_stage_b_soft"
OUT_IMAGES = OUT_ROOT / "images"
OUT_LABELS = OUT_ROOT / "labels"
DATASET_YAML = OUT_ROOT / "dataset.yaml"

MIN_SIDE = 10

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def xywh_to_yolo(x, y, w, h, img_w, img_h):
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    wn = w / img_w
    hn = h / img_h
    return xc, yc, wn, hn

def write_label(dst_label: Path, bbox_xywh, img_w, img_h):
    x, y, w, h = bbox_xywh
    xc, yc, wn, hn = xywh_to_yolo(x, y, w, h, img_w, img_h)
    with open(dst_label, "w", encoding="utf-8") as f:
        f.write(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

def main():
    hr_data = load_json(HR_MANIFEST)
    acc_train = load_json(ACC_TRAIN)["records"]
    acc_val = load_json(ACC_VAL)["records"]
    acc_test = load_json(ACC_TEST)["records"]

    hr_by_ann = {rec["annotation_id"]: rec for rec in hr_data["crops"]}

    counts = {
        "original_train": 0, "original_val": 0, "original_test": 0,
        "realesrgan_stage_b_soft_train": 0, "realesrgan_stage_b_soft_val": 0, "realesrgan_stage_b_soft_test": 0,
    }

    for rec in hr_data["crops"]:
        split_name = rec["split"]
        crop_file_name = rec["crop_file_name"]
        crop_w = rec["crop_width"]
        crop_h = rec["crop_height"]
        if crop_w < MIN_SIDE or crop_h < MIN_SIDE:
            continue

        src_img = HR_DIR / split_name / crop_file_name
        dst_img = OUT_IMAGES / split_name / crop_file_name
        dst_label = OUT_LABELS / split_name / (Path(crop_file_name).with_suffix(".txt").name)

        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)

        if src_img.exists() and not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        write_label(dst_label, rec["crop_bbox_xywh"], crop_w, crop_h)
        counts[f"original_{split_name}"] += 1

    for records, split_name in [(acc_train, "train"), (acc_val, "val"), (acc_test, "test")]:
        for rec in records:
            ann_id = rec["annotation_id"]
            if ann_id not in hr_by_ann:
                continue

            hr_rec = hr_by_ann[ann_id]
            crop_w = hr_rec["crop_width"]
            crop_h = hr_rec["crop_height"]
            if crop_w < MIN_SIDE or crop_h < MIN_SIDE:
                continue

            bbox_xywh = hr_rec["crop_bbox_xywh"]
            sr_file_name = rec["sr_file_name"]
            src_img = SR_DIR / split_name / sr_file_name

            dst_img = OUT_IMAGES / split_name / sr_file_name
            dst_label = OUT_LABELS / split_name / (Path(sr_file_name).with_suffix(".txt").name)

            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_label.parent.mkdir(parents=True, exist_ok=True)

            if src_img.exists() and not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            write_label(dst_label, bbox_xywh, crop_w, crop_h)
            counts[f"realesrgan_stage_b_soft_{split_name}"] += 1

    yaml_text = f"""path: {OUT_ROOT}
train: images/train
val: images/val
test: images/test

names:
  0: ship
"""
    with open(DATASET_YAML, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print("Finished.")
    print(counts)
    print(f"dataset.yaml: {DATASET_YAML}")

if __name__ == "__main__":
    main()
    