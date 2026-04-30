from __future__ import annotations
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]

HR_MANIFEST = ROOT / "data/crops_hr/crops_hr_manifest.json"
HR_DIR = ROOT / "data/crops_hr/images"

SR_DIR = ROOT / "data/crops_sr_realesrgan_full/images/x2"

ACC_TRAIN = ROOT / "data/crops_sr_realesrgan_full/manifests/accepted_realesrgan_stage_b_soft_full_train_manifest.json"
ACC_VAL = ROOT / "data/crops_sr_realesrgan_full/manifests/accepted_realesrgan_stage_b_soft_full_val_manifest.json"
ACC_TEST = ROOT / "data/crops_sr_realesrgan_full/manifests/accepted_realesrgan_stage_b_soft_full_test_manifest.json"

OUT_ROOT = ROOT / "data/detector_yolo_exports/newdata_full_realesrgan_x2_stagebsoft_v1_dataset"

REAL_ROOT = OUT_ROOT / "real_only"
SYN_ROOT = OUT_ROOT / "synthetic_only"
COMBINED_ROOT = OUT_ROOT / "combined_yolo"

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
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_label, "w", encoding="utf-8") as f:
        f.write(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")


def copy_if_needed(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)


def write_dataset_yaml(root: Path):
    yaml_text = f"""path: {root}
train: images/train
val: images/val
test: images/test

names:
  0: ship
"""
    with open(root / "dataset.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_text)


def export_one_image_and_label(src_img: Path, out_root: Path, split_name: str, out_file_name: str, bbox_xywh, img_w, img_h):
    dst_img = out_root / "images" / split_name / out_file_name
    dst_label = out_root / "labels" / split_name / (Path(out_file_name).with_suffix(".txt").name)

    copy_if_needed(src_img, dst_img)
    write_label(dst_label, bbox_xywh, img_w, img_h)


def main():
    hr_data = load_json(HR_MANIFEST)
    acc_train = load_json(ACC_TRAIN)["records"]
    acc_val = load_json(ACC_VAL)["records"]
    acc_test = load_json(ACC_TEST)["records"]

    hr_by_ann = {rec["annotation_id"]: rec for rec in hr_data["crops"]}

    counts = {
        "real_only_train": 0,
        "real_only_val": 0,
        "real_only_test": 0,
        "synthetic_only_train": 0,
        "synthetic_only_val": 0,
        "synthetic_only_test": 0,
        "combined_train": 0,
        "combined_val": 0,
        "combined_test": 0,
    }

    # -------------------------------------------------
    # Export original / real crops
    # -------------------------------------------------
    for rec in hr_data["crops"]:
        split_name = rec["split"]
        crop_file_name = rec["crop_file_name"]
        crop_w = rec["crop_width"]
        crop_h = rec["crop_height"]

        if crop_w < MIN_SIDE or crop_h < MIN_SIDE:
            continue

        src_img = HR_DIR / split_name / crop_file_name
        bbox_xywh = rec["crop_bbox_xywh"]

        # real_only
        export_one_image_and_label(
            src_img=src_img,
            out_root=REAL_ROOT,
            split_name=split_name,
            out_file_name=crop_file_name,
            bbox_xywh=bbox_xywh,
            img_w=crop_w,
            img_h=crop_h,
        )
        counts[f"real_only_{split_name}"] += 1

        # combined_yolo
        export_one_image_and_label(
            src_img=src_img,
            out_root=COMBINED_ROOT,
            split_name=split_name,
            out_file_name=crop_file_name,
            bbox_xywh=bbox_xywh,
            img_w=crop_w,
            img_h=crop_h,
        )
        counts[f"combined_{split_name}"] += 1

    # -------------------------------------------------
    # Export accepted synthetic crops
    # -------------------------------------------------
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

            # synthetic_only
            export_one_image_and_label(
                src_img=src_img,
                out_root=SYN_ROOT,
                split_name=split_name,
                out_file_name=sr_file_name,
                bbox_xywh=bbox_xywh,
                img_w=crop_w,
                img_h=crop_h,
            )
            counts[f"synthetic_only_{split_name}"] += 1

            # combined_yolo
            export_one_image_and_label(
                src_img=src_img,
                out_root=COMBINED_ROOT,
                split_name=split_name,
                out_file_name=sr_file_name,
                bbox_xywh=bbox_xywh,
                img_w=crop_w,
                img_h=crop_h,
            )
            counts[f"combined_{split_name}"] += 1

    # write dataset yamls
    write_dataset_yaml(REAL_ROOT)
    write_dataset_yaml(SYN_ROOT)
    write_dataset_yaml(COMBINED_ROOT)

    # save a summary json also
    summary_path = OUT_ROOT / "export_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)

    print("Finished.")
    print(json.dumps(counts, indent=2))
    print(f"real_only dataset.yaml: {REAL_ROOT / 'dataset.yaml'}")
    print(f"synthetic_only dataset.yaml: {SYN_ROOT / 'dataset.yaml'}")
    print(f"combined_yolo dataset.yaml: {COMBINED_ROOT / 'dataset.yaml'}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()