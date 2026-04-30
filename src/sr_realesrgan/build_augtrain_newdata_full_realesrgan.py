from __future__ import annotations
from pathlib import Path
import shutil
from PIL import Image

ROOT = Path("/Users/gaurav/newdata")

SRC_ROOT = ROOT / "data/detector_yolo_exports/newdata_full_realesrgan_x2_stagebsoft_v1_dataset/combined_yolo"
SYN_ROOT = ROOT / "data/detector_yolo_exports/newdata_full_realesrgan_x2_stagebsoft_v1_dataset/synthetic_only"
OUT_ROOT = ROOT / "data/detector_yolo_exports/newdata_full_realesrgan_x2_stagebsoft_v1_augtrain/combined_yolo"

AUG_N = 5000  # controlled count to stay near 30k total train images


def copy_tree(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            copy_tree(item, target)
        else:
            if not target.exists():
                shutil.copy2(item, target)


def read_yolo_label(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    cls_id, xc, yc, w, h = line.split()
    return int(cls_id), float(xc), float(yc), float(w), float(h)


def write_yolo_label(path: Path, cls_id, xc, yc, w, h):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def main():
    # 1. Copy the full clean combined dataset first
    print("Copying clean combined dataset...")
    copy_tree(SRC_ROOT, OUT_ROOT)

    syn_train_img_dir = SYN_ROOT / "images" / "train"
    syn_train_lbl_dir = SYN_ROOT / "labels" / "train"

    out_train_img_dir = OUT_ROOT / "images" / "train"
    out_train_lbl_dir = OUT_ROOT / "labels" / "train"

    syn_images = sorted(syn_train_img_dir.iterdir())
    syn_images = [p for p in syn_images if p.is_file()][:AUG_N]

    added = 0
    for img_path in syn_images:
        stem = img_path.stem
        suffix = img_path.suffix
        lbl_path = syn_train_lbl_dir / f"{stem}.txt"
        if not lbl_path.exists():
            continue

        aug_img_name = f"{stem}_augflip{suffix}"
        aug_lbl_name = f"{stem}_augflip.txt"

        out_img = out_train_img_dir / aug_img_name
        out_lbl = out_train_lbl_dir / aug_lbl_name

        img = Image.open(img_path).convert("RGB")
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        out_img.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_img)

        cls_id, xc, yc, w, h = read_yolo_label(lbl_path)
        xc_new = 1.0 - xc
        write_yolo_label(out_lbl, cls_id, xc_new, yc, w, h)

        added += 1
        if added % 500 == 0:
            print(f"Added {added} augmented train images...")

    yaml_text = f"""path: {OUT_ROOT}
train: images/train
val: images/val
test: images/test

names:
  0: ship
"""
    with open(OUT_ROOT / "dataset.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print("Finished.")
    print(f"Added augmented train images: {added}")
    print(f"dataset.yaml: {OUT_ROOT / 'dataset.yaml'}")


if __name__ == "__main__":
    main()