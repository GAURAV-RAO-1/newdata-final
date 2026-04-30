from __future__ import annotations
from pathlib import Path
import json
import math

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

HR_DIR = ROOT / "data/crops_hr/images"
LR_DIR = ROOT / "data/crops_lr/images"

GAN_MANIFEST = ROOT / "data/crops_sr_realesrgan/manifests/sr_realesrgan_x2_train_inference_manifest.json"
GAN_DIR = ROOT / "data/crops_sr_realesrgan/images/x2"

OUT_TRAIN_JSON = ROOT / "data/crops_sr_realesrgan/manifests/quality_filter_realesrgan_x2_train_results.json"
OUT_SUMMARY = ROOT / "reports/sr_realesrgan/x2/quality_filter_train_summary.json"


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    err = mse(a, b)
    if err == 0:
        return float("inf")
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(err)


def ssim_single_channel(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = img1.var()
    sigma2 = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2)

    if denominator == 0:
        return 1.0
    return float(numerator / denominator)


def ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    return float(sum(ssim_single_channel(a[..., c], b[..., c]) for c in range(3)) / 3.0)


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def img_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)


def evaluate_record(rec: dict) -> dict | None:
    split_name = rec["split"]
    if split_name != "train":
        return None

    lr_file = rec["lr_file_name"]
    hr_file = rec["hr_file_name"]
    sr_file = rec["sr_file_name"]

    lr_path = LR_DIR / split_name / lr_file
    hr_path = HR_DIR / split_name / hr_file
    sr_path = GAN_DIR / split_name / sr_file

    if not lr_path.exists() or not hr_path.exists() or not sr_path.exists():
        return None

    hr_img = load_rgb(hr_path)
    lr_img = load_rgb(lr_path)
    sr_img = load_rgb(sr_path)

    hr_w, hr_h = hr_img.size
    bicubic_img = lr_img.resize((hr_w, hr_h), Image.BICUBIC)

    if sr_img.size != (hr_w, hr_h):
        sr_img = sr_img.resize((hr_w, hr_h), Image.BICUBIC)

    hr_np = img_to_np(hr_img)
    bicubic_np = img_to_np(bicubic_img)
    sr_np = img_to_np(sr_img)

    psnr_b = psnr(hr_np, bicubic_np)
    ssim_b = ssim_rgb(hr_np, bicubic_np)
    psnr_s = psnr(hr_np, sr_np)
    ssim_s = ssim_rgb(hr_np, sr_np)

    accepted = (psnr_s >= psnr_b) and (ssim_s >= ssim_b)

    return {
        "scale": 2,
        "split": split_name,
        "lr_file_name": lr_file,
        "hr_file_name": hr_file,
        "sr_file_name": sr_file,
        "source_dataset": rec["source_dataset"],
        "parent_image_id": rec["parent_image_id"],
        "annotation_id": rec["annotation_id"],
        "psnr_bicubic": psnr_b,
        "ssim_bicubic": ssim_b,
        "psnr_sr": psnr_s,
        "ssim_sr": ssim_s,
        "accepted": accepted,
    }


def main():
    with open(GAN_MANIFEST, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for rec in data["records"]:
        out = evaluate_record(rec)
        if out is not None:
            results.append(out)
        if len(results) and len(results) % 200 == 0:
            print(f"Processed {len(results)} records...")

    summary = {
        "train_total": len(results),
        "train_accepted": sum(r["accepted"] for r in results),
    }

    OUT_TRAIN_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_TRAIN_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Finished.")
    print(json.dumps(summary, indent=2))
    print(f"Saved train results to: {OUT_TRAIN_JSON}")
    print(f"Saved summary to: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()