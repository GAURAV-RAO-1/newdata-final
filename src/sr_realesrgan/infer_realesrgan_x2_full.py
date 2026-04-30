from __future__ import annotations
from pathlib import Path
import json

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = ROOT / "data/crops_lr_full/crops_lr_full_manifest_x2.json"
LR_DIR = ROOT / "data/crops_lr_full/images"
CKPT_PATH = ROOT / "checkpoints/sr_realesrgan/x2/best.pt"

OUT_DIR = ROOT / "data/crops_sr_realesrgan_full/images/x2"
OUT_MANIFEST = ROOT / "data/crops_sr_realesrgan_full/manifests/sr_realesrgan_x2_full_inference_manifest.json"

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().clamp(0, 1).numpy()
    x = np.transpose(x, (1, 2, 0))
    x = (x * 255.0).round().astype(np.uint8)
    return Image.fromarray(x)


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x + 0.2 * x5


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + 0.2 * out


class RRDBNetX2(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=8, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.rrdb_trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.hr_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.rrdb_trunk(fea))
        fea = fea + trunk
        fea = F.interpolate(fea, scale_factor=2, mode="nearest")
        fea = self.lrelu(self.upconv1(fea))
        out = self.conv_last(self.lrelu(self.hr_conv(fea)))
        return torch.clamp(out, 0, 1)


def main():
    device = get_device()
    print(f"Using device: {device}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = [p for p in data["pairs"] if p["split"] in ("val", "test")]

    model = RRDBNetX2().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["generator_state_dict"])
    model.eval()

    out_records = []
    count = 0

    with torch.no_grad():
        for p in records:
            split_name = p["split"]
            lr_file_name = p["lr_file_name"]
            hr_file_name = p["hr_file_name"]

            lr_path = LR_DIR / split_name / lr_file_name
            if not lr_path.exists():
                continue

            lr_img = Image.open(lr_path).convert("RGB")
            lr_t = pil_to_tensor(lr_img).unsqueeze(0).to(device)

            sr_t = model(lr_t)
            sr_img = tensor_to_pil(sr_t.squeeze(0))

            sr_file_name = lr_file_name.replace("lr_x2_", "sr_realesrgan_x2_")
            out_path = OUT_DIR / split_name / sr_file_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sr_img.save(out_path)

            out_records.append({
                "scale": 2,
                "split": split_name,
                "lr_file_name": lr_file_name,
                "hr_file_name": hr_file_name,
                "sr_file_name": sr_file_name,
                "checkpoint": str(CKPT_PATH),
                "source_dataset": p["source_dataset"],
                "parent_image_id": p["parent_image_id"],
                "annotation_id": p["annotation_id"],
            })

            count += 1
            if count % 200 == 0:
                print(f"Saved {count} Real-ESRGAN train SR images...")

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump({
            "model": "realesrgan_x2_best_train",
            "num_outputs": len(out_records),
            "records": out_records
        }, f, indent=2)

    print("\nFinished.")
    print(f"Total Real-ESRGAN  outputs saved: {len(out_records)}")
    print(f"Manifest saved to: {OUT_MANIFEST}")


if __name__ == "__main__":
    main()