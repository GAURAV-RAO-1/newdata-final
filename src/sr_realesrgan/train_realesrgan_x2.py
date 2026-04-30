from __future__ import annotations
from pathlib import Path
import json
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = ROOT / "data/crops_lr/crops_lr_manifest_x2.json"
LR_DIR = ROOT / "data/crops_lr/images"
HR_DIR = ROOT / "data/crops_hr/images"

CKPT_DIR = ROOT / "checkpoints/sr_realesrgan/x2"
LOG_PATH = ROOT / "logs/sr_realesrgan/x2_log.json"
REPORT_PATH = ROOT / "reports/sr_realesrgan/x2/metrics.json"
SAMPLE_DIR = ROOT / "data/sr_realesrgan_samples/x2"

SEED = 42
BATCH_SIZE = 8
NUM_EPOCHS = 20
LR_G = 5e-5
LR_D = 5e-5
NUM_WORKERS = 0
PATCH_SIZE = 96

ADV_WEIGHT = 2e-4
PIXEL_WEIGHT = 1.0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


@dataclass
class PairRecord:
    lr_file_name: str
    hr_file_name: str
    split: str


class SRPairDataset(Dataset):
    def __init__(self, records, split: str):
        self.records = records
        self.split = split

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        lr_path = LR_DIR / rec.split / rec.lr_file_name
        hr_path = HR_DIR / rec.split / rec.hr_file_name

        lr = Image.open(lr_path).convert("RGB")
        hr = Image.open(hr_path).convert("RGB")

        lr_t = pil_to_tensor(lr)
        hr_t = pil_to_tensor(hr)

        if self.split == "train":
            hr_h, hr_w = hr_t.shape[1], hr_t.shape[2]
            lr_patch = PATCH_SIZE // 2

            if hr_h < PATCH_SIZE or hr_w < PATCH_SIZE:
                hr_t = F.interpolate(
                    hr_t.unsqueeze(0),
                    size=(PATCH_SIZE, PATCH_SIZE),
                    mode="bicubic",
                    align_corners=False
                ).squeeze(0)
                lr_t = F.interpolate(
                    lr_t.unsqueeze(0),
                    size=(lr_patch, lr_patch),
                    mode="bicubic",
                    align_corners=False
                ).squeeze(0)
            else:
                top = random.randint(0, hr_h - PATCH_SIZE)
                left = random.randint(0, hr_w - PATCH_SIZE)

                hr_t = hr_t[:, top:top + PATCH_SIZE, left:left + PATCH_SIZE]
                lr_top = top // 2
                lr_left = left // 2
                lr_t = lr_t[:, lr_top:lr_top + lr_patch, lr_left:lr_left + lr_patch]

        return lr_t, hr_t, rec.lr_file_name, rec.hr_file_name


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


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_ch, out_ch, stride):
            return [
                nn.Conv2d(in_ch, out_ch, 3, stride, 1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        layers = [
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            *block(64, 64, 2),
            *block(64, 128, 1),
            *block(128, 128, 2),
            *block(128, 256, 1),
            *block(256, 256, 2),
            *block(256, 512, 1),
            *block(512, 512, 2),
        ]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def collate_fn(batch):
    lrs, hrs, lr_names, hr_names = zip(*batch)
    return torch.stack(lrs), torch.stack(hrs), lr_names, hr_names


def psnr(sr, hr):
    mse = F.mse_loss(sr, hr).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def evaluate(generator, loader, device, save_samples=False):
    generator.eval()
    l1s = []
    psnrs = []
    saved = 0

    with torch.no_grad():
        for lr, hr, lr_names, _ in loader:
            lr = lr.to(device)
            hr = hr.to(device)

            sr = generator(lr)

            if sr.shape[-2:] != hr.shape[-2:]:
                sr = F.interpolate(sr, size=hr.shape[-2:], mode="bicubic", align_corners=False)

            sr = torch.clamp(sr, 0, 1)

            l1s.append(F.l1_loss(sr, hr).item())

            for i in range(sr.size(0)):
                psnrs.append(psnr(sr[i:i+1], hr[i:i+1]))

                if save_samples and saved < 40:
                    img = tensor_to_pil(sr[i])
                    img.save(SAMPLE_DIR / f"realesrgan_x2_{saved:03d}_{lr_names[i]}")
                    saved += 1

    return {
        "l1": float(np.mean(l1s)) if l1s else None,
        "psnr": float(np.mean(psnrs)) if psnrs else None,
    }


def main():
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = [
        PairRecord(
            lr_file_name=p["lr_file_name"],
            hr_file_name=p["hr_file_name"],
            split=p["split"]
        )
        for p in data["pairs"]
    ]

    train_records = [p for p in pairs if p.split == "train"]
    val_records = [p for p in pairs if p.split == "val"]

    print(f"Train pairs: {len(train_records)}")
    print(f"Val pairs: {len(val_records)}")

    train_ds = SRPairDataset(train_records, split="train")
    val_ds = SRPairDataset(val_records, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    G = RRDBNetX2().to(device)
    D = Discriminator().to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.9, 0.99))
    opt_d = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.9, 0.99))

    bce = nn.BCEWithLogitsLoss()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    history = []
    best_psnr = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        G.train()
        D.train()

        g_losses = []
        d_losses = []

        for step, (lr, hr, _, _) in enumerate(train_loader, start=1):
            lr = lr.to(device)
            hr = hr.to(device)

            fake = G(lr)

            # D
            real_logits = D(hr)
            fake_logits = D(fake.detach())

            real_targets = torch.ones_like(real_logits)
            fake_targets = torch.zeros_like(fake_logits)

            d_loss_real = bce(real_logits, real_targets)
            d_loss_fake = bce(fake_logits, fake_targets)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # G
            fake_logits_for_g = D(fake)
            adv_targets = torch.ones_like(fake_logits_for_g)

            pixel_loss = F.l1_loss(fake, hr)
            adv_loss = bce(fake_logits_for_g, adv_targets)
            g_loss = PIXEL_WEIGHT * pixel_loss + ADV_WEIGHT * adv_loss

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            if step % 100 == 0:
                print(
                    f"Epoch {epoch} Step {step}/{len(train_loader)} "
                    f"G {g_loss.item():.4f} D {d_loss.item():.4f} "
                    f"Pix {pixel_loss.item():.4f} Adv {adv_loss.item():.4f}"
                )

        val_metrics = evaluate(G, val_loader, device, save_samples=(epoch == NUM_EPOCHS))

        epoch_record = {
            "epoch": epoch,
            "train_g_loss": float(np.mean(g_losses)) if g_losses else None,
            "train_d_loss": float(np.mean(d_losses)) if d_losses else None,
            "val_l1": val_metrics["l1"],
            "val_psnr": val_metrics["psnr"],
        }
        history.append(epoch_record)

        print(f"\nEpoch {epoch} summary:")
        print(json.dumps(epoch_record, indent=2))

        torch.save({
            "epoch": epoch,
            "generator_state_dict": G.state_dict(),
            "discriminator_state_dict": D.state_dict(),
            "opt_g_state_dict": opt_g.state_dict(),
            "opt_d_state_dict": opt_d.state_dict(),
            "val_psnr": val_metrics["psnr"],
        }, CKPT_DIR / "last.pt")

        if val_metrics["psnr"] is not None and val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            torch.save({
                "epoch": epoch,
                "generator_state_dict": G.state_dict(),
                "discriminator_state_dict": D.state_dict(),
                "opt_g_state_dict": opt_g.state_dict(),
                "opt_d_state_dict": opt_d.state_dict(),
                "val_psnr": val_metrics["psnr"],
            }, CKPT_DIR / "best.pt")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    final_report = {
        "device": str(device),
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr_g": LR_G,
        "lr_d": LR_D,
        "adv_weight": ADV_WEIGHT,
        "pixel_weight": PIXEL_WEIGHT,
        "train_pairs": len(train_records),
        "val_pairs": len(val_records),
        "best_val_psnr": max(r["val_psnr"] for r in history if r["val_psnr"] is not None),
        "history": history,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\nTraining finished.")
    print(f"Best checkpoint: {CKPT_DIR / 'best.pt'}")
    print(f"Last checkpoint: {CKPT_DIR / 'last.pt'}")
    print(f"Log saved to: {LOG_PATH}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()