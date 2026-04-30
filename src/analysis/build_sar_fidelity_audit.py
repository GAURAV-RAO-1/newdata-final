from pathlib import Path
from PIL import Image
import numpy as np
import csv
import json
import math
import random

ROOT = Path("/Users/gaurav/newdata")

DATASET_ROOT = ROOT / "data/detector_yolo_exports/newdata_full_realesrgan_x2_stagebsoft_v1_dataset"
REAL_ONLY = DATASET_ROOT / "real_only/images"
COMBINED = DATASET_ROOT / "combined_yolo/images"

OUT_DIR = ROOT / "reports/sar_fidelity_audit_current"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ["train", "val", "test"]

# Use None for all images. Use an integer like 5000 if you want faster runtime.
MAX_PER_GROUP_PER_SPLIT = None
RANDOM_SEED = 0


def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


def load_gray(path: Path):
    img = Image.open(path).convert("L")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def entropy_from_hist(hist):
    p = hist.astype(np.float64)
    p = p / (p.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def image_features(path: Path):
    x = load_gray(path)

    mean = float(x.mean())
    std = float(x.std())
    p5 = float(np.percentile(x, 5))
    p95 = float(np.percentile(x, 95))
    contrast = p95 - p5

    hist, _ = np.histogram(x, bins=256, range=(0.0, 1.0))
    ent = entropy_from_hist(hist)

    # Non-overlapping 8x8 local variance as a lightweight SAR texture proxy
    h, w = x.shape
    hh = (h // 8) * 8
    ww = (w // 8) * 8
    if hh >= 8 and ww >= 8:
        blocks = x[:hh, :ww].reshape(hh // 8, 8, ww // 8, 8)
        local_var = float(blocks.var(axis=(1, 3)).mean())
    else:
        local_var = float(x.var())

    # ENL-style proxy: mean^2 / variance
    enl_proxy = float((mean * mean) / (std * std + 1e-8))

    # Simple edge/texture energy
    gx = np.diff(x, axis=1)
    gy = np.diff(x, axis=0)
    edge_energy = float((np.mean(np.abs(gx)) + np.mean(np.abs(gy))) / 2.0)

    return {
        "mean_intensity": mean,
        "std_intensity": std,
        "contrast_p95_p5": contrast,
        "entropy": ent,
        "local_var_8x8": local_var,
        "enl_proxy": enl_proxy,
        "edge_energy": edge_energy,
        "hist": hist.tolist(),
    }


def mean_std(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) <= 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def cohen_d(a_vals, b_vals):
    a_vals = [v for v in a_vals if v is not None]
    b_vals = [v for v in b_vals if v is not None]
    if len(a_vals) < 2 or len(b_vals) < 2:
        return None

    ma, sa = mean_std(a_vals)
    mb, sb = mean_std(b_vals)
    pooled = math.sqrt(((len(a_vals) - 1) * sa * sa + (len(b_vals) - 1) * sb * sb) / (len(a_vals) + len(b_vals) - 2))
    if pooled < 1e-12:
        return 0.0
    return (mb - ma) / pooled


def aggregate_hist(records):
    h = np.zeros(256, dtype=np.float64)
    for r in records:
        h += np.asarray(r["hist"], dtype=np.float64)
    h = h / (h.sum() + 1e-12)
    return h


def hist_l1(h1, h2):
    return float(np.abs(h1 - h2).sum())


def classify_combined_split(split):
    real_imgs = list_images(REAL_ONLY / split)
    combined_imgs = list_images(COMBINED / split)

    real_names = {p.name for p in real_imgs}
    real_stems = {p.stem for p in real_imgs}

    accepted_sr = []
    combined_real = []

    for p in combined_imgs:
        # Main rule: files already present in real_only are real;
        # extra files in combined_yolo are accepted synthetic SR.
        if p.name in real_names or p.stem in real_stems:
            combined_real.append(p)
        else:
            accepted_sr.append(p)

    return real_imgs, accepted_sr, combined_real


def maybe_sample(paths, max_n):
    if max_n is None or len(paths) <= max_n:
        return paths
    random.seed(RANDOM_SEED)
    return sorted(random.sample(paths, max_n))


def summarize_group(records, split, group):
    features = [
        "mean_intensity",
        "std_intensity",
        "contrast_p95_p5",
        "entropy",
        "local_var_8x8",
        "enl_proxy",
        "edge_energy",
    ]

    row = {
        "split": split,
        "group": group,
        "n": len(records),
    }

    for f in features:
        m, s = mean_std([r[f] for r in records])
        row[f + "_mean"] = m
        row[f + "_std"] = s

    return row


def fmt(x, digits=4):
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def main():
    random.seed(RANDOM_SEED)

    all_records = []
    group_records = {}
    split_counts = []

    for split in SPLITS:
        real_imgs, accepted_sr, combined_real = classify_combined_split(split)

        real_imgs = maybe_sample(real_imgs, MAX_PER_GROUP_PER_SPLIT)
        accepted_sr = maybe_sample(accepted_sr, MAX_PER_GROUP_PER_SPLIT)

        split_counts.append({
            "split": split,
            "real_only_found": len(real_imgs),
            "accepted_sr_found": len(accepted_sr),
            "combined_real_found": len(combined_real),
        })

        for group, paths in [
            ("real_only", real_imgs),
            ("accepted_stagebsoft_sr", accepted_sr),
        ]:
            key = (split, group)
            group_records[key] = []

            print(f"Processing {split} | {group} | n={len(paths)}")

            for p in paths:
                try:
                    feats = image_features(p)
                    rec = {
                        "split": split,
                        "group": group,
                        "path": str(p),
                        **feats,
                    }
                    group_records[key].append(rec)
                    all_records.append(rec)
                except Exception as e:
                    print(f"SKIP {p}: {e}")

    # Group summary
    summary_rows = []
    for split in SPLITS:
        for group in ["real_only", "accepted_stagebsoft_sr"]:
            records = group_records.get((split, group), [])
            summary_rows.append(summarize_group(records, split, group))

    # Add all-split summaries
    for group in ["real_only", "accepted_stagebsoft_sr"]:
        records = [r for r in all_records if r["group"] == group]
        summary_rows.append(summarize_group(records, "all", group))

    # Comparison rows
    features = [
        "mean_intensity",
        "std_intensity",
        "contrast_p95_p5",
        "entropy",
        "local_var_8x8",
        "enl_proxy",
        "edge_energy",
    ]

    comparison_rows = []

    for split in SPLITS + ["all"]:
        if split == "all":
            real = [r for r in all_records if r["group"] == "real_only"]
            sr = [r for r in all_records if r["group"] == "accepted_stagebsoft_sr"]
        else:
            real = group_records.get((split, "real_only"), [])
            sr = group_records.get((split, "accepted_stagebsoft_sr"), [])

        if not real or not sr:
            continue

        h_real = aggregate_hist(real)
        h_sr = aggregate_hist(sr)
        l1 = hist_l1(h_real, h_sr)

        for f in features:
            real_vals = [r[f] for r in real]
            sr_vals = [r[f] for r in sr]
            real_mean, real_std = mean_std(real_vals)
            sr_mean, sr_std = mean_std(sr_vals)

            delta = sr_mean - real_mean
            delta_pct = (delta / (abs(real_mean) + 1e-12)) * 100.0
            d = cohen_d(real_vals, sr_vals)

            comparison_rows.append({
                "split": split,
                "feature": f,
                "real_mean": real_mean,
                "real_std": real_std,
                "sr_mean": sr_mean,
                "sr_std": sr_std,
                "delta_abs": delta,
                "delta_pct": delta_pct,
                "cohen_d_sr_minus_real": d,
                "histogram_l1_distance": l1,
                "n_real": len(real),
                "n_sr": len(sr),
            })

    # Save JSON
    json_out = {
        "note": "This audit checks image-level statistical consistency. It is not a full SAR radiometric/phase-level physical validation.",
        "dataset_root": str(DATASET_ROOT),
        "split_counts": split_counts,
        "summary_rows": summary_rows,
        "comparison_rows": comparison_rows,
    }

    (OUT_DIR / "sar_fidelity_audit.json").write_text(json.dumps(json_out, indent=2), encoding="utf-8")

    # Save CSVs
    with open(OUT_DIR / "sar_fidelity_group_summary.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(OUT_DIR / "sar_fidelity_comparison.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(comparison_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    # Markdown report
    md = []
    md.append("# SAR Fidelity / Statistical Audit - newdata-final\n")
    md.append("This audit compares original real SAR crops against accepted Stage B-soft super-resolved synthetic crops.\n")
    md.append("Important: this is an image-level distribution and morphology/texture audit. It is not a full raw SAR/SLC radiometric or phase-level physical validation.\n")

    md.append("## Split Counts\n")
    md.append("| Split | Real-only images used | Accepted Stage B-soft SR images used | Combined real images found |\n")
    md.append("|---|---:|---:|---:|\n")
    for r in split_counts:
        md.append(f"| {r['split']} | {r['real_only_found']} | {r['accepted_sr_found']} | {r['combined_real_found']} |\n")

    md.append("\n## Group Summary\n")
    md.append("| Split | Group | n | Mean intensity | Std intensity | Contrast p95-p5 | Entropy | Local var 8x8 | ENL proxy | Edge energy |\n")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")

    for r in summary_rows:
        md.append(
            f"| {r['split']} | {r['group']} | {r['n']} | "
            f"{fmt(r['mean_intensity_mean'])} | "
            f"{fmt(r['std_intensity_mean'])} | "
            f"{fmt(r['contrast_p95_p5_mean'])} | "
            f"{fmt(r['entropy_mean'])} | "
            f"{fmt(r['local_var_8x8_mean'])} | "
            f"{fmt(r['enl_proxy_mean'])} | "
            f"{fmt(r['edge_energy_mean'])} |\n"
        )

    md.append("\n## Real vs Accepted SR Comparison\n")
    md.append("Positive delta means accepted SR has a higher value than real-only images. Cohen's d near 0 indicates small shift; larger absolute values indicate stronger distributional shift.\n\n")
    md.append("| Split | Feature | Real mean | SR mean | Delta % | Cohen's d | Hist L1 | n real | n SR |\n")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")

    for r in comparison_rows:
        md.append(
            f"| {r['split']} | {r['feature']} | "
            f"{fmt(r['real_mean'])} | {fmt(r['sr_mean'])} | "
            f"{fmt(r['delta_pct'], 2)} | {fmt(r['cohen_d_sr_minus_real'])} | "
            f"{fmt(r['histogram_l1_distance'])} | {r['n_real']} | {r['n_sr']} |\n"
        )

    md.append("\n## Suggested Paper Interpretation\n")
    md.append("The accepted SR samples are evaluated for image-level statistical consistency with the real SAR crops using intensity, contrast, entropy, local variance, ENL-style proxy, and edge-energy statistics. This supports the claim that the proposed Stage B-soft filter does not blindly add arbitrary SR outputs, but retains detector-useful samples while monitoring distributional shift. However, because raw complex SAR/SLC data are not available, this audit should not be described as full SAR physical or radiometric validation.\n")

    (OUT_DIR / "sar_fidelity_audit.md").write_text("".join(md), encoding="utf-8")

    print("\nDONE")
    print(OUT_DIR / "sar_fidelity_audit.md")
    print(OUT_DIR / "sar_fidelity_group_summary.csv")
    print(OUT_DIR / "sar_fidelity_comparison.csv")


if __name__ == "__main__":
    main()