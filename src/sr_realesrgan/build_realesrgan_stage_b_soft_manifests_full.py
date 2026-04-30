from __future__ import annotations
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]

TRAIN_RESULTS = ROOT / "data/crops_sr_realesrgan_full/manifests/stage_b_soft_realesrgan_x2_full_train_results.json"
VAL_RESULTS = ROOT / "data/crops_sr_realesrgan_full/manifests/stage_b_soft_realesrgan_x2_full_val_results.json"
TEST_RESULTS = ROOT / "data/crops_sr_realesrgan_full/manifests/stage_b_soft_realesrgan_x2_full_test_results.json"

OUT_TRAIN = ROOT / "data/crops_sr_realesrgan_full/manifests/accepted_realesrgan_stage_b_soft_full_train_manifest.json"
OUT_VAL = ROOT / "data/crops_sr_realesrgan_full/manifests/accepted_realesrgan_stage_b_soft_full_val_manifest.json"
OUT_TEST = ROOT / "data/crops_sr_realesrgan_full/manifests/accepted_realesrgan_stage_b_soft_full_test_manifest.json"

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path: Path, name: str, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": name,
                "num_records": len(records),
                "records": records,
            },
            f,
            indent=2,
        )


def main():
    train = load_json(TRAIN_RESULTS)
    val = load_json(VAL_RESULTS)
    test = load_json(TEST_RESULTS)

    train_acc = [r for r in train if r["accepted_stage_b_soft"]]
    val_acc = [r for r in val if r["accepted_stage_b_soft"]]
    test_acc = [r for r in test if r["accepted_stage_b_soft"]]

    save_manifest(OUT_TRAIN, "accepted_realesrgan_stage_b_soft_train", train_acc)
    save_manifest(OUT_VAL, "accepted_realesrgan_stage_b_soft_val", val_acc)
    save_manifest(OUT_TEST, "accepted_realesrgan_stage_b_soft_test", test_acc)

    print(f"Train accepted: {len(train_acc)}")
    print(f"Val accepted: {len(val_acc)}")
    print(f"Test accepted: {len(test_acc)}")
    print(f"Saved: {OUT_TRAIN}")
    print(f"Saved: {OUT_VAL}")
    print(f"Saved: {OUT_TEST}")


if __name__ == "__main__":
    main()