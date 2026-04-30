# Dataset Release Information

The heavy dataset archive is stored externally because GitHub is not suitable for large dataset binaries.

## Dataset Archive

- File name: `release_dataset_newdata_final_v1.tar.gz`
- SHA256: `7060499f2f24cf0fa9c98f42d024e983bd275d68ec52c2412011bebccbef8dd8`
- Google Drive: https://drive.google.com/drive/folders/11uX0ZFEIOKeEqdOmo-dobozEsPh3fvtk

## Contents

The release archive contains:

- final detector-ready dataset
- `real_only` split
- `combined_yolo` split
- accepted Stage B-soft synthetic SR samples
- final result tables
- RT-DETR validation reports
- YOLO11s ablation reports
- YOLOv8s multi-seed stability reports
- SAR fidelity/statistical audit outputs
- qualitative audit outputs
- scripts and configurations used in the project

## Important Note

The dataset is derived from public SAR ship detection datasets and synthetic SR processing. Users must respect the licenses and usage terms of the original datasets. If redistribution is restricted by any upstream dataset license, users should regenerate the dataset using the provided scripts and source datasets instead of redistributing derived images.
