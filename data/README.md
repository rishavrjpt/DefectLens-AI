# MVTec Anomaly Detection (MVTec AD) Dataset

This directory contains data for **DefectLens AI**.

## Official Dataset Source & Citation

- **Website:** [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- **Reference Paper:**
  > Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger.
  > *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.*
  > In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).
  - Non-commercial educational and research use only.
  - Commercial use requires authorization from MVTec Software GmbH.

## Expected Directory Layout

The default category is `bottle`, but DefectLens AI supports any MVTec category (e.g., `capsule`, `carpet`, `hazelnut`, `metal_nut`, `cable`):

```text
data/
└── mvtec/
    └── bottle/
        ├── train/
        │   └── good/
        │       ├── 000.png
        │       └── ...
        ├── test/
        │   ├── good/
        │   │   ├── 000.png
        │   │   └── ...
        │   ├── broken_large/
        │   ├── broken_small/
        │   └── contamination/
        └── ground_truth/
            ├── broken_large/
            │   ├── 000_mask.png
            │   └── ...
            ├── broken_small/
            └── contamination/
```

## How to Obtain and Prepare the Dataset

You can download and extract the dataset automatically or manually:

### Option A: Automatic via Script
```bash
python src/data/prepare_dataset.py --category bottle --download
```

### Option B: Manual Download
1. Visit https://www.mvtec.com/company/research/datasets/mvtec-ad
2. Download the `bottle.tar.xz` archive (or full MVTec AD dataset).
3. Place `bottle.tar.xz` inside `data/` or extract it into `data/mvtec/bottle/`.
4. Validate the directory structure and generate the summary:
```bash
python src/data/prepare_dataset.py --category bottle --data-dir data/mvtec
```
