"""Dataset preparation and validation script for DefectLens AI (MVTec AD).

Supports:
1. Validating directory structure, image integrity, and mask correspondences.
2. Generating a comprehensive dataset summary JSON.
3. Extracting local MVTec AD archives (tar.xz / tar / zip).
4. Downloading standard category splits from open Hugging Face mirrors.
5. Generating synthetic industrial surface smoke-test data for offline testing.
"""

import argparse
import glob
import io
import json
import os
import shutil
import sys
import tarfile
from typing import Any, Dict, List, Optional
from PIL import Image, ImageDraw


def validate_dataset(category_dir: str) -> Dict[str, Any]:
    """Validate directory structure, image readability, and mask alignment.

    Args:
        category_dir: Path to category root (e.g. data/mvtec/bottle)

    Returns:
        Dictionary containing dataset validation summary and metrics
    """
    category_dir = os.path.abspath(category_dir)
    category_name = os.path.basename(category_dir)

    train_good_dir = os.path.join(category_dir, "train", "good")
    test_dir = os.path.join(category_dir, "test")
    gt_dir = os.path.join(category_dir, "ground_truth")

    summary = {
        "dataset_name": "MVTec Anomaly Detection (MVTec AD)",
        "category": category_name,
        "category_path": category_dir,
        "valid_structure": False,
        "train_good_count": 0,
        "test_good_count": 0,
        "test_defect_counts": {},
        "ground_truth_mask_count": 0,
        "total_images": 0,
        "image_resolution": None,
        "color_channels": 3,
        "corrupted_files": [],
        "missing_masks": [],
        "status": "pending",
    }

    if not os.path.exists(train_good_dir):
        summary["status"] = f"Missing train/good directory at {train_good_dir}"
        return summary

    if not os.path.exists(test_dir):
        summary["status"] = f"Missing test directory at {test_dir}"
        return summary

    # 1. Validate train/good images
    train_files = sorted(glob.glob(os.path.join(train_good_dir, "*.*")))
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    train_files = [f for f in train_files if f.lower().endswith(valid_exts)]

    for fpath in train_files:
        try:
            with Image.open(fpath) as img:
                img.verify()
            with Image.open(fpath) as img:
                if summary["image_resolution"] is None:
                    summary["image_resolution"] = list(img.size)
            summary["train_good_count"] += 1
        except Exception as e:
            summary["corrupted_files"].append(f"{fpath}: {str(e)}")

    # 2. Validate test images
    subdirs = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    for sdir in subdirs:
        spath = os.path.join(test_dir, sdir)
        test_files = sorted(glob.glob(os.path.join(spath, "*.*")))
        test_files = [f for f in test_files if f.lower().endswith(valid_exts)]

        count = 0
        for fpath in test_files:
            try:
                with Image.open(fpath) as img:
                    img.verify()
                count += 1
            except Exception as e:
                summary["corrupted_files"].append(f"{fpath}: {str(e)}")

            # Check matching mask for defect samples
            if sdir != "good" and os.path.exists(gt_dir):
                base_name = os.path.splitext(os.path.basename(fpath))[0]
                mask_candidates = [
                    os.path.join(gt_dir, sdir, f"{base_name}_mask.png"),
                    os.path.join(gt_dir, sdir, f"{base_name}.png"),
                ]
                if not any(os.path.exists(m) for m in mask_candidates):
                    summary["missing_masks"].append(fpath)

        if sdir == "good":
            summary["test_good_count"] = count
        else:
            summary["test_defect_counts"][sdir] = count

    # 3. Validate ground truth masks
    if os.path.exists(gt_dir):
        gt_files = sorted(glob.glob(os.path.join(gt_dir, "**", "*.*"), recursive=True))
        gt_files = [f for f in gt_files if f.lower().endswith(valid_exts)]
        for fpath in gt_files:
            try:
                with Image.open(fpath) as img:
                    img.verify()
                summary["ground_truth_mask_count"] += 1
            except Exception as e:
                summary["corrupted_files"].append(f"{fpath}: {str(e)}")

    summary["total_images"] = (
        summary["train_good_count"]
        + summary["test_good_count"]
        + sum(summary["test_defect_counts"].values())
    )

    if (
        summary["train_good_count"] > 0
        and (summary["test_good_count"] > 0 or len(summary["test_defect_counts"]) > 0)
        and len(summary["corrupted_files"]) == 0
    ):
        summary["valid_structure"] = True
        summary["status"] = "verified"
    else:
        summary["status"] = (
            f"Validation issues found: {len(summary['corrupted_files'])} corrupted files, "
            f"{summary['train_good_count']} train images, {summary['total_images']} total."
        )

    return summary


def extract_archive_if_present(archive_path: str, target_dir: str) -> bool:
    """Extract .tar.xz, .tar, or .zip archive if found."""
    if not os.path.exists(archive_path):
        return False

    print(f"[*] Extracting {archive_path} to {target_dir}...")
    os.makedirs(target_dir, exist_ok=True)
    if archive_path.endswith((".tar.xz", ".tar.gz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(path=target_dir)
        print(f"[+] Successfully extracted {archive_path}")
        return True
    elif archive_path.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(path=target_dir)
        print(f"[+] Successfully extracted {archive_path}")
        return True
    return False


def download_category_from_mirror(category: str, target_dir: str) -> bool:
    """Download category from open Hugging Face mirror."""
    import urllib.request
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("[!] pyarrow is required to unpack mirror parquet files. Please install pyarrow.")
        return False

    os.makedirs(target_dir, exist_ok=True)
    base_url = "https://huggingface.co/datasets/TheoM55/mvtec_all_objects_split/resolve/main/data"

    split_urls = {
        "train": f"{base_url}/{category}.train-00000-of-00001.parquet",
        "test": f"{base_url}/{category}.test-00000-of-00001.parquet",
    }

    cache_dir = os.path.join(target_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    category_dir = os.path.join(target_dir, category)
    os.makedirs(category_dir, exist_ok=True)

    for split_name, url in split_urls.items():
        parquet_file = os.path.join(cache_dir, f"{category}.{split_name}.parquet")
        if not os.path.exists(parquet_file):
            print(f"[*] Downloading {split_name} data for '{category}'...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(parquet_file, "wb") as f:
                shutil.copyfileobj(resp, f)
            print(f"[+] Downloaded {parquet_file}")

        print(f"[*] Unpacking {parquet_file} into {category_dir}...")
        table = pq.read_table(parquet_file)
        rows = table.to_pylist()

        for idx, row in enumerate(rows):
            defect_type = row.get("defect") or row.get("defect_type") or ("good" if split_name == "train" else "anomaly")
            defect_type = str(defect_type).lower().replace(" ", "_")

            # Extract image filename and bytes
            img_data = row.get("image_path") or row.get("image")
            img_name = None
            img_bytes = None
            if isinstance(img_data, dict):
                img_bytes = img_data.get("bytes")
                img_name = img_data.get("path")
            elif isinstance(img_data, bytes):
                img_bytes = img_data

            if not img_name:
                img_name = f"{idx:03d}.png"
            else:
                img_name = os.path.basename(img_name)

            if split_name == "train":
                out_img_dir = os.path.join(category_dir, "train", "good")
            else:
                out_img_dir = os.path.join(category_dir, "test", defect_type)

            os.makedirs(out_img_dir, exist_ok=True)
            img_out_path = os.path.join(out_img_dir, img_name)

            if img_bytes:
                with open(img_out_path, "wb") as f:
                    f.write(img_bytes)

            # Mask unpacking for defective test samples
            if split_name == "test" and defect_type != "good":
                mask_data = row.get("mask_path") or row.get("mask")
                mask_name = None
                mask_bytes = None
                if isinstance(mask_data, dict):
                    mask_bytes = mask_data.get("bytes")
                    mask_name = mask_data.get("path")
                elif isinstance(mask_data, bytes):
                    mask_bytes = mask_data

                if not mask_name:
                    base = os.path.splitext(img_name)[0]
                    mask_name = f"{base}_mask.png"
                else:
                    mask_name = os.path.basename(mask_name)

                if mask_bytes:
                    gt_dir = os.path.join(category_dir, "ground_truth", defect_type)
                    os.makedirs(gt_dir, exist_ok=True)
                    mask_out_path = os.path.join(gt_dir, mask_name)
                    with open(mask_out_path, "wb") as f:
                        f.write(mask_bytes)


    print(f"[+] Successfully prepared {category} dataset in {category_dir}")
    return True


def create_synthetic_smoke_data(category_dir: str) -> None:
    """Generate synthetic normal and defective surface images for pipeline smoke tests.

    Creates realistic synthetic industrial bottle surfaces with cylindrical bodies,
    specular reflections, and synthetic flaws (broken edges, scratches, contamination)
    with accurate ground-truth binary masks.
    """
    os.makedirs(category_dir, exist_ok=True)
    train_good_dir = os.path.join(category_dir, "train", "good")
    test_good_dir = os.path.join(category_dir, "test", "good")
    test_broken_dir = os.path.join(category_dir, "test", "broken_large")
    test_contam_dir = os.path.join(category_dir, "test", "contamination")
    gt_broken_dir = os.path.join(category_dir, "ground_truth", "broken_large")
    gt_contam_dir = os.path.join(category_dir, "ground_truth", "contamination")

    for d in [train_good_dir, test_good_dir, test_broken_dir, test_contam_dir, gt_broken_dir, gt_contam_dir]:
        os.makedirs(d, exist_ok=True)

    img_size = (256, 256)

    def draw_normal_surface(seed_val: int) -> Image.Image:
        # Background: dark metallic slate
        img = Image.new("RGB", img_size, color=(35, 38, 45))
        draw = ImageDraw.Draw(img)

        # Smooth bottle body: rounded rectangle / cylinder
        # Center x: 128, y: 30 to 226
        offset = (seed_val % 5) - 2
        body_box = [70 + offset, 60, 186 + offset, 225]
        neck_box = [105 + offset, 25, 151 + offset, 60]

        # Draw glass bottle body with smooth gradient
        draw.rectangle(neck_box, fill=(110, 150, 160))
        draw.rounded_rectangle(body_box, radius=18, fill=(130, 180, 195))

        # Specular highlight reflection stripe
        draw.line([95 + offset, 70, 95 + offset, 215], fill=(210, 235, 245), width=4)
        draw.line([102 + offset, 75, 102 + offset, 210], fill=(175, 215, 230), width=2)
        return img

    # 1. Generate normal training images
    print("[*] Generating synthetic normal training samples...")
    for i in range(25):
        img = draw_normal_surface(seed_val=i)
        img.save(os.path.join(train_good_dir, f"{i:03d}.png"))

    # 2. Generate normal test images
    print("[*] Generating synthetic normal test samples...")
    for i in range(10):
        img = draw_normal_surface(seed_val=100 + i)
        img.save(os.path.join(test_good_dir, f"{i:03d}.png"))

    # 3. Generate defective test images + masks: broken large
    print("[*] Generating synthetic broken large defect samples...")
    for i in range(8):
        img = draw_normal_surface(seed_val=200 + i)
        mask = Image.new("L", img_size, color=0)

        img_draw = ImageDraw.Draw(img)
        mask_draw = ImageDraw.Draw(mask)

        # Severe break / notch on edge
        notch_box = [65, 120 + i * 8, 98, 155 + i * 8]
        img_draw.polygon(
            [(65, 120 + i * 8), (98, 138 + i * 8), (75, 155 + i * 8), (60, 140 + i * 8)],
            fill=(35, 38, 45)  # reveals background
        )
        img_draw.line([(65, 120 + i * 8), (98, 138 + i * 8), (75, 155 + i * 8)], fill=(255, 255, 255), width=2)

        mask_draw.polygon(
            [(65, 120 + i * 8), (98, 138 + i * 8), (75, 155 + i * 8), (60, 140 + i * 8)],
            fill=255
        )

        img.save(os.path.join(test_broken_dir, f"{i:03d}.png"))
        mask.save(os.path.join(gt_broken_dir, f"{i:03d}_mask.png"))

    # 4. Generate defective test images + masks: contamination
    print("[*] Generating synthetic contamination defect samples...")
    for i in range(8):
        img = draw_normal_surface(seed_val=300 + i)
        mask = Image.new("L", img_size, color=0)

        img_draw = ImageDraw.Draw(img)
        mask_draw = ImageDraw.Draw(mask)

        cx, cy = 135 + (i % 3) * 10, 130 + (i * 9)
        r = 12
        img_draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(40, 20, 15))
        mask_draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

        img.save(os.path.join(test_contam_dir, f"{i:03d}.png"))
        mask.save(os.path.join(gt_contam_dir, f"{i:03d}_mask.png"))

    print(f"[+] Synthetic dataset successfully generated in: {category_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare and validate MVTec AD dataset for DefectLens AI")
    parser.add_argument("--category", type=str, default="bottle", help="Dataset category (e.g. bottle, capsule)")
    parser.add_argument("--data-dir", type=str, default="data/mvtec", help="Directory where MVTec data is stored")
    parser.add_argument("--output-summary", type=str, default="data/dataset_summary.json", help="Summary JSON output path")
    parser.add_argument("--download", action="store_true", help="Download category from public mirror if absent")
    parser.add_argument("--smoke-test-data", action="store_true", help="Generate synthetic smoke data for testing")
    args = parser.parse_args()

    category_dir = os.path.join(args.data_dir, args.category)

    if args.smoke_test_data:
        create_synthetic_smoke_data(category_dir)

    # Check for local tar.xz archive in data-dir
    parent_dir = os.path.dirname(os.path.abspath(category_dir))
    archive_candidates = [
        os.path.join(parent_dir, f"{args.category}.tar.xz"),
        os.path.join(parent_dir, "mvtec_anomaly_detection.tar.xz"),
        os.path.join("data", f"{args.category}.tar.xz"),
        os.path.join("data", "mvtec_anomaly_detection.tar.xz"),
    ]
    for cand in archive_candidates:
        if os.path.exists(cand):
            extract_archive_if_present(cand, parent_dir)
            break

    # If missing and download requested
    if args.download and not os.path.exists(os.path.join(category_dir, "train", "good")):
        success = download_category_from_mirror(args.category, args.data_dir)
        if not success:
            print("[!] Download failed or was interrupted.")

    print(f"[*] Validating dataset at: {category_dir}")
    summary = validate_dataset(category_dir)

    # Write summary file
    os.makedirs(os.path.dirname(os.path.abspath(args.output_summary)), exist_ok=True)
    with open(args.output_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[+] Dataset summary written to {args.output_summary}")
    print(f"    Category: {summary['category']}")
    print(f"    Train good images: {summary['train_good_count']}")
    print(f"    Test good images: {summary['test_good_count']}")
    print(f"    Test defect images: {sum(summary['test_defect_counts'].values())} ({summary['test_defect_counts']})")
    print(f"    Status: {summary['status']}")

    if not summary["valid_structure"]:
        print("\n[!] Dataset is not ready for training yet.")
        print(f"    To obtain the official MVTec AD dataset:")
        print(f"    1. Download from https://www.mvtec.com/company/research/datasets/mvtec-ad")
        print(f"    2. Place {args.category}.tar.xz into 'data/' and re-run this script.")
        print(f"    3. Or use --download to fetch from public mirror.")
        print(f"    4. Or use --smoke-test-data to run with synthetic industrial test images.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
