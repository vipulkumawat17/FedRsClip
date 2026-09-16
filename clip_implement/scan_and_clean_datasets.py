"""Scan all 5 prepared datasets for corrupted/unreadable images and remove
those rows from train.jsonl/val.jsonl before training hits them mid-epoch.

For each dataset, backs up the original jsonl files (once) to *.jsonl.bak,
then writes a cleaned version with bad rows dropped. Prints a summary of
what was removed. Safe to re-run - it always scans from the .bak if one
exists, so re-running doesn't compound removals or lose data.

Usage:
    python scan_and_clean_datasets.py --datasets-root /home/mazaveri/hpc-prog/Vipul/data
"""

import argparse
import json
import os
import shutil

from PIL import Image

# (dataset_dir, image_root_subdir) - must match prepared_jsonl_datasets.py
DATASETS = [
    ("cifar_100", "prepared_images"),
    ("eurosat", "rgb_from_allbands"),
    ("flower_102", "raw/jpg"),
    ("food_101", "raw/images"),
    ("caltech_101", "raw/caltech-101/101_ObjectCategories"),
]


def is_readable(path):
    try:
        with Image.open(path) as im:
            im.convert("RGB")
        return True
    except Exception:
        return False


def clean_jsonl(jsonl_path, image_root):
    bak_path = jsonl_path + ".bak"
    if os.path.exists(bak_path):
        source_path = bak_path
    else:
        shutil.copy2(jsonl_path, bak_path)
        source_path = bak_path

    kept, dropped, malformed = [], [], []
    with open(source_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                malformed.append((lineno, str(e)))
                continue
            impath = os.path.join(image_root, row["image"])
            if os.path.exists(impath) and is_readable(impath):
                kept.append(line)
            else:
                dropped.append(row["image"])

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")

    return len(kept), dropped, malformed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", required=True)
    args = parser.parse_args()

    total_dropped = 0
    for dataset_dir, image_root_subdir in DATASETS:
        base = os.path.join(args.datasets_root, dataset_dir)
        image_root = os.path.join(base, image_root_subdir)
        print(f"\n=== {dataset_dir} (image_root={image_root}) ===")

        for split in ("train.jsonl", "val.jsonl"):
            jsonl_path = os.path.join(base, split)
            if not os.path.exists(jsonl_path):
                print(f"  {split}: not found, skipping")
                continue

            kept, dropped, malformed = clean_jsonl(jsonl_path, image_root)
            print(f"  {split}: {kept} kept, {len(dropped)} dropped, {len(malformed)} malformed lines")
            for d in dropped[:10]:
                print(f"    - {d}")
            if len(dropped) > 10:
                print(f"    ... and {len(dropped) - 10} more")
            for lineno, err in malformed[:10]:
                print(f"    [malformed] line {lineno}: {err}")
            if len(malformed) > 10:
                print(f"    ... and {len(malformed) - 10} more malformed lines")
            total_dropped += len(dropped) + len(malformed)

    print(f"\nTotal dropped across all datasets/splits: {total_dropped}")
    print("Originals backed up as train.jsonl.bak / val.jsonl.bak in each dataset folder.")


if __name__ == "__main__":
    main()