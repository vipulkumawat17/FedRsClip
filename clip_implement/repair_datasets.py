"""Repair broken train/val manifests and re-extract/re-convert images for the
dissertation datasets, using the patched (crash-safe, self-healing)
dataset_factory.py.

Usage:
    python repair_datasets.py --datasets-root /home/mazaveri/hpc-prog/Vipul/data

By default this repairs the four datasets your scan flagged as broken:
cifar_100, eurosat, flower_102, caltech_101. food_101 was clean and is left
untouched. Pass --datasets to override which ones run.

What this does, per dataset:
  1. Deletes only the *derived* artifacts that were found to be stale/corrupt
     (train.jsonl, val.jsonl, test.jsonl, classnames.txt, image_root.txt, and
     any cached prepared-image / converted-RGB / extracted-jpg directories).
     Downloaded raw archives (zips, .tgz, .mat, pickle batches, etc.) are
     NEVER touched, so nothing has to be re-downloaded.
  2. Calls prepare_dataset_source(..., force_extract=True, force_prepare=True)
     from the patched dataset_factory module so everything is rebuilt from
     the raw sources on disk.

Run this from the same folder as run_experiment.py (or pass --repo-root),
so `clip_implement.dataset_factory` (patched copy) can be imported.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Make sure we import the PATCHED dataset_factory, not an old cached one.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_factory import PrepareConfig, prepare_dataset_source, DEFAULT_TEMPLATES  # noqa: E402


# Which derived artifacts to wipe per dataset before re-running, relative to
# the dataset's work_dir (datasets_root / dataset_folder_name).
# "raw/..." entries are intermediate re-derivable data (extracted jpgs,
# converted RGB pngs) -- NOT the original downloaded archives.
REPAIR_TARGETS: dict[str, list[str]] = {
    "cifar_100": [
        "train.jsonl", "val.jsonl", "test.jsonl",
        "classnames.txt", "image_root.txt",
        "prepared_images",  # regenerated cheaply from the cifar-100 pickle batches
    ],
    "eurosat": [
        "train.jsonl", "val.jsonl", "test.jsonl",
        "classnames.txt", "image_root.txt",
        "rgb_from_allbands",  # includes the stale .conversion_complete marker
    ],
    "flower_102": [
        "train.jsonl", "val.jsonl", "test.jsonl",
        "classnames.txt", "image_root.txt",
        "raw/jpg",  # partially-extracted 102flowers.tgz output (only 32/8189 present)
    ],
    "caltech_101": [
        "train.jsonl", "val.jsonl", "test.jsonl",
        "classnames.txt", "image_root.txt",
        # raw/ is left alone: it's the source archive extraction and appears intact,
        # the manifest just pointed at the wrong / stale root. Re-prepare relocates it.
    ],
}

DATASET_FOLDER_NAMES = {
    "cifar_100": "cifar_100",
    "eurosat": "eurosat",
    "flower_102": "flower_102",
    "food_101": "food_101",
    "caltech_101": "caltech_101",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair broken dataset manifests/images.")
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(REPAIR_TARGETS.keys()),
        default=list(REPAIR_TARGETS.keys()),
        help="Which datasets to repair (default: all four broken ones).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--convert-multiband-tiff", action="store_true", default=True,
                         help="Re-derive EuroSAT RGB pngs from the allBands TIFFs (default on).")
    parser.add_argument("--rgb-bands", default="4,3,2")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted/rebuilt, do nothing.")
    return parser.parse_args()


def wipe_targets(work_dir: Path, relative_targets: list[str], dry_run: bool) -> None:
    for rel in relative_targets:
        target = work_dir / rel
        if not target.exists():
            continue
        if dry_run:
            print(f"  [DRY-RUN] would remove {target}")
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        print(f"  removed {target}")


def main() -> None:
    args = parse_args()
    config = PrepareConfig(
        datasets_root=args.datasets_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        force_extract=True,
        force_prepare=True,
        convert_multiband_tiff=args.convert_multiband_tiff,
        rgb_bands=args.rgb_bands,
    )

    for dataset_name in args.datasets:
        work_dir = args.datasets_root / DATASET_FOLDER_NAMES[dataset_name]
        print(f"=== Repairing {dataset_name} ({work_dir}) ===")
        if not work_dir.exists():
            print(f"  [SKIP] {work_dir} does not exist, nothing to repair")
            continue

        wipe_targets(work_dir, REPAIR_TARGETS[dataset_name], args.dry_run)

        if args.dry_run:
            print(f"  [DRY-RUN] would re-run prepare_dataset_source on {work_dir}")
            continue

        prepared = prepare_dataset_source(work_dir, config, DEFAULT_TEMPLATES)
        print(
            f"  [OK] {prepared.name}: classes={prepared.class_count} "
            f"train={prepared.train_count} val={prepared.val_count} "
            f"image_root={prepared.image_root}"
        )

    print("Done. Re-run scan_and_clean_datasets.py to confirm 0 dropped rows.")


if __name__ == "__main__":
    main()