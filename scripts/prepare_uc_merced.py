from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CLASS_DISPLAY_NAMES: Dict[str, str] = {
    "agricultural": "agricultural land",
    "airplane": "airplanes",
    "baseballdiamond": "a baseball diamond",
    "beach": "a beach",
    "buildings": "buildings",
    "chaparral": "chaparral",
    "denseresidential": "dense residential area",
    "forest": "a forest",
    "freeway": "a freeway",
    "golfcourse": "a golf course",
    "harbor": "a harbor",
    "intersection": "an intersection",
    "mediumresidential": "medium density residential area",
    "mobilehomepark": "a mobile home park",
    "overpass": "an overpass",
    "parkinglot": "a parking lot",
    "river": "a river",
    "runway": "a runway",
    "sparseresidential": "sparse residential area",
    "storagetanks": "storage tanks",
    "tenniscourt": "a tennis court",
}

PROMPTS = [
    "a satellite photo of {}.",
    "an aerial image of {}.",
    "a remote sensing image of {}.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare UC Merced Land Use for CLIP experiments")
    parser.add_argument("--zip", required=True, type=Path, help="Path to UC_merced_land_datsets.zip")
    parser.add_argument("--output", type=Path, default=Path("data/uc_merced"), help="Output dataset directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    extract_dir = args.output / "raw"
    images_dir = extract_dir / "UCMerced_LandUse" / "Images"

    if args.force_extract or not images_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {args.zip} -> {extract_dir}")
        with zipfile.ZipFile(args.zip) as archive:
            archive.extractall(extract_dir)
    else:
        print(f"Using existing extracted data at {images_dir}")

    class_dirs = sorted(path for path in images_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No class folders found under {images_dir}")

    rng = random.Random(args.seed)
    splits = {"train": [], "val": [], "test": []}
    for class_dir in class_dirs:
        images = sorted(list(class_dir.glob("*.tif")) + list(class_dir.glob("*.tiff")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
        rng.shuffle(images)
        train_end = int(len(images) * args.train_ratio)
        val_end = train_end + int(len(images) * args.val_ratio)
        label = class_dir.name
        splits["train"].extend(make_rows(images[:train_end], label, images_dir, rng))
        splits["val"].extend(make_rows(images[train_end:val_end], label, images_dir, rng))
        splits["test"].extend(make_rows(images[val_end:], label, images_dir, rng))

    for split_name, rows in splits.items():
        path = args.output / f"{split_name}.jsonl"
        write_jsonl(path, rows)
        print(f"Wrote {len(rows)} rows to {path}")

    classnames = [CLASS_DISPLAY_NAMES.get(path.name, path.name) for path in class_dirs]
    (args.output / "classnames.txt").write_text("\n".join(classnames) + "\n", encoding="utf-8")
    (args.output / "labels.txt").write_text("\n".join(path.name for path in class_dirs) + "\n", encoding="utf-8")
    print(f"Wrote {len(classnames)} class names to {args.output / 'classnames.txt'}")
    print(f"Image root for training: {images_dir}")


def make_rows(images: Iterable[Path], label: str, image_root: Path, rng: random.Random) -> List[dict]:
    display_name = CLASS_DISPLAY_NAMES.get(label, label)
    rows = []
    for image in images:
        rows.append(
            {
                "image": image.relative_to(image_root).as_posix(),
                "caption": rng.choice(PROMPTS).format(display_name),
                "label": label,
                "class_name": display_name,
            }
        )
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
