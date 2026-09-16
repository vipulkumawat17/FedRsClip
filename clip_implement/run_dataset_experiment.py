"""
Prepare and (optionally) train CLIP on one or more local datasets.

This merges the two scripts you had:
  - run_experiment.txt          -> gave you the --datasets shortcut (cifar_100,
                                    eurosat, flower_102, food_101, caltech_101)
  - run_dataset_experiment.py   -> gave you --report-samples for training reports

Both are kept here as clearly separated blocks so you can use whichever you need:

  BLOCK 1 - Dataset selection: pick datasets either the easy way with
            --datasets NAME [NAME ...] (uses DEFAULT_DATASET_ALIASES to find
            them under --datasets-root), or manually with --dataset-folder /
            --dataset-zip / --all.

  BLOCK 2 - Prepare only: pass --prepare-only to just build/refresh the
            train.jsonl / val.jsonl / classnames.txt files (this is what you
            want after fixing the flower_102 classnames bug -- run with
            --force-prepare --prepare-only to regenerate the JSONL/classnames
            without training).

  BLOCK 3 - Train: without --prepare-only, each prepared dataset is trained
            sequentially by shelling out to `python -m <module>` with all the
            standard training flags, including --report-samples.

Examples
--------
Just prepare (or re-prepare) flower_102 after the classnames fix, no training:
    python run_experiment.py --datasets flower_102 --force-prepare --prepare-only

Prepare + train all five dissertation datasets:
    python run_experiment.py --datasets cifar_100 eurosat flower_102 food_101 caltech_101

Dry-run to see the exact training commands without executing them:
    python run_experiment.py --all --dry-run
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clip_implement.dataset_factory import DEFAULT_TEMPLATES, PrepareConfig, PreparedDataset, prepare_dataset_source, slugify

# ---------------------------------------------------------------------------
# BLOCK 1: dataset selection helpers
# ---------------------------------------------------------------------------

DEFAULT_DATASET_ALIASES = {
    "cifar_100": ["cifar_100"],
    "eurosat": ["eurosat"],
    "flower_102": ["flower_102"],
    "food_101": ["food_101"],
    "caltech_101": ["caltech_101"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare local CLIP datasets and, optionally, train each one sequentially. "
            "Supports registered TorchVision datasets, prepared JSONL folders, "
            "ImageFolder datasets, and zip files that extract into ImageFolder datasets."
        )
    )
    parser.add_argument("--datasets-root", type=Path, default=Path("data"), help="Parent folder containing dataset folders or zip files")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["cifar_100", "eurosat", "flower_102", "food_101", "caltech_101"],
        default=None,
        help="Shortcut for the five dissertation datasets, e.g. --datasets cifar_100 eurosat flower_102 food_101 caltech_101",
    )
    parser.add_argument(
        "--dataset-folder",
        action="append",
        default=None,
        help="Dataset folder name under --datasets-root, or an absolute/relative path. Repeat to run several.",
    )
    parser.add_argument(
        "--dataset-zip",
        action="append",
        default=None,
        help="Dataset zip name under --datasets-root, or an absolute/relative path. Repeat to run several.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every immediate dataset folder and .zip file found under --datasets-root.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("checkpoints") / "datasets")
    parser.add_argument("--module", default="clip_implement.train", help="Training module to execute with python -m")
    parser.add_argument("--model", default="ViT-B-32", choices=["RN50", "RN101", "ViT-B-32", "ViT-B-16"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=76)
    parser.add_argument("--precision", choices=["fp32", "amp"], default="amp")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--report-samples", type=int, default=8, help="Number of qualitative samples to log per report (train mode only)")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--template",
        action="append",
        default=None,
        help="Prompt template used for generated captions and validation. Use {} for the class name. Repeat for several.",
    )
    parser.add_argument("--force-extract", action="store_true", help="Extract zip files even if a raw folder already exists")
    parser.add_argument("--force-prepare", action="store_true", help="Regenerate train/val/test JSONL and classnames.txt files")
    parser.add_argument(
        "--convert-multiband-tiff",
        action="store_true",
        help="Convert 13-band TIFF datasets such as EuroSAT allBands into RGB PNGs before training.",
    )
    parser.add_argument(
        "--rgb-bands",
        default="4,3,2",
        help="1-based band numbers used for --convert-multiband-tiff. Default 4,3,2 is natural-color Sentinel-2 RGB.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip a dataset when clip_last.pt already exists")

    # BLOCK 2 / BLOCK 3 switch: prepare-only vs prepare-and-train
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="BLOCK 2: only (re)build JSONL/classnames files; do not start training. "
        "Use with --force-prepare to regenerate after fixing a data/classnames bug.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print training commands without executing them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    templates = args.template or DEFAULT_TEMPLATES
    sources = collect_sources(args)
    if not sources:
        raise SystemExit("No datasets selected. Use --datasets, --dataset-folder, --dataset-zip, or --all.")

    config = PrepareConfig(
        datasets_root=args.datasets_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        force_extract=args.force_extract,
        force_prepare=args.force_prepare,
        convert_multiband_tiff=args.convert_multiband_tiff,
        rgb_bands=args.rgb_bands,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "run_summary.csv"
    had_success = False

    for source in sources:
        # -------------------------------------------------------------
        # BLOCK 2: prepare (always runs, for both prepare-only and train modes)
        # -------------------------------------------------------------
        try:
            prepared = prepare_dataset_source(source, config, templates)
        except Exception as exc:
            dataset_label = source.stem if source.is_file() else source.name
            print(f"[WARNING] Unsupported dataset {dataset_label}: {exc}", flush=True)
            continue

        print_prepared_summary(prepared)
        output_dir = dataset_output_dir(prepared.name, args)
        clip_last = output_dir / "clip_last.pt"

        if args.prepare_only:
            print(f"[PREPARED] {prepared.name}: train={prepared.train_jsonl} image_root={prepared.image_root}", flush=True)
            append_summary(summary_path, prepared, output_dir, "prepared")
            had_success = True
            continue

        if args.skip_existing and clip_last.exists():
            print(f"[SKIP] {prepared.name}: {clip_last} already exists", flush=True)
            append_summary(summary_path, prepared, output_dir, "skipped_existing")
            had_success = True
            continue

        # -------------------------------------------------------------
        # BLOCK 3: train
        # -------------------------------------------------------------
        command = build_train_command(prepared, args, templates)
        print(" ".join(quote_arg(part) for part in command), flush=True)
        if args.dry_run:
            append_summary(summary_path, prepared, output_dir, "dry_run")
            had_success = True
            continue

        result = subprocess.run(command, check=False)
        status = "ok" if result.returncode == 0 else f"failed_{result.returncode}"
        append_summary(summary_path, prepared, output_dir, status)
        if result.returncode != 0:
            print(f"[WARNING] Training failed for {prepared.name} with exit code {result.returncode}", flush=True)
            continue
        had_success = True

    if not had_success:
        raise SystemExit("No datasets were prepared or trained successfully.")


def collect_sources(args: argparse.Namespace) -> list[Path]:
    sources: list[Path] = []

    if args.datasets:
        for dataset_name in args.datasets:
            sources.append(resolve_named_dataset(dataset_name, args.datasets_root))

    if args.all:
        root = args.datasets_root
        if not root.exists():
            raise FileNotFoundError(root)
        sources.extend(sorted(path for path in root.iterdir() if path.is_dir()))
        sources.extend(sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".zip"))

    for value in args.dataset_folder or []:
        sources.append(resolve_dataset_path(value, args.datasets_root))

    for value in args.dataset_zip or []:
        sources.append(resolve_dataset_path(value, args.datasets_root))

    return dedupe_sources_by_name(dedupe_paths(sources))


def resolve_named_dataset(dataset_name: str, datasets_root: Path) -> Path:
    for candidate_name in DEFAULT_DATASET_ALIASES[dataset_name]:
        candidate = datasets_root / candidate_name
        if candidate.exists():
            return candidate

    candidates = ", ".join(DEFAULT_DATASET_ALIASES[dataset_name])
    raise FileNotFoundError(
        f"Prepared dataset '{dataset_name}' was not found under {datasets_root}. "
        f"Expected one of: {candidates}"
    )


def resolve_dataset_path(value: str, datasets_root: Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    rooted = datasets_root / path
    if rooted.exists():
        return rooted
    raise FileNotFoundError(f"Could not find dataset source: {value}")


def dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(path)
    return result


def dedupe_sources_by_name(paths: Iterable[Path]) -> list[Path]:
    by_name: dict[str, Path] = {}
    order: list[str] = []
    for path in paths:
        key = slugify(path.stem if path.is_file() else path.name)
        if key not in by_name:
            by_name[key] = path
            order.append(key)
            continue
        if path.is_dir() and by_name[key].is_file():
            by_name[key] = path
    return [by_name[key] for key in order]


# ---------------------------------------------------------------------------
# BLOCK 3: training command construction (only used when --prepare-only is not set)
# ---------------------------------------------------------------------------

def build_train_command(prepared: PreparedDataset, args: argparse.Namespace, templates: Sequence[str]) -> list[str]:
    output_dir = dataset_output_dir(prepared.name, args)
    report_dir = output_dir / "reports"
    command = [
        sys.executable,
        "-m",
        args.module,
        "--train-jsonl",
        str(prepared.train_jsonl),
        "--image-root",
        str(prepared.image_root),
        "--model",
        args.model,
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-steps",
        str(args.warmup_steps),
        "--workers",
        str(args.workers),
        "--context-length",
        str(args.context_length),
        "--precision",
        args.precision,
        "--save-every",
        str(args.save_every),
        "--save-every-steps",
        str(args.save_every_steps),
        "--eval-every",
        str(args.eval_every),
        "--report-samples",
        str(args.report_samples),
        "--output-dir",
        str(output_dir),
        "--report-dir",
        str(report_dir),
    ]
    if prepared.val_jsonl:
        command.extend(["--val-jsonl", str(prepared.val_jsonl)])
    if prepared.classnames:
        command.extend(["--classnames", str(prepared.classnames)])
    for template in templates:
        command.extend(["--template", template])
    return command


def dataset_output_dir(dataset_name: str, args: argparse.Namespace) -> Path:
    return args.output_root / dataset_name / args.model


def append_summary(summary_path: Path, prepared: PreparedDataset, output_dir: Path, status: str) -> None:
    row = {
        "dataset": prepared.name,
        "dataset_type": prepared.dataset_type,
        "status": status,
        "classes": prepared.class_count,
        "train_images": prepared.train_count,
        "validation_images": prepared.val_count,
        "train_jsonl": str(prepared.train_jsonl),
        "val_jsonl": str(prepared.val_jsonl or ""),
        "classnames": str(prepared.classnames or ""),
        "image_root": str(prepared.image_root),
        "output_dir": str(output_dir),
    }
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def print_prepared_summary(prepared: PreparedDataset) -> None:
    print("==========================", flush=True)
    print(f"Dataset: {prepared.name}", flush=True)
    print(f"Type: {prepared.dataset_type}", flush=True)
    print(f"Classes: {prepared.class_count}", flush=True)
    print(f"Train Images: {prepared.train_count}", flush=True)
    print(f"Validation Images: {prepared.val_count}", flush=True)
    print("==========================", flush=True)


def quote_arg(value: str) -> str:
    if not value or any(char.isspace() for char in value):
        return f'"{value}"'
    return value


if __name__ == "__main__":
    main()