from __future__ import annotations

import argparse
import csv
import math
import random
import textwrap
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]

BLUE = "#2563eb"
BLUE_DARK = "#1d4ed8"
BLUE_LIGHT = "#dbeafe"
INK = "#111827"
MUTED = "#64748b"
BORDER = "#dbe4f0"
GRID = "#e8eef7"
CARD = "#ffffff"
BG = "#ffffff"
GREEN = "#16a34a"
GREEN_SOFT = "#dcfce7"
RED = "#dc2626"
RED_SOFT = "#fee2e2"
AMBER = "#f59e0b"
VIOLET = "#7c3aed"
CYAN = "#0891b2"


@dataclass(frozen=True)
class MetricRow:
    epoch: int
    global_step: int
    train_loss: float
    train_image_to_text_top1: float
    train_text_to_image_top1: float
    val_zero_shot_top1: float
    val_zero_shot_top5: float


@dataclass(frozen=True)
class PredictionRow:
    image_path: Path
    true_class: str
    predicted_class: str
    is_correct: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-quality CLIP evaluation dashboard figures for one or more datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset keys to process, e.g. --datasets uc_merced resisc45 eurosat. "
        "If omitted, every subfolder under --checkpoints-root that contains reports/metrics.csv is used.",
    )
    parser.add_argument("--checkpoints-root", default="checkpoints", help="Root folder containing <dataset>/reports and <dataset>/zero_shot_reports")
    parser.add_argument("--data-root", default="data", help="Root folder containing <dataset>/classnames.txt")
    parser.add_argument("--skip-trained", action="store_true", help="Skip locally fine-tuned datasets (those with reports/metrics.csv)")
    parser.add_argument("--metrics-subpath", default="reports/metrics.csv")
    parser.add_argument("--predictions-subpath", default="zero_shot_reports/predictions.csv")
    parser.add_argument("--summary-subpath", default="zero_shot_reports/summary.csv")
    parser.add_argument("--classnames-name", default="classnames.txt")
    parser.add_argument("--output-subdir", default="evaluation_dashboard", help="Subfolder created under each dataset's checkpoint dir for the generated PNGs")
    # Pretrained zero-shot datasets: checkpoints/pretrained_zero_shot/<dataset>/<variant>/{predictions,summary}.csv,
    # e.g. checkpoints/pretrained_zero_shot/caltech_101/ViT-B-32_openai/. No metrics.csv, so no training curves page.
    parser.add_argument("--pretrained-subdir", default="pretrained_zero_shot", help="Subfolder of --checkpoints-root holding zero-shot-only datasets")
    parser.add_argument("--pretrained-datasets", nargs="+", default=None, help="Pretrained zero-shot dataset keys to process. Default: auto-discover.")
    parser.add_argument("--skip-pretrained", action="store_true", help="Skip OpenAI-pretrained zero-shot datasets entirely")
    parser.add_argument("--pretrained-predictions-name", default="predictions.csv")
    parser.add_argument("--pretrained-summary-name", default="summary.csv")
    # Single-dataset overrides. When any of these are set, only that one dataset run is produced
    # (matches the original script's behaviour) instead of iterating over --datasets.
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--classnames", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-label", default=None, help="Display name for the dataset when using single-dataset overrides")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=4000)
    parser.add_argument("--height", type=int, default=3200)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def discover_datasets(checkpoints_root: Path, metrics_subpath: str) -> list[str]:
    if not checkpoints_root.exists():
        return []
    found = []
    for child in sorted(checkpoints_root.iterdir()):
        if child.is_dir() and (child / metrics_subpath).exists():
            found.append(child.name)
    return found


def discover_pretrained_datasets(pretrained_root: Path, predictions_name: str) -> list[tuple[str, str]]:
    if not pretrained_root.exists():
        return []
    found = []
    for dataset_dir in sorted(pretrained_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for variant_dir in sorted(dataset_dir.iterdir()):
            if variant_dir.is_dir() and (variant_dir / predictions_name).exists():
                found.append((dataset_dir.name, variant_dir.name))
    return found


def classnames_or_infer(classnames_path: Path, predictions: list[PredictionRow]) -> list[str]:
    names = read_classnames(classnames_path)
    if names:
        return names
    return sorted({row.true_class for row in predictions if row.true_class})


def dataset_label(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip().title()


def main() -> None:
    args = parse_args()

    # Backwards-compatible single-dataset mode: explicit --metrics/--predictions/etc.
    if args.metrics or args.predictions or args.summary or args.classnames or args.output_dir:
        label = args.dataset_label or "Dataset"
        run_dashboard(
            label=label,
            metrics_path=Path(args.metrics or f"{args.checkpoints_root}/uc_merced/{args.metrics_subpath}"),
            predictions_path=Path(args.predictions or f"{args.checkpoints_root}/uc_merced/{args.predictions_subpath}"),
            summary_path=Path(args.summary or f"{args.checkpoints_root}/uc_merced/{args.summary_subpath}"),
            classnames_path=Path(args.classnames or f"{args.data_root}/uc_merced/{args.classnames_name}"),
            output_dir=Path(args.output_dir or f"{args.checkpoints_root}/uc_merced/{args.output_subdir}"),
            seed=args.seed,
            width=args.width,
            height=args.height,
            dpi=args.dpi,
        )
        return

    checkpoints_root = Path(args.checkpoints_root)
    data_root = Path(args.data_root)

    jobs: list[tuple[str, "Any"]] = []

    if not args.skip_trained:
        trained = args.datasets or discover_datasets(checkpoints_root, args.metrics_subpath)
        for name in trained:
            jobs.append((
                name,
                lambda n=name: run_dashboard(
                    label=dataset_label(n),
                    metrics_path=checkpoints_root / n / args.metrics_subpath,
                    predictions_path=checkpoints_root / n / args.predictions_subpath,
                    summary_path=checkpoints_root / n / args.summary_subpath,
                    classnames_path=data_root / n / args.classnames_name,
                    output_dir=checkpoints_root / n / args.output_subdir,
                    seed=args.seed,
                    width=args.width,
                    height=args.height,
                    dpi=args.dpi,
                ),
            ))

    if not args.skip_pretrained:
        pretrained_root = checkpoints_root / args.pretrained_subdir
        pretrained = args.pretrained_datasets or discover_pretrained_datasets(pretrained_root, args.pretrained_predictions_name)
        for name, variant in pretrained:
            variant_dir = pretrained_root / name / variant
            label = f"{dataset_label(name)} ({variant})"
            model_name = f"OpenAI CLIP ({variant})"
            jobs.append((
                label,
                lambda vd=variant_dir, n=name, l=label, mn=model_name: run_pretrained_dashboard(
                    label=l,
                    predictions_path=vd / args.pretrained_predictions_name,
                    summary_path=vd / args.pretrained_summary_name,
                    classnames_path=data_root / n / args.classnames_name,
                    output_dir=vd / args.output_subdir,
                    seed=args.seed,
                    width=args.width,
                    height=args.height,
                    dpi=args.dpi,
                    model_name=mn,
                ),
            ))

    if not jobs:
        raise ValueError(
            f"No datasets found under {checkpoints_root} (looked for */{args.metrics_subpath} and "
            f"{args.pretrained_subdir}/*/*/{args.pretrained_predictions_name}). "
            "Pass --datasets/--pretrained-datasets explicitly if your layout differs."
        )

    print(f"Generating evaluation dashboards for {len(jobs)} dataset run(s): {', '.join(name for name, _ in jobs)}")

    failures: list[tuple[str, Exception]] = []
    for name, job in jobs:
        print(f"\n=== {name} ===")
        try:
            job()
        except Exception as exc:  # keep going so one bad dataset doesn't block the rest
            print(f"  FAILED: {exc}")
            failures.append((name, exc))

    if failures:
        print(f"\n{len(failures)} of {len(jobs)} dataset run(s) failed:")
        for name, exc in failures:
            print(f"  - {name}: {exc}")


def run_dashboard(
    label: str,
    metrics_path: Path,
    predictions_path: Path,
    summary_path: Path,
    classnames_path: Path,
    output_dir: Path,
    seed: int,
    width: int,
    height: int,
    dpi: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = read_metrics(metrics_path)
    metrics = latest_run(all_metrics)
    predictions = read_predictions(predictions_path)
    summary = read_summary(summary_path)
    classnames = read_classnames(classnames_path)

    overview_path = output_dir / "final_model_evaluation_dashboard_overview.png"
    curves_path = output_dir / "final_model_evaluation_training_curves.png"
    grid_path = output_dir / "final_model_evaluation_zero_shot_grid.png"

    render_overview(overview_path, width, height, metrics, predictions, summary, classnames, label)
    render_curves(curves_path, width, height, dpi, metrics, label)
    render_classification_grid(
        grid_path,
        width,
        height,
        predictions,
        summary,
        classnames,
        rounds=len(metrics),
        seed=seed,
        dataset_label=label,
    )

    print(f"  {overview_path.resolve()}")
    print(f"  {curves_path.resolve()}")
    print(f"  {grid_path.resolve()}")


def run_pretrained_dashboard(
    label: str,
    predictions_path: Path,
    summary_path: Path,
    classnames_path: Path,
    output_dir: Path,
    seed: int,
    width: int,
    height: int,
    dpi: int,
    model_name: str,
) -> None:
    """Same two-page dashboard as run_dashboard, minus the training-curves page, since
    pretrained zero-shot datasets have no metrics.csv (no local training happened)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_predictions(predictions_path)
    summary = read_summary(summary_path)
    classnames = classnames_or_infer(classnames_path, predictions)

    overview_path = output_dir / "final_model_evaluation_dashboard_overview.png"
    grid_path = output_dir / "final_model_evaluation_zero_shot_grid.png"

    render_zero_shot_overview(overview_path, width, height, predictions, summary, classnames, label, model_name)
    render_classification_grid(
        grid_path,
        width,
        height,
        predictions,
        summary,
        classnames,
        rounds=0,
        seed=seed,
        dataset_label=label,
        model_name=model_name,
        federated_learning="Not applicable (zero-shot)",
        prompt_learning="Not applicable (zero-shot)",
        rounds_label="N/A",
    )

    print(f"  {overview_path.resolve()}")
    print(f"  {grid_path.resolve()}")
    print("  (no training curves page — pretrained zero-shot dataset has no metrics.csv)")


def read_metrics(path: Path) -> list[MetricRow]:
    rows: list[MetricRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("epoch"):
                continue
            rows.append(
                MetricRow(
                    epoch=int(float(row["epoch"])),
                    global_step=int(float(row["global_step"])),
                    train_loss=float(row["train_loss"]),
                    train_image_to_text_top1=float(row["train_image_to_text_top1"]),
                    train_text_to_image_top1=float(row["train_text_to_image_top1"]),
                    val_zero_shot_top1=float(row["val_zero_shot_top1"] or 0),
                    val_zero_shot_top5=float(row["val_zero_shot_top5"] or 0),
                )
            )
    if not rows:
        raise ValueError(f"No metric rows found in {path}")
    return rows


def latest_run(rows: list[MetricRow]) -> list[MetricRow]:
    runs: list[list[MetricRow]] = []
    current: list[MetricRow] = []
    previous_epoch: int | None = None
    for row in rows:
        if previous_epoch is not None and row.epoch <= previous_epoch and current:
            runs.append(current)
            current = []
        current.append(row)
        previous_epoch = row.epoch
    if current:
        runs.append(current)
    return runs[-1]


def read_predictions(path: Path) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_value = row.get("image_path") or row.get("input") or row.get("path")
            if not image_value:
                continue
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = ROOT / image_path
            rows.append(
                PredictionRow(
                    image_path=image_path,
                    true_class=row.get("true_class", ""),
                    predicted_class=row.get("predicted_class", ""),
                    is_correct=str(row.get("is_correct", "")).strip().lower() == "true",
                )
            )
    if not rows:
        raise ValueError(f"No prediction rows found in {path}")
    return rows


def read_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def read_classnames(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canvas(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), BG)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_header(draw: ImageDraw.ImageDraw, width: int, eyebrow: str | None = None) -> None:
    draw.text((150, 105), "Final Model Evaluation Dashboard", fill=INK, font=font(78, True))
    draw.text((154, 205), "CLIP-based Federated Remote Sensing Scene Classification", fill=MUTED, font=font(36))
    if eyebrow:
        eyebrow_font = font(28, True)
        text_w = draw.textlength(eyebrow, font=eyebrow_font)
        pad_x = 40
        box_w = min(width - 300, text_w + pad_x * 2)
        x1 = width - 150
        x0 = x1 - box_w
        draw.rounded_rectangle((x0, 120, x1, 190), radius=34, fill=BLUE_LIGHT)
        draw.text((x0 + pad_x, 138), eyebrow, fill=BLUE_DARK, font=eyebrow_font)
    draw.line((150, 285, width - 150, 285), fill=GRID, width=4)


def card(base: Image.Image, box: tuple[int, int, int, int], radius: int = 28, fill: str = CARD, outline: str = BORDER) -> ImageDraw.ImageDraw:
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    shadow = ImageDraw.Draw(layer)
    sx0, sy0, sx1, sy1 = box[0] + 10, box[1] + 14, box[2] + 10, box[3] + 14
    shadow.rounded_rectangle((sx0, sy0, sx1, sy1), radius=radius, fill=(15, 23, 42, 26))
    layer = layer.filter(ImageFilter.GaussianBlur(14))
    base.paste(Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB"))
    draw = ImageDraw.Draw(base)
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)
    return draw


def pct(value: float) -> str:
    return f"{value:.2f}%"


def num(value: float) -> str:
    return f"{value:.4f}"


def display_class(name: str) -> str:
    cleaned = name.strip()
    for prefix in ("a ", "an "):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    if cleaned == "airplanes":
        cleaned = "airplane"
    return cleaned.title()


def best_model_row(metrics: list[MetricRow]) -> MetricRow:
    return max(metrics, key=lambda row: (row.val_zero_shot_top1, row.val_zero_shot_top5, -row.train_loss))


def render_overview(
    path: Path,
    width: int,
    height: int,
    metrics: list[MetricRow],
    predictions: list[PredictionRow],
    summary: dict[str, str],
    classnames: list[str],
    dataset_label: str = "Dataset",
) -> None:
    img = canvas(width, height)
    draw = ImageDraw.Draw(img)
    draw_header(draw, width, f"{dataset_label} · Evaluation Overview")

    final = metrics[-1]
    best = best_model_row(metrics)
    lowest_loss = min(metrics, key=lambda row: row.train_loss)
    best_top5 = max(metrics, key=lambda row: row.val_zero_shot_top5)

    draw_best_card(img, (150, 380, width - 150, 760), best, lowest_loss, best_top5)

    metric_cards = [
        ("Final Epoch", str(final.epoch), "Last completed training epoch", BLUE, "calendar"),
        ("Final Train Loss", num(final.train_loss), "Contrastive training objective", AMBER, "loss"),
        ("Train Image-to-Text Top-1", pct(final.train_image_to_text_top1), "Image retrieval against text prompts", CYAN, "image"),
        ("Train Text-to-Image Top-1", pct(final.train_text_to_image_top1), "Text retrieval against image embeddings", VIOLET, "text"),
        ("Validation Zero-Shot Top-1", pct(final.val_zero_shot_top1), "Best class at rank 1", GREEN, "target"),
        ("Validation Zero-Shot Top-5", pct(final.val_zero_shot_top5), "Ground truth inside top 5", BLUE_DARK, "stack"),
    ]
    grid_y = 895
    gap = 44
    card_w = (width - 300 - 2 * gap) // 3
    card_h = 360
    for index, item in enumerate(metric_cards):
        row = index // 3
        col = index % 3
        x0 = 150 + col * (card_w + gap)
        y0 = grid_y + row * (card_h + 44)
        draw_metric_card(img, (x0, y0, x0 + card_w, y0 + card_h), *item)

    summary_top = 1780
    draw.text((150, summary_top), "Bottom Summary", fill=INK, font=font(48, True))
    draw.text((150, summary_top + 64), "Full zero-shot test-set summary and experiment configuration", fill=MUTED, font=font(28))

    stats = bottom_summary_items(predictions, summary, classnames, len(metrics), dataset_label)
    draw_summary_grid(img, (150, summary_top + 140, width - 150, height - 160), stats, columns=4)
    img.save(path, quality=95)


def draw_best_card(img: Image.Image, box: tuple[int, int, int, int], best: MetricRow, lowest_loss: MetricRow, best_top5: MetricRow) -> None:
    draw = card(img, box, radius=36)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0 + 34, y0 + 42, x0 + 120, y0 + 128), radius=24, fill=BLUE_LIGHT)
    draw_icon(draw, "target", x0 + 55, y0 + 62, BLUE)
    draw.text((x0 + 150, y0 + 46), "Best Model", fill=INK, font=font(42, True))
    draw.text((x0 + 150, y0 + 102), "Selected by highest validation zero-shot Top-1 accuracy", fill=MUTED, font=font(25))
    items = [
        ("Best Epoch", str(best.epoch)),
        ("Global Step", str(best.global_step)),
        ("Lowest Training Loss", f"{lowest_loss.train_loss:.4f} (epoch {lowest_loss.epoch})"),
        ("Best Validation Zero-Shot Top-1 Accuracy", pct(best.val_zero_shot_top1)),
        ("Best Validation Zero-Shot Top-5 Accuracy", f"{best_top5.val_zero_shot_top5:.2f}% (epoch {best_top5.epoch})"),
    ]
    usable_w = x1 - x0 - 80
    cell_w = usable_w // len(items)
    for index, (label, value) in enumerate(items):
        cx = x0 + 42 + index * cell_w
        draw.text((cx, y0 + 210), value, fill=BLUE_DARK if index in (0, 1, 3, 4) else INK, font=font(32, True))
        wrapped = textwrap.wrap(label, width=24)
        for offset, line in enumerate(wrapped[:2]):
            draw.text((cx, y0 + 260 + offset * 30), line, fill=MUTED, font=font(22))


def render_zero_shot_overview(
    path: Path,
    width: int,
    height: int,
    predictions: list[PredictionRow],
    summary: dict[str, str],
    classnames: list[str],
    dataset_label: str,
    model_name: str,
) -> None:
    """Overview page for pretrained zero-shot datasets: no training happened, so there's
    no epoch/loss/best-epoch story here — just the off-the-shelf zero-shot result."""
    img = canvas(width, height)
    draw = ImageDraw.Draw(img)
    draw_header(draw, width, f"{dataset_label} · Zero-Shot Evaluation")

    total = int(float(summary.get("samples", "0") or "0")) or len(predictions)
    correct = sum(row.is_correct for row in predictions)
    wrong = max(0, total - correct)
    top1 = float(summary.get("top1", "0") or 0)
    top5 = float(summary.get("top5", "0") or 0)

    draw_zero_shot_card(img, (150, 380, width - 150, 760), model_name, top1, top5, total)

    metric_cards = [
        ("Top-1 Accuracy", pct(top1), "Best class at rank 1", GREEN, "target"),
        ("Top-5 Accuracy", pct(top5), "Ground truth inside top 5", BLUE_DARK, "stack"),
        ("Number of Test Images", f"{total:,}", "Zero-shot evaluation set size", CYAN, "image"),
        ("Correct Predictions", f"{correct:,}", "Matched the ground-truth class", BLUE, "calendar"),
        ("Wrong Predictions", f"{wrong:,}", "Misclassified by the pretrained model", RED, "loss"),
        ("Number of Classes", str(len(classnames)), "Zero-shot label set size", VIOLET, "text"),
    ]
    grid_y = 895
    gap = 44
    card_w = (width - 300 - 2 * gap) // 3
    card_h = 360
    for index, item in enumerate(metric_cards):
        row = index // 3
        col = index % 3
        x0 = 150 + col * (card_w + gap)
        y0 = grid_y + row * (card_h + 44)
        draw_metric_card(img, (x0, y0, x0 + card_w, y0 + card_h), *item)

    summary_top = 1780
    draw.text((150, summary_top), "Bottom Summary", fill=INK, font=font(48, True))
    draw.text((150, summary_top + 64), "Zero-shot test-set summary and evaluation configuration", fill=MUTED, font=font(28))

    stats = bottom_summary_items(
        predictions, summary, classnames, rounds=0, dataset_label=dataset_label,
        model_name=model_name, federated_learning="Not applicable (zero-shot)",
        prompt_learning="Not applicable (zero-shot)", rounds_label="N/A",
    )
    draw_summary_grid(img, (150, summary_top + 140, width - 150, height - 160), stats, columns=4)
    img.save(path, quality=95)


def draw_zero_shot_card(img: Image.Image, box: tuple[int, int, int, int], model_name: str, top1: float, top5: float, total: int) -> None:
    draw = card(img, box, radius=36)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0 + 34, y0 + 42, x0 + 120, y0 + 128), radius=24, fill=BLUE_LIGHT)
    draw_icon(draw, "target", x0 + 55, y0 + 62, BLUE)
    draw.text((x0 + 150, y0 + 46), "Zero-Shot Result", fill=INK, font=font(42, True))
    draw.text((x0 + 150, y0 + 102), "Off-the-shelf pretrained inference, no fine-tuning on this dataset", fill=MUTED, font=font(25))
    items = [
        ("Model", model_name),
        ("Top-1 Accuracy", pct(top1)),
        ("Top-5 Accuracy", pct(top5)),
        ("Test Images", f"{total:,}"),
    ]
    usable_w = x1 - x0 - 80
    cell_w = usable_w // len(items)
    for index, (label, value) in enumerate(items):
        cx = x0 + 42 + index * cell_w
        draw.text((cx, y0 + 210), value, fill=BLUE_DARK, font=font(32, True))
        wrapped = textwrap.wrap(label, width=24)
        for offset, line in enumerate(wrapped[:2]):
            draw.text((cx, y0 + 260 + offset * 30), line, fill=MUTED, font=font(22))


def draw_metric_card(
    img: Image.Image,
    box: tuple[int, int, int, int],
    title: str,
    value: str,
    description: str,
    color: str,
    icon_name: str,
) -> None:
    draw = card(img, box)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0 + 32, y0 + 34, x0 + 100, y0 + 102), radius=20, fill=soft_color(color))
    draw_icon(draw, icon_name, x0 + 50, y0 + 52, color)
    draw.ellipse((x1 - 78, y0 + 50, x1 - 42, y0 + 86), fill=color)
    draw.text((x0 + 34, y0 + 135), value, fill=INK, font=font(50, True))
    draw.text((x0 + 36, y0 + 215), title, fill=INK, font=font(25, True))
    for offset, line in enumerate(textwrap.wrap(description, width=42)[:2]):
        draw.text((x0 + 36, y0 + 260 + offset * 31), line, fill=MUTED, font=font(22))


def draw_icon(draw: ImageDraw.ImageDraw, name: str, x: int, y: int, color: str) -> None:
    if name == "calendar":
        draw.rounded_rectangle((x, y + 4, x + 34, y + 38), radius=5, outline=color, width=4)
        draw.line((x, y + 15, x + 34, y + 15), fill=color, width=4)
        draw.line((x + 9, y, x + 9, y + 10), fill=color, width=4)
        draw.line((x + 25, y, x + 25, y + 10), fill=color, width=4)
    elif name == "loss":
        pts = [(x, y + 36), (x + 9, y + 25), (x + 18, y + 28), (x + 28, y + 12), (x + 38, y + 18)]
        draw.line(pts, fill=color, width=5, joint="curve")
        draw.ellipse((x + 24, y + 8, x + 32, y + 16), fill=color)
    elif name == "image":
        draw.rounded_rectangle((x, y + 4, x + 38, y + 34), radius=5, outline=color, width=4)
        draw.ellipse((x + 25, y + 10, x + 31, y + 16), fill=color)
        draw.line((x + 4, y + 30, x + 15, y + 20, x + 23, y + 27, x + 34, y + 16), fill=color, width=4)
    elif name == "text":
        for yy, w in [(4, 38), (16, 30), (28, 36)]:
            draw.line((x, y + yy, x + w, y + yy), fill=color, width=5)
    elif name == "target":
        draw.ellipse((x, y, x + 40, y + 40), outline=color, width=4)
        draw.ellipse((x + 10, y + 10, x + 30, y + 30), outline=color, width=4)
        draw.ellipse((x + 18, y + 18, x + 22, y + 22), fill=color)
    else:
        for i in range(3):
            draw.rounded_rectangle((x + i * 7, y + i * 8, x + 32 + i * 7, y + 20 + i * 8), radius=4, outline=color, width=3)


def soft_color(color: str) -> str:
    return {
        BLUE: BLUE_LIGHT,
        BLUE_DARK: BLUE_LIGHT,
        GREEN: GREEN_SOFT,
        RED: RED_SOFT,
        AMBER: "#fef3c7",
        VIOLET: "#ede9fe",
        CYAN: "#cffafe",
    }.get(color, BLUE_LIGHT)


def bottom_summary_items(
    predictions: list[PredictionRow],
    summary: dict[str, str],
    classnames: list[str],
    rounds: int,
    dataset_label: str = "Dataset",
    model_name: str = "OpenAI CLIP ViT-B/32",
    federated_learning: str = "FedAvg",
    prompt_learning: str = "Enabled",
    rounds_label: str | None = None,
) -> list[tuple[str, str]]:
    total = int(float(summary.get("samples", "0") or "0")) or len(predictions)
    correct = sum(row.is_correct for row in predictions)
    wrong = max(0, total - correct)
    top1 = float(summary.get("top1", "0") or 0)
    top5 = float(summary.get("top5", "0") or 0)
    return [
        ("Overall Accuracy", pct(top1)),
        ("Top-1 Accuracy", pct(top1)),
        ("Top-5 Accuracy", pct(top5)),
        ("Number of Test Images", f"{total:,}"),
        ("Number of Correct Predictions", f"{correct:,}"),
        ("Number of Wrong Predictions", f"{wrong:,}"),
        ("Inference Time per Image", "Not logged"),
        ("Model", model_name),
        ("Federated Learning", federated_learning),
        ("Prompt Learning", prompt_learning),
        ("Dataset", dataset_label),
        ("Number of Classes", str(len(classnames))),
        ("Communication Rounds", rounds_label if rounds_label is not None else str(rounds)),
    ]


def draw_summary_grid(img: Image.Image, box: tuple[int, int, int, int], items: list[tuple[str, str]], columns: int) -> None:
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    gap = 28
    rows = math.ceil(len(items) / columns)
    cell_w = (x1 - x0 - gap * (columns - 1)) // columns
    cell_h = (y1 - y0 - gap * (rows - 1)) // rows
    for index, (label, value) in enumerate(items):
        row = index // columns
        col = index % columns
        cx = x0 + col * (cell_w + gap)
        cy = y0 + row * (cell_h + gap)
        draw.rounded_rectangle((cx, cy, cx + cell_w, cy + cell_h), radius=22, fill="#f8fbff", outline=BORDER, width=2)
        draw.text((cx + 24, cy + 26), value, fill=INK, font=font(31, True))
        draw.text((cx + 24, cy + 78), label, fill=MUTED, font=font(21))
        draw.rectangle((cx, cy + cell_h - 8, cx + cell_w, cy + cell_h), fill=BLUE if index % 3 != 1 else CYAN)


def render_curves(path: Path, width: int, height: int, _dpi: int, metrics: list[MetricRow], dataset_label: str = "Dataset") -> None:
    img = canvas(width, height)
    draw = ImageDraw.Draw(img)
    draw_header(draw, width, f"{dataset_label} · Training Curves")
    draw.text((150, 335), "Training Curves", fill=INK, font=font(46, True))
    draw.text((150, 395), "Five smooth epoch-wise performance traces from metrics.csv", fill=MUTED, font=font(26))

    boxes = [
        (150, 500, 1930, 1460),
        (2070, 500, 3850, 1460),
        (150, 1640, 1265, 2920),
        (1442, 1640, 2558, 2920),
        (2735, 1640, 3850, 2920),
    ]
    series = [
        ("Train Loss vs Epoch", "Train Loss", [row.train_loss for row in metrics], "Loss", BLUE),
        ("Train Image-to-Text Top-1 vs Epoch", "Image-to-Text Top-1", [row.train_image_to_text_top1 for row in metrics], "Accuracy (%)", CYAN),
        ("Train Text-to-Image Top-1 vs Epoch", "Text-to-Image Top-1", [row.train_text_to_image_top1 for row in metrics], "Accuracy (%)", VIOLET),
        ("Validation Zero-Shot Top-1 vs Epoch", "Zero-Shot Top-1", [row.val_zero_shot_top1 for row in metrics], "Accuracy (%)", GREEN),
        ("Validation Zero-Shot Top-5 vs Epoch", "Zero-Shot Top-5", [row.val_zero_shot_top5 for row in metrics], "Accuracy (%)", BLUE_DARK),
    ]
    epochs = [row.epoch for row in metrics]
    for box, item in zip(boxes, series):
        draw_line_chart(img, box, epochs, item)
    img.save(path, quality=95)


def draw_line_chart(
    img: Image.Image,
    box: tuple[int, int, int, int],
    epochs: list[int],
    series: tuple[str, str, list[float], str, str],
) -> None:
    title, label, values, ylabel, color = series
    draw = card(img, box, radius=28)
    x0, y0, x1, y1 = box
    draw.text((x0 + 42, y0 + 34), title, fill=INK, font=font(28, True))
    draw.line((x0 + 44, y0 + 88, x0 + 104, y0 + 88), fill=color, width=8)
    draw.text((x0 + 122, y0 + 74), label, fill=MUTED, font=font(22))

    px0, py0 = x0 + 110, y0 + 150
    px1, py1 = x1 - 70, y1 - 125
    draw.rectangle((px0, py0, px1, py1), outline=BORDER, width=2)

    y_min = min(values)
    y_max = max(values)
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    pad = (y_max - y_min) * 0.16
    y_min -= pad
    y_max += pad

    x_min = min(epochs)
    x_max = max(epochs)
    if x_min == x_max:
        x_min -= 1
        x_max += 1

    for i in range(6):
        yy = py0 + i * (py1 - py0) / 5
        value = y_max - i * (y_max - y_min) / 5
        draw.line((px0, int(yy), px1, int(yy)), fill=GRID, width=2)
        draw.text((x0 + 30, int(yy) - 13), f"{value:.1f}", fill=MUTED, font=font(18))
    for epoch in epochs:
        xx = map_value(epoch, x_min, x_max, px0, px1)
        draw.line((xx, py0, xx, py1), fill="#f1f5fb", width=2)
        draw.text((xx - 10, py1 + 18), str(epoch), fill=MUTED, font=font(19))

    points = [
        (map_value(epoch, x_min, x_max, px0, px1), map_value(value, y_min, y_max, py1, py0))
        for epoch, value in zip(epochs, values)
    ]
    smooth_points = catmull_rom(points, samples=28)
    if len(smooth_points) >= 2:
        draw.line(smooth_points, fill=color, width=8, joint="curve")
    for px, py in points:
        draw.ellipse((px - 13, py - 13, px + 13, py + 13), fill="white", outline=color, width=6)

    best_index = int(np.argmin(values) if "Loss" in title else np.argmax(values))
    bx, by = points[best_index]
    draw.ellipse((bx - 24, by - 24, bx + 24, by + 24), fill=soft_color(color), outline=INK, width=4)
    draw.ellipse((bx - 9, by - 9, bx + 9, by + 9), fill=color)
    draw.text((px0 + (px1 - px0) // 2 - 58, y1 - 66), "Epoch", fill=MUTED, font=font(22, True))
    draw.text((x0 + 28, py0 - 45), "Metric", fill=MUTED, font=font(21, True))
    draw.text((px1 - 245, py0 - 45), ylabel, fill=MUTED, font=font(21))


def map_value(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> int:
    ratio = (value - src_min) / (src_max - src_min)
    return int(round(dst_min + ratio * (dst_max - dst_min)))


def catmull_rom(points: list[tuple[int, int]], samples: int = 24) -> list[tuple[int, int]]:
    if len(points) < 3:
        return points
    extended = [points[0], *points, points[-1]]
    output: list[tuple[int, int]] = []
    for index in range(1, len(extended) - 2):
        p0 = np.array(extended[index - 1], dtype=float)
        p1 = np.array(extended[index], dtype=float)
        p2 = np.array(extended[index + 1], dtype=float)
        p3 = np.array(extended[index + 2], dtype=float)
        for step in range(samples):
            t = step / samples
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            output.append((int(round(point[0])), int(round(point[1]))))
    output.append(points[-1])
    return output


def render_classification_grid(
    path: Path,
    width: int,
    height: int,
    predictions: list[PredictionRow],
    summary: dict[str, str],
    classnames: list[str],
    rounds: int,
    seed: int,
    dataset_label: str = "Dataset",
    model_name: str = "OpenAI CLIP ViT-B/32",
    federated_learning: str = "FedAvg",
    prompt_learning: str = "Enabled",
    rounds_label: str | None = None,
) -> None:
    img = canvas(width, height)
    draw = ImageDraw.Draw(img)
    draw_header(draw, width, f"{dataset_label} · Zero-Shot Classification")
    draw.text((150, 335), "Zero-Shot Classification Visualization", fill=INK, font=font(46, True))
    draw.text((150, 395), f"Five randomly selected {dataset_label} classes with five test images per class", fill=MUTED, font=font(26))

    samples = select_grid_samples(predictions, seed)
    classes = list(samples)
    table_x = 150
    table_y = 495
    table_w = width - 300
    class_w = 360
    gap = 24
    img_w = (table_w - class_w - gap * 5) // 5
    row_h = 365
    thumb_h = 240

    draw.text((table_x + 18, table_y - 55), "Class", fill=MUTED, font=font(24, True))
    for i in range(5):
        x = table_x + class_w + gap + i * (img_w + gap)
        draw.text((x + img_w // 2 - 48, table_y - 55), f"Image {i + 1}", fill=MUTED, font=font(24, True))

    for row_index, class_name in enumerate(classes):
        y = table_y + row_index * row_h
        draw.rounded_rectangle((table_x, y, table_x + table_w, y + row_h - 22), radius=26, fill="#fbfdff", outline=BORDER, width=2)
        draw.text((table_x + 30, y + 135), display_class(class_name), fill=INK, font=font(31, True))
        draw.text((table_x + 30, y + 184), "Ground Truth Class", fill=MUTED, font=font(20))
        for col_index, row in enumerate(samples[class_name]):
            x = table_x + class_w + gap + col_index * (img_w + gap)
            render_prediction_tile(img, (x, y + 28, x + img_w, y + row_h - 52), row, thumb_h)

    footer_y = table_y + len(classes) * row_h + 35
    summary_items = bottom_summary_items(
        predictions, summary, classnames, rounds, dataset_label,
        model_name=model_name, federated_learning=federated_learning,
        prompt_learning=prompt_learning, rounds_label=rounds_label,
    )
    draw.text((150, footer_y), "Evaluation Summary", fill=INK, font=font(40, True))
    draw_summary_grid(img, (150, footer_y + 70, width - 150, height - 140), summary_items, columns=4)
    img.save(path, quality=95)


def select_grid_samples(predictions: list[PredictionRow], seed: int) -> dict[str, list[PredictionRow]]:
    rng = random.Random(seed)
    grouped: dict[str, list[PredictionRow]] = {}
    for row in predictions:
        if row.image_path.exists():
            grouped.setdefault(row.true_class, []).append(row)
    eligible = [name for name, rows in grouped.items() if len(rows) >= 5]
    preferred = ["airplanes", "a forest", "a harbor", "a parking lot", "a runway"]
    if all(name in eligible for name in preferred):
        selected = preferred
    else:
        selected = rng.sample(sorted(eligible), k=min(5, len(eligible)))
    result: dict[str, list[PredictionRow]] = {}
    for name in selected[:5]:
        rows = grouped[name][:]
        correct = [row for row in rows if row.is_correct]
        wrong = [row for row in rows if not row.is_correct]
        rng.shuffle(correct)
        rng.shuffle(wrong)
        mixed = (correct[:2] + wrong[:5])[:5]
        if len(mixed) < 5:
            remaining = [row for row in rows if row not in mixed]
            rng.shuffle(remaining)
            mixed.extend(remaining[: 5 - len(mixed)])
        rng.shuffle(mixed)
        result[name] = mixed[:5]
    return result


def render_prediction_tile(img: Image.Image, box: tuple[int, int, int, int], row: PredictionRow, thumb_h: int) -> None:
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    color = GREEN if row.is_correct else RED
    side = min(thumb_h, x1 - x0 - 28)
    thumb_x0 = x0 + (x1 - x0 - side) // 2
    thumb_box = (thumb_x0, y0 + 14, thumb_x0 + side, y0 + 14 + side)
    try:
        thumb = load_fit(row.image_path, (thumb_box[2] - thumb_box[0], thumb_box[3] - thumb_box[1]))
    except Exception:
        thumb = Image.new("RGB", (thumb_box[2] - thumb_box[0], thumb_box[3] - thumb_box[1]), "#eef2f7")
    img.paste(thumb, thumb_box)
    draw.rounded_rectangle(thumb_box, radius=14, outline=color, width=4)
    gt_text = f"GT: {display_class(row.true_class)}"
    pred_text = f"Pred: {display_class(row.predicted_class)}"
    text_x = max(x0 + 20, thumb_box[0] - 8)
    draw.text((text_x, y0 + side + 38), gt_text, fill=INK, font=font(20, True))
    for offset, line in enumerate(textwrap.wrap(pred_text, width=24)[:2]):
        draw.text((text_x, y0 + side + 72 + offset * 26), line, fill=color, font=font(20, True))


def load_fit(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    target_w, target_h = size
    src_w, src_h = image.size
    scale = min(target_w / src_w, target_h / src_h)
    resized = image.resize((math.floor(src_w * scale), math.floor(src_h * scale)), Image.Resampling.LANCZOS)
    fitted = Image.new("RGB", (target_w, target_h), "#f8fafc")
    left = (target_w - resized.width) // 2
    top = (target_h - resized.height) // 2
    fitted.paste(resized, (left, top))
    return fitted


if __name__ == "__main__":  
    main()