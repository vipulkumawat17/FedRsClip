from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clip_implement.data import CLIP_MEAN, CLIP_STD, read_classnames
from clip_implement.model import build_model
from clip_implement.tokenizer import ByteTokenizer


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_TEMPLATES = ["a satellite photo of {}.", "an aerial image of {}.", "a remote sensing image of {}."]

INK = (30, 37, 48)
MUTED = (90, 99, 112)
LIGHT = (241, 245, 249)
BORDER = (203, 213, 225)
GREEN = (22, 163, 74)
BLUE = (37, 99, 235)
RED = (220, 38, 38)
GOLD = (202, 138, 4)


@dataclass
class SampleRow:
    image_path: Path
    class_name: str
    caption: str


@dataclass
class ForwardOutputs:
    image_features: torch.Tensor
    text_features: torch.Tensor
    activations: dict[str, torch.Tensor]
    tokens: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-quality CLIP research figures for one or more datasets.")

    # Batch mode (default): discover datasets under a checkpoints root.
    parser.add_argument("--checkpoints-root", default="checkpoints", help="Root folder holding <dataset>/clip_epoch_*.pt and pretrained_zero_shot/<dataset>/<variant>/")
    parser.add_argument("--data-root", default="data", help="Root folder containing <dataset>/classnames.txt")
    parser.add_argument("--datasets", nargs="+", default=None, help="Locally fine-tuned dataset keys (checkpoints/<name>/clip_epoch_*.pt). Default: auto-discover.")
    parser.add_argument("--skip-trained", action="store_true", help="Skip locally fine-tuned datasets entirely")
    parser.add_argument("--pretrained-subdir", default="pretrained_zero_shot", help="Subfolder of --checkpoints-root holding zero-shot-only datasets")
    parser.add_argument("--pretrained-datasets", nargs="+", default=None, help="Pretrained zero-shot dataset keys. Default: auto-discover.")
    parser.add_argument("--skip-pretrained", action="store_true", help="Skip OpenAI-pretrained zero-shot datasets entirely")
    parser.add_argument("--pretrained-model", default="ViT-B-32", help="open_clip model name for pretrained zero-shot datasets")
    parser.add_argument("--pretrained-tag", default="openai", help="open_clip pretrained tag, matches the *_openai folder naming")
    parser.add_argument("--checkpoint-select", choices=["best", "latest"], default="best", help="Which clip_epoch_*.pt to use for locally fine-tuned datasets")
    parser.add_argument("--metrics-subpath", default="reports/metrics.csv")
    parser.add_argument("--predictions-name", default="predictions.csv", help="Filename read from zero_shot_reports/ (trained) or the variant folder (pretrained)")
    parser.add_argument("--zero-shot-subdir", default="zero_shot_reports", help="Subfolder holding predictions.csv for locally fine-tuned datasets")
    parser.add_argument("--classnames-name", default="classnames.txt")
    parser.add_argument("--output-subdir", default="research_figures", help="Subfolder created for the generated figures, matching the uc_merced layout")
    parser.add_argument("--sample-index", type=int, default=0, help="Which predictions.csv row (0-based, among rows whose image file exists) to explain in Figures 1-4/7/8")

    # Shared figure-generation options.
    parser.add_argument("--template", action="append", default=None, help="Prompt template. Use {} for class name. Repeatable.")
    parser.add_argument("--context-length", type=int, default=76, help="Only used for the locally fine-tuned ByteTokenizer model")
    parser.add_argument("--max-matrix-images", type=int, default=12, help="Rows in Figure 5")
    parser.add_argument("--max-embedding-images", type=int, default=48, help="Images in Figure 6")
    parser.add_argument("--max-matrix-classes", type=int, default=24, help="Columns in Figure 5")
    parser.add_argument("--device", default=None, help="cuda, cpu, etc. Defaults to cuda when available")
    parser.add_argument("--dpi", type=int, default=300)

    # Single-run overrides: set any of these to fall back to the original one-checkpoint behaviour.
    parser.add_argument("--checkpoint", default=None, help="Path to a trained CLIP checkpoint (single-run mode)")
    parser.add_argument("--image", default=None, help="Specific image to explain. Defaults to the first --val-jsonl row.")
    parser.add_argument("--ground-truth", default=None, help="Ground-truth class for --image")
    parser.add_argument("--text", default=None, help="Text prompt to inspect. Defaults to the ground-truth template.")
    parser.add_argument("--val-jsonl", default=None, help="Validation/test JSONL with image and class_name/label fields")
    parser.add_argument("--image-root", default=".", help="Root directory for relative image paths")
    parser.add_argument("--classnames", default=None, help="classnames.txt (single-run mode)")
    parser.add_argument("--output-dir", default=None, help="Output directory (single-run mode)")
    parser.add_argument("--model", default=None, choices=["RN50", "RN101", "ViT-B-32", "ViT-B-16"], help="Defaults to checkpoint args (single-run mode)")
    parser.add_argument("--dataset-label", default=None, help="Display name used in figure titles")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Backwards-compatible single-run mode: explicit --checkpoint/--val-jsonl/--image.
    if args.checkpoint or args.val_jsonl or args.image:
        run_single_checkpoint(args, device)
        return

    checkpoints_root = Path(args.checkpoints_root)
    data_root = Path(args.data_root)

    jobs: list[tuple[str, "callable"]] = []

    if not args.skip_trained:
        trained = args.datasets or discover_trained_datasets(checkpoints_root)
        for name in trained:
            jobs.append((name, lambda n=name: process_trained_dataset(n, checkpoints_root, data_root, args, device)))

    if not args.skip_pretrained:
        pretrained_root = checkpoints_root / args.pretrained_subdir
        pretrained = args.pretrained_datasets or discover_pretrained_datasets(pretrained_root, args.predictions_name)
        for name, variant in pretrained:
            jobs.append((f"{name} ({variant})", lambda n=name, v=variant: process_pretrained_dataset(n, v, pretrained_root, data_root, args, device)))

    if not jobs:
        raise ValueError(
            f"No datasets found under {checkpoints_root} (looked for */clip_epoch_*.pt and "
            f"{args.pretrained_subdir}/*/*/{args.predictions_name}). Pass --datasets/--pretrained-datasets explicitly if your layout differs."
        )

    print(f"Generating research figures for {len(jobs)} dataset run(s): {', '.join(name for name, _ in jobs)}")
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


def discover_trained_datasets(checkpoints_root: Path) -> list[str]:
    if not checkpoints_root.exists():
        return []
    found = []
    for child in sorted(checkpoints_root.iterdir()):
        if child.is_dir() and list(child.glob("clip_epoch_*.pt")):
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


def dataset_label(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip().title()


def extract_epoch(path: Path) -> int:
    match = re.search(r"clip_epoch_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def best_epoch_from_metrics(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scored = [row for row in rows if row.get("epoch") and row.get("val_zero_shot_top1")]
    if not scored:
        return None
    best = max(scored, key=lambda row: float(row["val_zero_shot_top1"] or 0))
    return int(float(best["epoch"]))


def find_checkpoint(dataset_dir: Path, select: str, metrics_path: Path) -> Path | None:
    candidates = sorted(dataset_dir.glob("clip_epoch_*.pt"), key=extract_epoch)
    if not candidates:
        return None
    if select == "best":
        best_epoch = best_epoch_from_metrics(metrics_path)
        if best_epoch is not None:
            for candidate in candidates:
                if extract_epoch(candidate) == best_epoch:
                    return candidate
    return candidates[-1]


def default_templates_for(name: str, cli_templates: list[str] | None) -> list[str]:
    if cli_templates:
        return cli_templates
    remote_sensing = {"uc_merced", "resisc45", "eurosat", "aid", "whu_rs19", "pattern_net"}
    if name.lower() in remote_sensing:
        return DEFAULT_TEMPLATES
    return ["a photo of a {}.", "a photo of the {}.", "an image of a {}."]


def build_rows_from_predictions(predictions_path: Path, image_root: Path) -> list[SampleRow]:
    rows: list[SampleRow] = []
    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_value = row.get("image_path") or row.get("input") or row.get("path")
            if not image_value:
                continue
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = image_root / image_path
            class_name = str(row.get("true_class") or row.get("class_name") or row.get("label") or "").strip()
            rows.append(SampleRow(image_path=image_path, class_name=class_name, caption=""))
    if not rows:
        raise ValueError(f"No prediction rows found in {predictions_path}")
    return rows


def classnames_or_infer(classnames_path: Path, rows: list[SampleRow]) -> list[str]:
    names = read_classnames(classnames_path)
    if names:
        return names
    return sorted({row.class_name for row in rows if row.class_name})


def pick_sample(rows: list[SampleRow], sample_index: int) -> SampleRow:
    existing = [row for row in rows if row.image_path.exists()]
    pool = existing or rows
    if not pool:
        raise ValueError("No usable rows to select a sample image from")
    return pool[min(max(sample_index, 0), len(pool) - 1)]


def process_trained_dataset(name: str, checkpoints_root: Path, data_root: Path, args: argparse.Namespace, device: str) -> None:
    dataset_dir = checkpoints_root / name
    metrics_path = dataset_dir / args.metrics_subpath
    checkpoint_path = find_checkpoint(dataset_dir, args.checkpoint_select, metrics_path)
    if checkpoint_path is None:
        raise ValueError(f"No clip_epoch_*.pt checkpoint found in {dataset_dir}")

    predictions_path = dataset_dir / args.zero_shot_subdir / args.predictions_name
    classnames_path = data_root / name / args.classnames_name
    output_dir = dataset_dir / args.output_subdir
    label = args.dataset_label or dataset_label(name)

    rows = build_rows_from_predictions(predictions_path, ROOT)
    classnames = classnames_or_infer(classnames_path, rows)
    templates = default_templates_for(name, args.template)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    model_name = args.model or checkpoint_args.get("model", "ViT-B-32")
    tokenizer = ByteTokenizer(context_length=args.context_length)
    model = build_model(model_name, vocab_size=tokenizer.vocab_size, context_length=tokenizer.context_length)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()

    print(f"  checkpoint: {checkpoint_path.name}")
    generate_all_figures(output_dir, label, model, model_name, tokenizer, rows, classnames, templates, args, device)


def process_pretrained_dataset(name: str, variant: str, pretrained_root: Path, data_root: Path, args: argparse.Namespace, device: str) -> None:
    variant_dir = pretrained_root / name / variant
    predictions_path = variant_dir / args.predictions_name
    classnames_path = data_root / name / args.classnames_name
    output_dir = variant_dir / args.output_subdir
    label = f"{args.dataset_label or dataset_label(name)} ({variant})"

    rows = build_rows_from_predictions(predictions_path, ROOT)
    classnames = classnames_or_infer(classnames_path, rows)
    templates = default_templates_for(name, args.template)

    model, tokenizer, model_name = load_openclip_model(args.pretrained_model, args.pretrained_tag, device)
    print(f"  pretrained weights: open_clip {args.pretrained_model} / {args.pretrained_tag}")
    generate_all_figures(output_dir, label, model, model_name, tokenizer, rows, classnames, templates, args, device)


def run_single_checkpoint(args: argparse.Namespace, device: str) -> None:
    """Single-image mode: explains one image, never loops over other datasets/images.
    Uses a local checkpoint if --checkpoint is given, otherwise falls back to
    open_clip pretrained weights (--pretrained-model/--pretrained-tag) — the latter is
    what pretrained_zero_shot datasets (no local .pt file) need."""
    output_dir = Path(args.output_dir or "research_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    classnames = read_classnames(args.classnames) or []

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        checkpoint_args = checkpoint.get("args", {})
        model_name = args.model or checkpoint_args.get("model", "ViT-B-32")
        tokenizer = ByteTokenizer(context_length=args.context_length)
        model = build_model(model_name, vocab_size=tokenizer.vocab_size, context_length=tokenizer.context_length)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device)
        model.eval()
        print(f"  checkpoint: {args.checkpoint}")
    else:
        model_name = args.model or args.pretrained_model
        model, tokenizer, model_name = load_openclip_model(model_name, args.pretrained_tag, device)
        print(f"  pretrained weights: open_clip {model_name} / {args.pretrained_tag}")

    templates = args.template or DEFAULT_TEMPLATES

    rows = load_rows(args.val_jsonl, args.image_root, classnames)
    sample = select_sample(args, rows, classnames)
    matrix_rows = rows[: max(1, args.max_matrix_images)] if rows else [sample]
    embedding_rows = stratified_rows(rows, args.max_embedding_images) if rows else [sample]

    generate_figures_for_sample(
        output_dir, args.dataset_label or "Dataset", model, model_name, tokenizer, sample,
        args.text, matrix_rows, embedding_rows, classnames, templates, args, device,
    )
    print(f"Saved research figures to: {output_dir.resolve()}")


def generate_all_figures(
    output_dir: Path,
    label: str,
    model: torch.nn.Module,
    model_name: str,
    tokenizer: Any,
    rows: list[SampleRow],
    classnames: list[str],
    templates: list[str],
    args: argparse.Namespace,
    device: str,
) -> None:
    sample = pick_sample(rows, args.sample_index)
    matrix_rows = rows[: max(1, args.max_matrix_images)]
    embedding_rows = stratified_rows(rows, args.max_embedding_images)
    generate_figures_for_sample(
        output_dir, label, model, model_name, tokenizer, sample, None,
        matrix_rows, embedding_rows, classnames, templates, args, device,
    )
    print(f"  saved to: {output_dir.resolve()}")


def generate_figures_for_sample(
    output_dir: Path,
    label: str,
    model: torch.nn.Module,
    model_name: str,
    tokenizer: Any,
    sample: SampleRow,
    text_override: str | None,
    matrix_rows: list[SampleRow],
    embedding_rows: list[SampleRow],
    classnames: list[str],
    templates: list[str],
    args: argparse.Namespace,
    device: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = text_override or templates[0].format(sample.class_name)
    image, image_tensor, preprocessed_image = load_preprocessed_image(sample.image_path, model.input_resolution, device)

    layer_names = default_research_layers(model, model_name)
    outputs = run_forward_with_hooks(model, tokenizer, image_tensor, prompt, layer_names, device)
    classifier = build_zero_shot_classifier(model, tokenizer, classnames, templates, device) if classnames else None
    top_predictions = predict_topk(outputs.image_features, classifier, classnames, top_k=5) if classifier is not None else []

    matrix_data = compute_dataset_similarities(model, classifier, classnames, matrix_rows, model.input_resolution, device) if classifier is not None else None
    embedding_data = compute_image_embeddings(model, embedding_rows, model.input_resolution, device)
    gradcam_maps = compute_gradcam_maps(model, image_tensor, classifier, classnames, top_predictions, model_name, device) if classifier is not None else {}

    save_figure_1(output_dir / "figure_1_input_image_processing.png", image, preprocessed_image, sample, model.input_resolution, args.dpi, label)
    save_figure_2(output_dir / "figure_2_vision_encoder_pipeline.png", image, outputs, args.dpi)
    save_figure_3(output_dir / "figure_3_text_encoder_pipeline.png", prompt, tokenizer, outputs, args.dpi)
    save_figure_4(output_dir / "figure_4_image_text_similarity.png", image, sample, top_predictions, args.dpi)
    if matrix_data is not None:
        save_figure_5(output_dir / "figure_5_correlation_matrix.png", matrix_data, classnames[: args.max_matrix_classes], args.dpi)
    save_figure_6(output_dir / "figure_6_embedding_visualization.png", embedding_rows, embedding_data, args.dpi)
    save_figure_7(output_dir / "figure_7_attention_visualization.png", image, outputs, gradcam_maps, args.dpi)
    save_figure_8(output_dir / "figure_8_complete_clip_pipeline.png", image, prompt, outputs, top_predictions, args.dpi)
    write_summary(output_dir / "figure_generation_summary.csv", sample, prompt, model_name, classnames, top_predictions)


def load_rows(jsonl_path: str | None, image_root: str, classnames: list[str]) -> list[SampleRow]:
    if not jsonl_path:
        return []
    root = Path(image_root)
    rows: list[SampleRow] = []
    with Path(jsonl_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            image_value = first_present(item, ("image", "image_path", "path"))
            if image_value is None:
                continue
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = root / image_path
            class_name = str(item.get("class_name") or item.get("label") or "").replace("_", " ").strip()
            if not class_name and classnames:
                class_name = classnames[0]
            caption = str(item.get("caption") or item.get("text") or item.get("title") or "")
            rows.append(SampleRow(image_path=image_path, class_name=class_name, caption=caption))
    return rows


def select_sample(args: argparse.Namespace, rows: list[SampleRow], classnames: list[str]) -> SampleRow:
    if args.image:
        class_name = args.ground_truth or infer_class_from_path(Path(args.image), classnames)
        return SampleRow(image_path=Path(args.image), class_name=class_name, caption=args.text or "")
    if rows:
        return rows[0]
    raise ValueError("Provide --image or --val-jsonl so the script can choose an input sample.")


class TextTower(torch.nn.Module):
    """Exposes an open_clip model's text-side submodules under a `text.` prefix so the
    existing layer-name conventions (`text.token_embedding`, `text.transformer.resblocks.N`,
    `text.ln_final`) and forward hooks work unchanged."""

    def __init__(self, clip_model: torch.nn.Module) -> None:
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final


class OpenCLIPAdapter(torch.nn.Module):
    """Wraps an open_clip pretrained model so it matches the duck-typed interface
    (`encode_image(x, normalize=...)`, `encode_text(x, normalize=...)`, `.input_resolution`,
    `visual.*` / `text.*` submodule names) that the rest of this script expects from the
    locally fine-tuned `clip_implement` model."""

    def __init__(self, clip_model: torch.nn.Module, input_resolution: int) -> None:
        super().__init__()
        # Stored in a plain list (not an nn.Module attribute) so PyTorch's automatic
        # submodule registration doesn't re-visit/dedupe visual.*/text.* under this name too.
        self._clip_model_ref = [clip_model]
        self.visual = clip_model.visual
        self.text = TextTower(clip_model)
        self.input_resolution = input_resolution

    @property
    def _clip_model(self) -> torch.nn.Module:
        return self._clip_model_ref[0]

    def encode_image(self, image: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        features = self._clip_model.encode_image(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, tokens: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        features = self._clip_model.encode_text(tokens)
        return F.normalize(features, dim=-1) if normalize else features


def load_openclip_model(model_name: str, pretrained_tag: str, device: str) -> tuple[torch.nn.Module, Any, str]:
    """Loads an off-the-shelf OpenAI CLIP checkpoint via open_clip (no local .pt file
    required) and returns (adapter_model, tokenizer, model_name)."""
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "open_clip_torch is required for pretrained zero-shot datasets. Install it with "
            "`pip install open_clip_torch`."
        ) from exc

    clip_model = open_clip.create_model(model_name, pretrained=pretrained_tag)
    clip_model.to(device)
    clip_model.eval()
    image_size = getattr(clip_model.visual, "image_size", 224)
    input_resolution = image_size[0] if isinstance(image_size, (tuple, list)) else int(image_size)
    adapter = OpenCLIPAdapter(clip_model, input_resolution).to(device)
    adapter.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return adapter, tokenizer, model_name



    for key in keys:
        if key in item:
            return item[key]
    return None


def infer_class_from_path(image_path: Path, classnames: list[str]) -> str:
    parent = image_path.parent.name.replace("_", " ").replace("-", " ").strip()
    for classname in classnames:
        if classname.lower() == parent.lower():
            return classname
    return parent or (classnames[0] if classnames else "unknown")


def load_preprocessed_image(image_path: Path, image_size: int, device: str) -> tuple[Image.Image, torch.Tensor, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    cropped = resize_center_crop(image, image_size)
    array = np.asarray(cropped).astype(np.float32) / 255.0
    normalized = (array - np.asarray(CLIP_MEAN, dtype=np.float32)) / np.asarray(CLIP_STD, dtype=np.float32)
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(device)
    return image, tensor, cropped


def resize_center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    scale = size / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
    left = max(0, (resized.width - size) // 2)
    top = max(0, (resized.height - size) // 2)
    return resized.crop((left, top, left + size, top + size))


def default_research_layers(model: torch.nn.Module, model_name: str) -> list[str]:
    modules = dict(model.named_modules())
    if model_name.startswith("ViT"):
        names = [
            "visual.conv1",
            "visual.ln_pre",
            "visual.transformer.resblocks.0",
            "visual.transformer.resblocks.3",
            "visual.transformer.resblocks.6",
            "visual.transformer.resblocks.9",
            "visual.transformer.resblocks.11",
            "visual.ln_post",
            "text.token_embedding",
            "text.transformer.resblocks.0",
            "text.transformer.resblocks.3",
            "text.transformer.resblocks.6",
            "text.transformer.resblocks.9",
            "text.transformer.resblocks.11",
            "text.ln_final",
        ]
    else:
        names = [
            "visual.conv1",
            "visual.layer1",
            "visual.layer2",
            "visual.layer3",
            "visual.layer4",
            "visual.attnpool",
            "text.token_embedding",
            "text.transformer.resblocks.0",
            "text.transformer.resblocks.3",
            "text.transformer.resblocks.6",
            "text.transformer.resblocks.9",
            "text.transformer.resblocks.11",
            "text.ln_final",
        ]
    return [name for name in names if name in modules]


def register_hooks(model: torch.nn.Module, layer_names: Iterable[str]) -> tuple[dict[str, torch.Tensor], list[Any]]:
    activations: dict[str, torch.Tensor] = {}
    hooks = []
    modules = dict(model.named_modules())
    for layer_name in layer_names:
        module = modules[layer_name]

        def hook_fn(_module, _inputs, output, name=layer_name):
            if isinstance(output, tuple):
                output = output[0]
            activations[name] = output.detach().float().cpu()

        hooks.append(module.register_forward_hook(hook_fn))
    return activations, hooks


@torch.no_grad()
def run_forward_with_hooks(
    model: torch.nn.Module,
    tokenizer: ByteTokenizer,
    image_tensor: torch.Tensor,
    prompt: str,
    layer_names: list[str],
    device: str,
) -> ForwardOutputs:
    activations, hooks = register_hooks(model, layer_names)
    tokens = tokenizer([prompt]).to(device)
    try:
        image_features = model.encode_image(image_tensor, normalize=True)
        text_features = model.encode_text(tokens, normalize=True)
    finally:
        for hook in hooks:
            hook.remove()
    return ForwardOutputs(
        image_features=image_features.detach().cpu(),
        text_features=text_features.detach().cpu(),
        activations=activations,
        tokens=tokens.detach().cpu(),
    )


@torch.no_grad()
def build_zero_shot_classifier(
    model: torch.nn.Module,
    tokenizer: ByteTokenizer,
    classnames: list[str],
    templates: list[str],
    device: str,
) -> torch.Tensor:
    weights = []
    for classname in classnames:
        texts = [template.format(classname) for template in templates]
        tokens = tokenizer(texts).to(device)
        text_features = model.encode_text(tokens, normalize=True)
        weights.append(F.normalize(text_features.mean(dim=0), dim=0))
    return torch.stack(weights, dim=1)


def predict_topk(image_features: torch.Tensor, classifier: torch.Tensor, classnames: list[str], top_k: int) -> list[dict[str, Any]]:
    scores = image_features.to(classifier.device) @ classifier
    top_scores, top_indices = scores.topk(min(top_k, len(classnames)), dim=1)
    rows = []
    for rank, (score, index) in enumerate(zip(top_scores[0].detach().cpu(), top_indices[0].detach().cpu()), start=1):
        rows.append({"rank": rank, "class_name": classnames[int(index)], "score": float(score), "index": int(index)})
    return rows


@torch.no_grad()
def compute_dataset_similarities(
    model: torch.nn.Module,
    classifier: torch.Tensor,
    classnames: list[str],
    rows: list[SampleRow],
    image_size: int,
    device: str,
) -> dict[str, Any]:
    features = []
    for row in rows:
        _, tensor, _ = load_preprocessed_image(row.image_path, image_size, device)
        features.append(model.encode_image(tensor, normalize=True).detach().cpu())
    image_features = torch.cat(features, dim=0)
    matrix = image_features.to(classifier.device) @ classifier
    predictions = matrix.argmax(dim=1).detach().cpu().tolist()
    targets = [classnames.index(row.class_name) if row.class_name in classnames else -1 for row in rows]
    return {
        "rows": rows,
        "matrix": matrix.detach().cpu().numpy(),
        "predictions": predictions,
        "targets": targets,
    }


@torch.no_grad()
def compute_image_embeddings(
    model: torch.nn.Module,
    rows: list[SampleRow],
    image_size: int,
    device: str,
) -> np.ndarray:
    features = []
    for row in rows:
        _, tensor, _ = load_preprocessed_image(row.image_path, image_size, device)
        features.append(model.encode_image(tensor, normalize=True).detach().cpu().numpy()[0])
    return np.asarray(features, dtype=np.float32)


def stratified_rows(rows: list[SampleRow], limit: int) -> list[SampleRow]:
    if len(rows) <= limit:
        return rows
    by_class: dict[str, list[SampleRow]] = {}
    for row in rows:
        by_class.setdefault(row.class_name, []).append(row)
    selected: list[SampleRow] = []
    class_names = sorted(by_class)
    cursor = 0
    while len(selected) < limit and class_names:
        name = class_names[cursor % len(class_names)]
        bucket = by_class[name]
        index = len([row for row in selected if row.class_name == name])
        if index < len(bucket):
            selected.append(bucket[index])
        if all(len([row for row in selected if row.class_name == key]) >= len(value) for key, value in by_class.items()):
            break
        cursor += 1
    return selected[:limit]


def compute_gradcam_maps(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    classifier: torch.Tensor,
    classnames: list[str],
    top_predictions: list[dict[str, Any]],
    model_name: str,
    device: str,
) -> dict[str, np.ndarray]:
    if not top_predictions:
        return {}
    target_index = int(top_predictions[0]["index"])
    layer_names = [
        "visual.transformer.resblocks.3",
        "visual.transformer.resblocks.6",
        "visual.transformer.resblocks.9",
        "visual.transformer.resblocks.11",
    ]
    if not model_name.startswith("ViT"):
        layer_names = ["visual.layer2", "visual.layer3", "visual.layer4", "visual.attnpool"]
    modules = dict(model.named_modules())
    layer_names = [name for name in layer_names if name in modules]
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    hooks = []

    for layer_name in layer_names:
        module = modules[layer_name]

        def forward_hook(_module, _inputs, output, name=layer_name):
            if isinstance(output, tuple):
                output = output[0]
            activations[name] = output
            if output.requires_grad:
                output.register_hook(lambda grad, key=name: gradients.__setitem__(key, grad))

        hooks.append(module.register_forward_hook(forward_hook))

    try:
        model.zero_grad(set_to_none=True)
        tensor = image_tensor.detach().clone().to(device).requires_grad_(True)
        image_features = model.encode_image(tensor, normalize=True)
        score = (image_features @ classifier[:, target_index].to(device)).sum()
        score.backward()
    finally:
        for hook in hooks:
            hook.remove()

    maps: dict[str, np.ndarray] = {}
    for layer_name in layer_names:
        if layer_name not in activations or layer_name not in gradients:
            continue
        maps[layer_name] = gradcam_from_activation(activations[layer_name], gradients[layer_name])
    return maps


def gradcam_from_activation(activation: torch.Tensor, gradient: torch.Tensor) -> np.ndarray:
    act = activation.detach().float().cpu()
    grad = gradient.detach().float().cpu()
    if act.ndim == 4:
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1)[0]
    elif act.ndim == 3:
        if act.shape[1] == 1:
            act_tokens = act[:, 0, :]
            grad_tokens = grad[:, 0, :]
        elif act.shape[0] == 1:
            act_tokens = act[0]
            grad_tokens = grad[0]
        else:
            act_tokens = act[0]
            grad_tokens = grad[0]
        token_scores = (act_tokens * grad_tokens).mean(dim=-1)
        cam = tokens_to_grid(token_scores)
    elif act.ndim == 2:
        cam = tokens_to_grid((act * grad).mean(dim=-1))
    else:
        cam = (act * grad).flatten().unsqueeze(0)
    cam = F.relu(cam)
    return normalize(cam.numpy())


def activation_map(activation: torch.Tensor) -> np.ndarray:
    tensor = activation.detach().float().cpu()
    if tensor.ndim == 4:
        values = tensor[0].abs().mean(dim=0)
    elif tensor.ndim == 3:
        if tensor.shape[1] == 1:
            tokens = tensor[:, 0, :]
        elif tensor.shape[0] == 1:
            tokens = tensor[0]
        else:
            tokens = tensor[0]
        values = tokens_to_grid(tokens.abs().mean(dim=-1))
    elif tensor.ndim == 2:
        values = tokens_to_grid(tensor.abs().mean(dim=-1))
    else:
        values = tensor.flatten().abs().unsqueeze(0)
    return normalize(values.numpy())


def tokens_to_grid(values: torch.Tensor) -> torch.Tensor:
    values = values.flatten()
    if values.numel() > 1:
        without_cls = values[1:]
        grid = int(math.sqrt(without_cls.numel()))
        if grid * grid == without_cls.numel():
            return without_cls.reshape(grid, grid)
    grid = int(math.sqrt(values.numel()))
    if grid * grid == values.numel():
        return values.reshape(grid, grid)
    return values.unsqueeze(0)


def normalize(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    array = array - float(array.min())
    return array / (float(array.max()) + 1e-8)


def stats_for_tensor(tensor: torch.Tensor | np.ndarray) -> dict[str, str]:
    if isinstance(tensor, torch.Tensor):
        values = tensor.detach().float().cpu().numpy()
        shape = "x".join(str(size) for size in tensor.shape)
    else:
        values = np.asarray(tensor, dtype=np.float32)
        shape = "x".join(str(size) for size in values.shape)
    return {
        "shape": shape,
        "mean": f"{float(values.mean()):.4f}",
        "std": f"{float(values.std()):.4f}",
        "min": f"{float(values.min()):.4f}",
        "max": f"{float(values.max()):.4f}",
    }


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = ["arialbd.ttf" if bold else "arial.ttf", "calibrib.ttf" if bold else "calibri.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    return image, draw


def save_png(image: Image.Image, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(dpi, dpi))


def draw_title(draw: ImageDraw.ImageDraw, title: str, subtitle: str | None, width: int) -> None:
    draw.text((60, 36), title, fill=INK, font=font(42, bold=True))
    if subtitle:
        draw.text((60, 92), subtitle, fill=MUTED, font=font(24))
    draw.line((60, 130, width - 60, 130), fill=BORDER, width=2)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int = 22, fill: tuple[int, int, int] = INK, bold: bool = False) -> None:
    draw.text(xy, text, fill=fill, font=font(size, bold=bold))


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    max_chars: int,
    line_height: int,
    size: int = 22,
    fill: tuple[int, int, int] = INK,
) -> int:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font(size))
        y += line_height
    return y


def draw_image_box(
    base: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    box: tuple[int, int, int, int],
    label: str | None = None,
    border: tuple[int, int, int] = BORDER,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=8, outline=border, width=2, fill=(248, 250, 252))
    fitted = fit_image(image, x1 - x0 - 16, y1 - y0 - 42 if label else y1 - y0 - 16)
    base.paste(fitted, (x0 + (x1 - x0 - fitted.width) // 2, y0 + 8))
    if label:
        draw.text((x0 + 12, y1 - 30), label, fill=INK, font=font(18))


def fit_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    output = image.convert("RGB").copy()
    output.thumbnail((max_width, max_height), Image.Resampling.BICUBIC)
    return output


def draw_stat_table(draw: ImageDraw.ImageDraw, x: int, y: int, stats: dict[str, str], title: str | None = None) -> None:
    if title:
        draw.text((x, y), title, fill=INK, font=font(20, bold=True))
        y += 30
    for key in ["shape", "mean", "std", "min", "max"]:
        draw.text((x, y), f"{key}", fill=MUTED, font=font(18))
        draw.text((x + 88, y), stats.get(key, ""), fill=INK, font=font(18))
        y += 26


def colormap(value: float) -> tuple[int, int, int]:
    stops = [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ]
    value = min(1.0, max(0.0, value))
    scaled = value * (len(stops) - 1)
    index = min(len(stops) - 2, int(scaled))
    frac = scaled - index
    a = stops[index]
    b = stops[index + 1]
    return tuple(round(a[channel] + (b[channel] - a[channel]) * frac) for channel in range(3))


def heatmap_image(values: np.ndarray, size: tuple[int, int]) -> Image.Image:
    values = normalize(values)
    rgb = np.zeros((values.shape[0], values.shape[1], 3), dtype=np.uint8)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            rgb[row, col] = colormap(float(values[row, col]))
    return Image.fromarray(rgb, mode="RGB").resize(size, Image.Resampling.BICUBIC)


def overlay_map(image: Image.Image, values: np.ndarray, size: tuple[int, int]) -> Image.Image:
    base = image.convert("RGB").resize(size, Image.Resampling.BICUBIC)
    heat = heatmap_image(values, size).convert("RGBA")
    alpha_values = np.uint8(np.clip(normalize(values), 0.0, 1.0) * 165)
    alpha = Image.fromarray(alpha_values, mode="L").resize(size, Image.Resampling.BICUBIC)
    heat.putalpha(alpha)
    return Image.alpha_composite(base.convert("RGBA"), heat).convert("RGB")


def embedding_strip(feature: torch.Tensor | np.ndarray, size: tuple[int, int]) -> Image.Image:
    values = feature.detach().float().cpu().numpy() if isinstance(feature, torch.Tensor) else np.asarray(feature)
    values = values.reshape(1, -1)
    return heatmap_image(values, size)


def save_figure_1(path: Path, image: Image.Image, preprocessed: Image.Image, sample: SampleRow, image_size: int, dpi: int, dataset_label: str = "Dataset") -> None:
    fig, draw = canvas(1800, 1200)
    draw_title(draw, "Figure 1. Input Image Processing", f"{dataset_label} sample preparation before CLIP encoding", 1800)
    draw_image_box(fig, draw, image, (80, 190, 780, 940), "Original RGB image")
    draw_image_box(fig, draw, preprocessed, (960, 190, 1660, 940), f"Preprocessed image ({image_size}x{image_size})")
    draw.line((810, 560, 930, 560), fill=MUTED, width=4)
    draw.polygon([(930, 560), (902, 544), (902, 576)], fill=MUTED)
    details = [
        ("Filename", sample.image_path.name),
        ("Ground-truth class", sample.class_name),
        ("Original image size", f"{image.width} x {image.height} pixels"),
        ("Preprocessing", "Resize shortest side, center crop, RGB conversion, CLIP mean/std normalization"),
    ]
    y = 980
    for label, value in details:
        draw.text((90, y), label, fill=MUTED, font=font(22))
        draw.text((330, y), value, fill=INK, font=font(22, bold=label == "Ground-truth class"))
        y += 46
    save_png(fig, path, dpi)


def save_figure_2(path: Path, image: Image.Image, outputs: ForwardOutputs, dpi: int) -> None:
    stages = [
        ("Original image", None, "Input RGB pixels"),
        ("Patch embedding", "visual.conv1", "Patch/token feature map"),
        ("Position embedding + pre-LN", "visual.ln_pre", "Position-aware token sequence"),
        ("Transformer Block 1", "visual.transformer.resblocks.0", "Contextual visual tokens"),
        ("Transformer Block 3", "visual.transformer.resblocks.3", "Intermediate visual representation"),
        ("Transformer Block 6", "visual.transformer.resblocks.6", "Mid-level visual representation"),
        ("Transformer Block 9", "visual.transformer.resblocks.9", "High-level visual representation"),
        ("Transformer Block 11", "visual.transformer.resblocks.11", "Final transformer representation"),
        ("LayerNorm", "visual.ln_post", "Normalized CLS representation"),
        ("Final 512-dimensional image embedding", None, "Projected image vector"),
    ]
    row_h = 250
    fig, draw = canvas(2200, 210 + row_h * len(stages))
    draw_title(draw, "Figure 2. Vision Encoder Pipeline", "Stage-wise ViT feature maps with dimensions and statistics", 2200)
    thumb = image.convert("RGB")
    y = 170
    for stage_index, (label, layer_name, note) in enumerate(stages):
        draw.text((70, y + 10), f"{stage_index + 1}. {label}", fill=INK, font=font(24, bold=True))
        draw.text((70, y + 44), note, fill=MUTED, font=font(18))
        draw_image_box(fig, draw, thumb, (470, y, 690, y + 210), "Original")
        if layer_name and layer_name in outputs.activations:
            values = activation_map(outputs.activations[layer_name])
            feature = overlay_map(image, values, (330, 210)) if values.shape[0] > 1 else heatmap_image(values, (330, 210))
            stats = stats_for_tensor(outputs.activations[layer_name])
        elif label.startswith("Final"):
            feature = embedding_strip(outputs.image_features[0], (330, 210))
            stats = stats_for_tensor(outputs.image_features)
        else:
            feature = fit_image(image, 330, 210)
            stats = {"shape": f"1x3x{image.height}x{image.width}", "mean": "", "std": "", "min": "", "max": ""}
        draw_image_box(fig, draw, feature, (760, y, 1110, y + 210), "Feature representation")
        draw_stat_table(draw, 1180, y + 22, stats, "Feature statistics")
        if stage_index < len(stages) - 1:
            draw.line((350, y + 190, 350, y + row_h + 5), fill=BORDER, width=3)
            draw.polygon([(350, y + row_h + 5), (338, y + row_h - 18), (362, y + row_h - 18)], fill=BORDER)
        y += row_h
    save_png(fig, path, dpi)


def save_figure_3(path: Path, prompt: str, tokenizer: Any, outputs: ForwardOutputs, dpi: int) -> None:
    stages = [
        ("Tokenization", None),
        ("Token embedding", "text.token_embedding"),
        ("Transformer Block 0", "text.transformer.resblocks.0"),
        ("Transformer Block 3", "text.transformer.resblocks.3"),
        ("Transformer Block 6", "text.transformer.resblocks.6"),
        ("Transformer Block 9", "text.transformer.resblocks.9"),
        ("Transformer Block 11", "text.transformer.resblocks.11"),
        ("LayerNorm", "text.ln_final"),
        ("Final text embedding (512D)", None),
    ]
    row_h = 245
    fig, draw = canvas(2200, 260 + row_h * len(stages))
    draw_title(draw, "Figure 3. Text Encoder Pipeline", "Prompt tokenization, text-transformer activations, and final embedding", 2200)
    draw.text((70, 160), "Prompt", fill=MUTED, font=font(22))
    draw_wrapped_text(draw, (180, 160), f'"{prompt}"', 95, 30, size=22, fill=INK)
    tokens = readable_tokens(tokenizer, prompt)
    y = 235
    for stage_index, (label, layer_name) in enumerate(stages):
        draw.text((70, y + 10), f"{stage_index + 1}. {label}", fill=INK, font=font(24, bold=True))
        if label == "Tokenization":
            draw_token_boxes(draw, tokens, 500, y + 22, 1240, 150)
            stats = {"shape": f"1x{len(tokens)} tokens", "mean": "", "std": "", "min": "", "max": ""}
        elif layer_name and layer_name in outputs.activations:
            feature = heatmap_image(activation_map(outputs.activations[layer_name]), (700, 150))
            fig.paste(feature, (500, y + 24))
            draw.rectangle((500, y + 24, 1200, y + 174), outline=BORDER, width=2)
            stats = stats_for_tensor(outputs.activations[layer_name])
        else:
            feature = embedding_strip(outputs.text_features[0], (700, 150))
            fig.paste(feature, (500, y + 24))
            draw.rectangle((500, y + 24, 1200, y + 174), outline=BORDER, width=2)
            stats = stats_for_tensor(outputs.text_features)
        draw_stat_table(draw, 1280, y + 24, stats, "Representation statistics")
        if stage_index < len(stages) - 1:
            draw.line((330, y + 174, 330, y + row_h - 12), fill=BORDER, width=3)
            draw.polygon([(330, y + row_h - 12), (318, y + row_h - 34), (342, y + row_h - 34)], fill=BORDER)
        y += row_h
    save_png(fig, path, dpi)


def readable_tokens(tokenizer: Any, prompt: str) -> list[str]:
    if hasattr(tokenizer, "sos_token_id"):
        return readable_tokens_byte(tokenizer, prompt)
    if hasattr(tokenizer, "sot_token_id") and hasattr(tokenizer, "decoder"):
        return readable_tokens_bpe(tokenizer, prompt)
    # Fallback: best-effort whitespace/punctuation split, purely for display purposes.
    return prompt.replace(".", " .").split()


def readable_tokens_byte(tokenizer: ByteTokenizer, prompt: str) -> list[str]:
    ids = tokenizer.encode(prompt)
    output = []
    for token_id in ids:
        if token_id == tokenizer.sos_token_id:
            output.append("<SOS>")
        elif token_id == tokenizer.eos_token_id:
            output.append("<EOS>")
        elif token_id >= tokenizer.byte_offset:
            value = token_id - tokenizer.byte_offset
            char = bytes([value]).decode("utf-8", errors="replace")
            output.append(char if char.strip() else "<space>")
        else:
            output.append(str(token_id))
    return output


def readable_tokens_bpe(tokenizer: Any, prompt: str) -> list[str]:
    ids = tokenizer.encode(prompt)
    output = []
    for token_id in [tokenizer.sot_token_id, *ids, tokenizer.eot_token_id]:
        piece = tokenizer.decoder.get(token_id, str(token_id))
        piece = piece.replace("</w>", "")
        if token_id == tokenizer.sot_token_id:
            output.append("<SOS>")
        elif token_id == tokenizer.eot_token_id:
            output.append("<EOS>")
        else:
            output.append(piece if piece.strip() else "<space>")
    return output


def draw_token_boxes(draw: ImageDraw.ImageDraw, tokens: list[str], x: int, y: int, max_width: int, max_height: int) -> None:
    cursor_x = x
    cursor_y = y
    for token in tokens[:60]:
        width = max(42, 15 * len(token) + 20)
        if cursor_x + width > x + max_width:
            cursor_x = x
            cursor_y += 44
        if cursor_y + 36 > y + max_height:
            draw.text((cursor_x, cursor_y), "...", fill=MUTED, font=font(18))
            break
        draw.rounded_rectangle((cursor_x, cursor_y, cursor_x + width, cursor_y + 32), radius=6, fill=LIGHT, outline=BORDER)
        draw.text((cursor_x + 10, cursor_y + 7), token, fill=INK, font=font(15))
        cursor_x += width + 8


def save_figure_4(path: Path, image: Image.Image, sample: SampleRow, top_predictions: list[dict[str, Any]], dpi: int) -> None:
    fig, draw = canvas(1800, 1200)
    draw_title(draw, "Figure 4. Image vs Text Similarity", "Zero-shot top-5 prediction scores from cosine similarity", 1800)
    draw_image_box(fig, draw, image, (80, 190, 640, 850), sample.image_path.name)
    draw.text((720, 210), "Ground truth", fill=MUTED, font=font(24))
    draw.text((720, 250), sample.class_name, fill=INK, font=font(34, bold=True))
    draw.text((720, 330), "Top-5 predicted classes", fill=INK, font=font(28, bold=True))
    max_score = max([item["score"] for item in top_predictions], default=1.0)
    min_score = min([item["score"] for item in top_predictions], default=0.0)
    span = max(1e-6, max_score - min_score)
    y = 400
    for item in top_predictions:
        score = item["score"]
        normalized = (score - min_score) / span if span else 1.0
        bar_w = int(680 * normalized)
        color = GREEN if item["class_name"] == sample.class_name else BLUE
        draw.text((720, y + 8), f"{item['rank']}. {item['class_name']}", fill=INK, font=font(24, bold=item["rank"] == 1))
        draw.rounded_rectangle((1080, y, 1080 + 680, y + 38), radius=10, fill=LIGHT, outline=BORDER)
        draw.rounded_rectangle((1080, y, 1080 + bar_w, y + 38), radius=10, fill=color)
        draw.text((1088 + min(bar_w + 10, 600), y + 6), f"{score:.3f}", fill=INK, font=font(20))
        y += 78
    draw.text((720, 860), "Bars show relative confidence within the top-5 classes.", fill=MUTED, font=font(20))
    save_png(fig, path, dpi)


def save_figure_5(path: Path, matrix_data: dict[str, Any], classnames: list[str], dpi: int) -> None:
    rows: list[SampleRow] = matrix_data["rows"]
    matrix = matrix_data["matrix"][:, : len(classnames)]
    predictions = matrix_data["predictions"]
    targets = matrix_data["targets"]
    cell_w = 54
    row_h = 118
    left = 460
    top = 250
    width = max(1900, left + cell_w * len(classnames) + 100)
    height = top + row_h * len(rows) + 100
    fig, draw = canvas(width, height)
    draw_title(draw, "Figure 5. Correlation Matrix", "Rows combine image thumbnails, filenames, ground truth, and text-prompt similarities", width)
    for col, name in enumerate(classnames):
        x = left + col * cell_w
        draw.text((x + 6, top - 92), shorten(name, 10), fill=INK, font=font(15))
    normalized = normalize(matrix)
    for row_index, row in enumerate(rows):
        y = top + row_index * row_h
        correct = predictions[row_index] == targets[row_index]
        border = GREEN if correct else RED
        thumb = Image.open(row.image_path).convert("RGB")
        draw_image_box(fig, draw, thumb, (60, y + 8, 158, y + 106), None, border=border)
        draw.text((175, y + 20), row.image_path.name, fill=INK, font=font(18, bold=True))
        draw.text((175, y + 52), f"GT: {row.class_name}", fill=MUTED, font=font(17))
        pred_name = classnames[predictions[row_index]] if predictions[row_index] < len(classnames) else "outside view"
        draw.text((175, y + 80), f"Pred: {pred_name}", fill=border, font=font(17))
        row_min = int(np.argmin(matrix[row_index]))
        row_max = int(np.argmax(matrix[row_index]))
        for col in range(len(classnames)):
            x = left + col * cell_w
            fill = colormap(float(normalized[row_index, col]))
            draw.rectangle((x, y + 12, x + cell_w - 4, y + row_h - 12), fill=fill, outline=(255, 255, 255), width=1)
            outline = None
            if col == row_max:
                outline = GREEN
            elif col == row_min:
                outline = BLUE
            if outline:
                draw.rectangle((x + 2, y + 14, x + cell_w - 6, y + row_h - 14), outline=outline, width=4)
    draw.text((60, height - 58), "Green row border: correct prediction. Red row border: incorrect prediction. Green cell: highest row similarity. Blue cell: lowest row similarity.", fill=MUTED, font=font(18))
    save_png(fig, path, dpi)


def save_figure_6(path: Path, rows: list[SampleRow], embeddings: np.ndarray, dpi: int) -> None:
    fig, draw = canvas(1900, 1400)
    draw_title(draw, "Figure 6. Embedding Visualization", "PCA projection of image embeddings with thumbnails and class boundaries", 1900)
    if len(rows) == 0 or embeddings.size == 0:
        draw.text((80, 220), "No embedding rows available.", fill=INK, font=font(28))
        save_png(fig, path, dpi)
        return
    points = pca_2d(embeddings)
    x0, y0, x1, y1 = 150, 230, 1720, 1240
    draw.rectangle((x0, y0, x1, y1), outline=BORDER, width=2)
    draw.text((x0, y1 + 28), "PC1", fill=INK, font=font(22))
    draw.text((x0 - 70, y0 - 30), "PC2", fill=INK, font=font(22))
    scaled = scale_points(points, (x0 + 70, y0 + 70, x1 - 70, y1 - 70))
    by_class: dict[str, list[tuple[float, float]]] = {}
    for row, point in zip(rows, scaled):
        by_class.setdefault(row.class_name, []).append(point)
    palette = [GREEN, BLUE, GOLD, RED, (14, 165, 233), (168, 85, 247), (236, 72, 153)]
    for index, (class_name, pts) in enumerate(sorted(by_class.items())):
        if len(pts) < 2:
            continue
        xs = [point[0] for point in pts]
        ys = [point[1] for point in pts]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        rx = max(50, np.std(xs) * 2.2 + 45)
        ry = max(50, np.std(ys) * 2.2 + 45)
        color = palette[index % len(palette)]
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=color, width=3)
        draw.text((cx + rx + 6, cy - 10), shorten(class_name, 18), fill=color, font=font(16))
    for row, point in zip(rows, scaled):
        thumb = Image.open(row.image_path).convert("RGB")
        fitted = fit_image(thumb, 58, 58)
        x = int(point[0] - fitted.width / 2)
        y = int(point[1] - fitted.height / 2)
        draw.rectangle((x - 2, y - 2, x + fitted.width + 2, y + fitted.height + 2), outline=(255, 255, 255), width=4)
        fig.paste(fitted, (x, y))
    save_png(fig, path, dpi)


def pca_2d(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) == 1:
        return np.zeros((1, 2), dtype=np.float32)
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    points = centered @ components
    if points.shape[1] == 1:
        points = np.concatenate([points, np.zeros_like(points)], axis=1)
    return points[:, :2]


def scale_points(points: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = box
    if len(points) == 1:
        return [((x0 + x1) / 2, (y0 + y1) / 2)]
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    normalized = (points - mins) / span
    return [(x0 + value[0] * (x1 - x0), y1 - value[1] * (y1 - y0)) for value in normalized]


def save_figure_7(path: Path, image: Image.Image, outputs: ForwardOutputs, gradcam_maps: dict[str, np.ndarray], dpi: int) -> None:
    fig, draw = canvas(1900, 1450)
    draw_title(draw, "Figure 7. Attention Visualization", "Grad-CAM style transparent heatmap overlays from selected ViT layers", 1900)
    stages = [
        ("Original Image", None),
        ("Attention Layer 3", "visual.transformer.resblocks.3"),
        ("Attention Layer 6", "visual.transformer.resblocks.6"),
        ("Attention Layer 9", "visual.transformer.resblocks.9"),
        ("Final Attention", "visual.transformer.resblocks.11"),
    ]
    x_positions = [70, 430, 790, 1150, 1510]
    for index, (label, layer_name) in enumerate(stages):
        x = x_positions[index]
        if layer_name is None:
            display = image
        else:
            values = gradcam_maps.get(layer_name)
            if values is None and layer_name in outputs.activations:
                values = activation_map(outputs.activations[layer_name])
            display = overlay_map(image, values, (300, 300)) if values is not None else image
        draw_image_box(fig, draw, display, (x, 250, x + 310, 610), label)
        if index < len(stages) - 1:
            draw.line((x + 320, 420, x + 350, 420), fill=MUTED, width=4)
            draw.polygon([(x + 350, 420), (x + 332, 408), (x + 332, 432)], fill=MUTED)
    draw.text((80, 700), "Warmer regions indicate stronger class-discriminative visual evidence for the predicted zero-shot class.", fill=MUTED, font=font(22))
    save_png(fig, path, dpi)


def save_figure_8(path: Path, image: Image.Image, prompt: str, outputs: ForwardOutputs, top_predictions: list[dict[str, Any]], dpi: int) -> None:
    fig, draw = canvas(2000, 1400)
    draw_title(draw, "Figure 8. Complete CLIP Pipeline", "Image and text encoders meet through cosine similarity for zero-shot prediction", 2000)
    draw_image_box(fig, draw, image, (80, 220, 440, 600), "Original image")
    draw_pipeline_box(draw, (560, 280, 920, 370), "Vision Transformer")
    draw_pipeline_box(draw, (560, 470, 920, 560), "Image Embedding (512D)")
    draw_arrow(draw, (440, 410), (560, 325))
    draw_arrow(draw, (740, 370), (740, 470))
    draw_pipeline_box(draw, (80, 820, 440, 930), "Text Prompt")
    draw_wrapped_text(draw, (110, 858), prompt, 26, 25, size=18)
    draw_pipeline_box(draw, (560, 825, 920, 915), "Text Transformer")
    draw_pipeline_box(draw, (560, 1015, 920, 1105), "Text Embedding (512D)")
    draw_arrow(draw, (440, 875), (560, 870))
    draw_arrow(draw, (740, 915), (740, 1015))
    draw_pipeline_box(draw, (1110, 610, 1510, 715), "Cosine Similarity")
    draw_arrow(draw, (920, 515), (1110, 640))
    draw_arrow(draw, (920, 1060), (1110, 690))
    similarity = float((outputs.image_features @ outputs.text_features.t()).item())
    draw.text((1210, 745), f"image-text score = {similarity:.3f}", fill=INK, font=font(24, bold=True))
    matrix = np.asarray([[item["score"] for item in top_predictions]], dtype=np.float32) if top_predictions else np.zeros((1, 1), dtype=np.float32)
    sim_img = heatmap_image(matrix, (420, 86))
    fig.paste(sim_img, (1160, 850))
    draw.rectangle((1160, 850, 1580, 936), outline=BORDER, width=2)
    draw.text((1160, 810), "Similarity matrix", fill=INK, font=font(24, bold=True))
    pred = top_predictions[0]["class_name"] if top_predictions else "unknown"
    draw_pipeline_box(draw, (1160, 1030, 1580, 1135), f"Predicted Class: {pred}", border=GREEN)
    draw_arrow(draw, (1370, 715), (1370, 850))
    draw_arrow(draw, (1370, 936), (1370, 1030))
    save_png(fig, path, dpi)


def draw_pipeline_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, border: tuple[int, int, int] = BORDER) -> None:
    draw.rounded_rectangle(box, radius=14, fill=(248, 250, 252), outline=border, width=3)
    x0, y0, x1, y1 = box
    draw_wrapped_text(draw, (x0 + 26, y0 + 26), text, 30, 28, size=22, fill=INK)


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line((start[0], start[1], end[0], end[1]), fill=MUTED, width=4)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 18
    left = (end[0] - length * math.cos(angle - 0.5), end[1] - length * math.sin(angle - 0.5))
    right = (end[0] - length * math.cos(angle + 0.5), end[1] - length * math.sin(angle + 0.5))
    draw.polygon([end, left, right], fill=MUTED)


def write_summary(path: Path, sample: SampleRow, prompt: str, model_name: str, classnames: list[str], top_predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item", "value"])
        writer.writeheader()
        rows = {
            "sample_image": str(sample.image_path),
            "ground_truth": sample.class_name,
            "prompt": prompt,
            "model": model_name,
            "class_count": len(classnames),
            "top1_prediction": top_predictions[0]["class_name"] if top_predictions else "",
            "top1_score": f"{top_predictions[0]['score']:.6f}" if top_predictions else "",
        }
        for item, value in rows.items():
            writer.writerow({"item": item, "value": value})


def shorten(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


if __name__ == "__main__":
    main()