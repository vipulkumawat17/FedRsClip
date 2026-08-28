from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import gridspec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Ellipse, Rectangle
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import cv2
except ModuleNotFoundError:  # OpenCV is optional; PIL/Matplotlib fallbacks keep the script portable.
    cv2 = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clip_implement.data import CLIP_MEAN, CLIP_STD, read_classnames
from clip_implement.model import build_model
from clip_implement.tokenizer import ByteTokenizer


DEFAULT_TEMPLATES = ["a satellite photo of {}.", "an aerial image of {}.", "a remote sensing image of {}."]
VISION_LAYERS = [
    ("Patch Embedding", "visual.conv1"),
    ("Position Embedding", "visual.ln_pre"),
    ("Transformer Block 1", "visual.transformer.resblocks.0"),
    ("Transformer Block 3", "visual.transformer.resblocks.3"),
    ("Transformer Block 6", "visual.transformer.resblocks.6"),
    ("Transformer Block 9", "visual.transformer.resblocks.9"),
    ("Transformer Block 11", "visual.transformer.resblocks.11"),
    ("LayerNorm", "visual.ln_post"),
]
TEXT_LAYERS = [
    ("Token Embedding", "text.token_embedding"),
    ("Transformer Block 0", "text.transformer.resblocks.0"),
    ("Transformer Block 3", "text.transformer.resblocks.3"),
    ("Transformer Block 6", "text.transformer.resblocks.6"),
    ("Transformer Block 9", "text.transformer.resblocks.9"),
    ("Transformer Block 11", "text.transformer.resblocks.11"),
    ("LayerNorm", "text.ln_final"),
]
ATTENTION_LAYERS = [
    ("Layer 3", "visual.transformer.resblocks.3"),
    ("Layer 6", "visual.transformer.resblocks.6"),
    ("Layer 9", "visual.transformer.resblocks.9"),
    ("Final Layer", "visual.transformer.resblocks.11"),
]


@dataclass(frozen=True)
class Sample:
    image_path: Path
    ground_truth: str
    caption: str = ""


@dataclass
class EncoderOutputs:
    image_embedding: torch.Tensor
    text_embeddings: torch.Tensor
    activations: dict[str, torch.Tensor]
    token_ids: torch.Tensor


@dataclass
class Prediction:
    rank: int
    class_name: str
    similarity: float
    confidence: float
    index: int


@dataclass
class DashboardData:
    sample: Sample
    original_image: np.ndarray
    preprocessed_image: np.ndarray
    image_tensor: torch.Tensor
    prompt: str
    outputs: EncoderOutputs
    predictions: list[Prediction]
    similarity_matrix: np.ndarray
    matrix_rows: list[Sample]
    matrix_predictions: list[int]
    matrix_targets: list[int]
    embedding_rows: list[Sample]
    embedding_points: np.ndarray
    embedding_images: list[np.ndarray]
    attention_maps: dict[str, np.ndarray]
    cosine_similarity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one IEEE/Springer-style CLIP interpretability dashboard per test image.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--image", action="append", default=None, help="Specific image path. Repeat to generate several dashboards.")
    parser.add_argument("--ground-truth", action="append", default=None, help="Ground truth for each --image, in the same order")
    parser.add_argument("--test-jsonl", default=None, help="JSONL test/validation manifest used when --image is omitted")
    parser.add_argument("--image-root", default=".", help="Root for relative JSONL image paths")
    parser.add_argument("--classnames", required=True, help="classnames.txt")
    parser.add_argument("--template", action="append", default=None, help="Zero-shot prompt template. Use {} for class name.")
    parser.add_argument("--output-dir", default="checkpoints/uc_merced/dashboards", help="Output directory")
    parser.add_argument("--model", default=None, choices=["RN50", "RN101", "ViT-B-32", "ViT-B-16"], help="Defaults to checkpoint args")
    parser.add_argument("--device", default=None, help="cuda, cpu, etc. Defaults to cuda when available")
    parser.add_argument("--context-length", type=int, default=76)
    parser.add_argument("--max-images", type=int, default=1, help="Number of JSONL images to process when --image is omitted")
    parser.add_argument("--matrix-samples", type=int, default=10, help="Rows in correlation heatmap")
    parser.add_argument("--embedding-samples", type=int, default=48, help="Images in embedding PCA/t-SNE panel")
    parser.add_argument("--embedding-method", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    templates = args.template or DEFAULT_TEMPLATES
    classnames = read_classnames(args.classnames) or []
    if not classnames:
        raise ValueError("No class names were loaded. Provide a valid --classnames file.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    model_name = args.model or checkpoint_args.get("model", "ViT-B-32")
    tokenizer = ByteTokenizer(context_length=args.context_length)
    model = build_model(model_name, vocab_size=tokenizer.vocab_size, context_length=tokenizer.context_length).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    rows = load_samples(args.test_jsonl, args.image_root) if args.test_jsonl else []
    dashboard_samples = explicit_samples(args, classnames) if args.image else rows[: args.max_images]
    if not dashboard_samples:
        raise ValueError("Provide --image or --test-jsonl so dashboard images can be selected.")

    classifier, class_text_embeddings = build_zero_shot_classifier(model, tokenizer, classnames, templates, device)
    matrix_rows = stratified_samples(rows, args.matrix_samples) if rows else dashboard_samples[:1]
    embedding_rows = stratified_samples(rows, args.embedding_samples) if rows else dashboard_samples[:1]
    matrix_data = compute_similarity_matrix(model, classifier, classnames, matrix_rows, device)
    embedding_vectors, embedding_images = compute_embedding_panel_data(model, embedding_rows, device)
    embedding_points = reduce_embeddings(embedding_vectors, args.embedding_method)

    for sample in dashboard_samples:
        data = build_dashboard_data(
            model=model,
            tokenizer=tokenizer,
            classifier=classifier,
            class_text_embeddings=class_text_embeddings,
            classnames=classnames,
            templates=templates,
            sample=sample,
            matrix_data=matrix_data,
            embedding_rows=embedding_rows,
            embedding_points=embedding_points,
            embedding_images=embedding_images,
            device=device,
        )
        output_path = output_dir / f"{safe_stem(sample.image_path)}_clip_dashboard.png"
        saved_paths = plot_dashboard(data, classnames, output_path, dpi=args.dpi)
        for saved_path in saved_paths:
            print(f"Saved dashboard page: {saved_path.resolve()}")


def load_samples(jsonl_path: str | None, image_root: str) -> list[Sample]:
    if not jsonl_path:
        return []
    root = Path(image_root)
    samples: list[Sample] = []
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
            label = str(item.get("class_name") or item.get("label") or "").replace("_", " ").strip()
            caption = str(item.get("caption") or item.get("text") or item.get("title") or "")
            samples.append(Sample(image_path=image_path, ground_truth=label, caption=caption))
    return samples


def explicit_samples(args: argparse.Namespace, classnames: list[str]) -> list[Sample]:
    labels = args.ground_truth or []
    samples = []
    for index, image_value in enumerate(args.image or []):
        image_path = Path(image_value)
        label = labels[index] if index < len(labels) else infer_class_from_path(image_path, classnames)
        samples.append(Sample(image_path=image_path, ground_truth=label))
    return samples


def first_present(item: dict[str, Any], keys: Iterable[str]) -> Any | None:
    for key in keys:
        if key in item:
            return item[key]
    return None


def infer_class_from_path(image_path: Path, classnames: list[str]) -> str:
    parent = image_path.parent.name.replace("_", " ").replace("-", " ").strip()
    for class_name in classnames:
        if class_name.lower() == parent.lower() or class_name.lower().rstrip("s") == parent.lower().rstrip("s"):
            return class_name
    return parent


def stratified_samples(samples: list[Sample], limit: int) -> list[Sample]:
    if len(samples) <= limit:
        return samples
    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.ground_truth, []).append(sample)
    selected: list[Sample] = []
    class_names = sorted(grouped)
    cursor = 0
    while len(selected) < limit:
        class_name = class_names[cursor % len(class_names)]
        bucket = grouped[class_name]
        used = sum(1 for sample in selected if sample.ground_truth == class_name)
        if used < len(bucket):
            selected.append(bucket[used])
        if all(sum(1 for sample in selected if sample.ground_truth == name) >= len(bucket) for name, bucket in grouped.items()):
            break
        cursor += 1
    return selected[:limit]


def load_image_rgb(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.asarray(image)


def preprocess_image(image: np.ndarray, image_size: int, device: str) -> tuple[np.ndarray, torch.Tensor]:
    resized = resize_center_crop(image, image_size)
    array = resized.astype(np.float32) / 255.0
    normalized = (array - np.asarray(CLIP_MEAN, dtype=np.float32)) / np.asarray(CLIP_STD, dtype=np.float32)
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(device)
    return resized, tensor


def resize_center_crop(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = size / min(width, height)
    resized = resize_rgb(image, (round(width * scale), round(height * scale)), cubic=True)
    top = max(0, (resized.shape[0] - size) // 2)
    left = max(0, (resized.shape[1] - size) // 2)
    return resized[top : top + size, left : left + size]


def resize_rgb(image: np.ndarray, size: tuple[int, int], cubic: bool = False) -> np.ndarray:
    if cv2 is not None:
        interpolation = cv2.INTER_CUBIC if cubic else cv2.INTER_AREA
        return cv2.resize(image, size, interpolation=interpolation)
    pil_mode = Image.Resampling.BICUBIC if cubic else Image.Resampling.LANCZOS
    return np.asarray(Image.fromarray(image.astype(np.uint8)).resize(size, pil_mode))


def model_layer_names(model: torch.nn.Module) -> list[str]:
    modules = dict(model.named_modules())
    names = [name for _label, name in VISION_LAYERS + TEXT_LAYERS if name in modules]
    return names


def register_hooks(model: torch.nn.Module, layer_names: list[str]) -> tuple[dict[str, torch.Tensor], list[Any]]:
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
def encode_with_activations(
    model: torch.nn.Module,
    tokenizer: ByteTokenizer,
    image_tensor: torch.Tensor,
    prompts: list[str],
) -> EncoderOutputs:
    activations, hooks = register_hooks(model, model_layer_names(model))
    token_ids = tokenizer(prompts).to(image_tensor.device)
    try:
        image_embedding = model.encode_image(image_tensor, normalize=True)
        text_embeddings = model.encode_text(token_ids, normalize=True)
    finally:
        for hook in hooks:
            hook.remove()
    return EncoderOutputs(
        image_embedding=image_embedding.detach().cpu(),
        text_embeddings=text_embeddings.detach().cpu(),
        activations=activations,
        token_ids=token_ids.detach().cpu(),
    )


@torch.no_grad()
def build_zero_shot_classifier(
    model: torch.nn.Module,
    tokenizer: ByteTokenizer,
    classnames: list[str],
    templates: list[str],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = []
    class_text_embeddings = []
    for class_name in classnames:
        prompts = [template.format(class_name) for template in templates]
        tokens = tokenizer(prompts).to(device)
        embeddings = model.encode_text(tokens, normalize=True)
        class_text_embeddings.append(embeddings.detach().cpu())
        weights.append(F.normalize(embeddings.mean(dim=0), dim=0))
    return torch.stack(weights, dim=1), torch.stack([item.mean(dim=0) for item in class_text_embeddings], dim=0)


def top_predictions(image_embedding: torch.Tensor, classifier: torch.Tensor, classnames: list[str], top_k: int = 5) -> list[Prediction]:
    similarity = image_embedding.to(classifier.device) @ classifier
    confidence = F.softmax(100.0 * similarity, dim=1)
    top_scores, indices = similarity.topk(min(top_k, len(classnames)), dim=1)
    rows = []
    for rank, (score, index) in enumerate(zip(top_scores[0].detach().cpu(), indices[0].detach().cpu()), start=1):
        idx = int(index)
        rows.append(
            Prediction(
                rank=rank,
                class_name=classnames[idx],
                similarity=float(score),
                confidence=float(confidence[0, idx].detach().cpu()),
                index=idx,
            )
        )
    return rows


@torch.no_grad()
def compute_similarity_matrix(
    model: torch.nn.Module,
    classifier: torch.Tensor,
    classnames: list[str],
    samples: list[Sample],
    device: str,
) -> dict[str, Any]:
    embeddings = []
    for sample in samples:
        image = load_image_rgb(sample.image_path)
        _, tensor = preprocess_image(image, model.input_resolution, device)
        embeddings.append(model.encode_image(tensor, normalize=True).detach().cpu())
    image_embeddings = torch.cat(embeddings, dim=0) if embeddings else torch.empty(0, classifier.shape[0])
    matrix = image_embeddings.to(classifier.device) @ classifier
    predictions = matrix.argmax(dim=1).detach().cpu().tolist() if len(samples) else []
    targets = [classnames.index(sample.ground_truth) if sample.ground_truth in classnames else -1 for sample in samples]
    return {
        "samples": samples,
        "matrix": matrix.detach().cpu().numpy(),
        "predictions": predictions,
        "targets": targets,
    }


@torch.no_grad()
def compute_embedding_panel_data(model: torch.nn.Module, samples: list[Sample], device: str) -> tuple[np.ndarray, list[np.ndarray]]:
    vectors = []
    images = []
    for sample in samples:
        image = load_image_rgb(sample.image_path)
        _, tensor = preprocess_image(image, model.input_resolution, device)
        vectors.append(model.encode_image(tensor, normalize=True).detach().cpu().numpy()[0])
        images.append(image)
    return np.asarray(vectors, dtype=np.float32), images


def reduce_embeddings(vectors: np.ndarray, method: str) -> np.ndarray:
    if len(vectors) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(vectors) == 1:
        return np.zeros((1, 2), dtype=np.float32)
    if method == "tsne" and len(vectors) >= 4:
        perplexity = min(30, max(2, len(vectors) // 3))
        return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42).fit_transform(vectors)
    return PCA(n_components=2, random_state=42).fit_transform(vectors)


def build_dashboard_data(
    model: torch.nn.Module,
    tokenizer: ByteTokenizer,
    classifier: torch.Tensor,
    class_text_embeddings: torch.Tensor,
    classnames: list[str],
    templates: list[str],
    sample: Sample,
    matrix_data: dict[str, Any],
    embedding_rows: list[Sample],
    embedding_points: np.ndarray,
    embedding_images: list[np.ndarray],
    device: str,
) -> DashboardData:
    image = load_image_rgb(sample.image_path)
    preprocessed, image_tensor = preprocess_image(image, model.input_resolution, device)
    prompt = sample.caption or templates[0].format(sample.ground_truth)
    outputs = encode_with_activations(model, tokenizer, image_tensor, [prompt])
    predictions = top_predictions(outputs.image_embedding, classifier, classnames)
    predicted_prompt = templates[0].format(predictions[0].class_name) if predictions else prompt
    predicted_text = tokenizer([predicted_prompt]).to(device)
    with torch.no_grad():
        predicted_embedding = model.encode_text(predicted_text, normalize=True).detach().cpu()
    cosine_similarity = float(outputs.image_embedding @ predicted_embedding.t())
    attention_maps = compute_attention_overlays(model, image_tensor, classifier, predictions, device)
    return DashboardData(
        sample=sample,
        original_image=image,
        preprocessed_image=preprocessed,
        image_tensor=image_tensor,
        prompt=prompt,
        outputs=outputs,
        predictions=predictions,
        similarity_matrix=matrix_data["matrix"],
        matrix_rows=matrix_data["samples"],
        matrix_predictions=matrix_data["predictions"],
        matrix_targets=matrix_data["targets"],
        embedding_rows=embedding_rows,
        embedding_points=embedding_points,
        embedding_images=embedding_images,
        attention_maps=attention_maps,
        cosine_similarity=cosine_similarity,
    )


def compute_attention_overlays(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    classifier: torch.Tensor,
    predictions: list[Prediction],
    device: str,
) -> dict[str, np.ndarray]:
    if not predictions:
        return {}
    target_index = predictions[0].index
    modules = dict(model.named_modules())
    layer_names = [layer for _label, layer in ATTENTION_LAYERS if layer in modules]
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
        image_embedding = model.encode_image(tensor, normalize=True)
        score = (image_embedding @ classifier[:, target_index].to(device)).sum()
        score.backward()
    finally:
        for hook in hooks:
            hook.remove()

    maps = {}
    for layer_name in layer_names:
        if layer_name in activations and layer_name in gradients:
            maps[layer_name] = gradcam_map(activations[layer_name], gradients[layer_name])
    return maps


def gradcam_map(activation: torch.Tensor, gradient: torch.Tensor) -> np.ndarray:
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
        cam = tokens_to_grid((act_tokens * grad_tokens).mean(dim=-1))
    else:
        cam = tokens_to_grid((act * grad).flatten())
    return normalize(F.relu(cam).numpy())


def activation_heatmap(activation: torch.Tensor) -> np.ndarray:
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


def normalize(values: np.ndarray) -> np.ndarray:
    array = values.astype(np.float32, copy=False)
    array = array - float(array.min())
    return array / (float(array.max()) + 1e-8)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap = normalize(heatmap)
    if cv2 is not None:
        resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        colored = cv2.applyColorMap(np.uint8(resized * 255), cv2.COLORMAP_VIRIDIS)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(image.astype(np.uint8), 1.0 - alpha, colored, alpha, 0)
    resized = np.asarray(Image.fromarray(np.uint8(heatmap * 255)).resize((image.shape[1], image.shape[0]), Image.Resampling.BICUBIC)) / 255.0
    colored = np.uint8(plt.cm.viridis(resized)[..., :3] * 255)
    blended = image.astype(np.float32) * (1.0 - alpha) + colored.astype(np.float32) * alpha
    return np.uint8(np.clip(blended, 0, 255))


def tensor_stats(tensor: torch.Tensor | np.ndarray) -> dict[str, str]:
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().float().cpu().numpy()
        shape = tuple(tensor.shape)
    else:
        arr = np.asarray(tensor, dtype=np.float32)
        shape = arr.shape
    return {
        "dim": "x".join(str(item) for item in shape),
        "mean": f"{float(arr.mean()):.4f}",
        "std": f"{float(arr.std()):.4f}",
        "min": f"{float(arr.min()):.4f}",
        "max": f"{float(arr.max()):.4f}",
    }


def readable_tokens(tokenizer: ByteTokenizer, prompt: str) -> list[str]:
    tokens = []
    for token_id in tokenizer.encode(prompt):
        if token_id == tokenizer.sos_token_id:
            tokens.append("<SOS>")
        elif token_id == tokenizer.eos_token_id:
            tokens.append("<EOS>")
        elif token_id >= tokenizer.byte_offset:
            char = bytes([token_id - tokenizer.byte_offset]).decode("utf-8", errors="replace")
            tokens.append(char if char.strip() else "<space>")
        else:
            tokens.append(str(token_id))
    return tokens


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#1f2937",
            "text.color": "#1f2937",
            "axes.edgecolor": "#cbd5e1",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )


def plot_dashboard(data: DashboardData, classnames: list[str], output_path: Path, dpi: int) -> list[Path]:
    setup_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = output_path.with_suffix("")
    page1_path = base.with_name(f"{base.name}_page1_encoders.png")
    page2_path = base.with_name(f"{base.name}_page2_decision.png")

    fig1 = plt.figure(figsize=(22, 18), dpi=dpi)
    page1 = gridspec.GridSpec(3, 1, figure=fig1, height_ratios=[1.1, 2.3, 2.1], hspace=0.55)
    fig1.suptitle("CLIP Interpretability Dashboard - Page 1: Input and Encoders", fontsize=24, fontweight="bold", y=0.985)
    plot_input_information(fig1, page1[0, 0], data)
    plot_vision_encoder(fig1, page1[1, 0], data)
    plot_text_encoder(fig1, page1[2, 0], data)
    fig1.savefig(page1_path, dpi=dpi, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(22, 20), dpi=dpi)
    page2 = gridspec.GridSpec(5, 2, figure=fig2, height_ratios=[1.15, 2.2, 2.25, 1.5, 1.2], hspace=0.58, wspace=0.28)
    fig2.suptitle("CLIP Interpretability Dashboard - Page 2: Similarity, Attention, and Decision", fontsize=24, fontweight="bold", y=0.985)
    plot_similarity(fig2, page2[0, 0], data)
    plot_final_decision(fig2, page2[0, 1], data)
    plot_embedding_visualization(fig2, page2[1, :], data)
    plot_correlation_heatmap(fig2, page2[2, :], data, classnames)
    plot_attention_maps(fig2, page2[3, :], data)
    plot_pipeline_summary(fig2, page2[4, :], data)
    fig2.savefig(page2_path, dpi=dpi, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig2)

    return [page1_path, page2_path]


def panel_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=10)


def clean_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_input_information(fig: plt.Figure, spec: gridspec.SubplotSpec, data: DashboardData) -> None:
    sub = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, width_ratios=[1.05, 1.35, 1.2], wspace=0.25)
    ax_img = fig.add_subplot(sub[0, 0])
    ax_info = fig.add_subplot(sub[0, 1])
    ax_pre = fig.add_subplot(sub[0, 2])
    panel_title(ax_img, "Section 1: Input Information")
    ax_img.imshow(data.original_image)
    ax_img.set_xlabel("Original RGB image", fontsize=11)
    clean_axis(ax_img)
    confidence = data.predictions[0].confidence if data.predictions else 0.0
    pred = data.predictions[0].class_name if data.predictions else "N/A"
    info = [
        ("Filename", data.sample.image_path.name),
        ("Ground Truth", data.sample.ground_truth),
        ("Predicted", pred),
        ("Prediction Confidence", f"{confidence * 100:.2f}%"),
        ("Image Resolution", f"{data.original_image.shape[1]} x {data.original_image.shape[0]} px"),
        ("Image Size", f"{data.sample.image_path.stat().st_size / 1024:.1f} KB"),
    ]
    ax_info.axis("off")
    panel_title(ax_info, "Metadata")
    y = 0.88
    for label, value in info:
        ax_info.text(0.02, y, label, fontsize=12, color="#64748b", transform=ax_info.transAxes)
        ax_info.text(0.40, y, value, fontsize=12, fontweight="bold", transform=ax_info.transAxes)
        y -= 0.13
    ax_pre.imshow(data.preprocessed_image)
    ax_pre.set_xlabel("Preprocessed input: resize + center crop + normalize", fontsize=11)
    panel_title(ax_pre, "224 x 224 Model Input")
    clean_axis(ax_pre)


def plot_vision_encoder(fig: plt.Figure, spec: gridspec.SubplotSpec, data: DashboardData) -> None:
    sub = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=spec, hspace=0.42, wspace=0.22)
    stage_items = [("Original Image", None), *VISION_LAYERS, ("Final Image Embedding", None)]
    for index, (label, layer_name) in enumerate(stage_items[:10]):
        ax = fig.add_subplot(sub[index // 5, index % 5])
        if index == 0:
            panel_title(ax, "Section 2: Vision Encoder")
        if layer_name is None and label == "Original Image":
            ax.imshow(data.original_image)
            stats = tensor_stats(data.preprocessed_image)
        elif layer_name is None:
            vec = data.outputs.image_embedding[0].numpy().reshape(1, -1)
            ax.imshow(vec, aspect="auto", cmap="viridis")
            stats = tensor_stats(data.outputs.image_embedding)
        else:
            activation = data.outputs.activations.get(layer_name)
            if activation is not None:
                heat = activation_heatmap(activation)
                if heat.shape[0] > 1:
                    ax.imshow(overlay_heatmap(data.preprocessed_image, heat))
                else:
                    ax.imshow(heat, aspect="auto", cmap="viridis")
                stats = tensor_stats(activation)
            else:
                ax.imshow(data.preprocessed_image)
                stats = {"dim": "N/A", "mean": "", "std": "", "min": "", "max": ""}
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.text(
            0.02,
            0.02,
            f"dim: {stats['dim']}\nmean {stats['mean']} | std {stats['std']}\nmin {stats['min']} | max {stats['max']}",
            transform=ax.transAxes,
            fontsize=7.5,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cbd5e1", alpha=0.88),
        )
        clean_axis(ax)


def plot_text_encoder(fig: plt.Figure, spec: gridspec.SubplotSpec, data: DashboardData) -> None:
    sub = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=spec, hspace=0.45, wspace=0.25)
    tokenizer = ByteTokenizer(context_length=data.outputs.token_ids.shape[1])
    tokens = readable_tokens(tokenizer, data.prompt)
    stages = [("Tokenization", None), *TEXT_LAYERS, ("Final Text Embedding", None)]
    for index, (label, layer_name) in enumerate(stages[:8]):
        ax = fig.add_subplot(sub[index // 4, index % 4])
        if index == 0:
            panel_title(ax, "Section 3: Text Encoder")
        if label == "Tokenization":
            ax.axis("off")
            ax.text(0.0, 0.92, f'Prompt: "{data.prompt}"', fontsize=10, fontweight="bold", transform=ax.transAxes)
            token_text = "  ".join(tokens[:55]) + (" ..." if len(tokens) > 55 else "")
            ax.text(0.0, 0.70, token_text, fontsize=8, wrap=True, transform=ax.transAxes)
            stats = {"dim": f"{len(tokens)} tokens", "mean": "", "std": "", "min": "", "max": ""}
        elif layer_name is None:
            ax.imshow(data.outputs.text_embeddings[0].numpy().reshape(1, -1), aspect="auto", cmap="magma")
            stats = tensor_stats(data.outputs.text_embeddings)
        else:
            activation = data.outputs.activations.get(layer_name)
            if activation is not None:
                ax.imshow(activation_heatmap(activation), aspect="auto", cmap="magma")
                stats = tensor_stats(activation)
            else:
                ax.text(0.5, 0.5, "Not captured", ha="center", va="center")
                stats = {"dim": "N/A", "mean": "", "std": "", "min": "", "max": ""}
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.text(
            0.02,
            0.02,
            f"dim: {stats['dim']}\nmean {stats['mean']} | std {stats['std']}\nmin {stats['min']} | max {stats['max']}",
            transform=ax.transAxes,
            fontsize=7.5,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cbd5e1", alpha=0.88),
        )
        clean_axis(ax)


def plot_similarity(fig: plt.Figure, spec: gridspec.SubplotSpec, data: DashboardData) -> None:
    ax = fig.add_subplot(spec)
    panel_title(ax, "Section 4: Image vs Text Similarity")
    predictions = sorted(data.predictions, key=lambda item: item.similarity)
    labels = [item.class_name for item in predictions]
    values = [item.similarity for item in predictions]
    colors = ["#16a34a" if item.rank == 1 else "#2563eb" for item in predictions]
    ax.barh(labels, values, color=colors, alpha=0.88)
    ax.set_xlabel("Cosine similarity")
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    for y, item in enumerate(predictions):
        ax.text(item.similarity + 0.01, y, f"{item.similarity:.3f}  ({item.confidence * 100:.1f}%)", va="center", fontsize=9)


def plot_final_decision(fig: plt.Figure, spec: gridspec.SubplotSpec, data: DashboardData) -> None:
    ax = fig.add_subplot(spec)
    panel_title(ax, "Section 8: Final Decision")
    ax.axis("off")
    pred = data.predictions[0] if data.predictions else None
    predicted = pred.class_name if pred else "N/A"
    correct = predicted == data.sample.ground_truth
    status = "Correct" if correct else "Incorrect"
    status_color = "#16a34a" if correct else "#dc2626"
    rows = [
        ("Ground Truth", data.sample.ground_truth),
        ("Predicted Class", predicted),
        ("Confidence", f"{pred.confidence * 100:.2f}%" if pred else "N/A"),
        ("Correct / Incorrect", status),
        ("Cosine Similarity", f"{data.cosine_similarity:.4f}"),
        ("Top-5 Classes", ", ".join(item.class_name for item in data.predictions)),
        ("Confusion Information", "Prediction matches ground truth" if correct else f"Confused with {predicted}"),
    ]
    y = 0.88
    for label, value in rows:
        ax.text(0.02, y, label, fontsize=11, color="#64748b", transform=ax.transAxes)
        wrapped_value = "\n".join(textwrap.wrap(value, width=48)) if len(value) > 48 else value
        ax.text(
            0.38,
            y,
            wrapped_value,
            fontsize=10.5,
            fontweight="bold",
            color=status_color if label == "Correct / Incorrect" else "#1f2937",
            transform=ax.transAxes,
            va="top",
        )
        y -= 0.12 + 0.055 * max(0, wrapped_value.count("\n"))


def plot_embedding_visualization(fig: plt.Figure, spec: gridspec.SubplotSpec, data: DashboardData) -> None:
    ax = fig.add_subplot(spec)
    panel_title(ax, "Section 5: Embedding Visualization")
    points = data.embedding_points
    if len(points) == 0:
        ax.text(0.5, 0.5, "No embedding samples available", ha="center", va="center")
        return
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(color="#e2e8f0", linewidth=0.7)
    by_class: dict[str, list[np.ndarray]] = {}
    for sample, point in zip(data.embedding_rows, points):
        by_class.setdefault(sample.ground_truth, []).append(point)
    palette = plt.cm.tab20(np.linspace(0, 1, max(1, len(by_class))))
    for color, (class_name, class_points) in zip(palette, sorted(by_class.items())):
        class_arr = np.asarray(class_points)
        ax.scatter(class_arr[:, 0], class_arr[:, 1], s=18, color=color, alpha=0.25, label=class_name)
        if len(class_arr) >= 2:
            center = class_arr.mean(axis=0)
            radius_x = max(0.05, class_arr[:, 0].std() * 2.2)
            radius_y = max(0.05, class_arr[:, 1].std() * 2.2)
            ellipse = Ellipse(center, radius_x * 2, radius_y * 2, fill=False, edgecolor=color, linewidth=1.4)
            ax.add_patch(ellipse)
    for image, point in zip(data.embedding_images, points):
        thumbnail = resize_rgb(image, (42, 42))
        ab = AnnotationBbox(OffsetImage(thumbnail, zoom=0.8), point, frameon=True, pad=0.05)
        ax.add_artist(ab)
    if len(by_class) <= 12:
        ax.legend(loc="upper right", fontsize=7, frameon=True)


def plot_correlation_heatmap(fig: plt.Figure, spec: gridspec.SubplotSpec, data: DashboardData, classnames: list[str]) -> None:
    sub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[1.55, 3.45], wspace=0.08)
    ax_meta = fig.add_subplot(sub[0, 0])
    ax_heat = fig.add_subplot(sub[0, 1])
    panel_title(ax_meta, "Section 6: Correlation Heatmap")
    ax_meta.set_xlim(0, 1)
    ax_meta.set_ylim(len(data.matrix_rows), 0)
    ax_meta.axis("off")
    visible_classes = classnames[: min(len(classnames), data.similarity_matrix.shape[1])]
    matrix = data.similarity_matrix[:, : len(visible_classes)]
    for row_index, sample in enumerate(data.matrix_rows):
        image = resize_rgb(load_image_rgb(sample.image_path), (44, 44))
        ab = AnnotationBbox(OffsetImage(image, zoom=0.72), (0.07, row_index + 0.5), frameon=True, pad=0.04)
        ax_meta.add_artist(ab)
        pred_index = data.matrix_predictions[row_index]
        pred = classnames[pred_index] if 0 <= pred_index < len(classnames) else "N/A"
        correct = pred_index == data.matrix_targets[row_index]
        color = "#16a34a" if correct else "#dc2626"
        ax_meta.text(0.16, row_index + 0.28, sample.image_path.name, fontsize=7.5, fontweight="bold")
        ax_meta.text(0.16, row_index + 0.52, f"GT: {sample.ground_truth}", fontsize=6.8, color="#64748b")
        ax_meta.text(0.16, row_index + 0.76, f"Pred: {pred}", fontsize=6.8, color=color)
    im = ax_heat.imshow(matrix, aspect="auto", cmap="viridis")
    ax_heat.set_xticks(range(len(visible_classes)))
    ax_heat.set_xticklabels(visible_classes, rotation=60, ha="right", fontsize=7)
    ax_heat.set_yticks(range(len(data.matrix_rows)))
    ax_heat.set_yticklabels([""] * len(data.matrix_rows))
    ax_heat.set_xlabel("Text prompts / classes")
    ax_heat.set_ylabel("Images")
    for row in range(matrix.shape[0]):
        min_col = int(np.argmin(matrix[row]))
        max_col = int(np.argmax(matrix[row]))
        ax_heat.add_patch(Rectangle((max_col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="#16a34a", linewidth=2.2))
        ax_heat.add_patch(Rectangle((min_col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="#2563eb", linewidth=2.2))
        pred_col = data.matrix_predictions[row]
        correct = pred_col == data.matrix_targets[row]
        border_color = "#16a34a" if correct else "#dc2626"
        ax_heat.add_patch(Rectangle((-0.5, row - 0.5), matrix.shape[1], 1, fill=False, edgecolor=border_color, linewidth=1.2))
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.01)
    cbar.set_label("Cosine similarity")


def plot_attention_maps(fig: plt.Figure, spec: gridspec.SubplotSpec, data: DashboardData) -> None:
    sub = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=spec, wspace=0.18)
    stages = [("Original Image", None), *ATTENTION_LAYERS]
    for index, (label, layer_name) in enumerate(stages):
        ax = fig.add_subplot(sub[0, index])
        if index == 0:
            panel_title(ax, "Section 7: Attention Map")
        if layer_name is None:
            ax.imshow(data.original_image)
        else:
            heat = data.attention_maps.get(layer_name)
            if heat is None and layer_name in data.outputs.activations:
                heat = activation_heatmap(data.outputs.activations[layer_name])
            ax.imshow(overlay_heatmap(data.preprocessed_image, heat) if heat is not None else data.preprocessed_image)
        ax.set_title(label, fontsize=10, fontweight="bold")
        clean_axis(ax)


def plot_pipeline_summary(fig: plt.Figure, spec: gridspec.SubplotSpec, data: DashboardData) -> None:
    ax = fig.add_subplot(spec)
    panel_title(ax, "Complete CLIP Decision Flow")
    ax.axis("off")
    pred = data.predictions[0].class_name if data.predictions else "N/A"
    items = [
        ("Original Image", "Vision Transformer", "Image Embedding (512D)"),
        ("Text Prompt", "Text Transformer", "Text Embedding (512D)"),
    ]
    y_values = [0.68, 0.30]
    for y, row in zip(y_values, items):
        x_positions = [0.06, 0.30, 0.56]
        for x, text in zip(x_positions, row):
            ax.text(x, y, text, fontsize=12, fontweight="bold", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.45", fc="#f8fafc", ec="#cbd5e1"))
        ax.annotate("", xy=(0.22, y), xytext=(0.14, y), arrowprops=dict(arrowstyle="->", lw=1.6, color="#64748b"))
        ax.annotate("", xy=(0.48, y), xytext=(0.38, y), arrowprops=dict(arrowstyle="->", lw=1.6, color="#64748b"))
    ax.text(0.76, 0.49, "Cosine\nSimilarity", fontsize=12, fontweight="bold", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.55", fc="#f8fafc", ec="#cbd5e1"))
    ax.text(0.93, 0.49, f"Predicted\n{pred}", fontsize=12, fontweight="bold", color="#16a34a", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.55", fc="#f0fdf4", ec="#16a34a"))
    ax.annotate("", xy=(0.72, 0.52), xytext=(0.62, 0.66), arrowprops=dict(arrowstyle="->", lw=1.6, color="#64748b"))
    ax.annotate("", xy=(0.72, 0.46), xytext=(0.62, 0.31), arrowprops=dict(arrowstyle="->", lw=1.6, color="#64748b"))
    ax.annotate("", xy=(0.88, 0.49), xytext=(0.80, 0.49), arrowprops=dict(arrowstyle="->", lw=1.6, color="#64748b"))


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_").replace(".", "_")


if __name__ == "__main__":
    main()
