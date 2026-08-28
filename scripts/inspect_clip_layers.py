from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clip_implement.data import image_transform, read_classnames
from clip_implement.model import build_model
from clip_implement.tokenizer import ByteTokenizer


DEFAULT_TEXT = "a satellite photo of agricultural land."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect CLIP layer activations and parameters for research figures")
    parser.add_argument("--checkpoint", required=True, help="Path to CLIP checkpoint .pt file")
    parser.add_argument("--image", required=True, help="Path to one image to inspect")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text prompt to inspect with the text encoder")
    parser.add_argument("--output-dir", default="layer_outputs/sample", help="Directory where PNG/CSV outputs are written")
    parser.add_argument("--model", default=None, choices=["RN50", "RN101", "ViT-B-32", "ViT-B-16"], help="Model name. Defaults to checkpoint args.")
    parser.add_argument("--classnames", default=None, help="Optional classnames.txt for zero-shot prediction")
    parser.add_argument("--template", action="append", default=None, help="Optional zero-shot template. Use {} for class name.")
    parser.add_argument("--layers", nargs="*", default=None, help="Specific module names to inspect. Defaults to useful visual/text layers.")
    parser.add_argument("--device", default=None, help="cuda, cpu, etc. Defaults to cuda when available.")
    parser.add_argument("--context-length", type=int, default=76)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    model_name = args.model or checkpoint_args.get("model", "ViT-B-32")

    tokenizer = ByteTokenizer(context_length=args.context_length)
    model = build_model(model_name, vocab_size=tokenizer.vocab_size, context_length=tokenizer.context_length)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()

    original_image, image_tensor = load_image(args.image, model.input_resolution, device)
    text_tokens = tokenizer([args.text]).to(device)
    original_image.save(output_dir / "input_image.png")

    layer_names = args.layers or default_layer_names(model, model_name)
    activations, hooks = register_hooks(model, layer_names)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor, normalize=True)
        text_features = model.encode_text(text_tokens, normalize=True)

    for hook in hooks:
        hook.remove()

    write_parameter_summary(model, output_dir / "parameter_summary.csv")
    write_module_parameter_summary(model, output_dir / "module_parameter_summary.csv")
    write_activation_summary(activations, output_dir / "activation_summary.csv")
    write_embedding_summary(image_features, text_features, output_dir / "embedding_summary.csv")

    for layer_name, activation in activations.items():
        heatmap = activation_to_heatmap(activation)
        safe_name = safe_filename(layer_name)
        heatmap_image = heatmap_to_image(heatmap)
        heatmap_image.save(output_dir / f"{safe_name}_heatmap.png")
        overlay = overlay_heatmap(original_image, heatmap_image)
        overlay.save(output_dir / f"{safe_name}_overlay.png")

    if args.classnames:
        write_zero_shot_prediction(
            model=model,
            tokenizer=tokenizer,
            image_features=image_features,
            classnames=read_classnames(args.classnames) or [],
            templates=args.template or ["a satellite photo of {}.", "an aerial image of {}.", "a remote sensing image of {}."],
            output_csv=output_dir / "zero_shot_prediction.csv",
            device=device,
        )

    print(f"Saved inspection outputs to: {output_dir.resolve()}")
    print(f"Inspected layers: {len(activations)}")
    print(f"Image embedding shape: {tuple(image_features.shape)}")
    print(f"Text embedding shape: {tuple(text_features.shape)}")


def load_image(image_path: str, image_size: int, device: str) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    transform = image_transform(image_size, train=False)
    tensor = transform(image).unsqueeze(0).to(device)
    return image, tensor


def default_layer_names(model: torch.nn.Module, model_name: str) -> list[str]:
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
    missing = [name for name in layer_names if name not in modules]
    if missing:
        raise ValueError(f"Unknown layer names: {missing}")

    for layer_name in layer_names:
        module = modules[layer_name]

        def hook_fn(_module, _inputs, output, name=layer_name):
            if isinstance(output, tuple):
                output = output[0]
            activations[name] = output.detach().float().cpu()

        hooks.append(module.register_forward_hook(hook_fn))
    return activations, hooks


def activation_to_heatmap(activation: torch.Tensor) -> np.ndarray:
    tensor = activation.detach().float().cpu()

    if tensor.ndim == 4:
        # CNN feature map: [batch, channels, height, width]
        heatmap = tensor[0].abs().mean(dim=0)
    elif tensor.ndim == 3:
        if tensor.shape[0] == 1:
            # Token sequence: [batch, tokens, dim]
            tokens = tensor[0]
        elif tensor.shape[1] == 1:
            # Transformer sequence: [tokens, batch, dim]
            tokens = tensor[:, 0, :]
        else:
            tokens = tensor[0]
        heatmap = tokens_to_heatmap(tokens)
    elif tensor.ndim == 2:
        heatmap = tokens_to_heatmap(tensor)
    else:
        heatmap = tensor.flatten().abs().unsqueeze(0)

    array = heatmap.numpy()
    array = array - float(array.min())
    array = array / (float(array.max()) + 1e-8)
    return array


def tokens_to_heatmap(tokens: torch.Tensor) -> torch.Tensor:
    values = tokens.abs().mean(dim=-1)
    if values.numel() > 1:
        without_cls = values[1:]
        grid_size = int(math.sqrt(without_cls.numel()))
        if grid_size * grid_size == without_cls.numel():
            return without_cls.reshape(grid_size, grid_size)
    grid_size = int(math.sqrt(values.numel()))
    if grid_size * grid_size == values.numel():
        return values.reshape(grid_size, grid_size)
    return values.unsqueeze(0)


def heatmap_to_image(heatmap: np.ndarray) -> Image.Image:
    heatmap_uint8 = np.uint8(np.clip(heatmap, 0.0, 1.0) * 255)
    red = heatmap_uint8
    green = np.uint8(255 - np.abs(heatmap_uint8.astype(np.int16) - 128) * 2)
    blue = 255 - heatmap_uint8
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def overlay_heatmap(image: Image.Image, heatmap_image: Image.Image) -> Image.Image:
    base = image.convert("RGB").resize((512, 512))
    heatmap = heatmap_image.resize(base.size, Image.Resampling.BICUBIC)
    return Image.blend(base, heatmap, alpha=0.45)


def write_parameter_summary(model: torch.nn.Module, output_csv: Path) -> None:
    rows = []
    for name, param in model.named_parameters():
        values = param.detach().float().cpu()
        rows.append(
            {
                "parameter": name,
                "shape": "x".join(str(size) for size in values.shape),
                "num_parameters": int(values.numel()),
                "requires_grad": bool(param.requires_grad),
                "mean": f"{float(values.mean()):.8f}",
                "std": f"{float(values.std(unbiased=False)):.8f}",
                "min": f"{float(values.min()):.8f}",
                "max": f"{float(values.max()):.8f}",
            }
        )
    write_csv(output_csv, rows)


def write_module_parameter_summary(model: torch.nn.Module, output_csv: Path) -> None:
    rows = []
    for module_name, module in model.named_modules():
        direct_params = list(module.parameters(recurse=False))
        if not direct_params:
            continue
        total = sum(param.numel() for param in direct_params)
        trainable = sum(param.numel() for param in direct_params if param.requires_grad)
        rows.append(
            {
                "module": module_name or "<root>",
                "module_type": module.__class__.__name__,
                "direct_parameters": int(total),
                "trainable_direct_parameters": int(trainable),
            }
        )
    write_csv(output_csv, rows)


def write_activation_summary(activations: dict[str, torch.Tensor], output_csv: Path) -> None:
    rows = []
    for name, tensor in activations.items():
        values = tensor.detach().float().cpu()
        rows.append(
            {
                "layer": name,
                "shape": "x".join(str(size) for size in values.shape),
                "mean": f"{float(values.mean()):.8f}",
                "std": f"{float(values.std(unbiased=False)):.8f}",
                "min": f"{float(values.min()):.8f}",
                "max": f"{float(values.max()):.8f}",
                "abs_mean": f"{float(values.abs().mean()):.8f}",
            }
        )
    write_csv(output_csv, rows)


def write_embedding_summary(image_features: torch.Tensor, text_features: torch.Tensor, output_csv: Path) -> None:
    similarity = image_features @ text_features.t()
    rows = [
        {
            "item": "image_embedding",
            "shape": "x".join(str(size) for size in image_features.shape),
            "l2_norm": f"{float(image_features.norm(dim=-1).mean()):.8f}",
            "mean": f"{float(image_features.mean()):.8f}",
            "std": f"{float(image_features.std(unbiased=False)):.8f}",
        },
        {
            "item": "text_embedding",
            "shape": "x".join(str(size) for size in text_features.shape),
            "l2_norm": f"{float(text_features.norm(dim=-1).mean()):.8f}",
            "mean": f"{float(text_features.mean()):.8f}",
            "std": f"{float(text_features.std(unbiased=False)):.8f}",
        },
        {
            "item": "image_text_cosine_similarity",
            "shape": "1x1",
            "l2_norm": "",
            "mean": f"{float(similarity.item()):.8f}",
            "std": "",
        },
    ]
    write_csv(output_csv, rows)


@torch.no_grad()
def write_zero_shot_prediction(
    model: torch.nn.Module,
    tokenizer: ByteTokenizer,
    image_features: torch.Tensor,
    classnames: list[str],
    templates: list[str],
    output_csv: Path,
    device: str,
) -> None:
    if not classnames:
        return
    weights = []
    for classname in classnames:
        texts = [template.format(classname) for template in templates]
        tokens = tokenizer(texts).to(device)
        text_features = model.encode_text(tokens, normalize=True)
        weights.append(torch.nn.functional.normalize(text_features.mean(dim=0), dim=0))
    classifier = torch.stack(weights, dim=1)
    logits = 100.0 * image_features @ classifier
    top_scores, top_indices = logits.topk(min(5, len(classnames)), dim=1)
    rows = []
    for rank, (index, score) in enumerate(zip(top_indices[0].detach().cpu(), top_scores[0].detach().cpu()), start=1):
        rows.append({"rank": rank, "class_name": classnames[int(index)], "logit": f"{float(score):.6f}"})
    write_csv(output_csv, rows)


def write_csv(output_csv: Path, rows: list[dict[str, Any]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_csv.write_text("", encoding="utf-8")
        return
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_filename(name: str) -> str:
    return name.replace(".", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")


if __name__ == "__main__":
    main()
