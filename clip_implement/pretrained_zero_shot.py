"""
Zero-shot CLIP evaluation using REAL pretrained weights via open_clip.

This does NOT use clip_implement's custom ByteTokenizer / build_model.
It loads an actual pretrained CLIP checkpoint (OpenAI's original weights,
or a LAION-trained open_clip checkpoint) and evaluates it zero-shot on
your prepared datasets, using the same JsonlClassificationDataset /
val.jsonl / classnames.txt files you already created.

No training happens here. This is purely to establish what real
pretrained-CLIP zero-shot accuracy looks like on your data, as a
baseline to compare your from-scratch training results against.

Install requirement (one-time):
    pip install open_clip_torch --break-system-packages


Swap --pretrained to a LAION tag (e.g. laion2b_s34b_b79k) to compare
a different pretraining source. Run `python -c "import open_clip;
print(open_clip.list_pretrained())"` to see all valid (model, tag) pairs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import open_clip
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "open_clip_torch is not installed. Run:\n"
        "    pip install open_clip_torch --break-system-packages"
    ) from exc

if __package__ in {None, ""}:
    from data import ClassificationCollator, JsonlClassificationDataset, read_classnames
    from reporting import write_csv
else:
    from .data import ClassificationCollator, JsonlClassificationDataset, read_classnames
    from .reporting import write_csv


DEFAULT_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
]

SPECIAL_TEMPLATES = {
    "food_101": ["a photo of {}, a type of food."],
    "flower_102": ["a photo of {}, a type of flower."],
    "eurosat": ["a centered satellite photo of {}.", "a satellite photo of {}."],
    "cifar_100": DEFAULT_TEMPLATES,
    "caltech_101": DEFAULT_TEMPLATES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation using real pretrained CLIP weights (open_clip)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="open_clip model architecture name, e.g. ViT-B-32, ViT-B-16, RN50",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="openai",
        help="Pretrained weights tag, e.g. 'openai' or a LAION tag like 'laion2b_s34b_b79k'",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cifar_100", "eurosat", "flower_102", "food_101", "caltech_101"],
    )
    parser.add_argument("--eval-jsonl", type=str, required=True)
    parser.add_argument("--image-root", type=str, default=".")
    parser.add_argument("--classnames", type=str, default=None)
    parser.add_argument("--template", action="append", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-predictions", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def build_zero_shot_classifier(
    model,
    tokenizer,
    classnames: Iterable[str],
    templates: Iterable[str],
    device: str,
) -> torch.Tensor:
    weights: List[torch.Tensor] = []
    for classname in classnames:
        texts = [template.format(classname.replace("_", " ")) for template in templates]
        tokenized = tokenizer(texts).to(device)
        embeddings = model.encode_text(tokenized)
        embeddings = F.normalize(embeddings, dim=-1)
        embedding = F.normalize(embeddings.mean(dim=0), dim=0)
        weights.append(embedding)
    return torch.stack(weights, dim=1)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading pretrained {args.model} ({args.pretrained})...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)
    model.eval()

    classnames = read_classnames(args.classnames)
    if not classnames:
        candidate = Path(args.eval_jsonl).with_name("classnames.txt")
        classnames = read_classnames(str(candidate))
    if not classnames:
        raise ValueError(
            "Provide --classnames or keep classnames.txt beside the eval-jsonl file."
        )

    dataset = JsonlClassificationDataset(
        args.eval_jsonl,
        image_root=args.image_root,
        classnames=classnames,
        transform=preprocess,
    )

    templates = args.template or SPECIAL_TEMPLATES.get(args.dataset.lower(), DEFAULT_TEMPLATES)
    classifier = build_zero_shot_classifier(model, tokenizer, classnames, templates, device)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device == "cuda",
        collate_fn=ClassificationCollator(),
    )

    correct1 = 0
    correct5 = 0
    total = 0
    prediction_rows: list[dict[str, str]] = []
    dataset_offset = 0

    with torch.no_grad():
        for images, targets, _paths, _captions, _class_names in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            logits = 100.0 * image_features @ classifier

            max_k = min(5, logits.shape[1])
            top_scores, pred = logits.topk(max_k, dim=1)
            correct = pred.eq(targets.view(-1, 1))
            correct1 += int(correct[:, :1].sum())
            correct5 += int(correct[:, :max_k].sum())
            total += targets.numel()

            if not args.no_predictions:
                for row_index in range(images.shape[0]):
                    sample_index = dataset_offset + row_index
                    predicted_index = int(pred[row_index, 0].detach().cpu())
                    target_index = int(targets[row_index].detach().cpu())
                    top_names = [classnames[int(index)] for index in pred[row_index].detach().cpu()]
                    top_values = [f"{float(score):.4f}" for score in top_scores[row_index].detach().cpu()]
                    input_path = str(dataset.rows[sample_index]["image_path"])

                    prediction_rows.append(
                        {
                            "sample_index": str(sample_index),
                            "input": input_path,
                            "target_index": str(target_index),
                            "true_class": classnames[target_index],
                            "predicted_index": str(predicted_index),
                            "predicted_class": classnames[predicted_index],
                            "is_correct": str(predicted_index == target_index),
                            "top_classes": " | ".join(top_names),
                            "top_logits": " | ".join(top_values),
                            "final_output": classnames[predicted_index],
                        }
                    )

            dataset_offset += images.shape[0]

    top1 = 100.0 * correct1 / max(1, total)
    topk_name = f"top{min(5, len(classnames))}"
    topk = 100.0 * correct5 / max(1, total)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("checkpoints") / "pretrained_zero_shot" / args.dataset / f"{args.model}_{args.pretrained}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "summary.csv",
        [{
            "dataset": args.dataset,
            "model": args.model,
            "pretrained": args.pretrained,
            "samples": total,
            "classes": len(classnames),
            "top1": f"{top1:.2f}",
            topk_name: f"{topk:.2f}",
            "final_output": f"{args.dataset} top1={top1:.2f} {topk_name}={topk:.2f}",
        }],
    )

    if not args.no_predictions:
        write_csv(
            output_dir / "predictions.csv",
            prediction_rows,
            [
                "sample_index", "input", "target_index", "true_class",
                "predicted_index", "predicted_class", "is_correct",
                "top_classes", "top_logits", "final_output",
            ],
        )

    print("=" * 60, flush=True)
    print(f"Dataset       : {args.dataset}", flush=True)
    print(f"Model         : {args.model} ({args.pretrained})", flush=True)
    print(f"Samples       : {total}", flush=True)
    print(f"Classes       : {len(classnames)}", flush=True)
    print(f"Top-1         : {top1:.2f}%", flush=True)
    print(f"{topk_name:<14}: {topk:.2f}%", flush=True)
    print(f"Output        : {output_dir.resolve()}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()