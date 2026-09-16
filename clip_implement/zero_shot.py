from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **_: x

if __package__ in {None, ""}:  # Allows: python clip_implement/zero_shot.py
    from data import ClassificationCollator, JsonlClassificationDataset, build_eval_dataset, image_transform, normalize_classnames, read_classnames
    from model import build_model
    from reporting import write_csv
    from tokenizer import ByteTokenizer
else:
    from .data import ClassificationCollator, JsonlClassificationDataset, build_eval_dataset, image_transform, normalize_classnames, read_classnames
    from .model import build_model
    from .reporting import write_csv
    from .tokenizer import ByteTokenizer


DEFAULT_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
]

SPECIAL_TEMPLATES = {
    "food_101": ["a photo of {}, a type of food."],
    "oxfordpets": ["a photo of {}, a type of pet."],
    "flowers_102": ["a photo of {}, a type of flower."],
    "fgvcaircraft": ["a photo of {}, a type of aircraft."],
    "aircraft": ["a photo of {}, a type of aircraft."],
    "eurosat": ["a centered satellite photo of {}.", "a satellite photo of {}."],
    "resisc45": ["a satellite photo of {}."],
    "mnist": ['a photo of the number "{}".'],
    "svhn": ['a photo of the number "{}".'],
    "gtsrb": ["a photo of a {} traffic sign."],
    "cifar_100": DEFAULT_TEMPLATES, 
    "caltech_101": DEFAULT_TEMPLATES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone zero-shot CLIP evaluation for the prepared datasets."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cifar_100", "eurosat", "flower_102", "flowers_102", "food_101", "caltech_101", "imagefolder"],
    )
    parser.add_argument(
        "--eval-jsonl",
        type=str,
        default=None,
        help="Prepared labeled test JSONL. Recommended for the dissertation datasets.",
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--image-root", type=str, default=".")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model", type=str, default=None, choices=["RN50", "RN101", "ViT-B-32", "ViT-B-16"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--classnames", type=str, default=None)
    parser.add_argument("--template", action="append", default=None, help="Prompt template. Use {} for class name.")
    parser.add_argument("--context-length", type=int, default=76)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-predictions", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def build_zero_shot_classifier(
    model,
    tokenizer: ByteTokenizer,
    classnames: Iterable[str],
    templates: Iterable[str],
    device: str,
) -> torch.Tensor:
    weights: List[torch.Tensor] = []
    for classname in classnames:
        texts = [template.format(classname.replace("_", " ")) for template in templates]
        tokenized = tokenizer(texts).to(device)
        embeddings = model.encode_text(tokenized, normalize=True)
        embedding = F.normalize(embeddings.mean(dim=0), dim=0)
        weights.append(embedding)
    return torch.stack(weights, dim=1)


def extract_labels(dataset, override_path: str | None) -> List[str]:
    override = read_classnames(override_path)
    if override:
        return override
    for attr in ("classes", "categories"):
        if hasattr(dataset, attr):
            names = normalize_classnames(getattr(dataset, attr))
            if names:
                return names
    if hasattr(dataset, "_labels"):
        labels = sorted(set(dataset._labels))
        return [str(label) for label in labels]
    raise ValueError("Could not infer class names. Provide --classnames.")


def dataset_reference(dataset, index: int) -> str:
    for attr in ("samples", "imgs"):
        values = getattr(dataset, attr, None)
        if values is not None and index < len(values):
            value = values[index]
            if isinstance(value, (tuple, list)) and value:
                return str(value[0])
            return str(value)
    return f"{dataset.__class__.__name__}[{index}]"


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    model_name = args.model or checkpoint_args.get("model", "RN50")

    tokenizer = ByteTokenizer(context_length=args.context_length)
    model = build_model(
        model_name,
        vocab_size=tokenizer.vocab_size,
        context_length=tokenizer.context_length,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    # Prefer the prepared test JSONL/classnames files so evaluation uses the
    # exact same split and class-index mapping created for training.
    if args.eval_jsonl:
        classnames = read_classnames(args.classnames)
        if not classnames:
            candidate = Path(args.eval_jsonl).with_name("classnames.txt")
            classnames = read_classnames(str(candidate))
        if not classnames:
            raise ValueError(
                "For --eval-jsonl, provide --classnames or keep classnames.txt "
                "beside the JSONL file."
            )

        dataset = JsonlClassificationDataset(
            args.eval_jsonl,
            image_root=args.image_root,
            classnames=classnames,
            transform=image_transform(model.input_resolution, train=False),
        )
    else:
        dataset = build_eval_dataset(
            args.dataset,
            args.data_root,
            args.split,
            model.input_resolution,
            args.download,
        )
        classnames = extract_labels(dataset, args.classnames)

    templates = args.template or SPECIAL_TEMPLATES.get(
        args.dataset.lower(),
        DEFAULT_TEMPLATES,
    )
    classifier = build_zero_shot_classifier(
        model,
        tokenizer,
        classnames,
        templates,
        device,
    )

    is_jsonl_dataset = isinstance(dataset, JsonlClassificationDataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device == "cuda",
        collate_fn=ClassificationCollator() if is_jsonl_dataset else None,
    )

    correct1 = 0
    correct5 = 0
    total = 0
    prediction_rows: list[dict[str, str]] = []
    dataset_offset = 0

    for batch in tqdm(loader, desc=f"zero-shot {args.dataset}", disable=True):
        if is_jsonl_dataset:
            images, targets, _paths, _captions, _class_names = batch
        else:
            images, targets = batch
        images = images.to(device, non_blocking=True)
        targets = torch.as_tensor(targets, device=device)

        image_features = model.encode_image(images, normalize=True)
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

                if isinstance(dataset, JsonlClassificationDataset):
                    input_path = str(dataset.rows[sample_index]["image_path"])
                else:
                    input_path = dataset_reference(dataset, sample_index)

                prediction_rows.append(
                    {
                        "sample_index": str(sample_index),
                        "input": input_path,
                        "target_index": str(target_index),
                        "true_class": classnames[target_index] if 0 <= target_index < len(classnames) else str(target_index),
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
        else Path(args.checkpoint).resolve().parent / "zero_shot"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "summary.csv",
        [{
            "dataset": args.dataset,
            "split": args.split,
            "samples": total,
            "classes": len(classnames),
            "top1": f"{top1:.2f}",
            topk_name: f"{topk:.2f}",
            "checkpoint": args.checkpoint,
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
    print(f"Checkpoint    : {args.checkpoint}", flush=True)
    print(f"Samples       : {total}", flush=True)
    print(f"Classes       : {len(classnames)}", flush=True)
    print(f"Top-1         : {top1:.2f}%", flush=True)
    print(f"{topk_name:<14}: {topk:.2f}%", flush=True)
    print(f"Output        : {output_dir.resolve()}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()