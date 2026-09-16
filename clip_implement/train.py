from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader  

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency during import
    tqdm = lambda x, **_: x

if __package__ in {None, ""}:  # Allows: python clip_implement/train.py
    from data import ClassificationCollator, JsonlClassificationDataset, TokenizeCollator, build_pretrain_dataset, image_transform, read_classnames
    from model import build_model, clip_loss
    from reporting import write_csv
    from tokenizer import ByteTokenizer
else:
    from .data import ClassificationCollator, JsonlClassificationDataset, TokenizeCollator, build_pretrain_dataset, image_transform, read_classnames
    from .model import build_model, clip_loss
    from .reporting import write_csv
    from .tokenizer import ByteTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CLIP on a prepared image-text dataset with numerical validation."
    )
    parser.add_argument("--train-jsonl", type=str, default=None, help="WIT-style JSONL training manifest")
    parser.add_argument("--coco-annotations", type=str, default=None, help="COCO captions JSON annotation file")
    parser.add_argument("--image-root", type=str, default=".", help="Root directory for relative image paths")
    parser.add_argument("--model", type=str, default="RN50", choices=["RN50", "RN101", "ViT-B-32", "ViT-B-16"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--context-length", type=int, default=76)
    parser.add_argument("--precision", choices=["fp32", "amp"], default="amp")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--val-jsonl", type=str, default=None, help="Optional labeled JSONL for validation")
    parser.add_argument("--classnames", type=str, default=None, help="Class names file for validation")
    parser.add_argument("--template", action="append", default=None, help="Zero-shot prompt template. Use {} for class name.")
    parser.add_argument("--report-dir", type=str, default=None, help="Directory for numerical metrics.csv")
    parser.add_argument("--eval-every", type=int, default=1, help="Validate every N epochs")

    # DEPRECATED: accepted only so old commands do not break.
    # These options have no effect because epoch visual reporting is disabled.
    parser.add_argument("--report-samples", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--report-correlation-samples", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--report-image", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--report-class", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-report-predictions", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-visual-reports", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(".bias") or "bn" in name.lower() or "ln" in name.lower() or name == "logit_scale":
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-6,
    )


@torch.no_grad()
def build_zero_shot_classifier(model: nn.Module, tokenizer: ByteTokenizer, classnames: list[str], templates: list[str], device: str) -> torch.Tensor:
    weights = []
    model.eval()
    for classname in classnames:
        texts = [template.format(classname) for template in templates]
        tokenized = tokenizer(texts).to(device)
        features = model.encode_text(tokenized, normalize=True)
        weights.append(F.normalize(features.mean(dim=0), dim=0))
    return torch.stack(weights, dim=1)


@torch.no_grad()
def evaluate_zero_shot(
    model: nn.Module,
    loader: DataLoader,
    classifier: torch.Tensor,
    device: str,
) -> tuple[float, float]:
    """Numerical validation only: return zero-shot top-1 and top-5 accuracy."""
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0

    for images, targets, _paths, _captions, _true_classes in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        image_features = model.encode_image(images, normalize=True)
        logits = 100.0 * image_features @ classifier

        max_k = min(5, logits.shape[1])
        _, top_indices = logits.topk(max_k, dim=1)
        correct = top_indices.eq(targets.view(-1, 1))

        correct1 += int(correct[:, :1].sum())
        correct5 += int(correct[:, :max_k].sum())
        total += targets.numel()

    return (
        100.0 * correct1 / max(1, total),
        100.0 * correct5 / max(1, total),
    )


def append_metrics(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = ByteTokenizer(context_length=args.context_length)
    model = build_model(args.model, vocab_size=tokenizer.vocab_size, context_length=tokenizer.context_length).to(device)

    dataset = build_pretrain_dataset(
        train_jsonl=args.train_jsonl,
        image_root=args.image_root,
        image_size=model.input_resolution,
        coco_annotations=args.coco_annotations,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device == "cuda",
        drop_last=True,
        collate_fn=TokenizeCollator(tokenizer),
    )

    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))

    total_steps = max(1, len(loader) * args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.precision == "amp" and device == "cuda"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir) if args.report_dir else output_dir / "reports"
    metrics_path = report_dir / "metrics.csv"

    if args.val_jsonl is None and args.train_jsonl:
        candidate_val = Path(args.train_jsonl).with_name("val.jsonl")
        if candidate_val.exists():
            args.val_jsonl = str(candidate_val)
    if args.classnames is None and args.train_jsonl:
        candidate_classnames = Path(args.train_jsonl).with_name("classnames.txt")
        if candidate_classnames.exists():
            args.classnames = str(candidate_classnames)

    val_loader = None
    classnames = read_classnames(args.classnames)
    templates = args.template or ["a satellite photo of {}.", "an aerial image of {}.", "a remote sensing image of {}."]
    if args.val_jsonl and classnames:
        val_dataset = JsonlClassificationDataset(
            args.val_jsonl,
            image_root=args.image_root,
            classnames=classnames,
            transform=image_transform(model.input_resolution, train=False),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device == "cuda",
            collate_fn=ClassificationCollator(),
        )

    print(f"[INFO] Dataset size: {len(dataset)} samples", flush=True)
    print(f"[INFO] Batches per epoch: {len(loader)}", flush=True)
    print(f"[INFO] Saving checkpoints to: {output_dir.resolve()}", flush=True)
    print(f"[INFO] Saving numerical metrics to: {metrics_path.resolve()}", flush=True)
    if val_loader is not None:
        print(f"[INFO] Validation dataset size: {len(val_loader.dataset)} samples", flush=True)
        print(f"[INFO] Validation prompts: {templates}", flush=True)
    if len(loader) == 0:
        raise RuntimeError("No training batches were created. Reduce --batch-size or check the dataset manifest.")
    save_checkpoint(output_dir / "clip_init.pt", model, optimizer, start_epoch - 1, global_step, args)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        running_loss = 0.0
        retrieval_i2t_correct = 0
        retrieval_t2i_correct = 0
        retrieval_total = 0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", disable=True)
        for images, texts in progress:
            lr = cosine_lr(global_step, total_steps, args.lr, args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            images = images.to(device, non_blocking=True)
            texts = texts.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(args.precision == "amp" and device == "cuda")):
                logits_per_image, logits_per_text, logit_scale = model(images, texts)
                loss = clip_loss(logits_per_image, logits_per_text)
            labels = torch.arange(logits_per_image.shape[0], device=device)
            retrieval_i2t_correct += int((logits_per_image.argmax(dim=1) == labels).sum().detach())
            retrieval_t2i_correct += int((logits_per_text.argmax(dim=1) == labels).sum().detach())
            retrieval_total += labels.numel()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                model.logit_scale.clamp_(max=math.log(100))

            running_loss += float(loss.detach())
            global_step += 1
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(loss=running_loss / max(1, progress.n + 1), lr=lr, scale=float(logit_scale.detach()))
            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                save_checkpoint(output_dir / f"clip_step_{global_step}.pt", model, optimizer, epoch, global_step, args)
                save_checkpoint(output_dir / "clip_last.pt", model, optimizer, epoch, global_step, args)

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(output_dir / f"clip_epoch_{epoch + 1}.pt", model, optimizer, epoch, global_step, args)
        save_checkpoint(output_dir / "clip_last.pt", model, optimizer, epoch, global_step, args)
        print(f"[INFO] Saved checkpoint: {output_dir / 'clip_last.pt'}", flush=True)
        epoch_loss = running_loss / max(1, len(loader))
        train_i2t = 100.0 * retrieval_i2t_correct / max(1, retrieval_total)
        train_t2i = 100.0 * retrieval_t2i_correct / max(1, retrieval_total)
        val_top1 = ""
        val_top5 = ""
        if val_loader is not None and (epoch + 1) % max(1, args.eval_every) == 0:
            classifier = build_zero_shot_classifier(model, tokenizer, classnames, templates, device)
            val_top1_float, val_top5_float = evaluate_zero_shot(
                model,
                val_loader,
                classifier,
                device,
            )
            val_top1 = f"{val_top1_float:.2f}"
            val_top5 = f"{val_top5_float:.2f}"

            # DISABLED FOR CURRENT MULTI-DATASET RUN:
            # - epoch sample CSVs
            # - epoch prediction CSVs
            # - input preview PNGs
            # - text embedding heatmaps
            # - input/output correlation PNGs
            # Final detailed zero-shot predictions are produced separately by
            # zero_shot.py.
            model.train()
        append_metrics(
            metrics_path,
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": f"{epoch_loss:.6f}",
                "train_image_to_text_top1": f"{train_i2t:.2f}",
                "train_text_to_image_top1": f"{train_t2i:.2f}",
                "val_zero_shot_top1": val_top1,
                "val_zero_shot_top5": val_top5,
            },
        )
        print(
            "[EPOCH]"
            f" {epoch + 1}/{args.epochs}"
            f" loss={epoch_loss:.4f}"
            f" train_i2t={train_i2t:.2f}%"
            f" train_t2i={train_t2i:.2f}%"
            f" val_top1={val_top1 or 'NA'}"
            f" val_top5={val_top5 or 'NA'}",
            flush=True,
        )


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, global_step: int, args: argparse.Namespace) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
        },
        path,
    )


if __name__ == "__main__":
    main()