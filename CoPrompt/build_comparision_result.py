"""Build dissertation-ready comparison outputs (CSV + chart) from:
  1. CoPrompt SLURM .out logs (base2new_test_coprompt.sh output, both the
     base-class and novel-class '=> result' blocks)
  2. Pretrained zero-shot CLIP summary.csv files (from clip_implement's
     evaluation pipeline, checkpoints/pretrained_zero_shot/<dataset>/<model>/)

Produces:
  - coprompt_base_novel_results.csv   (per-dataset base/novel/H breakdown)
  - zero_shot_vs_coprompt.csv         (merged comparison table)
  - zero_shot_vs_coprompt.png         (grouped bar chart)

Usage: edit the DATASETS config list below with the actual paths for all
5 datasets, then run:
    python build_comparison_results.py --out-dir results
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Fill in paths for all 5 datasets here ---
# coprompt_log: the .out file from `sbatch 02_run_training.sh <dataset> ...`
#               (must contain both the base and novel '=> result' blocks)
# zero_shot_summary: the summary.csv from checkpoints/pretrained_zero_shot/<dataset>/<model>/
DATASETS = [
    {
        "display_name": "Caltech-101",
        "coprompt_log": "CoPrompt/log_files/run_tranning_coprompt_caltech101_30211.out",
        "zero_shot_summary": "checkpoints/pretrained_zero_shot/caltech_101/ViT-B-16_openai/summary.csv",
    },
    {
        "display_name": "CIFAR-100",
        "coprompt_log": "CoPrompt/log_files/run_tranning_coprompt_cifar100_30117.out",
        "zero_shot_summary": "checkpoints/pretrained_zero_shot/cifar_100/ViT-B-16_openai/summary.csv",
    },
    {
        "display_name": "EuroSAT",
        "coprompt_log": "CoPrompt/log_files/run_tranning_coprompt_eurosat_30397.out",
        "zero_shot_summary": "checkpoints/pretrained_zero_shot/eurosat/ViT-B-16_openai/summary.csv",
    },
    {
        "display_name": "Flowers-102",
        "coprompt_log": "CoPrompt/log_files/run_tranning_coprompt_flowers102_30114.out",
        "zero_shot_summary": "checkpoints/pretrained_zero_shot/flower_102/ViT-B-16_openai/summary.csv",
    },
    {
        "display_name": "Food-101",
        "coprompt_log": "CoPrompt/log_files/run_tranning_coprompt_30090.out",
        "zero_shot_summary": "checkpoints/pretrained_zero_shot/food_101/ViT-B-16_openai/summary.csv",
    },
]

RESULT_BLOCK_RE = re.compile(
    r"=> result\s*\n"
    r"\* total: ([\d,]+)\s*\n"
    r"\* correct: ([\d,]+)\s*\n"
    r"\* accuracy: ([\d.]+)%\s*\n"
    r"\* error: ([\d.]+)%\s*\n"
    r"\* macro_f1: ([\d.]+)%"
)


def parse_coprompt_log(path):
    """Extract the base-class and novel-class result blocks from a CoPrompt
    .out log. The training phase runs first (base classes), followed by the
    test phase (novel classes) - so the first '=> result' block is base and
    the second is novel."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    matches = RESULT_BLOCK_RE.findall(text)
    if len(matches) < 2:
        raise ValueError(
            f"{path}: expected 2 '=> result' blocks (base, novel), found {len(matches)}"
        )

    def to_dict(m):
        total, correct, accuracy, error, macro_f1 = m
        return {
            "total": int(total.replace(",", "")),
            "correct": int(correct.replace(",", "")),
            "accuracy": float(accuracy),
            "error": float(error),
            "macro_f1": float(macro_f1),
        }

    base, novel = to_dict(matches[0]), to_dict(matches[1])
    return base, novel


def read_zero_shot_summary(path):
    with open(path, newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    return {
        "model": row["model"],
        "pretrained": row["pretrained"],
        "samples": int(row["samples"]),
        "classes": int(row["classes"]),
        "top1": float(row["top1"]),
        "top5": float(row["top5"]),
    }


def harmonic_mean(a, b):
    return 0.0 if (a + b) == 0 else 2 * a * b / (a + b)


def build_tables(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    comparison_rows = []

    for ds in DATASETS:
        name = ds["display_name"]
        base, novel = parse_coprompt_log(ds["coprompt_log"])
        h = harmonic_mean(base["accuracy"], novel["accuracy"])
        zs = read_zero_shot_summary(ds["zero_shot_summary"])

        detail_rows.append({
            "dataset": name,
            "base_total": base["total"], "base_correct": base["correct"],
            "base_accuracy": base["accuracy"], "base_error": base["error"],
            "base_macro_f1": base["macro_f1"],
            "novel_total": novel["total"], "novel_correct": novel["correct"],
            "novel_accuracy": novel["accuracy"], "novel_error": novel["error"],
            "novel_macro_f1": novel["macro_f1"],
            "harmonic_mean_H": round(h, 2),
        })

        comparison_rows.append({
            "dataset": name,
            "zero_shot_model": f"{zs['model']}_{zs['pretrained']}",
            "zero_shot_top1": zs["top1"],
            "zero_shot_top5": zs["top5"],
            "coprompt_base_accuracy": base["accuracy"],
            "coprompt_novel_accuracy": novel["accuracy"],
            "coprompt_harmonic_mean_H": round(h, 2),
            "improvement_over_zero_shot_base": round(base["accuracy"] - zs["top1"], 2),
            "improvement_over_zero_shot_novel": round(novel["accuracy"] - zs["top1"], 2),
        })

    detail_path = out_dir / "coprompt_base_novel_results.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    comparison_path = out_dir / "zero_shot_vs_coprompt.csv"
    with open(comparison_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"Wrote {detail_path}")
    print(f"Wrote {comparison_path}")
    return comparison_rows


def plot_comparison(comparison_rows, out_dir):
    out_dir = Path(out_dir)
    datasets = [r["dataset"] for r in comparison_rows]
    zero_shot = [r["zero_shot_top1"] for r in comparison_rows]
    base = [r["coprompt_base_accuracy"] for r in comparison_rows]
    novel = [r["coprompt_novel_accuracy"] for r in comparison_rows]
    h = [r["coprompt_harmonic_mean_H"] for r in comparison_rows]

    x = np.arange(len(datasets))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 2), 6))
    ax.bar(x - 1.5 * width, zero_shot, width, label="Zero-shot CLIP (top-1)")
    ax.bar(x - 0.5 * width, base, width, label="CoPrompt (base classes)")
    ax.bar(x + 0.5 * width, novel, width, label="CoPrompt (novel classes)")
    ax.bar(x + 1.5 * width, h, width, label="CoPrompt (harmonic mean H)")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Zero-shot CLIP vs. CoPrompt Base-to-Novel Generalization")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)

    fig.tight_layout()
    chart_path = out_dir / "zero_shot_vs_coprompt.png"
    fig.savefig(chart_path, dpi=200)
    print(f"Wrote {chart_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="checkpoints/CoPrompt_comparison_results/Vit-B-16_vs_Vit-B-16", help="output directory for CSV and chart")
    args = parser.parse_args()

    comparison_rows = build_tables(args.out_dir)
    plot_comparison(comparison_rows, args.out_dir)


if __name__ == "__main__":
    main()