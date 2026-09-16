"""Generate gpt_file/<dataset>_prompt.json for datasets already prepared by
clip_implement/dataset_factory.py, reading their classnames.txt files.

Same substitute-for-GPT-3 approach as before: uses CLIP's own 80-template
prompt ensemble per class name (inlined below, same list as
trainers/imagenet_templates.py - no import needed, so this runs from
anywhere with no sys.path issues). This is required by CoPrompt's
consistency loss (trainers/coprompt.py: gpt_clip_classifier).

Usage (run from the CoPrompt repo root, after copying this file there):
    python generate_gpt_prompts_from_classnames.py \\
        --datasets-root /path/to/data \\
        --dataset-dir cifar_100 --dataset-name cifar100

Or generate for all 5 at once:
    python generate_gpt_prompts_from_classnames.py --datasets-root /path/to/data --all
"""

import argparse
import json
import os

# source: https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
IMAGENET_TEMPLATES = [
    "a bad photo of a {}.", "a photo of many {}.", "a sculpture of a {}.",
    "a photo of the hard to see {}.", "a low resolution photo of the {}.",
    "a rendering of a {}.", "graffiti of a {}.", "a bad photo of the {}.",
    "a cropped photo of the {}.", "a tattoo of a {}.", "the embroidered {}.",
    "a photo of a hard to see {}.", "a bright photo of a {}.",
    "a photo of a clean {}.", "a photo of a dirty {}.", "a dark photo of the {}.",
    "a drawing of a {}.", "a photo of my {}.", "the plastic {}.",
    "a photo of the cool {}.", "a close-up photo of a {}.",
    "a black and white photo of the {}.", "a painting of the {}.",
    "a painting of a {}.", "a pixelated photo of the {}.",
    "a sculpture of the {}.", "a bright photo of the {}.",
    "a cropped photo of a {}.", "a plastic {}.", "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.", "a blurry photo of the {}.",
    "a photo of the {}.", "a good photo of the {}.", "a rendering of the {}.",
    "a {} in a video game.", "a photo of one {}.", "a doodle of a {}.",
    "a close-up photo of the {}.", "a photo of a {}.", "the origami {}.",
    "the {} in a video game.", "a sketch of a {}.", "a doodle of the {}.",
    "a origami {}.", "a low resolution photo of a {}.", "the toy {}.",
    "a rendition of the {}.", "a photo of the clean {}.",
    "a photo of a large {}.", "a rendition of a {}.", "a photo of a nice {}.",
    "a photo of a weird {}.", "a blurry photo of a {}.", "a cartoon {}.",
    "art of a {}.", "a sketch of the {}.", "a embroidered {}.",
    "a pixelated photo of a {}.", "itap of the {}.",
    "a jpeg corrupted photo of the {}.", "a good photo of a {}.",
    "a plushie {}.", "a photo of the nice {}.", "a photo of the small {}.",
    "a photo of the weird {}.", "the cartoon {}.", "art of the {}.",
    "a drawing of the {}.", "a photo of the large {}.",
    "a black and white photo of a {}.", "the plushie {}.",
    "a dark photo of a {}.", "itap of a {}.", "graffiti of the {}.",
    "a toy {}.", "itap of my {}.", "a photo of a cool {}.",
    "a photo of a small {}.", "a tattoo of the {}.",
]

ALL_DATASETS = [
    ("cifar_100", "cifar100"),
    ("eurosat", "eurosat"),
    ("flower_102", "flowers102"),
    ("food_101", "food101"),
    ("caltech_101", "caltech101"),
]


def generate_one(datasets_root, dataset_dir, dataset_name, out_dir):
    classnames_path = os.path.join(datasets_root, dataset_dir, "classnames.txt")
    if not os.path.exists(classnames_path):
        print(f"[SKIP] {classnames_path} not found")
        return

    with open(classnames_path, "r", encoding="utf-8") as f:
        classnames = [line.strip() for line in f if line.strip()]

    prompts = {}
    for c in classnames:
        prompts[c] = [t.format(c) for t in IMAGENET_TEMPLATES]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_name}_prompt.json")
    with open(out_path, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Wrote {len(classnames)} classes x {len(IMAGENET_TEMPLATES)} prompts to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", required=True)
    parser.add_argument("--dataset-dir")
    parser.add_argument("--dataset-name")
    parser.add_argument("--out-dir", default="gpt_file")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        for dataset_dir, dataset_name in ALL_DATASETS:
            generate_one(args.datasets_root, dataset_dir, dataset_name, args.out_dir)
    else:
        if not args.dataset_dir or not args.dataset_name:
            parser.error("Provide --dataset-dir and --dataset-name, or use --all")
        generate_one(args.datasets_root, args.dataset_dir, args.dataset_name, args.out_dir)


if __name__ == "__main__":
    main()
