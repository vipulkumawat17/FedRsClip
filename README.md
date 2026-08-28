CLIP Base Paper Implementation

This repository implements the base model from **Learning Transferable Visual Models From Natural Language Supervision** (OpenAI CLIP).

The paper trains from scratch on WIT, a private 400M image-text dataset. This implementation keeps the same core method and makes the data layer practical for public or local equivalents:

- WIT-style JSONL files with `image`/`caption` pairs
- COCO Captions, Flickr8k, Flickr30k, Conceptual Captions, LAION-style local manifests
- Zero-shot evaluation on paper-style datasets exposed through `torchvision` where available, such as ImageNet, CIFAR10/100, STL10, Food101, OxfordPets, Flowers102, DTD, EuroSAT, GTSRB, MNIST, SUN397, Caltech101, StanfordCars, FGVC Aircraft, Country211, and others

## Paper Features Implemented

- Dual image/text encoders trained from scratch
- Modified ResNet image encoder with ResNet-D stem, anti-aliased downsampling, and attention pooling
- Vision Transformer image encoder with patch embeddings and pre-transformer layer norm
- Causal Transformer text encoder
- Linear projections into a shared embedding space
- L2-normalized image and text embeddings
- Learned log-temperature parameter
- Symmetric image-to-text and text-to-image cross entropy loss
- Zero-shot classifier from text prompts such as `a photo of a {label}.`

The tokenizer is intentionally self-contained. The paper used a lower-cased BPE tokenizer with a 49,152 token vocabulary and context length 76. This repo uses a byte-level lower-cased tokenizer by default so the project can train without external tokenizer files. You can replace `clip_implement/tokenizer.py` with a BPE tokenizer if you have a CLIP-compatible vocab/merges file.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train On A WIT-Style Manifest

Create a JSONL file where each row contains an image path and text:

```json
{"image": "images/example.jpg", "caption": "a dog playing on grass"}
{"image": "images/food.jpg", "text": "a bowl of tomato soup"}
```

Then train:

```powershell
python -m clip_implement.train `
  --train-jsonl data/train.jsonl `
  --image-root data `
  --model RN50 `
  --batch-size 128 `
  --epochs 32 `
  --output-dir checkpoints
```

For smaller hardware, start with:

```powershell
python -m clip_implement.train --train-jsonl data/train.jsonl --image-root data --model ViT-B-32 --batch-size 32 --epochs 5
```

## Zero-Shot Evaluation

```powershell
python -m clip_implement.zero_shot `
  --checkpoint checkpoints/clip_last.pt `
  --dataset CIFAR10 `
  --data-root data/eval `
  --batch-size 128
```

Use `--download` for torchvision datasets that support it.

## UC Merced Land Use Quick Start

UC Merced is a labeled remote-sensing classification dataset, not an image-caption dataset. To use it with CLIP training, generate text from class labels:

```powershell
python scripts/prepare_uc_merced.py `
  --zip "D:\VS Code\M.Tech\Dissertation\FedRSCLIP_Core_CLI\data\UC_merced_land_datsets.zip" `
  --output data/uc_merced
```

Train on generated satellite-image captions:

```powershell
python -m clip_implement.train `
  --train-jsonl data/uc_merced/train.jsonl `
  --val-jsonl data/uc_merced/val.jsonl `
  --image-root data/uc_merced/raw/UCMerced_LandUse/Images `
  --classnames data/uc_merced/classnames.txt `
  --model ViT-B-32 `
  --batch-size 16 `
  --epochs 10 `
  --template "a satellite photo of {}." `
  --template "an aerial image of {}." `
  --template "a remote sensing image of {}." `
  --report-dir checkpoints/uc_merced/reports `
  --output-dir checkpoints/uc_merced
```

During training this writes research-friendly epoch reports:

- `checkpoints/uc_merced/reports/metrics.csv` - one row per epoch with loss, image-to-text retrieval accuracy, text-to-image retrieval accuracy, validation top-1, and validation top-5.
- `checkpoints/uc_merced/reports/epoch_001_samples.csv`, `epoch_002_samples.csv`, ... - sample image paths, generated captions, true classes, predicted classes, top-5 classes, and logits for each epoch.
- `checkpoints/uc_merced/reports/epoch_001_predictions.csv`, `epoch_002_predictions.csv`, ... - every validation zero-shot prediction, including true class, predicted class, correctness, top classes/logits, and final output.
- `checkpoints/uc_merced/reports/epoch_001_visuals/`, `epoch_002_visuals/`, ... - input sample previews, text-embedding views at selected text-transformer levels, final text-embedding summaries, and input-vs-output correlation matrices as CSV and PNG.

Use `--report-correlation-samples` to control how many validation inputs appear in the correlation matrix. Use `--no-report-predictions` or `--no-visual-reports` to turn off the larger per-epoch artifacts.

Evaluate with the ImageFolder layout:

```powershell
python -m clip_implement.zero_shot `
  --checkpoint checkpoints/uc_merced/clip_last.pt `
  --dataset ImageFolder `
  --data-root data/uc_merced/raw/UCMerced_LandUse/Images `
  --classnames data/uc_merced/classnames.txt `
  --template "a satellite photo of {}." `
  --batch-size 32 `
  --output-dir checkpoints/uc_merced/zero_shot_reports
```

With `--output-dir`, zero-shot evaluation writes `predictions.csv` for every input prediction, `summary.csv` for the final output metrics, and `input_output_correlation.csv`/`.png` for the sampled input-vs-class correlation matrix.

## Run Many Local Datasets One By One

Use `scripts/run_dataset_experiment.py` when your datasets are already downloaded locally as folders or zip files. The script accepts either:

- a prepared dataset folder containing `train.jsonl`, optional `val.jsonl`, and optional `classnames.txt`
- an ImageFolder-style dataset, where each class is a subfolder of images
- a `.zip` file that extracts into an ImageFolder-style dataset

Run one dataset by changing only the folder name:

```powershell
python scripts/run_dataset_experiment.py `
  --dataset-folder uc_merced `
  --model ViT-B-32 `
  --batch-size 16 `
  --epochs 10 `
  --output-root checkpoints/all_datasets
```

Run several datasets one after another:

```powershell
python scripts/run_dataset_experiment.py `
  --dataset-folder uc_merced `
  --dataset-folder eurosat `
  --dataset-folder resisc45 `
  --model ViT-B-32 `
  --batch-size 16 `
  --epochs 10 `
  --output-root checkpoints/all_datasets
```

Run every immediate dataset folder or zip under `data`:

```powershell
python scripts/run_dataset_experiment.py `
  --datasets-root data `
  --all `
  --model ViT-B-32 `
  --batch-size 16 `
  --epochs 10 `
  --output-root checkpoints/all_datasets
```

Outputs are separated automatically:

- checkpoints: `checkpoints/all_datasets/<dataset_name>/<model>/`
- reports: `checkpoints/all_datasets/<dataset_name>/<model>/reports/`
- summary CSV: `checkpoints/all_datasets/run_summary.csv`

For class-folder datasets, the script creates `train.jsonl`, `val.jsonl`, `test.jsonl`, `classnames.txt`, and `image_root.txt` in the dataset folder. Use repeated `--template` options to change generated captions, for example:

```powershell
python scripts/run_dataset_experiment.py `
  --dataset-folder eurosat `
  --template "a satellite photo of {}." `
  --template "a remote sensing image of {}."
```

## Layer Inspection Outputs

After training, generate sample layer-wise images, overlays, activation summaries, parameter summaries, and optional zero-shot class scores:

```powershell
python scripts/inspect_clip_layers.py `
  --checkpoint checkpoints/uc_merced_reports/clip_last.pt `
  --image data/uc_merced/raw/UCMerced_LandUse/Images/agricultural/agricultural00.tif `
  --text "a satellite photo of agricultural land." `
  --classnames data/uc_merced/classnames.txt `
  --output-dir layer_outputs/agricultural00
```

This creates:

- `input_image.png` - original sample image.
- `*_heatmap.png` - activation heatmap for each selected visual/text layer.
- `*_overlay.png` - activation heatmap overlaid on the input image.
- `parameter_summary.csv` - every parameter tensor with shape, count, mean, std, min, and max.
- `module_parameter_summary.csv` - module-wise direct parameter counts.
- `activation_summary.csv` - layer-wise activation shapes and statistics.
- `embedding_summary.csv` - final image/text embedding statistics and cosine similarity.
- `zero_shot_prediction.csv` - top class predictions for the inspected image.





## Important Datasets From The Paper

The base pre-training dataset is WIT: 400M image-text pairs collected from public internet sources using a broad query list. Since WIT is not public, use a local image-text manifest or public substitutes such as COCO Captions, Flickr8k/Flickr30k, Conceptual Captions, or LAION-derived subsets.

The paper evaluates zero-shot and linear-probe behavior across a broad benchmark suite. The datasets most directly represented in this implementation are:

- ImageNet, CIFAR10, CIFAR100, STL10, Caltech101
- Food101, Oxford-IIIT Pets, Flowers102, Stanford Cars, FGVC Aircraft
- SUN397, DTD, EuroSAT, RESISC45-style satellite datasets
- GTSRB, MNIST, SVHN
- Country211, UCF101, Kinetics-style video frame datasets when arranged as image folders

Some paper datasets are not directly distributed through `torchvision` or require custom labels. For those, use `ImageFolder` or JSONL manifests and provide class names with `--classnames`.
