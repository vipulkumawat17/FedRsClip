from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset

if __package__ in {None, ""}:  # Allows direct script imports from this folder.
    from tokenizer import ByteTokenizer
else:
    from .tokenizer import ByteTokenizer


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def open_rgb_image(image_path: str | Path) -> Image.Image:
    try:
        return Image.open(image_path).convert("RGB")
    except UnidentifiedImageError as exc:
        path = Path(image_path)
        if any(part.lower() in {"allbands", "all_bands", "all-band"} for part in path.parts):
            raise UnidentifiedImageError(
                f"Cannot decode multi-band TIFF image '{path}' with PIL. "
                "For EuroSAT allBands, prepare the dataset with "
                "--convert-multiband-tiff or use the EuroSAT RGB zip."
            ) from exc
        raise


def _require_torchvision():
    try:
        from torchvision import datasets, transforms
        from torchvision.transforms import InterpolationMode
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("torchvision is required for image transforms and evaluation datasets") from exc
    return datasets, transforms, InterpolationMode


def image_transform(image_size: int = 224, train: bool = True) -> Callable[[Image.Image], torch.Tensor]:
    _, transforms, InterpolationMode = _require_torchvision()
    if train:
        return transforms.Compose(
            [
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


class ImageTextJsonlDataset(Dataset):
    """WIT-style local manifest dataset.

    Each JSONL row should contain an image path under `image`, `image_path`, or
    `path`, and text under `caption`, `text`, or `title`.
    """

    image_keys = ("image", "image_path", "path")
    text_keys = ("caption", "text", "title")

    def __init__(
        self,
        jsonl_path: str | Path,
        image_root: str | Path = ".",
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root)
        self.transform = transform
        self.rows: List[Tuple[Path, str]] = []

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                image_value = _first_present(item, self.image_keys)
                text_value = _first_present(item, self.text_keys)
                if image_value is None or text_value is None:
                    raise ValueError(f"{self.jsonl_path}:{line_number} must contain image and caption/text fields")
                image_path = Path(image_value)
                if not image_path.is_absolute():
                    image_path = self.image_root / image_path
                self.rows.append((image_path, str(text_value)))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        image_path, text = self.rows[index]
        image = open_rgb_image(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, text


class CocoCaptionsJsonDataset(Dataset):
    """COCO captions dataset without requiring pycocotools."""

    def __init__(
        self,
        image_root: str | Path,
        annotations: str | Path,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.image_root = Path(image_root)
        self.transform = transform
        payload = json.loads(Path(annotations).read_text(encoding="utf-8"))
        id_to_file = {img["id"]: img["file_name"] for img in payload["images"]}
        self.rows = [(self.image_root / id_to_file[ann["image_id"]], ann["caption"]) for ann in payload["annotations"]]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        image_path, text = self.rows[index]
        image = open_rgb_image(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, text


def _first_present(item: Dict[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in item:
            return item[key]
    return None


class TokenizeCollator:
    def __init__(self, tokenizer: ByteTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: Sequence[Tuple[torch.Tensor, str]]) -> Tuple[torch.Tensor, torch.LongTensor]:
        images, texts = zip(*batch)
        return torch.stack(list(images), dim=0), self.tokenizer(texts)


class JsonlClassificationDataset(Dataset):
    """JSONL image dataset with labels for epoch-wise zero-shot reporting."""

    image_keys = ("image", "image_path", "path")
    text_keys = ("caption", "text", "title")

    def __init__(
        self,
        jsonl_path: str | Path,
        image_root: str | Path,
        classnames: Sequence[str],
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root)
        self.classnames = list(classnames)
        self.class_to_index = {name: index for index, name in enumerate(self.classnames)}
        self.transform = transform
        self.rows: List[Dict[str, Any]] = []

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                image_value = _first_present(item, self.image_keys)
                text_value = _first_present(item, self.text_keys) or ""
                class_name = str(item.get("class_name") or item.get("label") or "").replace("_", " ")
                if image_value is None or not class_name:
                    raise ValueError(f"{self.jsonl_path}:{line_number} must contain image and class_name/label fields")
                if class_name not in self.class_to_index:
                    raise ValueError(
                        f"{self.jsonl_path}:{line_number} class '{class_name}' not found in classnames. "
                        "Use the same classnames.txt created during dataset preparation."
                    )
                image_path = Path(image_value)
                if not image_path.is_absolute():
                    image_path = self.image_root / image_path
                self.rows.append(
                    {
                        "image_path": image_path,
                        "caption": str(text_value),
                        "class_name": class_name,
                        "target": self.class_to_index[class_name],
                    }
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str, str, str]:
        row = self.rows[index]
        image = open_rgb_image(row["image_path"])
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["target"]), str(row["image_path"]), row["caption"], row["class_name"]


class ClassificationCollator:
    def __call__(self, batch: Sequence[Tuple[torch.Tensor, int, str, str, str]]):
        images, targets, paths, captions, class_names = zip(*batch)
        return torch.stack(list(images), dim=0), torch.tensor(targets, dtype=torch.long), list(paths), list(captions), list(class_names)


def build_pretrain_dataset(
    train_jsonl: str | None,
    image_root: str,
    image_size: int,
    coco_annotations: str | None = None,
) -> Dataset:
    transform = image_transform(image_size, train=True)
    if train_jsonl:
        return ImageTextJsonlDataset(train_jsonl, image_root=image_root, transform=transform)
    if coco_annotations:
        return CocoCaptionsJsonDataset(image_root=image_root, annotations=coco_annotations, transform=transform)
    raise ValueError("Provide --train-jsonl for WIT-style manifests or --coco-annotations for COCO captions")


def read_classnames(path: str | None) -> List[str] | None:
    if path is None:
        return None
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_classnames(raw: Any) -> List[str]:
    if raw is None:
        return []
    names = []
    for item in raw:
        if isinstance(item, (tuple, list)):
            names.append(str(item[0]))
        else:
            names.append(str(item))
    return [name.replace("_", " ") for name in names]


def build_eval_dataset(name: str, root: str, split: str, image_size: int, download: bool = False) -> Dataset:
    datasets, _, _ = _require_torchvision()
    transform = image_transform(image_size, train=False)
    name_key = name.lower()

    if name_key == "imagefolder":
        return datasets.ImageFolder(root=root, transform=transform)
    if name_key == "imagenet":
        return datasets.ImageNet(root=root, split=split, transform=transform)
    if name_key == "cifar10":
        return datasets.CIFAR10(root=root, train=split == "train", transform=transform, download=download)
    if name_key in {"cifar100", "cifar-100","cifar_100"}:
        return datasets.CIFAR100(root=root, train=split == "train", transform=transform, download=download)
    if name_key == "stl10":
        return datasets.STL10(root=root, split="test" if split != "train" else "train", transform=transform, download=download)
    if name_key in {"food101", "food-101", "food_101"}:
        return datasets.Food101(root=root, split=split, transform=transform, download=download)
    if name_key in {"oxfordpets", "oxford-iiit-pets", "pets"}:
        return datasets.OxfordIIITPet(root=root, split="test" if split != "trainval" else "trainval", target_types="category", transform=transform, download=download)
    if name_key in {"flowers102", "flower102", "flowers", "flower-102", "flowers-102","flower_102", "flowers_102"}:
        return datasets.Flowers102(root=root, split="test" if split not in {"train", "val"} else split, transform=transform, download=download)
    if name_key == "dtd":
        return datasets.DTD(root=root, split="test" if split != "train" else "train", transform=transform, download=download)
    if name_key in {"eurosat", "euro-sat"}:
        return datasets.EuroSAT(root=root, transform=transform, download=download)
    if name_key == "gtsrb":
        return datasets.GTSRB(root=root, split="test" if split != "train" else "train", transform=transform, download=download)
    if name_key == "mnist":
        return datasets.MNIST(root=root, train=split == "train", transform=transform, download=download)
    if name_key == "svhn":
        return datasets.SVHN(root=root, split="test" if split != "train" else "train", transform=transform, download=download)
    if name_key in {"caltech101", "caltech-101", "caltech_101"}:
        return datasets.Caltech101(root=root, transform=transform, download=download)
    if name_key == "sun397":
        return datasets.SUN397(root=root, transform=transform, download=download)
    if name_key == "stanfordcars":
        return datasets.StanfordCars(root=root, split="test" if split != "train" else "train", transform=transform, download=download)
    if name_key in {"fgvcaircraft", "aircraft"}:
        return datasets.FGVCAircraft(root=root, split="test" if split != "train" else "train", transform=transform, download=download)
    if name_key == "country211":
        return datasets.Country211(root=root, split="test" if split != "train" else "train", transform=transform, download=download)
    raise ValueError(f"Unsupported dataset '{name}'. Use ImageFolder for custom paper datasets.")