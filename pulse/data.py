from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

try:
    from PIL import Image, ImageEnhance
except ModuleNotFoundError:  # pragma: no cover
    Image = None
    ImageEnhance = None

from .geometry import polygons_to_mask, xyxy_to_cxcywh
from .specs import TaskSpec, TaskType

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _require_image_stack() -> None:
    if np is None or Image is None:
        raise ModuleNotFoundError(
            "PULSE data loading requires `numpy` and `pillow`. "
            "Install dependencies from requirements.txt first."
        )


def _pil_to_tensor(image: "Image.Image") -> torch.Tensor:
    _require_image_stack()
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    tensor = torch.from_numpy(array.transpose(2, 0, 1)) / 255.0
    return tensor


def _normalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype).view(3, 1, 1)
    return (image - mean) / std


def _load_image(path: str, size: int) -> "Image.Image":
    _require_image_stack()
    image = Image.open(path).convert("RGB")
    return image.resize((size, size), Image.BILINEAR)


def _load_mask_image(path: str, size: int) -> "Image.Image":
    _require_image_stack()
    image = Image.open(path).convert("L")
    return image.resize((size, size), Image.NEAREST)


def _train_augment_image(image: "Image.Image", level: str = "standard") -> "Image.Image":
    if ImageEnhance is None:
        return image
    flip_prob = 0.5
    rotate_prob = 0.35
    brightness_prob = 0.4
    contrast_prob = 0.4
    rotate_limit = 12.0
    brightness_range = (0.9, 1.1)
    contrast_range = (0.9, 1.15)

    if level == "light":
        rotate_prob = 0.2
        brightness_prob = 0.25
        contrast_prob = 0.25
        rotate_limit = 8.0
        brightness_range = (0.95, 1.05)
        contrast_range = (0.95, 1.08)
    elif level == "strong":
        flip_prob = 0.55
        rotate_prob = 0.5
        brightness_prob = 0.5
        contrast_prob = 0.5
        rotate_limit = 18.0
        brightness_range = (0.85, 1.15)
        contrast_range = (0.85, 1.2)

    if random.random() < flip_prob:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < rotate_prob:
        angle = random.uniform(-rotate_limit, rotate_limit)
        image = image.rotate(angle, resample=Image.BILINEAR)
    if random.random() < brightness_prob:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(*brightness_range))
    if random.random() < contrast_prob:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(*contrast_range))
    return image


def _classification_sample_weights(
    samples,
    strategy: str = "balanced",
    sample_scores: Optional[Sequence[float] | torch.Tensor] = None,
) -> torch.Tensor | None:
    labels = [int(sample["label"]) for sample in samples if "label" in sample]
    if not labels:
        return None
    if strategy == "natural":
        return None
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long))
    counts = counts.clamp_min(1)
    if strategy == "minority_focus":
        class_weights = counts.sum().float() / counts.float()
    else:
        class_weights = (counts.sum().float() / counts.float()).sqrt()
    class_weights = class_weights / class_weights.mean().clamp_min(1e-6)
    sample_weights = torch.tensor([float(class_weights[label].item()) for label in labels], dtype=torch.float32)

    hard_component: torch.Tensor | None = None
    if sample_scores is not None:
        hard_component = torch.as_tensor(sample_scores, dtype=torch.float32).flatten()
        if hard_component.numel() == len(labels):
            hard_component = hard_component.clamp_min(1e-4)
            hard_component = hard_component / hard_component.mean().clamp_min(1e-6)
        else:
            hard_component = None

    if strategy in {"hard_example", "hybrid_hard"} and hard_component is not None:
        if strategy == "hard_example":
            sample_weights = hard_component
        else:
            sample_weights = (0.5 * sample_weights) + (0.5 * hard_component)
    sample_weights = sample_weights.to(torch.double)
    return sample_weights


class ClassificationDataset(Dataset):
    def __init__(self, samples, image_size: int, train: bool = False, augmentation_level: str = "standard"):
        self.samples = samples
        self.image_size = image_size
        self.train = train
        self.augmentation_level = augmentation_level

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = _load_image(sample["image"], self.image_size)
        if self.train:
            image = _train_augment_image(image, level=self.augmentation_level)
        image_tensor = _normalize(_pil_to_tensor(image))
        return {
            "image": image_tensor,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "meta": sample,
        }


class MultiModalDataset(Dataset):
    def __init__(self, samples, image_size: int, train: bool = False, augmentation_level: str = "standard"):
        self.samples = samples
        self.image_size = image_size
        self.train = train
        self.augmentation_level = augmentation_level

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        flip = self.train and random.random() < 0.5
        modalities: Dict[str, torch.Tensor] = {}
        contrast_low, contrast_high = 0.92, 1.12
        if self.augmentation_level == "light":
            contrast_low, contrast_high = 0.96, 1.08
        elif self.augmentation_level == "strong":
            contrast_low, contrast_high = 0.88, 1.18
        for name, path in sample["modalities"].items():
            image = _load_image(path, self.image_size)
            if flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if self.train and random.random() < 0.25 and ImageEnhance is not None:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(contrast_low, contrast_high))
            modalities[name] = _normalize(_pil_to_tensor(image))
        return {
            "modalities": modalities,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "meta": sample,
        }


class SegmentationDataset(Dataset):
    def __init__(self, samples, image_size: int, train: bool = False, segmentation_mode: str = "binary"):
        self.samples = samples
        self.image_size = image_size
        self.train = train
        self.segmentation_mode = segmentation_mode

    def __len__(self) -> int:
        return len(self.samples)

    def _render_mask(self, sample, image_size: Tuple[int, int]) -> "np.ndarray":
        _require_image_stack()
        if sample.get("mask_path"):
            return np.asarray(_load_mask_image(sample["mask_path"], self.image_size), dtype=np.uint8)
        if sample.get("mask_paths"):
            union = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            for path in sample["mask_paths"]:
                union = np.maximum(union, np.asarray(_load_mask_image(path, self.image_size), dtype=np.uint8))
            return union
        if sample.get("mask_polygons"):
            mask = polygons_to_mask((image_size[0], image_size[1]), sample["mask_polygons"])
            resized = Image.fromarray(mask).resize((self.image_size, self.image_size), Image.NEAREST)
            return np.asarray(resized, dtype=np.uint8)
        return np.zeros((self.image_size, self.image_size), dtype=np.uint8)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        original = Image.open(sample["image"]).convert("RGB")
        mask = self._render_mask(sample, original.size)
        image = original.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask_image = Image.fromarray(mask).resize((self.image_size, self.image_size), Image.NEAREST)

        if self.train and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask_image = mask_image.transpose(Image.FLIP_LEFT_RIGHT)

        image_tensor = _normalize(_pil_to_tensor(image))
        mask_array = np.asarray(mask_image, dtype=np.uint8)
        if self.segmentation_mode == "binary":
            mask_tensor = torch.from_numpy((mask_array > 0).astype("float32")).unsqueeze(0)
        else:
            mask_tensor = torch.from_numpy(mask_array.astype("int64"))
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_index": torch.tensor(index, dtype=torch.long),
            "meta": sample,
        }


class DetectionDataset(Dataset):
    def __init__(self, samples, image_size: int, train: bool = False):
        self.samples = samples
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        original = Image.open(sample["image"]).convert("RGB")
        width, height = original.size
        image = original.resize((self.image_size, self.image_size), Image.BILINEAR)

        bbox = list(sample["bbox"])
        sx = self.image_size / max(width, 1)
        sy = self.image_size / max(height, 1)
        bbox = [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]
        if self.train and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            bbox = [
                self.image_size - bbox[2],
                bbox[1],
                self.image_size - bbox[0],
                bbox[3],
            ]

        image_tensor = _normalize(_pil_to_tensor(image))
        bbox_tensor = torch.tensor(
            xyxy_to_cxcywh(bbox, self.image_size, self.image_size),
            dtype=torch.float32,
        )
        return {
            "image": image_tensor,
            "bbox": bbox_tensor,
            "label": torch.tensor(sample.get("label", 1), dtype=torch.long),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "meta": sample,
        }


class RegressionDataset(Dataset):
    def __init__(self, samples, image_size: int, train: bool = False):
        self.samples = samples
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = _load_image(sample["image"], self.image_size)
        if self.train and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image_tensor = _normalize(_pil_to_tensor(image))
        return {
            "image": image_tensor,
            "target": torch.tensor(float(sample["target"]), dtype=torch.float32),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "meta": sample,
        }


def _pulse_collate(batch):
    if not batch:
        return {}

    collated = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if key == "meta":
            collated[key] = values
        else:
            collated[key] = default_collate(values)
    return collated


def build_dataloaders(
    task: TaskSpec,
    batch_size: int,
    num_workers: int,
    image_size: int,
    balanced_sampling: Optional[bool] = None,
    sampling_strategy: Optional[str] = None,
    sample_score_overrides: Optional[Sequence[float] | torch.Tensor] = None,
    train_augmentation: str = "standard",
):
    train_sampler = None
    use_balanced_sampling = task.extras.get("balanced_sampling", True) if balanced_sampling is None else balanced_sampling
    effective_sampling_strategy = sampling_strategy or ("balanced" if use_balanced_sampling else "natural")
    if task.task_type == TaskType.CLASSIFICATION:
        dataset_cls = ClassificationDataset
        train_set = dataset_cls(
            task.train_samples,
            image_size=image_size,
            train=True,
            augmentation_level=train_augmentation,
        )
        val_set = dataset_cls(task.val_samples, image_size=image_size, train=False)
        test_set = dataset_cls(task.test_samples, image_size=image_size, train=False)
        if effective_sampling_strategy != "natural":
            sample_weights = _classification_sample_weights(
                task.train_samples,
                strategy=effective_sampling_strategy,
                sample_scores=sample_score_overrides,
            )
            if sample_weights is not None:
                train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    elif task.task_type == TaskType.MULTIMODAL:
        dataset_cls = MultiModalDataset
        train_set = dataset_cls(
            task.train_samples,
            image_size=image_size,
            train=True,
            augmentation_level=train_augmentation,
        )
        val_set = dataset_cls(task.val_samples, image_size=image_size, train=False)
        test_set = dataset_cls(task.test_samples, image_size=image_size, train=False)
        if effective_sampling_strategy != "natural":
            sample_weights = _classification_sample_weights(
                task.train_samples,
                strategy=effective_sampling_strategy,
                sample_scores=sample_score_overrides,
            )
            if sample_weights is not None:
                train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    elif task.task_type == TaskType.SEGMENTATION:
        mode = task.extras.get("segmentation_mode", "binary")
        train_set = SegmentationDataset(task.train_samples, image_size=image_size, train=True, segmentation_mode=mode)
        val_set = SegmentationDataset(task.val_samples, image_size=image_size, train=False, segmentation_mode=mode)
        test_set = SegmentationDataset(task.test_samples, image_size=image_size, train=False, segmentation_mode=mode)
    elif task.task_type == TaskType.DETECTION:
        train_set = DetectionDataset(task.train_samples, image_size=image_size, train=True)
        val_set = DetectionDataset(task.val_samples, image_size=image_size, train=False)
        test_set = DetectionDataset(task.test_samples, image_size=image_size, train=False)
    else:
        train_set = RegressionDataset(task.train_samples, image_size=image_size, train=True)
        val_set = RegressionDataset(task.val_samples, image_size=image_size, train=False)
        test_set = RegressionDataset(task.test_samples, image_size=image_size, train=False)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": _pulse_collate,
    }
    if train_sampler is not None:
        train_loader = DataLoader(train_set, sampler=train_sampler, shuffle=False, **loader_kwargs)
    else:
        train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs) if len(val_set) else None
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs) if len(test_set) else None
    return train_loader, val_loader, test_loader
