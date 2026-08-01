"""
Data loading utilities for UAMP-Net.

The dataset code supports ImageFolder-style datasets. For smoke testing it can also
construct deterministic synthetic image tensors and labels that follow the split and
class-count structure defined in the paper configuration.

The real training path intentionally avoids dataset-specific shortcuts so that
PlantVillage, PlantDoc, and CUB-200-2011 can be evaluated under a shared protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import os
import random

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
except Exception:  # pragma: no cover
    torch = None
    Dataset = object
    DataLoader = None
    ImageFolder = None
    T = None


@dataclass
class DatasetBundle:
    name: str
    train: object
    val: object
    test: object
    classes: List[str]
    target_mode: str
    metadata: Dict[str, object]


class DeterministicSyntheticDataset(Dataset):
    """
    A deterministic synthetic dataset used only for smoke tests and CI-like checks.
    It generates texture-like tensors with class-dependent local regions so that
    activation maps, prototype discovery, and explanation metrics can be exercised.
    """

    def __init__(
        self,
        name: str,
        n_samples: int,
        n_classes: int,
        image_size: int = 224,
        seed: int = 0,
        split: str = "train",
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required for DeterministicSyntheticDataset")
        self.name = name
        self.n_samples = int(n_samples)
        self.n_classes = int(n_classes)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.split = split
        self.classes = [f"{name}_class_{i:03d}" for i in range(n_classes)]

    def __len__(self) -> int:
        return self.n_samples

    def _make_texture(self, idx: int) -> "torch.Tensor":
        rng = np.random.default_rng(self.seed + idx * 7919)
        h = w = self.image_size
        label = idx % self.n_classes
        yy, xx = np.mgrid[0:h, 0:w]
        cx = int((0.18 + 0.64 * ((label * 37) % 101) / 100.0) * w)
        cy = int((0.18 + 0.64 * ((label * 53) % 101) / 100.0) * h)
        sigma = 12 + (label % 7) * 3
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
        stripe = 0.5 + 0.5 * np.sin((xx + label * 3) / (8 + label % 5))
        noise = rng.normal(0, 0.06, size=(h, w))
        base = 0.25 + 0.25 * stripe + 0.55 * blob + noise
        img = np.stack([
            base,
            0.7 * base + 0.2 * rng.random((h, w)),
            0.4 * base + 0.2 * np.cos(yy / 11.0),
        ], axis=0)
        img = np.clip(img, 0.0, 1.0).astype("float32")
        return torch.from_numpy(img)

    def __getitem__(self, idx: int):
        label = idx % self.n_classes
        return self._make_texture(idx), int(label), {"index": idx, "split": self.split}


def build_transforms(input_size: int, train: bool = True):
    if T is None:
        return None
    if train:
        return T.Compose([
            T.Resize((input_size + 32, input_size + 32)),
            T.RandomResizedCrop(input_size, scale=(0.75, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(12),
            T.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.10, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            T.RandomErasing(p=0.15, scale=(0.02, 0.07), ratio=(0.3, 3.3)),
        ])
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def build_imagefolder_bundle(dataset_spec, seed: int = 0) -> DatasetBundle:
    if ImageFolder is None:
        raise RuntimeError("torchvision is required for real dataset loading")
    root = Path(dataset_spec.root)
    paths = {split: root / split for split in ["train", "val", "test"]}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset split folders: {missing}")
    train = ImageFolder(paths["train"], transform=build_transforms(dataset_spec.input_size, True))
    val = ImageFolder(paths["val"], transform=build_transforms(dataset_spec.input_size, False))
    test = ImageFolder(paths["test"], transform=build_transforms(dataset_spec.input_size, False))
    return DatasetBundle(
        name=dataset_spec.name,
        train=train,
        val=val,
        test=test,
        classes=list(train.classes),
        target_mode=dataset_spec.target,
        metadata={"root": str(root), "seed": seed, "synthetic": False},
    )


def build_synthetic_bundle(dataset_spec, seed: int = 0, max_samples: int = 2048) -> DatasetBundle:
    n_classes = min(dataset_spec.classes, 12 if dataset_spec.name != "CUB-200-2011" else 20)
    train_n = min(dataset_spec.train, max_samples)
    val_n = min(dataset_spec.val, max_samples // 4)
    test_n = min(dataset_spec.test, max_samples // 4)
    train = DeterministicSyntheticDataset(dataset_spec.name, train_n, n_classes, dataset_spec.input_size, seed, "train")
    val = DeterministicSyntheticDataset(dataset_spec.name, val_n, n_classes, dataset_spec.input_size, seed + 17, "val")
    test = DeterministicSyntheticDataset(dataset_spec.name, test_n, n_classes, dataset_spec.input_size, seed + 29, "test")
    return DatasetBundle(
        name=dataset_spec.name,
        train=train,
        val=val,
        test=test,
        classes=train.classes,
        target_mode=dataset_spec.target,
        metadata={"seed": seed, "synthetic": True, "class_count_used": n_classes},
    )


def make_loaders(bundle: DatasetBundle, batch_size: int, workers: int = 4, seed: int = 0):
    if DataLoader is None:
        raise RuntimeError("PyTorch is required for DataLoader")
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(bundle.train, batch_size=batch_size, shuffle=True, num_workers=workers, generator=generator)
    val_loader = DataLoader(bundle.val, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(bundle.test, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, val_loader, test_loader


def class_distribution(dataset: object, n_classes: int) -> np.ndarray:
    counts = np.zeros(n_classes, dtype=np.int64)
    if hasattr(dataset, "targets"):
        for y in dataset.targets:
            counts[int(y)] += 1
    else:
        for idx in range(len(dataset)):
            _, y, *_ = dataset[idx]
            counts[int(y)] += 1
    return counts
