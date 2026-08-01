"""
Configuration and reproducibility utilities for UAMP-Net.

This module keeps all paper-level hyperparameters out of the training scripts.
The repository is designed so that the same scripts can be used for:
1. real dataset training;
2. deterministic smoke-test artifact generation;
3. table and figure regeneration from stored evaluation outputs.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

import numpy as np

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Install requirements.txt.") from exc


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: str
    image_count: int
    train: int
    val: int
    test: int
    classes: int
    input_size: int
    batch_size: int
    epochs: int
    k_max: int
    target: str

    @property
    def split_tuple(self) -> tuple[int, int, int]:
        return self.train, self.val, self.test

    @property
    def total_split_count(self) -> int:
        return self.train + self.val + self.test

    def validate(self) -> None:
        if self.total_split_count <= 0:
            raise ValueError(f"{self.name}: empty split")
        if self.classes <= 1:
            raise ValueError(f"{self.name}: expected >1 class")
        if self.k_max <= 0:
            raise ValueError(f"{self.name}: Kmax must be positive")
        if self.input_size != 224:
            raise ValueError(f"{self.name}: paper config expects 224 input")


@dataclass(frozen=True)
class ModelSpec:
    backbone: str
    pretrained: bool
    descriptor_dim: int
    local_pool_kernel: int
    arm_quantile: float
    prototype_activation_threshold: float
    prototype_update_interval: int
    top_l_candidates: int

    def validate(self) -> None:
        if not 0.0 < self.arm_quantile < 1.0:
            raise ValueError("arm_quantile must be in (0, 1)")
        if not 0.0 <= self.prototype_activation_threshold < 1.0:
            raise ValueError("prototype_activation_threshold must be in [0, 1)")
        if self.descriptor_dim < 32:
            raise ValueError("descriptor_dim is unexpectedly small")
        if self.local_pool_kernel % 2 == 0:
            raise ValueError("local_pool_kernel should be odd")


@dataclass(frozen=True)
class OptimizationSpec:
    optimizer: str
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    early_stopping_patience: int
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class LossWeightSpec:
    compactness: float
    separation: float
    activation_consistency: float
    uncertainty: float


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int
    output_dir: str
    checkpoint_dir: str
    prototype_dir: str
    datasets: Dict[str, DatasetSpec]
    model: ModelSpec
    optimization: OptimizationSpec
    loss_weights: LossWeightSpec
    raw: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for dataset in self.datasets.values():
            dataset.validate()
        self.model.validate()
        if self.optimization.learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        if not self.optimization.seeds:
            raise ValueError("at least one seed required")


def _read_yaml(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dataset_from_mapping(name: str, obj: Mapping[str, Any]) -> DatasetSpec:
    split = obj.get("split", {})
    classes = obj.get("classes", obj.get("disease_labels", 1))
    return DatasetSpec(
        name=name,
        root=str(obj["root"]),
        image_count=int(obj["image_count"]),
        train=int(split["train"]),
        val=int(split["val"]),
        test=int(split["test"]),
        classes=int(classes),
        input_size=int(obj["input_size"]),
        batch_size=int(obj["batch_size"]),
        epochs=int(obj["epochs"]),
        k_max=int(obj["k_max"]),
        target=str(obj.get("target", "class")),
    )


def load_config(path: str | os.PathLike[str]) -> ProjectConfig:
    raw = _read_yaml(path)
    project = raw["project"]
    datasets = {
        name: _dataset_from_mapping(name, value)
        for name, value in raw["datasets"].items()
    }
    model = ModelSpec(**raw["model"])
    optimization = OptimizationSpec(
        optimizer=raw["optimization"]["optimizer"],
        learning_rate=float(raw["optimization"]["learning_rate"]),
        weight_decay=float(raw["optimization"]["weight_decay"]),
        warmup_epochs=int(raw["optimization"]["warmup_epochs"]),
        early_stopping_patience=int(raw["optimization"]["early_stopping_patience"]),
        seeds=tuple(int(s) for s in raw["optimization"]["seeds"]),
    )
    loss_weights = LossWeightSpec(**raw["loss_weights"])
    cfg = ProjectConfig(
        name=str(project["name"]),
        seed=int(project["seed"]),
        output_dir=str(project["output_dir"]),
        checkpoint_dir=str(project["checkpoint_dir"]),
        prototype_dir=str(project["prototype_dir"]),
        datasets=datasets,
        model=model,
        optimization=optimization,
        loss_weights=loss_weights,
        raw=raw,
    )
    cfg.validate()
    return cfg


def set_global_seed(seed: int) -> np.random.Generator:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    return np.random.default_rng(seed)


def ensure_project_dirs(cfg: ProjectConfig) -> None:
    for path in [cfg.output_dir, cfg.checkpoint_dir, cfg.prototype_dir]:
        Path(path).mkdir(parents=True, exist_ok=True)
    for sub in ["figures", "tables", "raw", "logs", "xls"]:
        Path(cfg.output_dir, sub).mkdir(parents=True, exist_ok=True)


def config_digest(cfg: ProjectConfig) -> str:
    import json, hashlib
    payload = json.dumps(cfg.raw, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
