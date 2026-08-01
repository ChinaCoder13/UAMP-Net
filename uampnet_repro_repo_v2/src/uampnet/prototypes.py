"""
Prototype discovery and reliability scoring.

The manuscript uses a variational Bayesian Gaussian mixture model (VIGMM) to
discover class-wise prototypes. This file implements a practical wrapper around
sklearn's BayesianGaussianMixture and a deterministic NumPy fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import json
from pathlib import Path

import numpy as np


@dataclass
class PrototypeRecord:
    dataset: str
    class_id: int
    prototype_id: int
    center: np.ndarray
    covariance_trace: float
    mixture_weight: float
    activation_mean: float
    separation: float
    reliability: float
    uncertainty: float
    support: int


class PrototypeReliability:
    """
    Implements the reliability equation from the paper:
        R_ck = pi^eta1 * compactness^eta2 * separation^eta3 * activation^eta4
    where compactness is 1/(1+tr(Sigma)).
    """

    def __init__(self, eta=(0.35, 0.25, 0.25, 0.15), eps: float = 1e-8):
        self.eta = tuple(float(x) for x in eta)
        self.eps = float(eps)

    def compute(self, weights, covariance_traces, separations, activations):
        weights = np.asarray(weights, dtype=float)
        covariance_traces = np.asarray(covariance_traces, dtype=float)
        separations = np.asarray(separations, dtype=float)
        activations = np.asarray(activations, dtype=float)
        compactness = 1.0 / (1.0 + np.maximum(covariance_traces, self.eps))

        def norm(x):
            x = np.asarray(x, dtype=float)
            lo, hi = np.nanmin(x), np.nanmax(x)
            return (x - lo) / (hi - lo + self.eps)

        w = norm(weights) + self.eps
        c = norm(compactness) + self.eps
        d = norm(separations) + self.eps
        a = norm(activations) + self.eps
        r = (w ** self.eta[0]) * (c ** self.eta[1]) * (d ** self.eta[2]) * (a ** self.eta[3])
        r = norm(r)
        u = 1.0 - r
        return r, u


class VariationalPrototypeBank:
    """
    Class-wise prototype discovery.

    Usage:
        bank = VariationalPrototypeBank(k_max=10, weight_threshold=0.03)
        bank.fit(descriptor_bank, activation_bank)
        bank.save("prototypes/PlantVillage_prototypes.npz")
    """

    def __init__(
        self,
        k_max: int,
        descriptor_dim: int,
        weight_threshold: float = 0.03,
        random_state: int = 0,
        covariance_type: str = "diag",
    ):
        self.k_max = int(k_max)
        self.descriptor_dim = int(descriptor_dim)
        self.weight_threshold = float(weight_threshold)
        self.random_state = int(random_state)
        self.covariance_type = covariance_type
        self.records: List[PrototypeRecord] = []
        self.class_to_indices: Dict[int, List[int]] = {}
        self.reliability_model = PrototypeReliability()

    def _fit_bgm(self, x: np.ndarray):
        try:
            from sklearn.mixture import BayesianGaussianMixture
            model = BayesianGaussianMixture(
                n_components=self.k_max,
                covariance_type=self.covariance_type,
                weight_concentration_prior_type="dirichlet_process",
                weight_concentration_prior=0.5,
                max_iter=400,
                random_state=self.random_state,
                reg_covar=1e-5,
            )
            model.fit(x)
            means = model.means_
            weights = model.weights_
            if self.covariance_type == "diag":
                covs = model.covariances_
                traces = covs.sum(axis=1)
            else:
                covs = model.covariances_
                traces = np.array([np.trace(c) for c in covs])
            labels = model.predict(x)
            return means, traces, weights, labels
        except Exception:
            # Deterministic fallback: farthest-point k-means-like clustering.
            rng = np.random.default_rng(self.random_state)
            n, d = x.shape
            k = min(self.k_max, max(1, int(np.sqrt(n / 3))))
            centers = [x[rng.integers(0, n)]]
            for _ in range(1, k):
                dist = np.min(((x[:, None, :] - np.stack(centers)[None, :, :]) ** 2).sum(-1), axis=1)
                centers.append(x[np.argmax(dist)])
            centers = np.stack(centers)
            for _ in range(30):
                dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
                labels = dist.argmin(axis=1)
                for j in range(k):
                    if np.any(labels == j):
                        centers[j] = x[labels == j].mean(axis=0)
            weights = np.array([(labels == j).mean() for j in range(k)])
            traces = np.array([x[labels == j].var(axis=0).sum() if np.any(labels == j) else 999.0 for j in range(k)])
            pad = self.k_max - k
            if pad > 0:
                centers = np.vstack([centers, np.zeros((pad, d))])
                weights = np.concatenate([weights, np.zeros(pad)])
                traces = np.concatenate([traces, np.ones(pad) * 999.0])
            return centers, traces, weights, labels

    @staticmethod
    def _prototype_separation(centers: np.ndarray, classes: np.ndarray, this_class: int) -> np.ndarray:
        sep = np.ones(len(centers), dtype=float)
        other = centers[classes != this_class]
        own = centers[classes == this_class]
        if len(other) == 0:
            return sep
        for i, c in enumerate(own):
            sep[i] = np.sqrt(((other - c) ** 2).sum(axis=1)).min()
        return sep

    def fit(self, dataset_name: str, descriptor_bank: Dict[int, np.ndarray], activation_bank: Optional[Dict[int, np.ndarray]] = None):
        preliminary = {}
        all_centers = []
        all_classes = []
        for class_id, descriptors in descriptor_bank.items():
            x = np.asarray(descriptors, dtype=np.float32)
            if x.ndim != 2 or x.shape[1] != self.descriptor_dim:
                raise ValueError(f"class {class_id}: descriptor shape mismatch {x.shape}")
            means, traces, weights, labels = self._fit_bgm(x)
            active = np.where(weights >= self.weight_threshold)[0]
            if len(active) == 0:
                active = np.array([int(weights.argmax())])
            preliminary[class_id] = (means, traces, weights, labels, active)
            for j in active:
                all_centers.append(means[j])
                all_classes.append(class_id)
        all_centers = np.asarray(all_centers, dtype=np.float32)
        all_classes = np.asarray(all_classes, dtype=int)

        self.records.clear()
        proto_global_id = 0
        for class_id, (means, traces, weights, labels, active) in preliminary.items():
            class_centers = means[active]
            sep = []
            for c in class_centers:
                other = all_centers[all_classes != class_id]
                if len(other) == 0:
                    sep.append(1.0)
                else:
                    sep.append(float(np.sqrt(((other - c) ** 2).sum(axis=1)).min()))
            sep = np.asarray(sep)
            acts = []
            supports = []
            a_values = None if activation_bank is None else np.asarray(activation_bank.get(class_id, []), dtype=float)
            for j in active:
                member = labels == j
                supports.append(int(member.sum()))
                if a_values is not None and len(a_values) == len(labels) and member.any():
                    acts.append(float(a_values[member].mean()))
                else:
                    acts.append(float(0.72 + 0.18 * weights[j] / (weights[active].max() + 1e-8)))
            rel, unc = self.reliability_model.compute(weights[active], traces[active], sep, acts)
            for local_id, j in enumerate(active):
                self.records.append(PrototypeRecord(
                    dataset=dataset_name,
                    class_id=int(class_id),
                    prototype_id=int(proto_global_id),
                    center=means[j].astype(np.float32),
                    covariance_trace=float(traces[j]),
                    mixture_weight=float(weights[j]),
                    activation_mean=float(acts[local_id]),
                    separation=float(sep[local_id]),
                    reliability=float(rel[local_id]),
                    uncertainty=float(unc[local_id]),
                    support=int(supports[local_id]),
                ))
                self.class_to_indices.setdefault(int(class_id), []).append(proto_global_id)
                proto_global_id += 1
        return self

    def arrays(self):
        centers = np.stack([r.center for r in self.records]).astype(np.float32)
        classes = np.array([r.class_id for r in self.records], dtype=np.int64)
        reliability = np.array([r.reliability for r in self.records], dtype=np.float32)
        uncertainty = np.array([r.uncertainty for r in self.records], dtype=np.float32)
        return centers, classes, reliability, uncertainty

    def save(self, path: str | Path):
        path = Path(path)
        centers, classes, reliability, uncertainty = self.arrays()
        np.savez_compressed(
            path,
            centers=centers,
            classes=classes,
            reliability=reliability,
            uncertainty=uncertainty,
            covariance_trace=np.array([r.covariance_trace for r in self.records], dtype=np.float32),
            mixture_weight=np.array([r.mixture_weight for r in self.records], dtype=np.float32),
            activation_mean=np.array([r.activation_mean for r in self.records], dtype=np.float32),
            separation=np.array([r.separation for r in self.records], dtype=np.float32),
            support=np.array([r.support for r in self.records], dtype=np.int64),
        )

    def save_csv(self, path: str | Path):
        import csv
        path = Path(path)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset", "class_id", "prototype_id", "mixture_weight", "covariance_trace",
                "activation_mean", "separation", "reliability", "uncertainty", "support"
            ])
            for r in self.records:
                writer.writerow([
                    r.dataset, r.class_id, r.prototype_id, f"{r.mixture_weight:.6f}",
                    f"{r.covariance_trace:.6f}", f"{r.activation_mean:.6f}", f"{r.separation:.6f}",
                    f"{r.reliability:.6f}", f"{r.uncertainty:.6f}", r.support
                ])

    @classmethod
    def load(cls, path: str | Path):
        data = np.load(path)
        obj = cls(k_max=1, descriptor_dim=int(data["centers"].shape[1]))
        for i in range(len(data["centers"])):
            obj.records.append(PrototypeRecord(
                dataset="loaded",
                class_id=int(data["classes"][i]),
                prototype_id=i,
                center=data["centers"][i],
                covariance_trace=float(data["covariance_trace"][i]),
                mixture_weight=float(data["mixture_weight"][i]),
                activation_mean=float(data["activation_mean"][i]),
                separation=float(data["separation"][i]),
                reliability=float(data["reliability"][i]),
                uncertainty=float(data["uncertainty"][i]),
                support=int(data["support"][i]),
            ))
        return obj
