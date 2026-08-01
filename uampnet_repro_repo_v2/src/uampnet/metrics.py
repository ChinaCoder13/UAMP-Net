"""
Metrics used by UAMP-Net experiments.

This module provides classification, calibration, uncertainty, and explanation
metrics used to produce the manuscript tables and plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(logits)
    return e / np.sum(e, axis=axis, keepdims=True)


class ClassificationMetrics:
    @staticmethod
    def accuracy(y_true, y_pred) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float((y_true == y_pred).mean())

    @staticmethod
    def precision_recall_f1(y_true, y_pred, num_classes: Optional[int] = None) -> Dict[str, float]:
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        if num_classes is None:
            num_classes = int(max(y_true.max(), y_pred.max()) + 1)
        precisions, recalls, f1s, supports = [], [], [], []
        for c in range(num_classes):
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            support = np.sum(y_true == c)
            p = tp / (tp + fp + 1e-12)
            r = tp / (tp + fn + 1e-12)
            f = 2 * p * r / (p + r + 1e-12)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
            supports.append(support)
        supports = np.asarray(supports, dtype=float)
        weights = supports / (supports.sum() + 1e-12)
        return {
            "precision_macro": float(np.mean(precisions)),
            "recall_macro": float(np.mean(recalls)),
            "macro_f1": float(np.mean(f1s)),
            "weighted_f1": float(np.sum(weights * np.asarray(f1s))),
        }

    @staticmethod
    def topk_accuracy(y_true, prob, k: int = 5) -> float:
        y_true = np.asarray(y_true, dtype=int)
        prob = np.asarray(prob)
        top = np.argsort(prob, axis=1)[:, -k:]
        return float(np.mean([y_true[i] in top[i] for i in range(len(y_true))]))


class CalibrationMetrics:
    @staticmethod
    def expected_calibration_error(y_true, prob, n_bins: int = 15) -> float:
        y_true = np.asarray(y_true, dtype=int)
        prob = np.asarray(prob, dtype=float)
        conf = prob.max(axis=1)
        pred = prob.argmax(axis=1)
        correct = (pred == y_true).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf >= lo) & (conf < hi)
            if m.any():
                ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
        return float(ece)

    @staticmethod
    def brier_score(y_true, prob) -> float:
        y_true = np.asarray(y_true, dtype=int)
        prob = np.asarray(prob, dtype=float)
        y_one = np.zeros_like(prob)
        y_one[np.arange(len(y_true)), y_true] = 1.0
        return float(np.mean(np.sum((prob - y_one) ** 2, axis=1)))

    @staticmethod
    def reliability_curve(y_true, prob, n_bins: int = 12):
        y_true = np.asarray(y_true, dtype=int)
        prob = np.asarray(prob, dtype=float)
        conf = prob.max(axis=1)
        pred = prob.argmax(axis=1)
        correct = (pred == y_true).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        out = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf >= lo) & (conf < hi)
            if m.any():
                out.append((float((lo + hi) / 2), float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
            else:
                out.append((float((lo + hi) / 2), float("nan"), float("nan"), 0))
        return out


class ExplanationMetrics:
    @staticmethod
    def deletion_auc(prob_curve: np.ndarray) -> float:
        x = np.linspace(0, 1, len(prob_curve))
        return float(np.trapz(prob_curve, x))

    @staticmethod
    def insertion_score(prob_curve: np.ndarray) -> float:
        x = np.linspace(0, 1, len(prob_curve))
        return float(np.trapz(prob_curve, x))

    @staticmethod
    def background_activation_ratio(activation_map: np.ndarray, foreground_mask: np.ndarray) -> float:
        activation_map = np.asarray(activation_map, dtype=float)
        foreground_mask = np.asarray(foreground_mask, dtype=bool)
        total = activation_map.sum() + 1e-12
        background = activation_map[~foreground_mask].sum()
        return float(background / total)

    @staticmethod
    def deletion_impact_score(original_conf: float, removed_conf: float) -> float:
        return float(np.clip(original_conf - removed_conf, 0.0, 1.0))


class UncertaintyMetrics:
    @staticmethod
    def uncertainty_error_correlation(uncertainty, is_error) -> float:
        uncertainty = np.asarray(uncertainty, dtype=float)
        is_error = np.asarray(is_error, dtype=float)
        if uncertainty.std() < 1e-12 or is_error.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(uncertainty, is_error)[0, 1])

    @staticmethod
    def high_uncertainty_error_rate(uncertainty, is_error, quantile: float = 0.80) -> float:
        uncertainty = np.asarray(uncertainty, dtype=float)
        is_error = np.asarray(is_error, dtype=float)
        th = np.quantile(uncertainty, quantile)
        return float(is_error[uncertainty >= th].mean())

    @staticmethod
    def uncertainty_summary(uncertainty, is_error):
        u = np.asarray(uncertainty, dtype=float)
        e = np.asarray(is_error, dtype=bool)
        return {
            "uncertainty_correct": float(u[~e].mean()) if np.any(~e) else float("nan"),
            "uncertainty_error": float(u[e].mean()) if np.any(e) else float("nan"),
            "uncertainty_gap": float(u[e].mean() - u[~e].mean()) if np.any(e) and np.any(~e) else float("nan"),
            "uncertainty_error_corr": UncertaintyMetrics.uncertainty_error_correlation(u, e),
            "high_uncertainty_error_rate": UncertaintyMetrics.high_uncertainty_error_rate(u, e),
        }
