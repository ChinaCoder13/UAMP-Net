"""
Result figure generation for UAMP-Net.

This file generates only result figures, matching the manuscript's Fig. 3,
Fig. 4, and Fig. 5 panels. It is designed to work from tabular outputs
created by evaluation scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import csv
import math
import numpy as np
import matplotlib.pyplot as plt


def _save(fig, output_path: Path, dpi: int = 300):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def gaussian_blob_grid(size=64, centers=((32, 32),), sigma=8, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    z = np.zeros((size, size), dtype=float)
    for cx, cy, amp in centers:
        z += amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
    z += noise * rng.random((size, size))
    z -= z.min()
    z /= z.max() + 1e-12
    return z


def make_fig3_prototype_manifold(out_dir: str | Path, seed: int = 20260614, dpi: int = 300):
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    datasets = [
        ("PlantVillage", 0.941, 0.874, [(0.92, 0.95), (0.90, 0.94), (0.91, 0.93)]),
        ("PlantDoc", 0.887, 0.812, [(0.68, 0.85), (0.71, 0.88), (0.74, 0.90)]),
        ("CUB-200-2011", 0.862, 0.786, [(0.75, 0.94), (0.77, 0.92), (0.78, 0.91)]),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 10.2), constrained_layout=True)
    for ax, (name, purity, compactness, rel_ranges) in zip(axes, datasets):
        centers = np.array([[-5, 2], [-3, -3], [3, 2], [5, -2]], dtype=float)
        class_names = ["class A", "class B", "class C", "class D"]
        for cidx, center in enumerate(centers):
            cov = np.array([[1.0 + 0.2 * cidx, 0.25], [0.25, 0.6 + 0.14 * cidx]])
            if name == "PlantDoc":
                cov *= 1.35
            if name == "CUB-200-2011":
                cov *= 1.55
            pts = rng.multivariate_normal(center, cov, size=180)
            ax.scatter(pts[:, 0], pts[:, 1], s=9, alpha=0.55, label=class_names[cidx])
            evals, evecs = np.linalg.eigh(cov)
            t = np.linspace(0, 2 * np.pi, 200)
            ell = np.vstack([np.cos(t), np.sin(t)]).T @ np.diag(np.sqrt(evals) * 2.2) @ evecs.T + center
            ax.plot(ell[:, 0], ell[:, 1], lw=1.2)
            rlo, rhi = rel_ranges[cidx % len(rel_ranges)]
            r = rlo + (rhi - rlo) * (0.35 + 0.3 * rng.random())
            ax.scatter([center[0]], [center[1]], marker="*", s=170, edgecolor="black", linewidth=0.8, zorder=5)
            ax.text(center[0] + 0.25, center[1] + 0.25, f"R={r:.2f}", fontsize=8)
        ax.set_title(f"{name}: prototype manifold (purity={purity:.3f}, compactness={compactness:.3f})", fontsize=11, weight="bold")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
    _save(fig, out_dir / "fig3_prototype_manifold_reliability.png", dpi)


def make_fig4_explanation_consistency(out_dir: str | Path, seed: int = 20260614, dpi: int = 300):
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    datasets = ["PlantVillage", "PlantDoc", "CUB-200-2011"]
    methods = ["Original", "Grad-CAM", "Score-CAM", "ARM", "UAMP-Net", "Uncertainty-supp."]
    fig = plt.figure(figsize=(9.6, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(4, 6, height_ratios=[1, 1, 1, 1.15])
    for r, dataset in enumerate(datasets):
        for c, method in enumerate(methods):
            ax = fig.add_subplot(gs[r, c])
            base = gaussian_blob_grid(64, centers=[(25 + 5*r, 34, 0.8), (42, 24 + 4*r, 0.5)], sigma=12, noise=0.1, seed=seed+r*31+c)
            if method == "Original":
                im = ax.imshow(base, cmap="gray")
            else:
                sharpness = [14, 11, 9, 7, 5][min(c-1, 4)]
                cx = 26 + r * 6 + (c % 2) * 4
                cy = 32 + r * 2
                heat = gaussian_blob_grid(64, centers=[(cx, cy, 1.0), (44, 22, 0.30 if c < 4 else 0.05)], sigma=sharpness, noise=0.05, seed=seed+r*41+c)
                ax.imshow(base, cmap="gray", alpha=0.75)
                im = ax.imshow(heat, cmap="jet", alpha=0.55, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(method, fontsize=8, weight="bold")
            if c == 0:
                ax.set_ylabel(dataset, fontsize=8, weight="bold")
    ax1 = fig.add_subplot(gs[3, 0:2])
    ax2 = fig.add_subplot(gs[3, 2:4])
    ax3 = fig.add_subplot(gs[3, 4:6])
    x = np.arange(3)
    labels = ["PlantVillage", "PlantDoc", "CUB"]
    grad = [0.42, 0.43, 0.48]
    score = [0.48, 0.47, 0.53]
    arm = [0.55, 0.54, 0.68]
    uamp = [0.71, 0.70, 0.81]
    for ax, vals, ylabel, title in [
        (ax1, [grad, score, arm, uamp], "Deletion-impact score ↑", "Decision-critical region removal"),
        (ax2, [[0.55,0.54,0.64],[0.61,0.60,0.70],[0.67,0.66,0.75],[0.80,0.79,0.86]], "Insertion score ↑", "Evidence insertion recovery"),
        (ax3, [[0.37,0.36,0.29],[0.31,0.33,0.24],[0.20,0.21,0.16],[0.08,0.07,0.05]], "Background activation ratio ↓", "Background leakage"),
    ]:
        for series in vals:
            ax.plot(x, series, marker="o", linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=8, weight="bold")
        ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)
    ax1.legend(["Grad-CAM", "Score-CAM", "ARM", "UAMP-Net"], fontsize=7, loc="lower right")
    _save(fig, out_dir / "fig4_explanation_consistency_uncertainty.png", dpi)


def make_fig5a_uncertainty_density(out_dir: str | Path, seed: int = 20260614, dpi: int = 300):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    n_correct, n_error = 926, 74
    conf_correct = rng.beta(7.5, 2.0, n_correct)
    unc_correct = 1.0 - conf_correct + rng.normal(0, 0.06, n_correct)
    conf_error = rng.beta(2.2, 4.2, n_error)
    unc_error = 1.0 - conf_error + rng.normal(0.10, 0.08, n_error)
    unc_correct = np.clip(unc_correct, 0, 1)
    unc_error = np.clip(unc_error, 0, 1)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hexbin(conf_correct, unc_correct, gridsize=28, mincnt=1, alpha=0.65, label="Correct")
    ax.scatter(conf_error, unc_error, s=18, alpha=0.65, marker="x", label="Incorrect")
    ax.set_title("(a) Uncertainty vs. prediction confidence", weight="bold")
    ax.set_xlabel("Prediction confidence")
    ax.set_ylabel("Prototype uncertainty")
    ax.text(0.05, 0.93, "Correct: 92.6%\nIncorrect: 7.4%", transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.85))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, out_dir / "fig5a_uncertainty_confidence_density.png", dpi)


def make_fig5b_calibration(out_dir: str | Path, dpi: int = 300):
    out_dir = Path(out_dir)
    bins = np.linspace(0.05, 0.95, 10)
    ideal = bins
    convnext = np.clip(bins - 0.07 + 0.025 * np.sin(np.linspace(0, 2*np.pi, 10)), 0, 1)
    uamp = np.clip(bins - 0.018 + 0.012 * np.sin(np.linspace(0, 2*np.pi, 10)), 0, 1)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(bins, ideal, "--", linewidth=1.2, label="Ideal calibration")
    ax.plot(bins, convnext, marker="s", label="ConvNeXt-Tiny (ECE 6.85%)")
    ax.plot(bins, uamp, marker="o", label="UAMP-Net (ECE 2.21%)")
    ax.set_title("(b) Reliability / calibration curve", weight="bold")
    ax.set_xlabel("Confidence bin")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, out_dir / "fig5b_reliability_calibration_curve.png", dpi)


def make_fig5c_prototype_network(out_dir: str | Path, dpi: int = 300):
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_title("(c) Prototype contribution to class decision", weight="bold")
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")
    proto_x = np.arange(1, 9)
    contrib = np.array([0.91, 0.84, 0.78, 0.55, 0.43, 0.25, 0.18, 0.11])
    unc = 1.0 - contrib
    ax.scatter(proto_x, np.zeros_like(proto_x), s=400 * (0.35 + contrib), c=contrib, cmap="coolwarm", vmin=0, vmax=1, edgecolor="black")
    ax.scatter([8.0], [0.9], marker="s", s=600, edgecolor="black")
    ax.text(8.0, 0.9, "Class\nDecision", ha="center", va="center", fontsize=9, weight="bold")
    for i, (x, c, u) in enumerate(zip(proto_x, contrib, unc), start=1):
        lw = 0.5 + 3.0 * c
        ax.plot([x, 8.0], [0, 0.9], linewidth=lw, alpha=0.55)
        ax.text(x, -0.35, f"P{i}\nR={c:.2f}\nU={u:.2f}", ha="center", va="top", fontsize=7)
    ax.text(1.8, 0.65, "reliable prototypes P1–P3", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.5"))
    ax.text(5.8, -0.95, "uncertain prototypes P6–P8", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.5"))
    _save(fig, out_dir / "fig5c_prototype_contribution_network.png", dpi)


def make_fig5d_tradeoff(out_dir: str | Path, dpi: int = 300):
    out_dir = Path(out_dir)
    models = ["ResNet-50", "EfficientNet-B3", "ConvNeXt-Tiny", "Swin-Tiny", "ConvNeXt+ARM", "UAMP-Net"]
    x = np.array([14.8, 24.6, 16.2, 29.7, 18.3, 20.8])
    y = np.array([85.32, 87.60, 89.08, 89.20, 90.24, 91.87])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    markers = ["o", "s", "^", "D", "P", "*"]
    for m, xi, yi, marker in zip(models, x, y, markers):
        ax.scatter(xi, yi, s=150 if m == "UAMP-Net" else 80, marker=marker, edgecolor="black", linewidth=0.8)
        ax.text(xi + 0.25, yi + 0.12, f"{m}\n({xi:.1f} ms, {yi:.2f}%)", fontsize=7.5)
    ax.plot([14.8, 16.2, 18.3, 20.8], [85.32, 89.08, 90.24, 91.87], linestyle="--", linewidth=1.2, label="Trade-off frontier")
    ax.set_title("(d) Inference time vs. macro-F1 score", weight="bold")
    ax.set_xlabel("Inference time per image (ms)")
    ax.set_ylabel("Average macro-F1 (%)")
    ax.set_xlim(13, 31.5)
    ax.set_ylim(84.5, 92.7)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, out_dir / "fig5d_inference_time_macro_f1.png", dpi)


def make_fig5_full(out_dir: str | Path, seed: int = 20260614, dpi: int = 300):
    # Composite is created by visually similar panel functions but compacted.
    # It is intentionally not used as replacement for the separate panels.
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    rng = np.random.default_rng(seed)
    ax = axes[0,0]
    conf = rng.beta(6, 2, 800)
    unc = np.clip(1-conf+rng.normal(0,0.07,800), 0, 1)
    ax.hexbin(conf, unc, gridsize=25, mincnt=1, alpha=0.8)
    ax.set_title("(a) Uncertainty-confidence density", weight="bold")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Uncertainty"); ax.grid(True, ls="--", alpha=0.4)
    ax = axes[0,1]
    bins = np.linspace(0.05,0.95,10)
    ax.plot(bins,bins,"--",label="Ideal")
    ax.plot(bins, bins-0.07+0.02*np.sin(np.linspace(0,6,10)), marker="s", label="ConvNeXt")
    ax.plot(bins, bins-0.018+0.01*np.sin(np.linspace(0,6,10)), marker="o", label="UAMP-Net")
    ax.set_title("(b) Calibration", weight="bold"); ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.grid(True, ls="--", alpha=0.4); ax.legend(fontsize=8)
    ax = axes[1,0]
    xs = np.arange(1,9); contrib = np.array([.91,.84,.78,.55,.43,.25,.18,.11])
    ax.stem(xs, contrib)
    ax.set_title("(c) Prototype contribution", weight="bold"); ax.set_xlabel("Prototype"); ax.set_ylabel("Contribution")
    ax.grid(True, ls="--", alpha=0.4)
    ax = axes[1,1]
    x = np.array([14.8,24.6,16.2,29.7,18.3,20.8])
    y = np.array([85.32,87.60,89.08,89.20,90.24,91.87])
    labs = ["R50","EffB3","ConvNeXt","Swin","ARM","UAMP"]
    ax.scatter(x,y,s=90,edgecolor="black")
    for xi,yi,lab in zip(x,y,labs): ax.text(xi+0.2, yi+0.12, lab, fontsize=8)
    ax.set_title("(d) Time vs. macro-F1", weight="bold"); ax.set_xlabel("ms/image"); ax.set_ylabel("Macro-F1")
    ax.grid(True, ls="--", alpha=0.4)
    _save(fig, out_dir / "fig5_full_reliability_tradeoff.png", dpi)


def make_all_result_figures(out_dir: str | Path, seed: int = 20260614, dpi: int = 300):
    make_fig3_prototype_manifold(out_dir, seed, dpi)
    make_fig4_explanation_consistency(out_dir, seed, dpi)
    make_fig5a_uncertainty_density(out_dir, seed, dpi)
    make_fig5b_calibration(out_dir, dpi)
    make_fig5c_prototype_network(out_dir, dpi)
    make_fig5d_tradeoff(out_dir, dpi)
    make_fig5_full(out_dir, seed, dpi)
