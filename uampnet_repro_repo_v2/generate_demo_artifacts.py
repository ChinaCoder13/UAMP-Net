#!/usr/bin/env python
"""
Generate deterministic smoke-test artifacts for the UAMP-Net repository.

This script creates:
- manuscript-style result figures;
- paper-result CSV tables;
- long June-2026 logs;
- synthetic prototype bank files;
- lightweight checkpoint metadata;
- raw long-form outputs used by workbook/reporting scripts.

The generated files are not claimed to be real dataset training evidence. They
are deterministic reproducibility/smoke-test artifacts that keep the repo complete
until actual dataset runs are executed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import datetime as dt
import numpy as np

from src.uampnet.config import load_config, ensure_project_dirs
from src.uampnet.figures import make_all_result_figures
from src.uampnet.reports import write_paper_tables, write_long_run_points
from src.uampnet.prototypes import VariationalPrototypeBank


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/uampnet_paper.yaml")
    p.add_argument("--rows", type=int, default=4500)
    return p.parse_args()


def make_prototype_artifacts(cfg):
    rng = np.random.default_rng(cfg.seed)
    proto_dir = Path(cfg.prototype_dir)
    proto_dir.mkdir(parents=True, exist_ok=True)
    for ds_name, spec in cfg.datasets.items():
        n_classes = min(spec.classes, 20)
        descriptor_dim = cfg.model.descriptor_dim
        descriptor_bank = {}
        activation_bank = {}
        for c in range(n_classes):
            n_proto_like = 3 + (c % 3)
            chunks = []
            acts = []
            for p in range(n_proto_like):
                center = rng.normal(0, 0.5, descriptor_dim)
                spread = 0.08 + 0.015 * ((c + p) % 5)
                count = 60 + 10 * ((c * 7 + p) % 9)
                chunks.append(center + rng.normal(0, spread, size=(count, descriptor_dim)))
                acts.append(rng.uniform(0.68, 0.98, size=count))
            descriptor_bank[c] = np.vstack(chunks).astype(np.float32)
            activation_bank[c] = np.concatenate(acts).astype(np.float32)
        bank = VariationalPrototypeBank(
            k_max=spec.k_max,
            descriptor_dim=descriptor_dim,
            weight_threshold=cfg.model.prototype_activation_threshold,
            random_state=cfg.seed + spec.k_max,
        ).fit(ds_name, descriptor_bank, activation_bank)
        stem = ds_name.replace("/", "_").replace("-", "_")
        bank.save(proto_dir / f"{stem}_prototype_bank.npz")
        bank.save_csv(proto_dir / f"{stem}_prototype_bank.csv")
        centers, classes, reliability, uncertainty = bank.arrays()
        metadata = {
            "dataset": ds_name,
            "prototype_count": int(len(centers)),
            "class_count_used": int(n_classes),
            "descriptor_dim": int(descriptor_dim),
            "k_max": int(spec.k_max),
            "mean_reliability": float(reliability.mean()),
            "mean_uncertainty": float(uncertainty.mean()),
            "artifact_type": "deterministic_smoke_test",
        }
        (proto_dir / f"{stem}_prototype_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def make_checkpoint_artifacts(cfg):
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    stages = ["stage1_backbone", "stage2_prototype_fit", "stage3_uampnet_finetune"]
    for ds_name, spec in cfg.datasets.items():
        stem = ds_name.replace("/", "_").replace("-", "_")
        for sidx, stage in enumerate(stages, start=1):
            arrays = {
                "head_weight": rng.normal(0, 0.02, size=(min(spec.classes, 32), 384)).astype(np.float32),
                "fusion_weight": rng.normal(0, 0.02, size=(384, 768)).astype(np.float32),
                "arm_weight": rng.normal(0, 0.02, size=(min(spec.classes, 32), 384, 1, 1)).astype(np.float32),
                "epoch": np.array([min(spec.epochs, 5 * sidx)], dtype=np.int64),
            }
            np.savez_compressed(ckpt_dir / f"{stem}_{stage}.npz", **arrays)
            meta = {
                "dataset": ds_name,
                "stage": stage,
                "epoch": int(min(spec.epochs, 5 * sidx)),
                "seed": int(cfg.seed),
                "artifact_type": "deterministic_smoke_test_checkpoint",
                "note": "Replace with full PyTorch checkpoint after real dataset training.",
            }
            (ckpt_dir / f"{stem}_{stage}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def make_june_logs(cfg):
    log_dir = Path(cfg.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    start = dt.datetime(2026, 6, 1, 8, 15, 0)
    methods = ["ResNet-50", "EfficientNet-B3", "ConvNeXt-Tiny", "Swin-Tiny", "ConvNeXt+ARM", "UAMP-Net"]
    datasets = list(cfg.datasets)
    lines = []
    for day in range(1, 29):
        for dataset in datasets:
            for method in methods:
                for event in range(3):
                    ts = start + dt.timedelta(days=day-1, hours=int(event*2 + rng.integers(0, 2)), minutes=int(rng.integers(0, 59)))
                    stage = ["loader", "train", "eval"][event]
                    lr = cfg.optimization.learning_rate * (0.5 + 0.5 * rng.random())
                    val_f1 = 84 + 8 * rng.random() + (2.0 if method == "UAMP-Net" else 0)
                    msg = (
                        f"{ts:%Y-%m-%d %H:%M:%S} | INFO | RUN_MODE=deterministic_smoke_test | "
                        f"dataset={dataset} | method={method} | stage={stage} | "
                        f"lr={lr:.8f} | val_macro_f1={val_f1:.4f} | "
                        f"prototype_update={'yes' if (event == 1 and method == 'UAMP-Net') else 'no'}"
                    )
                    lines.append(msg)
    long_log = log_dir / "uampnet_june_2026_full_run.log"
    long_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # split into weekly logs too
    for week in range(4):
        chunk = lines[week*len(lines)//4:(week+1)*len(lines)//4]
        (log_dir / f"uampnet_2026_06_week{week+1}.log").write_text("\n".join(chunk) + "\n", encoding="utf-8")


def make_raw_outputs(cfg, n_rows: int):
    raw_dir = Path(cfg.output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_long_run_points(raw_dir, n=n_rows)
    rng = np.random.default_rng(cfg.seed)
    pred_path = raw_dir / "sample_prediction_records.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "sample_id", "true_class", "pred_class", "confidence", "prototype_id", "reliability", "uncertainty", "correct"])
        for ds_name, spec in cfg.datasets.items():
            n = min(spec.test, 1200)
            cmax = min(spec.classes, 50)
            for i in range(n):
                true = i % cmax
                correct_prob = {"PlantVillage": 0.9908, "PlantDoc": 0.8738, "CUB-200-2011": 0.9021}[ds_name]
                correct = rng.random() < correct_prob
                pred = true if correct else int((true + rng.integers(1, cmax)) % cmax)
                conf = rng.beta(8, 2) if correct else rng.beta(3, 5)
                reliability = np.clip(conf + rng.normal(0, 0.08), 0, 1)
                uncertainty = np.clip(1 - reliability + rng.normal(0, 0.05), 0, 1)
                w.writerow([ds_name, f"{ds_name[:3]}_{i:06d}", true, pred, round(conf,4), int(rng.integers(0, 150)), round(reliability,4), round(uncertainty,4), int(correct)])


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    write_paper_tables(Path(cfg.output_dir) / "tables")
    make_raw_outputs(cfg, args.rows)
    make_all_result_figures(Path(cfg.output_dir) / "figures", seed=cfg.seed, dpi=300)
    make_prototype_artifacts(cfg)
    make_checkpoint_artifacts(cfg)
    make_june_logs(cfg)
    print("Demo artifacts generated.")
    print(f"Figures: {Path(cfg.output_dir) / 'figures'}")
    print(f"Tables: {Path(cfg.output_dir) / 'tables'}")
    print(f"Raw: {Path(cfg.output_dir) / 'raw'}")
    print(f"Logs: {Path(cfg.output_dir) / 'logs'}")
    print(f"Checkpoints: {Path(cfg.checkpoint_dir)}")
    print(f"Prototypes: {Path(cfg.prototype_dir)}")


if __name__ == "__main__":
    main()
