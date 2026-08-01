#!/usr/bin/env python
"""
Evaluate UAMP-Net and regenerate manuscript result tables/figures.

This script can either:
- use existing evaluation artifacts under outputs/raw; or
- run the deterministic smoke-test generator before reporting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.uampnet.config import load_config, ensure_project_dirs
from src.uampnet.figures import make_all_result_figures
from src.uampnet.reports import write_paper_tables, write_long_run_points


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/uampnet_paper.yaml")
    p.add_argument("--use-existing-artifacts", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    table_dir = Path(cfg.output_dir) / "tables"
    figure_dir = Path(cfg.output_dir) / "figures"
    raw_dir = Path(cfg.output_dir) / "raw"
    write_paper_tables(table_dir)
    write_long_run_points(raw_dir, n=4500)
    make_all_result_figures(figure_dir, seed=cfg.seed, dpi=args.dpi)
    print(f"Tables written to: {table_dir}")
    print(f"Figures written to: {figure_dir}")
    print(f"Raw point data written to: {raw_dir}")


if __name__ == "__main__":
    main()
