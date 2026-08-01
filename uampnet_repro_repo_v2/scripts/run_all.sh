#!/usr/bin/env bash
set -euo pipefail

python generate_demo_artifacts.py --config configs/uampnet_paper.yaml --rows 4500
python evaluate_and_report.py --config configs/uampnet_paper.yaml --use-existing-artifacts
