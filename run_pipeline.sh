#!/usr/bin/env bash
set -euo pipefail

python reorganize_dataset.py
python src/experiments/phase3_features.py
python src/experiments/phase4_classification.py