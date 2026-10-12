#!/usr/bin/env bash
# Score VizWiz predictions. Run from the repository root.
# Usage: bash scripts/vizwiz/evaluate.sh <predictions.jsonl>
set -euo pipefail

uv run src/evaluate_results.py \
  --data vizwiz \
  --pred_file "$1"
