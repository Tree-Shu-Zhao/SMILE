#!/usr/bin/env bash
# Score CUB-200-2011 predictions. Run from the repository root.
# Usage: bash scripts/cub/evaluate.sh <predictions.jsonl>
set -euo pipefail

uv run src/evaluate_results.py \
  --data cub \
  --pred_file "$1"
