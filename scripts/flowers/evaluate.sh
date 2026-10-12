#!/usr/bin/env bash
# Score Oxford 102 Flowers predictions. Run from the repository root.
# Usage: bash scripts/flowers/evaluate.sh <predictions.jsonl>
set -euo pipefail

uv run src/evaluate_results.py \
  --data flowers \
  --pred_file "$1"
