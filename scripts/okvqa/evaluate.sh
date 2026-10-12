#!/usr/bin/env bash
# Score OK-VQA predictions. Run from the repository root.
# Usage: bash scripts/okvqa/evaluate.sh <predictions.jsonl>
set -euo pipefail

uv run src/evaluate_results.py \
  --data okvqa \
  --pred_file "$1"
