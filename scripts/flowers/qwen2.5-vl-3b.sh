#!/usr/bin/env bash
# Oxford 102 Flowers with Qwen/Qwen2.5-VL-3B-Instruct. Run from the repository root while the CLIP
# server and the VLM server (serving this model) are up. Extra Hydra overrides are passed through.
set -euo pipefail

uv run src/run_batch_inference.py \
  data=flowers \
  model=Qwen/Qwen2.5-VL-3B-Instruct \
  k=8 \
  negative_k=4 \
  num_runs=1 \
  "$@"
