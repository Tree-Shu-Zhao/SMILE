#!/usr/bin/env bash
# CUB-200-2011 with Qwen/Qwen2.5-VL-7B-Instruct. Run from the repository root while the CLIP
# server and the VLM server (serving this model) are up. Extra Hydra overrides are passed through.
set -euo pipefail

uv run src/run_batch_inference.py \
  data=cub \
  model=Qwen/Qwen2.5-VL-7B-Instruct \
  k=8 \
  negative_k=4 \
  num_runs=1 \
  "$@"
