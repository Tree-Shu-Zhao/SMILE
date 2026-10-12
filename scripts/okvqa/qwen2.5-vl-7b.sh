#!/usr/bin/env bash
# OK-VQA with Qwen/Qwen2.5-VL-7B-Instruct. Run from the repository root while the CLIP
# server and the VLM server (serving this model) are up. Extra Hydra overrides are passed through.
set -euo pipefail

uv run src/run_batch_inference.py \
  data=okvqa \
  model=Qwen/Qwen2.5-VL-7B-Instruct \
  k=12 \
  negative_k=8 \
  num_runs=1 \
  "$@"
