#!/usr/bin/env bash
# Oxford 102 Flowers with google/gemma-3-12b-it. Run from the repository root while the CLIP
# server and the VLM server (serving this model) are up. Extra Hydra overrides are passed through.
set -euo pipefail

uv run src/run_batch_inference.py \
  data=flowers \
  model=google/gemma-3-12b-it \
  k=8 \
  negative_k=4 \
  num_runs=1 \
  "$@"
