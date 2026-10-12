#!/usr/bin/env bash
# VizWiz with google/gemma-3-12b-it. Run from the repository root while the CLIP
# server and the VLM server (serving this model) are up. Extra Hydra overrides are passed through.
set -euo pipefail

uv run src/run_batch_inference.py \
  data=vizwiz \
  model=google/gemma-3-12b-it \
  k=24 \
  negative_k=8 \
  num_runs=1 \
  "$@"
