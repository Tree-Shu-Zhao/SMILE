#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for one model on port 8000. Run from the repository root.
# Usage: bash scripts/start_vlm_server.sh <model> [extra vLLM arguments]
#   e.g. bash scripts/start_vlm_server.sh Qwen/Qwen2.5-VL-7B-Instruct
# The Gemma-3 models are gated: accept their license on Hugging Face and export HF_TOKEN first.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: bash scripts/start_vlm_server.sh <model> [extra vLLM arguments]" >&2
  exit 1
fi
MODEL=$1
shift

docker rm -f vllm-server >/dev/null 2>&1 || true
docker run -d --name vllm-server --gpus all --ipc=host -p 8000:8000 \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -e HF_TOKEN \
  vllm/vllm-openai:v0.17.1 \
  --model "$MODEL" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 40960 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image": 36, "video": 0}' \
  "$@"

echo "Waiting for vLLM to serve $MODEL on port 8000 (the first start downloads the weights) ..."
until curl -sf localhost:8000/health >/dev/null; do
  if [ "$(docker inspect -f '{{.State.Running}}' vllm-server 2>/dev/null)" != true ]; then
    echo "vllm-server stopped; see: docker logs vllm-server" >&2
    exit 1
  fi
  sleep 5
done
echo "vLLM is up: http://localhost:8000/v1"
