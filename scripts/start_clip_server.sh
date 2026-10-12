#!/usr/bin/env bash
# Build and start the CLIP retrieval server on port 8001. Run from the repository root.
# Usage: bash scripts/start_clip_server.sh [dataset ...]   (default: cub flowers okvqa vizwiz)
# Serves the embeddings in ./clip_embeddings. CLIP_DEVICE=cpu runs the text encoder on the CPU.
set -euo pipefail

if [ $# -eq 0 ]; then set -- cub flowers okvqa vizwiz; fi

docker build -t clip-server ./clip_server
docker rm -f clip-server >/dev/null 2>&1 || true
volumes=(-v "$(pwd)/clip_embeddings:/clip_embeddings" -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface")
if [ "${CLIP_DEVICE:-cuda}" = cpu ]; then
  docker run -d --name clip-server -p 8001:8001 "${volumes[@]}" \
    clip-server --embeddings-dir /clip_embeddings --datasets "$@"
else
  docker run -d --name clip-server --gpus all -p 8001:8001 "${volumes[@]}" \
    clip-server --embeddings-dir /clip_embeddings --datasets "$@" --gpu
fi

echo "Waiting for the CLIP server on port 8001 (the first start downloads the CLIP weights) ..."
until curl -sf localhost:8001/health >/dev/null; do
  if [ "$(docker inspect -f '{{.State.Running}}' clip-server 2>/dev/null)" != true ]; then
    echo "clip-server stopped; see: docker logs clip-server" >&2
    exit 1
  fi
  sleep 5
done
echo "CLIP server is up: http://localhost:8001"
