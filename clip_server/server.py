"""CLIP retrieval server.

Serves RICES retrieval over precomputed image embeddings read from
``<embeddings-dir>/<dataset>_<split>_{image,ids}.pt`` (generate them with
``scripts/generate_clip_embeddings.py``), plus text encoding of predictions for hard-negative
selection, with CLIP ViT-g-14 (laion2b_s12b_b42k) on the GPU (``--gpu``) or the CPU.

Usage:
    python server.py --port 8001 --datasets cub --embeddings-dir ./clip_embeddings [--gpu]
"""

import argparse
import os
import time
from typing import Dict, List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_NAME = "ViT-g-14"
PRETRAINED = "laion2b_s12b_b42k"

_clip_model = None
_clip_tokenizer = None
_clip_device = None

_datasets: Dict[str, Dict] = {}


class RetrieveRequest(BaseModel):
    dataset: str
    query_id: str
    k: int = 16


class RetrieveResponse(BaseModel):
    ids: List[str]


class EncodeTextRequest(BaseModel):
    texts: List[str]


class EncodeTextResponse(BaseModel):
    embeddings: List[List[float]]


def _load_clip_model(device: str = "cuda") -> None:
    global _clip_model, _clip_tokenizer, _clip_device
    import open_clip

    _clip_device = device
    print(f"[startup] Loading open_clip {MODEL_NAME} (pretrained={PRETRAINED}) on {device}...")
    t0 = time.time()
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=device)
    _clip_tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    _clip_model = model.eval()
    print(f"[startup] CLIP model loaded in {time.time()-t0:.1f}s")


@torch.no_grad()
def _encode_texts(texts: List[str]) -> torch.Tensor:
    tokens = _clip_tokenizer(texts).to(_clip_device)
    embs = _clip_model.encode_text(tokens)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    return embs.cpu().float()


def _load_all(datasets: List[str], embeddings_dir: str) -> None:
    t0 = time.time()
    for ds_name in datasets:
        for split in ["train", "test"]:
            paths = {kind: os.path.join(embeddings_dir, f"{ds_name}_{split}_{kind}.pt") for kind in ("image", "ids")}
            for path in paths.values():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Embedding file not found: {path}. Download the released embeddings "
                                            f"or generate them with scripts/generate_clip_embeddings.py.")
            embeddings = torch.load(paths["image"], map_location="cpu")
            ids = torch.load(paths["ids"], map_location="cpu")
            _datasets[f"{ds_name}/{split}"] = {
                "ids": ids,
                "embeddings": embeddings,
                "id_to_idx": {sid: i for i, sid in enumerate(ids)},
            }
            print(f"[startup] {ds_name}/{split}: {len(ids)} images {tuple(embeddings.shape)}")
    print(f"[startup] All datasets indexed in {time.time()-t0:.1f}s: {list(_datasets.keys())}")


app = FastAPI(title="CLIP Retrieval Server")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "pretrained": PRETRAINED, "indexed_datasets": list(_datasets.keys())}


@app.post("/v1/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    """Top-k training images by cosine similarity to the test image ``query_id``."""
    train, test = _datasets.get(f"{req.dataset}/train"), _datasets.get(f"{req.dataset}/test")
    if train is None or test is None:
        raise HTTPException(status_code=404, detail=f"Dataset {req.dataset} not indexed")
    q_idx = test["id_to_idx"].get(req.query_id)
    if q_idx is None:
        raise HTTPException(status_code=404, detail=f"Query ID {req.query_id} not found in {req.dataset}/test")

    query_emb = test["embeddings"][q_idx:q_idx + 1]
    sims = (query_emb @ train["embeddings"].T).squeeze(0).numpy()
    topk = min(req.k, len(train["ids"]))
    top_indices = np.argsort(sims)[-topk:][::-1]
    return RetrieveResponse(ids=[train["ids"][i] for i in top_indices])


@app.post("/v1/encode_text", response_model=EncodeTextResponse)
def encode_text(req: EncodeTextRequest):
    return EncodeTextResponse(embeddings=_encode_texts(req.texts).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP retrieval server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--datasets", type=str, nargs="+", default=["cub"],
                        help="Datasets to load embeddings for")
    parser.add_argument("--embeddings-dir", type=str, default="clip_embeddings",
                        help="Directory holding the precomputed *_{image,ids}.pt files")
    parser.add_argument("--gpu", action="store_true", help="Run the text encoder on the GPU")
    args = parser.parse_args()

    _load_all(args.datasets, args.embeddings_dir)
    _load_clip_model(device="cuda" if args.gpu else "cpu")

    uvicorn.run(app, host=args.host, port=args.port)
