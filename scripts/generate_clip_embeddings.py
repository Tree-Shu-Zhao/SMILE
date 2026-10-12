"""Generate the CLIP image embeddings served by clip_server/server.py.

Saves ``<dataset>_<split>_{image,ids}.pt`` for the train and test splits under ``--out_dir``, with
CLIP ViT-g-14 (laion2b_s12b_b42k). Existing files are skipped.

Example:
    uv run --with torch==2.6.0 --with torchvision==0.21.0 --with open_clip_torch==3.3.0 \\
        scripts/generate_clip_embeddings.py \\
        --datasets cub flowers okvqa vizwiz --out_dir ./clip_embeddings
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import open_clip
import torch
from PIL import Image
from tqdm import tqdm

from load_data import load_dataset

MODEL_NAME = "ViT-g-14"
PRETRAINED = "laion2b_s12b_b42k"


def open_image(path, max_side=512):
    """RGB image with the longer side resized to at most ``max_side`` (bicubic); black if unreadable."""
    try:
        with Image.open(path) as f:
            img = f.convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), color="black")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return img


@torch.no_grad()
def encode_images(model, preprocess, paths, device, batch_size=64):
    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc="encode_images"):
        batch = torch.stack([preprocess(open_image(p)) for p in paths[i:i + batch_size]]).to(device)
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.cpu().float())
    return torch.cat(embs, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cub", "flowers", "okvqa", "vizwiz"])
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--out_dir", default="./clip_embeddings")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=args.device)
    model.eval()
    os.makedirs(args.out_dir, exist_ok=True)

    for ds_name in args.datasets:
        for split in ["train", "test"]:
            print(f"\n=== {ds_name}/{split} ===")
            ds = load_dataset(ds_name, data_folder=os.path.join(args.data_dir, ds_name), split=split)
            img_path = os.path.join(args.out_dir, f"{ds_name}_{split}_image.pt")
            ids_path = os.path.join(args.out_dir, f"{ds_name}_{split}_ids.pt")
            if os.path.exists(img_path):
                print(f"  {img_path} exists, skipped")
            else:
                print(f"  encoding {len(ds.imgpaths)} images...")
                torch.save(encode_images(model, preprocess, ds.imgpaths, args.device, args.batch_size), img_path)
            if not os.path.exists(ids_path):
                torch.save([str(i) for i in ds.ids], ids_path)

    print("\nAll embeddings generated.")


if __name__ == "__main__":
    main()
