"""Hard-negative mining via self-consistency voting.

For each training sample, runs ``num_votes`` zero-shot completions with
non-zero temperature, takes the majority vote as the model's "voted prediction",
and keeps samples whose voted prediction differs from the ground truth as
hard negatives.

Usage:
    uv run src/mine_hard_negatives.py data=cub model=Qwen/Qwen2.5-VL-7B-Instruct num_votes=3

The pool is mined from the model the VLM server serves (``model=`` must name it) and written
to ``hard_negatives/<data>/<org_model>/hard_negatives.jsonl`` unless ``save_root`` (a .jsonl path)
is given.
"""

import json
import os
import string
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from smile import build_zero_shot_content
from vlm_client import image_content


def normalize_prediction(pred: str) -> str:
    return pred.replace("_", " ").strip().lower()


def majority_vote(completions: List[str]) -> Tuple[str, int, int]:
    if not completions:
        return ("", 0, 0)
    normalized = [c.replace("_", " ").strip().lower() for c in completions]
    counter = Counter(normalized)
    winner, count = counter.most_common(1)[0]
    for c in completions:
        if c.replace("_", " ").strip().lower() == winner:
            return (c.strip(), count, len(completions))
    return (completions[0].strip(), count, len(completions))


def compute_token_f1(pred: str, gt: str) -> float:
    translator = str.maketrans("", "", string.punctuation)
    pred_tokens = pred.replace("_", " ").strip().lower().translate(translator).split()
    gt_tokens = gt.replace("_", " ").strip().lower().translate(translator).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def construct_hard_negatives(
    vlm_client,
    train_ids: List[str],
    train_items: List[Dict[str, Any]],
    num_votes: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 512,
    task: str = "vqa",
    options: Optional[List[str]] = None,
    max_side: int = 2048,
    predictions_path: Optional[str] = None,
    save_path: Optional[str] = None,
    max_concurrent: int = 64,
    use_tqdm: bool = False,
    sample_retries: int = 3,
) -> str:
    """Mine hard negatives via majority-vote self-consistency on the training set."""
    records: List[Optional[Dict[str, Any]]] = [None] * len(train_ids)

    def _process_one(idx: int) -> None:
        item = train_items[idx]
        query_id = train_ids[idx]
        completions = vlm_client.chat(
            build_zero_shot_content(item["question"], image_content(item["imgpath"], max_side), task, options),
            temperature=temperature,
            max_tokens=max_tokens,
            n=num_votes,
        )
        voted_pred, vote_count, total = majority_vote(completions)
        gt = item["answer"]
        is_correct = normalize_prediction(voted_pred) == normalize_prediction(gt)
        token_f1 = compute_token_f1(voted_pred, gt)
        rec = {
            "id": query_id,
            "question": item["question"],
            "imgpath": item.get("imgpath", ""),
            "gt_answer": gt,
            "voted_pred": voted_pred,
            "vote_count": vote_count,
            "total_votes": total,
            "token_f1": round(token_f1, 4),
            "is_correct": is_correct,
        }
        records[idx] = rec

    # Failed samples (server/API errors) are retried. No pool is written from an incomplete pass:
    # a missing training sample would silently change the hard negatives. Requests the server
    # rejects (4xx other than 429) and rounds in which nothing succeeds (the server is down) abort
    # at once.
    pending = list(range(len(train_ids)))
    for attempt in range(sample_retries + 1):
        if attempt:
            print(f"{len(pending)} samples failed; retry {attempt}/{sample_retries} in {30 * attempt}s")
            time.sleep(30 * attempt)
        failed, rejected = [], {}
        pool = ThreadPoolExecutor(max_workers=max_concurrent)
        try:
            futures = {pool.submit(_process_one, i): i for i in pending}
            if use_tqdm:
                from tqdm import tqdm
                futures_iter = tqdm(as_completed(futures), total=len(futures), desc="Mining hard negatives")
            else:
                futures_iter = as_completed(futures)
            done = 0
            for future in futures_iter:
                try:
                    future.result()
                except Exception as e:
                    failed.append(futures[future])
                    print(f"Warning: sample failed: {e}")
                    if getattr(e, "retryable", True) is False:
                        rejected[train_ids[futures[future]]] = str(e)
                done += 1
                if not use_tqdm and (done % 500 == 0 or done == len(pending)):
                    print(f"Mining hard negatives: {done}/{len(pending)}")
        except BaseException:  # e.g. Ctrl-C: drop the queued samples instead of running them all
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
        if rejected:
            raise RuntimeError(f"the server rejected {len(rejected)} training samples (re-running cannot help), "
                               f"e.g. {next(iter(rejected.items()))}; no hard-negative pool written.")
        if failed and len(failed) == len(pending):
            raise RuntimeError(f"all {len(pending)} pending training samples failed; is the VLM server up? "
                               f"No hard-negative pool written.")
        pending = failed
        if not pending:
            break
    if pending:
        raise RuntimeError(f"{len(pending)} training samples still failed after {sample_retries} retries; "
                           f"no hard-negative pool written. Re-run once the VLM server is healthy.")

    all_records = [r for r in records if r is not None]
    hard_negatives = [r for r in all_records if not r["is_correct"]]

    # Pool order: voted_pred groups by size; within a group, gt classes interleaved by vote_count.
    by_pred: Dict[str, Dict[str, List]] = {}
    for rec in hard_negatives:
        pred_key = normalize_prediction(rec["voted_pred"])
        gt_key = normalize_prediction(rec["gt_answer"])
        by_pred.setdefault(pred_key, {}).setdefault(gt_key, []).append(rec)
    for pred_key in by_pred:
        for gt_key in by_pred[pred_key]:
            by_pred[pred_key][gt_key].sort(key=lambda r: r["vote_count"], reverse=True)
    sorted_hn: List[Dict[str, Any]] = []
    for pred_key in sorted(by_pred, key=lambda k: -sum(len(v) for v in by_pred[k].values())):
        gt_groups = sorted(by_pred[pred_key].values(), key=lambda g: -g[0]["vote_count"])
        iterators = [iter(g) for g in gt_groups]
        while iterators:
            remaining = []
            for it in iterators:
                val = next(it, None)
                if val is not None:
                    sorted_hn.append(val)
                    remaining.append(it)
            iterators = remaining
    hard_negatives = sorted_hn

    if save_path is None:
        save_path = "hard_negatives.jsonl"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for rec in hard_negatives:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if predictions_path:
        with open(predictions_path, "w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    correct = [r for r in all_records if r["is_correct"]]
    incorrect = [r for r in all_records if not r["is_correct"]]
    f1_values = [r["token_f1"] for r in all_records]
    vote_counts = [r["vote_count"] for r in all_records]
    stats = {
        "total_train_samples": len(train_ids),
        "total_processed": len(all_records),
        "total_correct": len(correct),
        "total_incorrect": len(incorrect),
        "hard_negatives_count": len(hard_negatives),
        "num_votes": num_votes,
        "token_f1": {
            "mean": round(np.mean(f1_values).item(), 4) if f1_values else 0,
            "median": round(float(np.median(f1_values)), 4) if f1_values else 0,
            "min": round(min(f1_values), 4) if f1_values else 0,
            "max": round(max(f1_values), 4) if f1_values else 0,
        },
        "vote_count": {
            "mean": round(np.mean(vote_counts).item(), 4) if vote_counts else 0,
            "distribution": dict(Counter(vote_counts)),
        },
    }
    stats_path = os.path.splitext(save_path)[0] + "_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved mining stats to {stats_path}")

    return save_path


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig, OmegaConf
    from pathlib import Path

    from load_data import load_dataset
    from vlm_client import VLMClient

    SRC_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SRC_DIR.parent

    @hydra.main(version_base=None, config_path="../configs", config_name="mining")
    def cli_main(cfg: DictConfig) -> None:
        VALID_DATA = {"okvqa", "vizwiz", "cub", "flowers"}
        assert cfg.data in VALID_DATA, f"data={cfg.data} not in {VALID_DATA}"

        vlm_client = VLMClient(model=cfg.model, api_base=cfg.api_base)
        served = vlm_client.served_models()
        if cfg.model not in served:
            raise ValueError(f"The VLM server at {cfg.api_base} serves {served}, not model={cfg.model}.")

        if cfg.save_root:
            save_path = cfg.save_root
        else:
            save_path = str(REPO_ROOT / "hard_negatives" / cfg.data / cfg.model.replace("/", "_")
                            / "hard_negatives.jsonl")
        predictions_path = os.path.splitext(save_path)[0] + "_predictions.jsonl"

        train_ds = load_dataset(cfg.data, split="train")
        train_ids = [str(sid) for sid in train_ds.ids]
        train_items = [train_ds[i] for i in range(len(train_ds))]

        construct_hard_negatives(
            vlm_client=vlm_client,
            train_ids=train_ids,
            train_items=train_items,
            num_votes=cfg.num_votes,
            task=train_ds.task,
            options=train_ds.options,
            max_side=train_ds.max_side,
            predictions_path=predictions_path,
            save_path=save_path,
            max_concurrent=cfg.get("max_concurrent", 64),
            use_tqdm=cfg.tqdm,
            sample_retries=cfg.get("sample_retries", 3),
        )
        params_path = os.path.splitext(save_path)[0] + "_params.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2, ensure_ascii=False)
        print(f"Hard negatives saved to {save_path}")

    cli_main()
