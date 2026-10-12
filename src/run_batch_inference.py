"""Batch inference entry point.

Runs SMILE over a dataset's test split, supporting repeated runs (``num_runs``; greedy decoding,
no per-run seed) and partial sampling (``sample_ratio``). Writes per-sample predictions to JSONL
and a summary JSON.
"""

import fcntl
import glob
import hashlib
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set

import hydra
from omegaconf import DictConfig, OmegaConf

from clip_client import CLIPClient
from evaluate_results import POSTPROCESS_RULE, sample_indices, score_predictions
from load_data import load_dataset
from smile import SMILE
from vlm_client import VLMClient


SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent


def _load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    items, seen = [], set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sid = str(item.get("id", ""))
            if sid and sid not in seen:
                seen.add(sid)
                items.append(item)
    return items


def _load_processed_ids(path: str) -> Set[str]:
    return {str(item["id"]) for item in _load_jsonl(path)}


def _build_save_path(save_root: str, data: str, model_name: str,
                     k: int, negative_k: int,
                     sample_ratio: float = 1.0, sample_seed: int = 42) -> str:
    model_dir = model_name.replace("/", "_")
    save_dir = os.path.join(save_root, data, model_dir)
    os.makedirs(save_dir, exist_ok=True)
    fname = f"vicl_{k}_{negative_k}"
    if sample_ratio < 1.0:
        fname += f"_sample{sample_ratio}_seed{sample_seed}"
    fname += ".jsonl"
    return os.path.join(save_dir, fname)


def _to_record(item: Dict, output: Dict) -> Dict:
    return {
        "id": item["id"],
        "question": item["question"],
        "imgpath": item["imgpath"],
        "pred_answer": output.get("answer", ""),
        "gt_answer": item.get("answer", ""),
        "k": output.get("k"),
        "retrieved_ids": output.get("retrieved_ids"),
        "contrastive_ids": output.get("contrastive_ids"),
        "query_prediction": output.get("query_prediction", ""),
    }


def _resolve_pool(cfg: DictConfig) -> str:
    """``auto``: the pool of this (data, model) under hard_negatives/; otherwise a path."""
    path = cfg.hard_negatives_path
    if path == "auto":
        path = str(REPO_ROOT / "hard_negatives" / cfg.data / cfg.model.replace("/", "_") / "hard_negatives.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No hard-negative pool at {path}: download the released pools (see README), "
                                f"mine one (src/mine_hard_negatives.py) or set hard_negatives_path.")
    return path


# Settings that change the predictions, with the values assumed when a parameter file lacks them.
_PREDICTION_SETTINGS = {"model": None, "data": None, "k": None, "negative_k": None, "temperature": None,
                        "max_tokens": None, "sample_ratio": 1.0, "sample_seed": 42}


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _lock_output(stem: str):
    """Exclusive lock on one output, held until the process exits: two commands writing to the same
    predictions at once would mix them."""
    lock = open(stem + ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(f"Another run is writing to {stem}*; let it finish or give this run its own "
                           f"output (save_path=<file>).") from None
    return lock


def _check_resume(params_path: str, stem: str, ext: str, cfg: DictConfig, identity: Dict) -> None:
    """Refuse to resume (and score) predictions made with other settings, another pool or another
    vLLM version. ``<stem>.jsonl`` (num_runs=1) and ``<stem>_run<i>.jsonl`` share one parameter file."""
    existing = [p for p in [stem + ext] + sorted(glob.glob(glob.escape(stem) + "_run[0-9]*" + ext))
                if _load_processed_ids(p)]
    if not existing or not os.path.exists(params_path):
        return
    with open(params_path, "r", encoding="utf-8") as f:
        old = json.load(f)
    new = OmegaConf.to_container(cfg, resolve=True)
    diff = [f"{key}: {old.get(key, d)!r} -> {new.get(key, d)!r}" for key, d in _PREDICTION_SETTINGS.items()
            if old.get(key, d) != new.get(key, d)]
    old_identity = old.get("run_identity") or {}
    diff += [f"{key}: {old_identity[key]!r} -> {value!r}" for key, value in identity.items()
             if old_identity.get(key) is not None and value is not None and old_identity[key] != value]
    if diff:
        raise ValueError(f"{existing[0]} holds predictions made with other settings ({'; '.join(diff)}); "
                         f"resuming would mix them. Give this configuration its own output "
                         f"(save_path=<file>) or remove the old files.")


def _check_setup(cfg: DictConfig, vlm_client: VLMClient) -> None:
    """Fail before the first request on a configuration the servers cannot run."""
    max_images = int(cfg.get("max_images_per_prompt", 36))
    if cfg.k + cfg.negative_k + 1 > max_images:
        raise ValueError(f"k={cfg.k} + negative_k={cfg.negative_k} + 1 query image exceeds the {max_images} images "
                         f"per prompt that vLLM accepts (--limit-mm-per-prompt in scripts/start_vlm_server.sh).")
    served = vlm_client.served_models()
    if cfg.model not in served:
        raise ValueError(f"The VLM server at {cfg.api_base} serves {served}, not model={cfg.model}.")


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig) -> None:
    save_root = str(REPO_ROOT / cfg.save_root) if not os.path.isabs(cfg.save_root) else cfg.save_root

    clip_client = CLIPClient(api_base=cfg.clip_api_base)
    vlm_client = VLMClient(model=cfg.model, api_base=cfg.api_base)
    hn_path = _resolve_pool(cfg)
    _check_setup(cfg, vlm_client)
    vllm_version = vlm_client.vllm_version()

    train_dataset = load_dataset(cfg.data, split="train")
    test_dataset = load_dataset(cfg.data, split="test")

    smile = SMILE(
        clip_client=clip_client,
        vlm_client=vlm_client,
        train_dataset=train_dataset,
        dataset_name=cfg.data,
        hard_negatives_path=hn_path,
        k=cfg.k,
        negative_k=cfg.negative_k,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )

    if cfg.save_path:
        base_save_path = cfg.save_path
        os.makedirs(os.path.dirname(base_save_path) or ".", exist_ok=True)
    else:
        base_save_path = _build_save_path(
            save_root=save_root,
            data=cfg.data,
            model_name=cfg.model,
            k=cfg.k,
            negative_k=cfg.negative_k,
            sample_ratio=cfg.sample_ratio,
            sample_seed=cfg.sample_seed,
        )

    num_runs = cfg.num_runs
    stem, ext = os.path.splitext(base_save_path)
    run_paths = [base_save_path] if num_runs == 1 else [f"{stem}_run{i + 1}{ext}" for i in range(num_runs)]
    params_path = stem + "_params.json"
    output_lock = _lock_output(stem)  # noqa: F841  (held until the process exits)
    identity = {"hard_negatives_sha256": _file_sha256(hn_path), "vllm_version": vllm_version}
    _check_resume(params_path, stem, ext, cfg, identity)
    params = OmegaConf.to_container(cfg, resolve=True)
    params["run_identity"] = identity
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"Saved run parameters to {params_path}")

    indices = sample_indices(len(test_dataset), cfg.sample_ratio, cfg.sample_seed)
    if cfg.sample_ratio < 1.0:
        print(f"Sampled {len(indices)}/{len(test_dataset)} test samples "
              f"(sample_ratio={cfg.sample_ratio}, sample_seed={cfg.sample_seed})")

    def _run_once(save_path: str, run_label: str) -> Dict:
        write_lock = threading.Lock()

        def _process_one(idx: int) -> None:
            item = test_dataset[idx]
            record = _to_record(item, smile.predict(item))
            with write_lock:
                with open(save_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Failed samples (server/API errors) are retried; a sample that still fails is scored as
        # wrong, never dropped from the denominator. Requests the server rejects (4xx other than
        # 429, e.g. a prompt vLLM refuses) are not retried.
        retries = int(cfg.get("sample_retries", 3))
        rejected: Dict[int, str] = {}
        for attempt in range(retries + 1):
            processed_ids = _load_processed_ids(save_path)
            pending = [idx for idx in indices
                       if str(test_dataset[idx]["id"]) not in processed_ids and idx not in rejected]
            if not pending:
                break
            if attempt:
                print(f"[{run_label}] {len(pending)} samples failed; retry {attempt}/{retries} in {30 * attempt}s")
                time.sleep(30 * attempt)
            print(f"[{run_label}] Running {len(pending)} samples with up to {cfg.max_concurrent} concurrent requests")
            succeeded = 0
            pool = ThreadPoolExecutor(max_workers=cfg.max_concurrent)
            try:
                futures = {pool.submit(_process_one, idx): idx for idx in pending}
                if cfg.tqdm:
                    from tqdm import tqdm
                    futures_iter = tqdm(as_completed(futures), total=len(futures), desc=run_label)
                else:
                    futures_iter = as_completed(futures)
                for future in futures_iter:
                    try:
                        future.result()
                        succeeded += 1
                    except Exception as e:
                        print(f"Warning: sample failed: {e}")
                        if getattr(e, "retryable", True) is False:
                            rejected[futures[future]] = str(e)
            except BaseException:  # e.g. Ctrl-C: drop the queued samples instead of running them all
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            pool.shutdown(wait=True)
            if succeeded == 0:
                if attempt == 0 and not processed_ids:
                    why = (f"the server rejected every request, e.g. {next(iter(rejected.values()))}"
                           if len(rejected) == len(pending) else
                           f"are the VLM server ({cfg.api_base}) and the CLIP server up? Errors are printed above")
                    raise RuntimeError(f"[{run_label}] all {len(pending)} samples failed: {why}")
                print(f"[{run_label}] no sample succeeded in this round; giving up on the remaining failures")
                break

        res = score_predictions(_load_jsonl(save_path), [test_dataset[idx] for idx in indices], cfg.data)
        if res["missing"]:
            print(f"[{run_label}] WARNING: {res['missing']} samples still failed ({len(rejected)} rejected by the "
                  f"server, not retried) and are scored as wrong; re-running the same command retries only "
                  f"these samples.")
        print(f"[{run_label}] EM: {res['em'] * 100:.2f}% | F1: {res['f1'] * 100:.2f}% "
              f"(post-process changed {res['stripped']} answers)")
        res["rejected"] = len(rejected)
        return res

    results = []
    for run_idx, save_path in enumerate(run_paths):
        label = "Run" if num_runs == 1 else f"Run {run_idx + 1}/{num_runs}"
        results.append(_run_once(save_path, label))
    em_scores, f1_scores = [r["em"] for r in results], [r["f1"] for r in results]

    mean_em = sum(em_scores) / len(em_scores)
    mean_f1 = sum(f1_scores) / len(f1_scores)
    if len(em_scores) > 1:
        std_em = math.sqrt(sum((e - mean_em) ** 2 for e in em_scores) / (len(em_scores) - 1))
        std_f1 = math.sqrt(sum((e - mean_f1) ** 2 for e in f1_scores) / (len(f1_scores) - 1))
        print(f"\n=== Summary ({num_runs} runs) ===")
        print(f"EM per run: {['%.2f%%' % (e * 100) for e in em_scores]}")
        print(f"F1 per run: {['%.2f%%' % (e * 100) for e in f1_scores]}")
        print(f"EM mean: {mean_em * 100:.2f}% | std: {std_em * 100:.2f}%")
        print(f"F1 mean: {mean_f1 * 100:.2f}% | std: {std_f1 * 100:.2f}%")
    else:
        std_em = std_f1 = 0.0
        print(f"\nEM: {mean_em * 100:.2f}% | F1: {mean_f1 * 100:.2f}%")

    summary = {
        "num_runs": num_runs,
        "em_per_run": [round(e * 100, 2) for e in em_scores],
        "em_mean": round(mean_em * 100, 2),
        "em_std": round(std_em * 100, 2),
        "f1_per_run": [round(e * 100, 2) for e in f1_scores],
        "f1_mean": round(mean_f1 * 100, 2),
        "f1_std": round(std_f1 * 100, 2),
        "n_expected": len(indices),
        "failed_scored_wrong_per_run": [r["missing"] for r in results],
        "rejected_by_server_per_run": [r["rejected"] for r in results],
        "postprocess": POSTPROCESS_RULE,
        "postprocess_changed_per_run": [r["stripped"] for r in results],
        "vllm_version": vllm_version,
        "hard_negatives_path": hn_path,
    }
    summary_path = stem + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
