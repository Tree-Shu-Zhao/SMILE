import argparse
import json
import math
import os
import random
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import f1_score

from load_data import load_dataset
from vizwiz_text import demonstration_label, normalize_answer, processDigitArticle, processPunctuation


SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent

# Applied to every prediction before scoring: some VLMs (e.g. Gemma-3) copy the demonstration
# format "<question>\nAnswer: <label>" and reply "Answer: <label>".
POSTPROCESS_RULE = "strip a leading 'Answer:' (case-insensitive) from pred_answer"
_ANSWER_PREFIX = re.compile(r"^\s*answer\s*:\s*", re.IGNORECASE)


def postprocess_answer(pred) -> str:
    pred = str(pred)
    return _ANSWER_PREFIX.sub("", pred, count=1).strip() if _ANSWER_PREFIX.match(pred) else pred


def postprocess_items(items: List[Dict]) -> Tuple[List[Dict], int]:
    """Copies of ``items`` with post-processed ``pred_answer``, and the number of changed answers."""
    out = [dict(it, pred_answer=postprocess_answer(it.get("pred_answer", ""))) for it in items]
    changed = sum(a["pred_answer"] != b.get("pred_answer", "") for a, b in zip(out, items))
    return out, changed


def fill_missing(items: List[Dict], expected: List[Dict]) -> Tuple[List[Dict], int]:
    """Add each expected sample without a prediction as a ``missing`` record that scores 0 on
    every metric. ``expected`` holds dataset items (``id``, ``answer``)."""
    have = {str(it.get("id", "")) for it in items}
    missing = [{"id": e["id"], "pred_answer": "", "gt_answer": e.get("answer", ""), "missing": True}
               for e in expected if str(e["id"]) not in have]
    return items + missing, len(missing)


def sample_indices(total_size: int, sample_ratio: float, seed: int) -> List[int]:
    """Indices of the test samples a ``sample_ratio < 1`` run predicts and is scored on."""
    k = max(1, int(total_size * sample_ratio))
    return sorted(random.Random(seed).sample(range(total_size), k))


_RUN_SUFFIX = re.compile(r"_run\d+$")
_SAMPLE_TAG = re.compile(r"_sample([0-9.]+)_seed(\d+)$")


def run_sampling(pred_file: str) -> Tuple[float, int, str]:
    """(sample_ratio, sample_seed, source) of the run that wrote ``pred_file``: read from the
    ``<stem>_params.json`` that run_batch_inference.py writes next to its predictions, else from
    a ``_sample<r>_seed<s>`` file name; a file of neither kind covers the full test split."""
    stem = _RUN_SUFFIX.sub("", os.path.splitext(pred_file)[0])
    params_path = stem + "_params.json"
    if os.path.exists(params_path):
        with open(params_path, "r", encoding="utf-8") as handle:
            params = json.load(handle)
        return float(params.get("sample_ratio", 1.0)), int(params.get("sample_seed", 42)), params_path
    match = _SAMPLE_TAG.search(stem)
    if match:
        return float(match.group(1)), int(match.group(2)), "file name"
    return 1.0, 42, "no _params.json next to the file"


def expected_test_items(data: str, sample_ratio: float = 1.0, sample_seed: int = 42) -> List[Dict]:
    """The test samples a run is scored on: the full test split, or its sampled subset."""
    test_dataset = load_dataset(data, split="test", metadata_only=True)
    if sample_ratio < 1.0:
        indices = sample_indices(len(test_dataset), sample_ratio, sample_seed)
    else:
        indices = range(len(test_dataset))
    return [test_dataset[i] for i in indices]

def _compute_vizwiz_scores(items: List[Dict]) -> Tuple[float, float, Dict[str, Tuple[float, float]], int]:
    anno_file = REPO_ROOT / "data" / "vizwiz" / "val.json"
    if not anno_file.exists():
        raise FileNotFoundError(f"VizWiz annotation file not found: {anno_file}")

    with open(anno_file, "r", encoding="utf-8") as handle:
        annotations = json.load(handle)

    id2answers: Dict[str, List[Dict]] = {}
    id2type: Dict[str, str] = {}
    for ann in annotations:
        key = str(ann.get("image", ""))
        if not key:
            continue
        id2answers[key] = ann.get("answers", [])
        id2type[key] = ann.get("answer_type", "unknown")

    accQA: List[float] = []
    f1QA: List[float] = []
    type2acc: Dict[str, List[float]] = {}
    type2f1: Dict[str, List[float]] = {}
    skipped = 0

    for item in items:
        sid = str(item.get("id", ""))
        if sid not in id2answers:
            skipped += 1
            continue

        resAns = str(item.get("pred_answer", "")).replace("\n", " ").replace("\t", " ").strip()
        resAns = processDigitArticle(processPunctuation(resAns))

        correctness: List[float] = []
        f1: List[float] = []
        for ans in id2answers[sid]:
            if ans.get("answer_confidence") == "no":
                continue
            gt_ans = str(ans.get("answer", "")).replace("\n", " ").replace("\t", " ").strip()
            gt_ans = processDigitArticle(processPunctuation(gt_ans))
            correctness.append(1.0 if gt_ans == resAns else 0.0)
            f1.append(_compute_token_f1(resAns, gt_ans))

        if not correctness:
            skipped += 1
            continue

        # A missing prediction scores 0 even where an empty answer would match a blank annotation.
        best_acc = 0.0 if item.get("missing") else max(correctness)
        best_f1 = 0.0 if item.get("missing") else max(f1)
        accQA.append(best_acc)
        f1QA.append(best_f1)

        qtype = id2type.get(sid, "unknown")
        type2acc.setdefault(qtype, []).append(best_acc)
        type2f1.setdefault(qtype, []).append(best_f1)

    if not accQA:
        raise ValueError("No VizWiz records could be scored.")

    per_type: Dict[str, Tuple[float, float]] = {}
    for qtype in sorted(type2acc):
        per_type[qtype] = (
            sum(type2acc[qtype]) / len(type2acc[qtype]),
            sum(type2f1[qtype]) / len(type2f1[qtype]),
        )

    return sum(accQA) / len(accQA), sum(f1QA) / len(f1QA), per_type, skipped


def _compute_token_f1(pred: str, gt: str) -> float:
    translator = str.maketrans("", "", string.punctuation)
    pred_tokens = pred.strip().lower().translate(translator).split()
    gt_tokens = gt.strip().lower().translate(translator).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _compute_weighted_f1(items: List[Dict]) -> float:
    labels = sorted({it["gt_answer"].strip().lower() for it in items})
    label_to_idx = {label: i for i, label in enumerate(labels)}
    y_true = [label_to_idx[it["gt_answer"].strip().lower()] for it in items]
    y_pred = [label_to_idx[it["pred_answer"].strip().lower()]
              if not it.get("missing") and it["pred_answer"].strip().lower() in label_to_idx else -1
              for it in items]
    return f1_score(y_true, y_pred, average="weighted", labels=list(label_to_idx.values()))


# VizWiz answerability vote: the answer becomes "unanswerable" when at least half of the query's
# demonstrations (RICES and hard negatives) are labelled "unanswerable" in VizWiz train.
UNANSWERABLE = "unanswerable"


def vizwiz_train_unanswerable(path: Optional[str] = None) -> Dict[str, bool]:
    """Whether each VizWiz train image's demonstration label reads "unanswerable"."""
    path = path or str(REPO_ROOT / "data" / "vizwiz" / "train.json")
    with open(path, "r", encoding="utf-8") as handle:
        annotations = json.load(handle)
    return {str(ann["image"]): normalize_answer(demonstration_label(ann)) == UNANSWERABLE
            for ann in annotations}


def apply_answerability_vote(items: List[Dict], unanswerable: Dict[str, bool]) -> Tuple[List[Dict], int]:
    """Copies of post-processed ``items`` with the vote applied to ``pred_answer``; returns them and
    the number of answers set to "unanswerable"."""
    out, fired = [], 0
    for item in items:
        if item.get("missing"):
            out.append(item)
            continue
        pred = item["pred_answer"]
        ids = list(item.get("retrieved_ids") or []) + list(item.get("contrastive_ids") or [])
        votes = sum(unanswerable[str(i)] for i in ids)
        answer = pred
        if ids and votes >= math.ceil(len(ids) / 2):
            answer = UNANSWERABLE
            fired += normalize_answer(pred) != UNANSWERABLE
        out.append(dict(item, pred_answer=answer))
    return out, fired


def _exact(item: Dict) -> bool:
    return not item.get("missing") and item["pred_answer"].strip().lower() == item["gt_answer"].strip().lower()


def compute_metrics(items: List[Dict], data: str) -> Tuple[float, float, Dict[str, Tuple[float, float]], int]:
    """(EM or accuracy, F1, VizWiz per-type scores, VizWiz skipped) of post-processed items."""
    if data == "vizwiz":
        return _compute_vizwiz_scores(items)
    em = sum(_exact(it) for it in items) / len(items)
    if data in ("cub", "flowers"):
        return em, _compute_weighted_f1(items), {}, 0
    f1 = sum(0.0 if it.get("missing") else _compute_token_f1(it["pred_answer"], it["gt_answer"])
             for it in items) / len(items)
    return em, f1, {}, 0


def score_predictions(items: List[Dict], expected: Optional[List[Dict]], data: str) -> Dict:
    """Score predictions over exactly the ``expected`` samples: records of other ids are dropped,
    expected samples without a prediction score 0, and every answer is post-processed
    (``expected=None`` scores the records as they are). VizWiz: the answerability vote is applied
    after the post-process."""
    dropped = missing = 0
    if expected is not None:
        expected_ids = {str(e["id"]) for e in expected}
        kept = [it for it in items if str(it.get("id", "")) in expected_ids]
        dropped = len(items) - len(kept)
        items, missing = fill_missing(kept, expected)
    if not items:
        raise ValueError("No prediction records to score.")
    items, stripped = postprocess_items(items)
    extra = {}
    if data == "vizwiz":
        items, fired = apply_answerability_vote(items, vizwiz_train_unanswerable())
        extra = {"av_unanswerable": fired}
    em, f1, per_type, skipped = compute_metrics(items, data)
    return {"em": em, "f1": f1, "per_type": per_type, "skipped": skipped, "n": len(items),
            "dropped": dropped, "missing": missing, "stripped": stripped, **extra}


def _load_items(path: str) -> Tuple[List[Dict], int, int]:
    items: List[Dict] = []
    total_lines = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    dedup: List[Dict] = []
    seen = set()
    for item in items:
        sid = str(item.get("id", ""))
        if sid and sid in seen:
            continue
        seen.add(sid)
        dedup.append(item)
    duplicate_count = max(0, len(items) - len(dedup))
    return dedup, total_lines, duplicate_count


def _build_completeness_message(pred_count: int, expected_count: int) -> str:
    if expected_count <= 0:
        return f"Prediction count: {pred_count} (expected count unavailable)"
    if pred_count >= expected_count:
        return f"Prediction coverage: COMPLETE ({pred_count}/{expected_count})"
    missing = expected_count - pred_count
    return f"Prediction coverage: INCOMPLETE ({pred_count}/{expected_count}, missing {missing})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction jsonl")
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--data", type=str, required=True, choices=["okvqa", "vizwiz", "cub", "flowers"])
    parser.add_argument("--sample_ratio", type=float, default=None,
                        help="Sampled run (default: read from the _params.json next to the file)")
    parser.add_argument("--sample_seed", type=int, default=None)
    parser.add_argument("--allow_partial", action="store_true",
                        help="If the test split cannot be loaded, score only the predictions present "
                             "(default: error)")
    args = parser.parse_args()

    if not os.path.exists(args.pred_file):
        raise FileNotFoundError(f"Prediction file not found: {args.pred_file}")

    items, total_lines, duplicate_count = _load_items(args.pred_file)
    if not items:
        raise ValueError("No valid prediction records found.")

    ratio, seed, source = run_sampling(args.pred_file)
    if args.sample_ratio is not None:
        ratio, source = args.sample_ratio, "--sample_ratio"
    if args.sample_seed is not None:
        seed = args.sample_seed
    expected: Optional[List[Dict]] = None
    try:
        expected = expected_test_items(args.data, ratio, seed)
    except Exception as exc:
        if not args.allow_partial:
            raise SystemExit(f"Cannot load the {args.data} test split ({exc}), so missing predictions "
                             f"cannot be counted. Prepare the dataset (see README) or pass "
                             f"--allow_partial to score only the {len(items)} predictions present.")
        print(f"Warning: could not load the test split ({exc}); --allow_partial: the scores below "
              f"cover only the {len(items)} predictions present.")
    if ratio < 1.0:
        print(f"Sampled run (sample_ratio={ratio}, sample_seed={seed}; from {source}).")

    print(_build_completeness_message(pred_count=len(items),
                                      expected_count=len(expected) if expected is not None else -1))
    if duplicate_count > 0:
        print(f"Note: removed {duplicate_count} duplicate prediction records by id.")
    if total_lines != len(items):
        print(f"Evaluating {len(items)} unique records from {total_lines} file lines.")
    res = score_predictions(items, expected, args.data)
    if res["dropped"]:
        print(f"Note: ignored {res['dropped']} records whose id is not an expected test sample.")
    if res["missing"]:
        print(f"Note: {res['missing']} test samples have no prediction and are scored as wrong.")
    print(f"Post-process ({POSTPROCESS_RULE}): {res['stripped']} predictions changed.")
    if args.data == "vizwiz":
        print(f"Answerability vote: {res['av_unanswerable']} answers set to 'unanswerable'.")

    if args.data in {"cub", "flowers"}:
        print(f"Accuracy: {res['em'] * 100:.2f}% | Weighted F1: {res['f1'] * 100:.2f}%")
        return

    print(f"EM: {res['em'] * 100:.2f}% | F1: {res['f1'] * 100:.2f}%")
    for qtype in sorted(res["per_type"]):
        t_em, t_f1 = res["per_type"][qtype]
        print(f"[{qtype} questions] EM: {t_em * 100:.2f}% | F1: {t_f1 * 100:.2f}%")
    if res["skipped"] > 0:
        print(f"Note: skipped {res['skipped']} records that could not be matched/scored by VizWiz annotations.")


if __name__ == "__main__":
    main()
