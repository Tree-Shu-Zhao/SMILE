"""SMILE: visual in-context learning with self-mined hard negatives.

For each query:
  (1) retrieve the ``k`` training images most similar to the query image (RICES);
  (2) run one zero-shot pass to obtain the model's own prediction for the query;
  (3) select ``negative_k`` hard negatives from the mined pool (training examples the model got
      wrong), ranked by the CLIP text similarity between their wrong prediction and the query's;
  (4) build the prompt from both sets of demonstrations and query the VLM.

Retrieval runs on the CLIP server and the VLM calls go to an OpenAI-compatible server (vLLM).
"""

import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from vlm_client import image_content


def load_hard_negatives(path: str) -> List[Dict[str, Any]]:
    """Records of a mined pool (``hard_negatives.jsonl``), without those flagged ``is_correct``."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("is_correct", False):
                records.append(rec)
    return records


def format_examples(examples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prompt parts for demonstrations: each image followed by its question and label."""
    content: List[Dict[str, Any]] = []
    for item in examples:
        content.append(item["image"])
        content.append({"type": "text", "text": f"{item['question']}\nAnswer: {item['answer']}"})
    return content


def build_prompt_content(
    question: str,
    query_image_content: Dict[str, Any],
    retrieved: List[Dict[str, Any]],
    hard_negatives: List[Dict[str, Any]],
    task: str,
    options: Optional[Sequence[str]],
    query_prediction: str,
    dataset_name: str,
) -> List[Dict[str, Any]]:
    """Build the full ICL prompt as a list of content parts."""
    task_name = "Visual Question Answering" if task == "vqa" else "Image Classification"
    prefix = f"Your task is to perform {task_name}."
    if task == "cls" and options:
        prefix += f" You need to choose one of the following options: {list(options)}"

    content: List[Dict[str, Any]] = [{"type": "text", "text": prefix}]
    content.append({"type": "text", "text": "Here are some reference examples:"})
    content.extend(format_examples(retrieved))

    if hard_negatives:
        if query_prediction:
            content.append({
                "type": "text",
                "text": f'Here are some tricky examples that were previously confused with "{query_prediction}":',
            })
        else:
            content.append({"type": "text", "text": "Here are some tricky examples that are often answered incorrectly:"})
        content.extend(format_examples(hard_negatives))

    content.append({"type": "text", "text": "Now look at the following image and answer the question."})
    content.append(query_image_content)
    content.append({"type": "text", "text": question})

    if task == "cls":
        answer_instruction = (
            "Compare the query image with the reference images above and choose the option "
            "whose visual features most closely match. Output exactly one option from the provided list."
        )
    elif dataset_name == "vizwiz":
        answer_instruction = "Answer the question using a single word or phrase."
    else:
        answer_instruction = "Answer with just one or two words. Use singular form."
    content.append({"type": "text", "text": answer_instruction})

    return content


def build_zero_shot_content(
    question: str,
    image_content: Dict[str, Any],
    task: str = "vqa",
    options: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Build a zero-shot (no ICL examples) prompt."""
    task_name = "Visual Question Answering" if task == "vqa" else "Image Classification"
    prefix = f"Your task is to perform {task_name}."
    if task == "cls" and options:
        prefix += f" You need to choose one of the following options: {list(options)}"
    return [
        {"type": "text", "text": prefix},
        image_content,
        {"type": "text", "text": f"Question: {question}"},
        {"type": "text", "text": "Please provide your response by directly outputting the answer."},
    ]


class SMILE:
    """SMILE inference for one dataset and one VLM, with that VLM's mined hard-negative pool."""

    def __init__(
        self,
        clip_client,
        vlm_client,
        train_dataset,
        dataset_name: str,
        hard_negatives_path: str,
        k: int = 8,
        negative_k: int = 4,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self._clip_client = clip_client
        self._vlm_client = vlm_client
        self._train = train_dataset
        self._train_index = {str(sid): i for i, sid in enumerate(train_dataset.ids)}
        # VizWiz demonstrations are labelled with the most frequent confident annotator answer
        self._labels = train_dataset.consensus_answers if dataset_name == "vizwiz" else train_dataset.answers
        self.task = train_dataset.task
        self.options = train_dataset.options
        self.max_side = train_dataset.max_side
        self.dataset_name = dataset_name
        self.k = k
        self.negative_k = negative_k
        self.temperature = temperature
        self.max_tokens = max_tokens

        # CLIP text embeddings of the pool's voted (wrong) predictions
        pool = load_hard_negatives(hard_negatives_path)
        preds = [rec.get("voted_pred", "") for rec in pool]
        embeddings = []
        for i in range(0, len(preds), 64):
            embeddings.extend(self._clip_client.encode_text(preds[i:i + 64]))
        embeddings = np.array(embeddings, dtype=np.float32)
        if dataset_name == "vizwiz":  # hard negatives predicted as "unanswerable" are excluded
            keep = [i for i, pred in enumerate(preds) if "unanswerable" not in pred.lower()]
            pool, embeddings = [pool[i] for i in keep], embeddings[keep]
        self._pool = pool
        self._pool_embeddings = embeddings
        print(f"[SMILE] {len(pool)} hard negatives from {hard_negatives_path}")

    def _select_hard_negatives(self, prediction: str, exclude: set) -> List[Dict[str, Any]]:
        """The ``negative_k`` pool records whose wrong prediction is closest to ``prediction``,
        skipping the retrieved examples (``exclude``)."""
        query_emb = np.array(self._clip_client.encode_text([prediction]), dtype=np.float32)
        similarities = (query_emb @ self._pool_embeddings.T).squeeze(0)
        selected: List[Dict[str, Any]] = []
        for idx in np.argsort(-similarities):
            rec = self._pool[idx]
            if str(rec.get("id", "")) not in exclude:
                selected.append(rec)
                if len(selected) >= self.negative_k:
                    break
        return selected

    def _demonstration(self, sid: str) -> Dict[str, Any]:
        i = self._train_index[sid]
        return {
            "image": image_content(self._train.imgpaths[i], self.max_side),
            "question": self._train.questions[i],
            "answer": self._labels[i],
        }

    def predict(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Run RICES retrieval, hard-negative selection, and ICL inference for one test item."""
        query_id = str(item["id"])
        query_image = image_content(item["imgpath"], self.max_side)

        retrieved_ids = self._clip_client.retrieve(self.dataset_name, query_id, k=self.k)

        query_prediction = self._vlm_client.chat(
            build_zero_shot_content(item["question"], query_image, self.task, self.options),
            temperature=self.temperature, max_tokens=self.max_tokens)[0]
        hard_negatives = self._select_hard_negatives(query_prediction, exclude=set(retrieved_ids))
        contrastive_ids = [str(rec.get("id", "")) for rec in hard_negatives]

        content = build_prompt_content(
            question=item["question"],
            query_image_content=query_image,
            retrieved=[self._demonstration(sid) for sid in retrieved_ids],
            hard_negatives=[self._demonstration(sid) for sid in contrastive_ids],
            task=self.task,
            options=self.options,
            query_prediction=query_prediction,
            dataset_name=self.dataset_name,
        )
        answer = self._vlm_client.chat(content, temperature=self.temperature, max_tokens=self.max_tokens)[0]

        return {
            "answer": answer,
            "retrieved_ids": retrieved_ids,
            "contrastive_ids": contrastive_ids,
            "query_prediction": query_prediction,
            "k": self.k,
        }
