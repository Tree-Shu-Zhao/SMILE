# The Value of Being Wrong: Self-Mined Visual In-Context Learning from Errors

[![arXiv](https://img.shields.io/badge/arXiv-2610.16977-b31b1b.svg)](https://arxiv.org/abs/2610.16977)
[![Python](https://img.shields.io/badge/Python-3.12+-green.svg)](https://www.python.org/)
[![vLLM](https://img.shields.io/badge/vLLM-0.17.1-blue.svg)](https://github.com/vllm-project/vllm)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-SMILE--data-yellow.svg)](https://huggingface.co/datasets/TreezzZ/SMILE-data)

## 📖 Overview

This repository contains the official implementation of **SMILE** (Self-Mined Visual In-Context Learning from Errors), NeurIPS 2026 Spotlight.

Visual in-context learning (ICL) lets vision-language models (VLMs) adapt at test time from demonstration examples, but its effectiveness depends critically on the examples selected. SMILE is grounded in a key observation: a model's own prediction errors are natural counterfactual demonstrations that directly encode its failure modes. Offline, SMILE records the target VLM's own mistakes as a model-specific pool of hard negatives. At test time, the model's prediction serves as a direction signal that surfaces, in the semantic embedding space, the past mistakes most likely to recur for the current query.

### 🔑 Key Features

- ✨ **Self-Mined Hard Negatives**: A model-specific pool built offline from the target VLM's own mistakes
- 🧭 **Prediction-Conditioned Retrieval**: The query's prediction surfaces the past mistakes most likely to recur
- 🧩 **One Mechanism for Classification and VQA**: Closed-set classification (CUB-200-2011, Oxford 102 Flowers) and open-ended VQA (OK-VQA, VizWiz)
- 🚀 **Higher Scores at Lower Latency**: +3.09 points average task-score gain over the SOTA baseline at 5x lower latency, across four benchmarks and four VLMs

## 📋 Table of Contents

- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Usage](#-usage)
  - [1. Start the Servers](#1-start-the-servers)
  - [2. Run SMILE](#2-run-smile)
  - [3. Evaluate](#3-evaluate)
  - [Mining Hard Negatives](#mining-hard-negatives)
- [Repository Layout](#-repository-layout)
- [Citation](#-citation)

## 🔧 Installation

### Prerequisites

- Linux with NVIDIA GPUs (we used H100 80 GB) and Docker with the NVIDIA Container Toolkit
- [uv](https://docs.astral.sh/uv/)
- About 70 GB of disk for the datasets and 60 GB for the model weights
- The Gemma-3 models are gated: accept their license on Hugging Face and `export HF_TOKEN=<your_token>`

### Setup Environment

```bash
git clone https://github.com/Tree-Shu-Zhao/SMILE.git
cd SMILE
uv sync
```

All commands below run from the repository root.

## 📊 Data Preparation

**Datasets.** Each dataset is downloaded from its official source into `data/<dataset>/` the first time it is used; no manual step is needed:

| Dataset | Source | Files |
|---------|--------|-------|
| CUB-200-2011 | [CaltechDATA](https://data.caltech.edu/records/65de6-vp158) | `CUB_200_2011.tgz` |
| Oxford 102 Flowers | [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) | `102flowers.tgz`, `imagelabels.mat`, `setid.mat` |
| OK-VQA | [OK-VQA](https://okvqa.allenai.org/download.html) | train and val questions and annotations, COCO images `train2014.zip`, `val2014.zip` |
| VizWiz | [VizWiz VQA](https://vizwiz.org/tasks-and-datasets/vqa/) | `train.zip`, `val.zip`, `Annotations.zip` |

The download URLs are in `src/load_data.py`. To use copies you already have, put these files in `data/<dataset>/` and the code uses them instead of downloading.

**CLIP embeddings, hard-negative pools and predictions.** The CLIP image embeddings served by the retrieval server, the hard-negative pool mined for every model and dataset, and our predictions are on the Hugging Face Hub at [`TreezzZ/SMILE-data`](https://huggingface.co/datasets/TreezzZ/SMILE-data). Download them into the repository root:

```bash
uvx --from "huggingface_hub>=1.0" hf download TreezzZ/SMILE-data --repo-type dataset --local-dir . \
  --include "clip_embeddings/*" --include "hard_negatives/*" --include "predictions/*"
```

To compute the CLIP embeddings yourself instead (CLIP ViT-g-14, one GPU; recomputed embeddings can differ from the released ones in the last digits, which reorders a few near-tied neighbors):

```bash
uv run --with torch==2.6.0 --with torchvision==0.21.0 --with open_clip_torch==3.3.0 scripts/generate_clip_embeddings.py
```

## 🚀 Usage

### 1. Start the Servers

```bash
bash scripts/start_clip_server.sh                              # CLIP retrieval server, port 8001
bash scripts/start_vlm_server.sh Qwen/Qwen2.5-VL-7B-Instruct  # vLLM server, port 8000
```

### 2. Run SMILE

With the VLM server serving the model:

```bash
bash scripts/<dataset>/<model>.sh    # e.g. bash scripts/cub/qwen2.5-vl-7b.sh
```

`<dataset>` is `cub`, `flowers`, `okvqa` or `vizwiz`; `<model>` is `qwen2.5-vl-7b`, `qwen2.5-vl-3b`, `gemma-3-12b` or `gemma-3-4b`. The predictions are written to `outputs/<dataset>/<org_model>/vicl_<k>_<negative_k>.jsonl`, with the settings (`_params.json`) and the scores (`_summary.json`) next to them.

### 3. Evaluate

```bash
bash scripts/cub/evaluate.sh outputs/cub/Qwen_Qwen2.5-VL-7B-Instruct/vicl_8_4.jsonl

# the released predictions
bash scripts/vizwiz/evaluate.sh predictions/vizwiz/Qwen_Qwen2.5-VL-7B-Instruct/vicl_12_8.jsonl
```

### Mining Hard Negatives

The released pools are ready to use. To mine a pool yourself, with the VLM server serving the model:

```bash
uv run src/mine_hard_negatives.py data=cub model=Qwen/Qwen2.5-VL-7B-Instruct
```

The pool is written to `hard_negatives/<dataset>/<org_model>/hard_negatives.jsonl`, replacing the downloaded one.

## 📁 Repository Layout

```
SMILE/
├── src/
│   ├── smile.py                  # SMILE: RICES, hard-negative selection, prompts
│   ├── mine_hard_negatives.py    # offline mining of a model's hard-negative pool
│   ├── run_batch_inference.py    # runs SMILE over a test split
│   ├── evaluate_results.py       # scoring
│   ├── load_data.py              # dataset download and loading
│   ├── vizwiz_text.py            # VizWiz answer normalization and demonstration labels
│   ├── clip_client.py            # client of the CLIP server
│   └── vlm_client.py             # client of the vLLM server
├── clip_server/                  # CLIP retrieval server (Docker)
├── configs/                      # Hydra configs of inference and mining
└── scripts/                      # server start scripts, one run script per dataset and model, CLIP embeddings
```

## 📚 Citation

If you find our paper useful, please cite it:

```bibtex
@article{zhao2026value,
  title={The Value of Being Wrong: Self-Mined Visual In-Context Learning from Errors},
  author={Zhao, Shu and Yu, Tan},
  journal={arXiv preprint arXiv:2610.16977},
  year={2026}
}
```
