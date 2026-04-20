# SemRank
The source code used for paper [Scientific Paper Retrieval with LLM-Guided Semantic-Based Ranking](https://arxiv.org/abs/2505.21815), published in EMNLP 2025.

## Overview
**SemRank** is an effective and efficient paper retrieval framework that combines LLM-guided query understanding with a concept-based semantic index. Each paper is indexed using multi-granular scientific concepts, including general research topics and detailed key phrases. At query time, an LLM identifies core concepts derived from the corpus to explicitly capture the query's information need. These identified concepts enable precise semantic matching, significantly enhancing retrieval accuracy.

Please refer to our paper for more details ([paper](https://arxiv.org/abs/2505.21815)).

<img src="./semrank-example.png" width="1000px"></img>

## Datasets
We use CSFCube, DORISMAE, and LitSearch in our experiments. We use the processed version of CSFCube and DORISMAE available [here](https://aclanthology.org/attachments/2024.emnlp-main.407.data.zip) and LitSearch from [HuggingFace](https://huggingface.co/datasets/princeton-nlp/LitSearch).

**Smaller corpora (CSFCube ~4.2k docs, DORISMAE ~8.5k docs):** download and unpack the supplement above. The loader recognizes the **official layout** automatically:

- **CSFCube:** `abstracts-csfcube-preds.jsonl` (or your own `corpus.jsonl` with `paper_id` / `corpus_id`, `title`, `abstract`).
- **DORISMAE:** pickle file `corpus` (or `corpus.jsonl` with the same schema as in `corpus_io.py`).

Point `--data_dir` at `./CSFCube` or `./DORISMAE` after copying files out of `Dataset/CSFCube` and `Dataset/DORISMAE`, **or** set `--data_dir` to the unpacked `Dataset/CSFCube` / `Dataset/DORISMAE` folder directly. Optional `--corpus_jsonl` overrides the corpus file path. Pipeline defaults are:

- LitSearch: `--dataset litsearch` → artifacts under `./LitSearch` (no local corpus file; uses HuggingFace).
- CSFCube: `--dataset csfcube` → `./CSFCube/`
- DORISMAE: `--dataset dorismae` → `./DORISMAE/`

## Build Index

Run the following commands to build the semantic index (swap `--dataset` / paths for CSFCube or DORISMAE as above).
```
# Predict candidate topic labels (GPU needed)
python eval_classifier.py

# Get LLM-assigned topic labels (OpenAI key needed)
python llm-topic.py

# Encode corpus + semantic labels (GPU needed)
python encoding.py
```

Example for **CSFCube** (after `corpus.jsonl` is in place under `./CSFCube`):

```
python eval_classifier.py --dataset csfcube
python llm-topic.py --dataset csfcube
python encoding.py --dataset csfcube
```

Our code by default loads and processes **LitSearch** with gpt-4.1-mini and specter2. Use `--dataset`, `--data_dir`, and `--corpus_jsonl` where documented in `eval_classifier.py`, `llm-topic.py`, and `encoding.py`.

We provide the trained topic classifier checkpoint on the CSRanking domain using [MAPLE](https://github.com/yuzhimanhua/MAPLE). The checkpoint can be [downloaded here](https://www.dropbox.com/scl/fi/tzg189k3n6tfxr2lzvjqj/topic_classifier_specter2.pt?rlkey=hnp2kfkxezubqeblpq4ym8kkd&st=btgz2a4s&dl=0) and please put it in the ```./classifier``` folder which also includes the complete label space. 

If you want to use semantic indexing in domains other than Computer Science, we recommend you to look at other available corpora from [MAPLE](https://github.com/yuzhimanhua/MAPLE) and check the text classifier training code by [TELEClass](https://github.com/yzhan238/TELEClass) which also supports training a hierarchical text classifier without labeled data.

## Run SemRank Retrieval

Please check ```SemRank.ipynb``` which includes step-by-step running of SemRank

## Citations

If you find our work useful for your research, please cite the following paper:
```
@inproceedings{zhang2025semrank,
    title={Scientific Paper Retrieval with LLM-Guided Semantic-Based Ranking},
    author={Yunyi Zhang and Ruozhen Yang and Siqi Jiao and SeongKu Kang and Jiawei Han},
    booktitle={Findings of EMNLP},
    year={2025}
}
```