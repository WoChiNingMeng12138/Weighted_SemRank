"""
Load corpus documents for SemRank (LitSearch via HuggingFace, or local files for CSFCube / DORISMAE).

**CSFCube** (JSONL): each line may use ``corpus_id`` / ``corpusid`` / ``paper_id``, ``title``, and
``abstract`` (string or list of paragraphs). The official
``abstracts-csfcube-preds.jsonl`` from the EMNLP supplement is supported.

**DORISMAE**: either ``corpus.jsonl`` (same keys as above) or the official pickle ``corpus`` file
(mapping ``c0``, ``c1``, ... → ``{\"text\": ...}``).

See README.md for download paths and default directory layout.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any

# Default output / artifact directories per --dataset
DATASET_DEFAULT_DATA_DIR = {
    "litsearch": "./LitSearch",
    "csfcube": "./CSFCube",
    "dorismae": "./DORISMAE",
}


def resolve_data_dir(dataset: str, data_dir: str | None) -> str:
    """If ``data_dir`` is None or empty, use the default folder for ``dataset``."""
    if data_dir is not None and str(data_dir).strip():
        return str(data_dir).strip()
    return DATASET_DEFAULT_DATA_DIR.get(dataset.lower().strip(), "./LitSearch")


def _abstract_to_str(abstract: Any) -> str:
    if abstract is None:
        return ""
    if isinstance(abstract, list):
        return " ".join(str(x).strip() for x in abstract if str(x).strip())
    return str(abstract).strip()


# Official SPECTER2 / SciRepEval string format (see allenai/specter2 README): title + sep_token + abstract
SPECTER2_SEP = "[SEP]"


def specter2_paper_text(title: str | None, abstract: Any) -> str:
    """Format a paper for SPECTER2 dense encoding (matches HuggingFace allenai/specter2 examples)."""
    title = (title or "").strip()
    abstract = _abstract_to_str(abstract) if abstract is not None else ""
    if title:
        if not abstract:
            return title
        return f"{title}{SPECTER2_SEP}{abstract}"
    return abstract


def specter2_encode_text_for_doc(doc: dict[str, Any]) -> str:
    """Build one string per corpus row for *retrieval* encoding (SEP format when title is present)."""
    title = (doc.get("title") or "").strip()
    abstract = doc.get("abstract")
    if title:
        return specter2_paper_text(title, abstract)
    text = (doc.get("text") or "").strip()
    if text:
        return text
    return _abstract_to_str(abstract)


def load_local_corpus_jsonl(path: str) -> tuple[dict[str, Any], list[str], list[str]]:
    """Return ``id2doc``, parallel ``corpus`` strings (title. abstract), and ``id2corpus_id`` order."""
    id2doc: dict[str, Any] = {}
    corpus: list[str] = []
    id2corpus_id: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("corpus_id") or rec.get("corpusid") or rec.get("paper_id")
            if cid is None:
                raise ValueError(f"Missing corpus_id/corpusid/paper_id in line: {line[:120]}...")
            cid = str(cid)
            title = (rec.get("title") or "").strip()
            abstract = _abstract_to_str(rec.get("abstract"))
            if not abstract:
                abstract = (rec.get("text") or "").strip()
            id2doc[cid] = {"corpusid": cid, "title": title, "abstract": abstract}
            if title:
                corpus.append(f"{title}. {abstract}")
            else:
                corpus.append(abstract)
            id2corpus_id.append(cid)
    return id2doc, corpus, id2corpus_id


def load_dorismae_corpus_pickle(path: str) -> tuple[dict[str, Any], list[str], list[str]]:
    """Official DORISMAE ``corpus`` pickle: keys ``c0``..``cN`` with ``{\"text\": ...}``."""
    with open(path, "rb") as f:
        obj: dict[str, Any] = pickle.load(f)

    def sort_key(k: str) -> tuple[int, str]:
        if len(k) >= 2 and k[0] == "c" and k[1:].isdigit():
            return (int(k[1:]), k)
        return (10**9, k)

    keys = sorted(obj.keys(), key=sort_key)
    id2doc: dict[str, Any] = {}
    corpus: list[str] = []
    id2corpus_id: list[str] = []
    for cid in keys:
        row = obj[cid]
        text = (row.get("text") or "").strip() if isinstance(row, dict) else str(row)
        id2doc[cid] = {"corpusid": cid, "title": "", "abstract": text}
        corpus.append(text)
        id2corpus_id.append(cid)
    return id2doc, corpus, id2corpus_id


def _first_existing(paths: list[str | None]) -> str | None:
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def _discover_csfcube_path(data_dir: str, corpus_override: str | None) -> str:
    p = _first_existing(
        [
            corpus_override,
            os.path.join(data_dir, "corpus.jsonl"),
            os.path.join(data_dir, "abstracts-csfcube-preds.jsonl"),
            os.path.join(data_dir, "Dataset", "CSFCube", "abstracts-csfcube-preds.jsonl"),
        ]
    )
    if p:
        return p
    raise FileNotFoundError(
        f"CSFCube: no corpus JSONL found under {data_dir!r}. Expected one of:\n"
        "  corpus.jsonl, abstracts-csfcube-preds.jsonl, or Dataset/CSFCube/abstracts-csfcube-preds.jsonl\n"
        "Download https://aclanthology.org/attachments/2024.emnlp-main.407.data.zip or pass --corpus_jsonl PATH."
    )


def _discover_dorismae_path(data_dir: str, corpus_override: str | None) -> tuple[str, str]:
    """Returns (path, kind) where kind is 'jsonl' or 'pickle'."""
    if corpus_override and os.path.isfile(corpus_override):
        if corpus_override.endswith(".jsonl"):
            return corpus_override, "jsonl"
        return corpus_override, "pickle"

    p_jsonl = _first_existing(
        [
            os.path.join(data_dir, "corpus.jsonl"),
        ]
    )
    if p_jsonl:
        return p_jsonl, "jsonl"

    p_pickle = _first_existing(
        [
            os.path.join(data_dir, "corpus"),
            os.path.join(data_dir, "Dataset", "DORISMAE", "corpus"),
        ]
    )
    if p_pickle:
        return p_pickle, "pickle"

    raise FileNotFoundError(
        f"DORISMAE: no corpus found under {data_dir!r}. Expected corpus.jsonl or pickle file `corpus` "
        f"(e.g. Dataset/DORISMAE/corpus from the EMNLP supplement). Pass --corpus_jsonl PATH to a file."
    )


def load_corpus(
    dataset: str,
    data_dir: str,
    *,
    corpus_jsonl: str | None = None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """
    Load corpus for topic classification and encoding.

    - ``litsearch``: HuggingFace ``princeton-nlp/LitSearch`` ``corpus_clean`` split.
    - ``csfcube``: JSONL (official ``abstracts-csfcube-preds.jsonl`` or ``corpus.jsonl``; see README).
    - ``dorismae``: JSONL or official pickle ``corpus``.
    """
    ds = dataset.lower().strip()
    if ds == "litsearch":
        import datasets

        corpus_data = datasets.load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full")
        id2doc = {str(doc["corpusid"]): doc for doc in corpus_data}
        corpus: list[str] = []
        id2corpus_id: list[str] = []
        for paper in corpus_data:
            i = str(paper["corpusid"])
            id2corpus_id.append(i)
            title = paper.get("title") or ""
            abstract = paper.get("abstract") or ""
            corpus.append(f"{title}. {abstract}")
        return id2doc, corpus, id2corpus_id

    if ds == "csfcube":
        path = _discover_csfcube_path(data_dir, corpus_jsonl)
        return load_local_corpus_jsonl(path)

    if ds == "dorismae":
        path, kind = _discover_dorismae_path(data_dir, corpus_jsonl)
        if kind == "jsonl":
            return load_local_corpus_jsonl(path)
        return load_dorismae_corpus_pickle(path)

    raise ValueError(f"Unknown dataset: {dataset!r} (use litsearch, csfcube, or dorismae)")
