#!/usr/bin/env python3
"""
Audit offline LLM topic artifacts (specter2-llm-topics.json → specter2_corpus_with-topic-terms.json).

Usage:
  python check_llm_topic_quality.py --data_dir ./CSFCube
  python check_llm_topic_quality.py --data_dir ./CSFCube --sample_ids 58885,10010426
  python check_llm_topic_quality.py --data_dir ./CSFCube --show_empty_topics 15
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter

from corpus_io import resolve_data_dir


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check quality of llm-topic JSON artifacts.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="csfcube",
        choices=("litsearch", "csfcube", "dorismae"),
        help="Used with resolve_data_dir when --data_dir is omitted.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory with specter2_topics.json (default per --dataset).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random samples.")
    parser.add_argument(
        "--show_empty_topics",
        type=int,
        default=10,
        metavar="N",
        help="Print N examples where mapped topics list is empty (name mismatch diagnostic).",
    )
    parser.add_argument(
        "--sample_ids",
        type=str,
        default="",
        help="Comma-separated corpus ids to dump in detail (optional).",
    )
    parser.add_argument(
        "--random_samples",
        type=int,
        default=3,
        help="Number of random ids to print for spot-checking.",
    )
    args = parser.parse_args()
    data_dir = resolve_data_dir(args.dataset, args.data_dir)
    if not os.path.isdir(data_dir):
        raise SystemExit(f"Not a directory: {data_dir!r}")

    topics_path = os.path.join(data_dir, "specter2_topics.json")
    llm_path = os.path.join(data_dir, "specter2-llm-topics.json")
    final_path = os.path.join(data_dir, "specter2_corpus_with-topic-terms.json")

    for p, label in ((topics_path, "specter2_topics.json"), (llm_path, "specter2-llm-topics.json"), (final_path, "specter2_corpus_with-topic-terms.json")):
        if not os.path.isfile(p):
            raise SystemExit(f"Missing {label}: {p}")

    topic_candidates = _load(topics_path)
    llm_raw = _load(llm_path)
    final = _load(final_path)

    n = len(topic_candidates)
    print(f"data_dir: {data_dir}")
    print(f"classifier papers (specter2_topics.json): {n}")
    print(f"llm file keys: {len(llm_raw)}  |  final index keys: {len(final)}")

    missing_llm = set(topic_candidates.keys()) - set(llm_raw.keys())
    missing_final = set(topic_candidates.keys()) - set(final.keys())
    print(f"missing from specter2-llm-topics.json: {len(missing_llm)}")
    print(f"missing from specter2_corpus_with-topic-terms.json: {len(missing_final)}")
    if missing_llm:
        print(f"  sample missing llm: {list(missing_llm)[:8]}")
    if missing_final:
        print(f"  sample missing final: {list(missing_final)[:8]}")

    n_err = sum(1 for v in llm_raw.values() if isinstance(v, dict) and v.get("llm_error"))
    print(f"llm_error entries in specter2-llm-topics.json: {n_err}")

    # Parsed llm_output validity
    no_output = 0
    bad_tags = 0
    for cid, v in llm_raw.items():
        if not isinstance(v, dict):
            continue
        if v.get("llm_error"):
            continue
        lo = v.get("llm_output")
        if not lo:
            no_output += 1
            continue
        if not all(t in lo for t in ("<top>", "</top>", "<kp>", "</kp>")):
            bad_tags += 1
    print(f"no llm_output (and no llm_error): {no_output}")
    print(f"llm_output missing proper tags: {bad_tags}")

    # Final index stats
    n_terms = [len(final[cid].get("terms") or []) for cid in final]
    n_top = [len(final[cid].get("topics") or []) for cid in final]
    empty_terms = sum(1 for x in n_terms if x == 0)
    empty_topics = sum(1 for x in n_top if x == 0)
    print("\n--- specter2_corpus_with-topic-terms.json ---")
    print(f"empty terms[]: {empty_terms} / {len(final)}")
    print(f"empty topics[] (no label matched after LLM): {empty_topics} / {len(final)} ({100 * empty_topics / max(len(final), 1):.1f}%)")
    if n_terms:
        print(f"terms per doc: min={min(n_terms)} median={sorted(n_terms)[len(n_terms) // 2]} max={max(n_terms)}")
    if n_top:
        print(f"topics per doc (mapped tuples): min={min(n_top)} median={sorted(n_top)[len(n_top) // 2]} max={max(n_top)}")

    # Topic name mismatch: LLM wrote <top> names that are not in topic2score keys
    mismatch_hits = 0
    mismatch_examples: list[tuple[str, str, list[str]]] = []
    for cid in final:
        if (final[cid].get("topics") or []):
            continue
        v = llm_raw.get(cid)
        if not isinstance(v, dict) or not v.get("llm_output"):
            continue
        lo = v["llm_output"]
        topic2score = {t[1]: t for t in v["topic_labels"]}
        keys_lower = {k.strip().lower() for k in topic2score.keys()}
        m = re.search(r"<top>(.*?)</top>", lo, re.DOTALL | re.IGNORECASE)
        raw_names = []
        if m:
            raw_names = [x.strip() for x in m.group(1).split(",") if x.strip()]
        mapped = [x for x in raw_names if x.strip().lower() in keys_lower]
        if raw_names and not mapped:
            mismatch_hits += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append((cid, lo[:400], raw_names[:12]))

    print(f"\n--- topic label mismatch (LLM named topics in <top> but none mapped): ~{mismatch_hits} docs (see samples below)")

    rng = random.Random(args.seed)
    # Empty-topic diagnostics
    empty_ids = [cid for cid in final if not (final[cid].get("topics") or [])]
    rng.shuffle(empty_ids)
    shown = 0
    print(f"\n--- Examples: empty topics[] (up to {args.show_empty_topics}) — compare <top> text to classifier label spellings ---")
    for cid in empty_ids:
        if shown >= args.show_empty_topics:
            break
        v = llm_raw.get(cid)
        if not isinstance(v, dict) or not v.get("llm_output"):
            continue
        lo = v["llm_output"]
        labels = [t[1] for t in v.get("topic_labels", [])[:20]]
        m = re.search(r"<top>(.*?)</top>", lo, re.DOTALL | re.IGNORECASE)
        top_raw = (m.group(1).strip()[:200] + "…") if m else "(no <top> match)"
        print(f"\ncorpusid={cid}")
        print(f"  candidate label sample: {labels}")
        print(f"  <top> excerpt: {top_raw}")
        shown += 1

    if mismatch_examples:
        print("\n--- Raw <top> names that failed lower() match (first 5) ---")
        for cid, snip, names in mismatch_examples:
            print(f"\ncorpusid={cid}  LLM topic tokens: {names}")
            print(f"  llm_output[:400]: {snip!r}")

    # Random spot-check
    ids = list(final.keys())
    pick = rng.sample(ids, min(args.random_samples, len(ids)))
    print(f"\n--- Random spot-check (n={len(pick)}, seed={args.seed}) ---")
    for cid in pick:
        t = final[cid]
        print(f"\ncorpusid={cid}  n_topics={len(t.get('topics') or [])}  n_terms={len(t.get('terms') or [])}")
        print(f"  terms[:6]: {(t.get('terms') or [])[:6]}")

    if args.sample_ids.strip():
        want = [x.strip() for x in args.sample_ids.split(",") if x.strip()]
        print(f"\n--- Requested ids: {want} ---")
        for cid in want:
            if cid not in final:
                print(f"{cid}: NOT IN final JSON")
                continue
            t = final[cid]
            v = llm_raw.get(cid, {})
            print(f"\ncorpusid={cid}")
            print(f"  title: {str((v.get('title') or t.get('title') or ''))[:120]}")
            print(f"  n_topics={len(t.get('topics') or [])}  n_terms={len(t.get('terms') or [])}")
            lo = v.get("llm_output")
            if lo:
                print(f"  llm_output[:900]:\n{lo[:900]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
