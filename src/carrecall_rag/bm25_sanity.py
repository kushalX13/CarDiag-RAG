"""CLI: BM25 baseline retrieval sanity check on random complaints (metadata-filtered)."""

import argparse
import json
import logging
import os
import random
import re

from rank_bm25 import BM25Okapi

from .config import PROCESSED_DIR
from .utils import jsonl_write

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MIN_SUBSET_SIZE = 50


def _tokenize(text: str) -> list[str]:
    """Simple tokenization: lowercase, split on non-letters/numbers."""
    text = (text or "").lower()
    return re.findall(r"[a-z0-9]+", text)


def _get_subset(corpus: list[dict], q_make_norm: str, q_model_key: str) -> tuple[list[dict], str]:
    """Return (subset, filter_used). filter_used: 'model' | 'make' | 'global'."""
    subset_make = [d for d in corpus if (d.get("make_norm") or "") == q_make_norm]
    subset_model = [d for d in subset_make if (d.get("model_key") or "") == q_model_key]

    if len(subset_model) >= MIN_SUBSET_SIZE:
        return subset_model, "model"
    if len(subset_make) >= MIN_SUBSET_SIZE:
        return subset_make, "make"
    return corpus, "global"


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 sanity check on complaints (metadata-filtered)")
    parser.add_argument("--n", type=int, default=30, help="Number of random complaints to sample")
    parser.add_argument("--k", type=int, default=10, help="Top-K docs to retrieve per complaint")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    args = parser.parse_args()

    complaints_path = os.path.join(PROCESSED_DIR, "complaints.jsonl")
    corpus_merged_path = os.path.join(PROCESSED_DIR, "corpus_merged.jsonl")
    corpus_path = os.path.join(PROCESSED_DIR, "corpus.jsonl")
    corpus_path = corpus_merged_path if os.path.exists(corpus_merged_path) else corpus_path
    if corpus_path == corpus_merged_path:
        logger.info("Using corpus_merged.jsonl")

    if not os.path.exists(complaints_path) or not os.path.exists(corpus_path):
        logger.error("Run build_corpus first. Missing: %s or %s", complaints_path, corpus_path)
        raise SystemExit(1)

    # Load complaints
    complaints: list[dict] = []
    with open(complaints_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                complaints.append(json.loads(line))

    # Load corpus
    corpus: list[dict] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                corpus.append(json.loads(line))

    # Ensure complaints and corpus have make_norm, model_norm, model_key (backfill if missing)
    from .utils import model_key, normalize_make, normalize_model
    for c in complaints:
        if "make_norm" not in c:
            c["make_norm"] = normalize_make(c.get("make", ""))
        if "model_norm" not in c:
            c["model_norm"] = normalize_model(c.get("model", ""))
        if "model_key" not in c:
            c["model_key"] = model_key(c.get("model", ""))
    for d in corpus:
        if "make_norm" not in d:
            d["make_norm"] = normalize_make(d.get("make", ""))
        if "model_norm" not in d:
            d["model_norm"] = normalize_model(d.get("model", ""))
        if "model_key" not in d:
            d["model_key"] = model_key(d.get("model", ""))

    if not complaints or not corpus:
        logger.error("Empty complaints or corpus")
        raise SystemExit(1)

    logger.info("Corpus size: %d chunks, %d complaints", len(corpus), len(complaints))

    # Sample N random complaints
    rng = random.Random(args.seed)
    n_sample = min(args.n, len(complaints))
    sampled = rng.sample(complaints, n_sample)

    results_rows: list[dict] = []

    for q in sampled:
        q_make_norm = q.get("make_norm", "")
        q_model_key = q.get("model_key", "")
        subset, filter_used = _get_subset(corpus, q_make_norm, q_model_key)

        tokenized_subset = [_tokenize(doc["text"]) for doc in subset]
        bm25 = BM25Okapi(tokenized_subset)

        query_text = q.get("complaint_text", "")
        query_tokens = _tokenize(query_text)
        scores = bm25.get_scores(query_tokens)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: args.k]

        topk = []
        for rank, idx in enumerate(top_indices, start=1):
            doc = subset[idx]
            score = float(scores[idx])
            text_snippet = (doc.get("text", ""))[:300]
            topk.append({
                "rank": rank,
                "doc_id": doc.get("doc_id", ""),
                "make": doc.get("make", ""),
                "model": doc.get("model", ""),
                "make_norm": doc.get("make_norm", ""),
                "model_key": doc.get("model_key", ""),
                "campaign_number": doc.get("campaign_number", ""),
                "component": doc.get("component", ""),
                "score": score,
                "text_snippet": text_snippet,
            })

        results_rows.append({
            "query_id": q.get("query_id", ""),
            "make": q.get("make", ""),
            "model": q.get("model", ""),
            "year": q.get("year", 0),
            "make_norm": q_make_norm,
            "model_key": q_model_key,
            "filter_used": filter_used,
            "subset_size": len(subset),
            "complaint_text": query_text,
            "topk": topk,
        })

    # Save results
    out_path = os.path.join(PROCESSED_DIR, "bm25_sanity_results.jsonl")
    jsonl_write(out_path, results_rows)
    logger.info("Saved %d results to %s", len(results_rows), out_path)

    # Summary over sampled N queries
    top1_same_make = sum(
        1 for r in results_rows
        if r["topk"] and (r["topk"][0].get("make_norm") or "") == (r.get("make_norm") or "")
    )
    top1_same_model = sum(
        1 for r in results_rows
        if r.get("filter_used") == "model" and r["topk"]
        and (r["topk"][0].get("model_key") or "") == (r.get("model_key") or "")
    )
    model_filter_count = sum(1 for r in results_rows if r.get("filter_used") == "model")
    top3_same_make = sum(
        1 for r in results_rows
        if any(
            (h.get("make_norm") or "") == (r.get("make_norm") or "")
            for h in r["topk"][:3]
        )
    )

    top1_same_make_rate = top1_same_make / len(results_rows) if results_rows else 0.0
    top1_same_model_rate = top1_same_model / model_filter_count if model_filter_count else 0.0
    top3_same_make_rate = top3_same_make / len(results_rows) if results_rows else 0.0

    logger.info("--- Summary (N=%d) ---", len(results_rows))
    logger.info("top1_same_make_rate: %.3f", top1_same_make_rate)
    logger.info("top1_same_model_rate (model filter, n=%d): %.3f", model_filter_count, top1_same_model_rate)
    logger.info("top3_same_make_rate: %.3f", top3_same_make_rate)

    # Preview first 5
    logger.info("--- Preview (first 5 queries) ---")
    for i, row in enumerate(results_rows[:5]):
        logger.info("\n[Query %d] %s %s %d (filter=%s)", i + 1, row["make"], row["model"], row["year"], row.get("filter_used", ""))
        complaint_preview = (row["complaint_text"] or "")[:200]
        if len(row["complaint_text"] or "") > 200:
            complaint_preview += "..."
        logger.info("Complaint: %s", complaint_preview)
        for hit in row["topk"][:3]:
            logger.info("  Top %d: %s %s (score=%.4f)", hit["rank"], hit.get("make", ""), hit.get("model", ""), hit["score"])


if __name__ == "__main__":
    main()
