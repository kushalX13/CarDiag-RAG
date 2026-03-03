"""CLI: Build weakly-labeled training triples for retriever training."""

import argparse
import hashlib
import json
import logging
import os
import random
import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from .config import PROCESSED_DIR
from .utils import jsonl_write

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MIN_POOL_SIZE = 50
TOP_K = 20
MIN_HITS = 2
GAP_FACTOR = 1.15
MAX_NEGS = 8
TRAIN_RATIO = 0.95  # 95% train, 5% val


def _tokenize(text: str) -> list[str]:
    """Simple tokenization: lowercase, split on non-letters/numbers."""
    text = (text or "").lower()
    return re.findall(r"[a-z0-9]+", text)


def _load_jsonl(path: str) -> list[dict]:
    """Load JSONL file into list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _group_corpus_by_make_model(corpus: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Group corpus docs by (make_norm, model_key)."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for doc in corpus:
        make_norm = doc.get("make_norm") or ""
        model_key = doc.get("model_key") or ""
        groups[(make_norm, model_key)].append(doc)
    return dict(groups)


def _build_bm25_cache(
    groups: dict[tuple[str, str], list[dict]],
) -> dict[tuple[str, str], tuple[BM25Okapi, list[dict]]]:
    """Pre-tokenize and build BM25 index per pool. Returns {(make, model): (bm25, docs)}."""
    cache = {}
    for key, docs in groups.items():
        tokenized = [_tokenize(d.get("text", "")) for d in docs]
        bm25 = BM25Okapi(tokenized)
        cache[key] = (bm25, docs)
    return cache


def _get_topk_with_scores(
    bm25: BM25Okapi,
    docs: list[dict],
    query_tokens: list[str],
    k: int,
) -> list[tuple[dict, float]]:
    """Return top-k (doc, score) pairs by BM25 score."""
    scores = bm25.get_scores(query_tokens)
    indexed = list(zip(docs, scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return indexed[:k]


def _is_unknown_campaign(campaign_number: str) -> bool:
    """Return True if campaign should be skipped (UNKNOWN or empty)."""
    cn = (campaign_number or "").strip().upper()
    return cn == "" or cn == "UNKNOWN"


def _select_positive_campaign(
    topk_with_scores: list[tuple[dict, float]],
) -> tuple[dict | None, list[dict]]:
    """
    Aggregate by campaign, compute campaign_score, select positive.
    Returns (positive_campaign_docs, all_other_campaign_docs) or (None, []) if fails.
    """
    # Group by campaign_number, skip UNKNOWN
    by_campaign: dict[str, list[tuple[dict, float]]] = defaultdict(list)
    for doc, score in topk_with_scores:
        cn = doc.get("campaign_number") or ""
        if _is_unknown_campaign(cn):
            continue
        by_campaign[cn].append((doc, score))

    if not by_campaign:
        return None, []

    # Compute campaign_score for each campaign
    campaign_scores: list[tuple[str, float, int, list[tuple[dict, float]]]] = []
    for cn, items in by_campaign.items():
        hits = len(items)
        score_sum = sum(s for _, s in items)
        score_max = max(s for _, s in items)
        campaign_score = score_sum + 0.1 * score_max + 0.5 * hits
        campaign_scores.append((cn, campaign_score, hits, items))

    # Sort by campaign_score descending
    campaign_scores.sort(key=lambda x: x[1], reverse=True)

    best_cn, best_score, best_hits, best_items = campaign_scores[0]

    # Confidence filters
    if best_hits < MIN_HITS:
        return None, []

    second_best_score = campaign_scores[1][1] if len(campaign_scores) > 1 else 0.0
    if second_best_score > 0 and best_score < second_best_score * GAP_FACTOR:
        return None, []

    # Positive campaign docs
    pos_docs = [d for d, _ in best_items]

    # All other campaign docs (for negatives)
    other_docs = []
    for cn, _, _, items in campaign_scores[1:]:
        other_docs.extend([(d, s) for d, s in items])

    return pos_docs, other_docs


def _pick_positive_passage(
    pos_docs: list[dict],
    topk_with_scores: list[tuple[dict, float]],
) -> dict | None:
    """Within positive campaign, pick the highest BM25-scoring doc."""
    pos_doc_ids = {d.get("doc_id") for d in pos_docs}
    best = None
    best_score = -1.0
    for doc, score in topk_with_scores:
        if doc.get("doc_id") in pos_doc_ids and score > best_score:
            best = doc
            best_score = score
    return best


def _pick_negatives(
    other_campaign_docs: list[tuple[dict, float]],
    pool_docs: list[dict],
    pos_campaign_number: str,
    rng: random.Random,
    max_negs: int = MAX_NEGS,
) -> list[dict]:
    """
    Pick up to max_negs hard negatives from other campaigns.
    a) Top retrieved from other campaigns (highest scores first)
    b) If not enough, sample random docs from other campaigns in pool
    Deduplicates by doc_id.
    """
    negs = []
    seen_ids: set[str] = set()
    # Sort other_campaign_docs by score descending
    other_sorted = sorted(other_campaign_docs, key=lambda x: x[1], reverse=True)
    for doc, _ in other_sorted:
        if len(negs) >= max_negs:
            break
        doc_id = doc.get("doc_id", "")
        cn = doc.get("campaign_number") or ""
        if (
            doc_id not in seen_ids
            and not _is_unknown_campaign(cn)
            and cn != pos_campaign_number
        ):
            negs.append(doc)
            seen_ids.add(doc_id)

    # If not enough, sample random from pool (other campaigns)
    if len(negs) < max_negs:
        pool_other = [
            d for d in pool_docs
            if (d.get("campaign_number") or "").strip() != pos_campaign_number
            and not _is_unknown_campaign(d.get("campaign_number") or "")
        ]
        candidates = [d for d in pool_other if d.get("doc_id") not in seen_ids]
        need = max_negs - len(negs)
        if candidates and need > 0:
            sampled = rng.sample(candidates, min(need, len(candidates)))
            for d in sampled:
                negs.append(d)
                seen_ids.add(d.get("doc_id", ""))

    return negs[:max_negs]


def _doc_to_triple_entry(doc: dict) -> dict:
    """Convert doc to triple pos/neg entry format."""
    return {
        "doc_id": doc.get("doc_id", ""),
        "campaign_number": doc.get("campaign_number", ""),
        "text": doc.get("text", ""),
    }


def _hash_to_split(query_id: str) -> str:
    """Deterministic train/val split by hash (95% train, 5% val). Returns 'train' or 'val'."""
    h = int(hashlib.sha256(query_id.encode("utf-8")).hexdigest(), 16)
    val_pct = int((1 - TRAIN_RATIO) * 100)  # 5
    return "val" if (h % 100) < val_pct else "train"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build weakly-labeled training triples for retriever training"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=20000,
        help="Maximum number of complaints to process",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    args = parser.parse_args()

    complaints_path = os.path.join(PROCESSED_DIR, "complaints.jsonl")
    corpus_path = os.path.join(PROCESSED_DIR, "corpus_merged.jsonl")
    train_path = os.path.join(PROCESSED_DIR, "train_triples.jsonl")
    val_path = os.path.join(PROCESSED_DIR, "val_triples.jsonl")
    report_path = os.path.join(PROCESSED_DIR, "triples_report.json")

    if not os.path.exists(complaints_path):
        logger.error("Missing complaints: %s", complaints_path)
        raise SystemExit(1)
    if not os.path.exists(corpus_path):
        logger.error("Missing corpus: %s", corpus_path)
        raise SystemExit(1)

    # 1) Load complaints + corpus
    logger.info("Loading complaints and corpus...")
    complaints = _load_jsonl(complaints_path)
    corpus = _load_jsonl(corpus_path)

    # Limit complaints
    rng = random.Random(args.seed)
    if len(complaints) > args.max_queries:
        complaints = rng.sample(complaints, args.max_queries)

    logger.info("Complaints: %d, Corpus: %d docs", len(complaints), len(corpus))

    # 2) Group corpus by (make_norm, model_key)
    groups = _group_corpus_by_make_model(corpus)
    logger.info("Corpus groups (make, model): %d", len(groups))

    # Build BM25 cache per pool
    bm25_cache = _build_bm25_cache(groups)

    # Stats
    total_processed = 0
    total_triples = 0
    skipped_low_pool = 0
    skipped_low_confidence = 0
    triples_per_make_model: dict[str, int] = defaultdict(int)
    negs_per_query: list[int] = []

    train_triples: list[dict] = []
    val_triples: list[dict] = []

    for q in complaints:
        total_processed += 1
        make_norm = q.get("make_norm") or ""
        model_key = q.get("model_key") or ""
        pool_key = (make_norm, model_key)

        # 3) Candidate pool
        pool_docs = groups.get(pool_key, [])
        if len(pool_docs) < MIN_POOL_SIZE:
            skipped_low_pool += 1
            continue

        bm25, docs = bm25_cache[pool_key]
        query_text = q.get("complaint_text", "")
        query_tokens = _tokenize(query_text)

        # 4) BM25 retrieval
        topk_with_scores = _get_topk_with_scores(bm25, docs, query_tokens, TOP_K)

        # 5-6) Campaign aggregation + select positive
        pos_docs, other_campaign_docs = _select_positive_campaign(topk_with_scores)
        if pos_docs is None:
            skipped_low_confidence += 1
            continue

        pos_campaign_number = pos_docs[0].get("campaign_number", "")

        # 7) Choose positive passage
        pos_passage = _pick_positive_passage(pos_docs, topk_with_scores)
        if pos_passage is None:
            skipped_low_confidence += 1
            continue

        # 8) Choose negatives
        negs = _pick_negatives(
            other_campaign_docs,
            pool_docs,
            pos_campaign_number,
            rng,
            MAX_NEGS,
        )

        # 9) Emit triple
        triple = {
            "query_id": q.get("query_id", ""),
            "query_text": query_text,
            "make_norm": make_norm,
            "model_key": model_key,
            "pos": _doc_to_triple_entry(pos_passage),
            "negs": [_doc_to_triple_entry(d) for d in negs],
        }

        # 10) Split train/val
        split = _hash_to_split(triple["query_id"])
        if split == "train":
            train_triples.append(triple)
        else:
            val_triples.append(triple)

        total_triples += 1
        mm_key = f"{make_norm}|{model_key}"
        triples_per_make_model[mm_key] += 1
        negs_per_query.append(len(negs))

    # Write outputs
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    jsonl_write(train_path, train_triples)
    jsonl_write(val_path, val_triples)
    logger.info("Wrote %s (%d triples)", train_path, len(train_triples))
    logger.info("Wrote %s (%d triples)", val_path, len(val_triples))

    # 11) Stats report
    avg_negs = sum(negs_per_query) / len(negs_per_query) if negs_per_query else 0.0
    pct_skipped_pool = 100.0 * skipped_low_pool / total_processed if total_processed else 0.0
    pct_skipped_confidence = (
        100.0 * skipped_low_confidence / total_processed if total_processed else 0.0
    )

    report = {
        "total_complaints_processed": total_processed,
        "total_triples_produced": total_triples,
        "train_triples": len(train_triples),
        "val_triples": len(val_triples),
        "triples_per_make_model": dict(triples_per_make_model),
        "avg_negs_per_query": round(avg_negs, 2),
        "skipped_low_pool_size": skipped_low_pool,
        "skipped_low_confidence": skipped_low_confidence,
        "pct_skipped_low_pool": round(pct_skipped_pool, 2),
        "pct_skipped_low_confidence": round(pct_skipped_confidence, 2),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Wrote %s", report_path)
    logger.info("--- Summary ---")
    logger.info("Total processed: %d", total_processed)
    logger.info("Total triples: %d (train=%d, val=%d)", total_triples, len(train_triples), len(val_triples))
    logger.info("Avg negs per query: %.2f", avg_negs)
    logger.info("Skipped (low pool): %d (%.1f%%)", skipped_low_pool, pct_skipped_pool)
    logger.info("Skipped (low confidence): %d (%.1f%%)", skipped_low_confidence, pct_skipped_confidence)


if __name__ == "__main__":
    main()
