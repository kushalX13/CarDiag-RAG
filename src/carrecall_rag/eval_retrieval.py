"""Evaluation pipeline: Recall@K metrics over a labeled test set."""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

from sentence_transformers import SentenceTransformer

from .config import DATA_DIR, PROCESSED_DIR
from .demo_retrieve import (
    _aggregate_by_campaign_with_scores,
    _build_query_from_context,
)
from .rerank import rerank
from .retrieve import (
    build_faiss_indexes,
    keyword_search,
    load_corpus,
    load_faiss_indexes,
    search,
)
from .utils import model_key, normalize_make

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = os.path.join(DATA_DIR, "models", "biencoder")
DEFAULT_CORPUS_PATH = os.path.join(PROCESSED_DIR, "corpus_merged.jsonl")
DEFAULT_CACHE_DIR = os.path.join(DATA_DIR, "indexes")


def run_retrieval(
    query_text: str,
    make: str,
    model: str,
    year: int | None,
    indexes: dict,
    encoder: SentenceTransformer,
    *,
    dense_topk: int = 100,
    keyword_topk: int = 150,
    alpha: float = 0.30,
    use_hybrid: bool = True,
    use_pool_indexes: bool = True,
    min_pool_docs: int = 50,
    rerank_tokens: int = 12,
    rerank_phrases: int = 10,
) -> list[str]:
    """
    Run full retrieval pipeline: search -> rerank -> aggregate_by_campaign.
    Returns list of campaign_numbers in rank order (top first).
    """
    make_norm = normalize_make(make) if make else ""
    mkey = model_key(model) if model else ""

    search_query = _build_query_from_context(make, model, year, query_text)

    # Dense search
    results = search(
        search_query,
        make_norm,
        mkey,
        indexes,
        encoder,
        top_k=dense_topk,
        use_pool_indexes=use_pool_indexes,
        min_pool_docs=min_pool_docs,
    )

    # Hybrid: union with keyword results
    if use_hybrid and keyword_topk > 0:
        kw_results = keyword_search(
            query_text,
            make_norm,
            mkey,
            indexes,
            top_k=keyword_topk,
            use_pool_indexes=use_pool_indexes,
            min_pool_docs=min_pool_docs,
        )
        seen: dict[str, tuple[dict, float]] = {}
        for doc, dense_score in results:
            did = doc.get("doc_id", "")
            if did and did not in seen:
                seen[did] = (doc, dense_score)
        for doc, _ in kw_results:
            did = doc.get("doc_id", "")
            if did and did not in seen:
                seen[did] = (doc, 0.0)
        results = list(seen.values())

    if not results:
        return []

    # Rerank
    results_for_agg = rerank(
        results,
        query_text,
        alpha=alpha,
        max_tokens=rerank_tokens,
        max_phrases=rerank_phrases,
        normalize_dense=use_hybrid,
    )

    # Aggregate by campaign
    campaigns = _aggregate_by_campaign_with_scores(
        results_for_agg,
        make_norm=make_norm if make_norm else None,
    )

    return [c["campaign_number"] for c in campaigns]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval: Recall@K over labeled test set"
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default="eval/recall_queries.jsonl",
        help="JSONL file with query, make, model, gold_campaign per line",
    )
    parser.add_argument(
        "--dense-topk",
        type=int,
        default=100,
        help="Dense retrieval topk when --hybrid",
    )
    parser.add_argument(
        "--keyword-topk",
        type=int,
        default=150,
        help="Keyword (BM25) retrieval topk when --hybrid",
    )
    parser.add_argument(
        "--topc",
        type=int,
        default=10,
        help="Number of top campaigns to consider for Recall@K (max K)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.30,
        help="Weight for keyword: combined = (1-alpha)*dense + alpha*kw_norm",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid (dense + keyword) retrieval",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Path to biencoder model",
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        default=DEFAULT_CORPUS_PATH,
        help="Path to corpus JSONL",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="Path to FAISS index cache",
    )
    parser.add_argument(
        "--no-pool",
        action="store_true",
        help="Force global index (disable pool/make indexes)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-query debug (query, gold, top-K, hit)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.eval_file):
        logger.error("Eval file not found: %s", args.eval_file)
        sys.exit(1)

    # Reduce log noise during eval unless verbose
    if not args.verbose:
        logging.getLogger("carrecall_rag").setLevel(logging.WARNING)

    # Load eval queries
    queries = []
    with open(args.eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    if not queries:
        logger.error("No queries in eval file: %s", args.eval_file)
        sys.exit(1)

    logger.info("Loaded %d eval queries from %s", len(queries), args.eval_file)

    # Load or build indexes
    indexes = load_faiss_indexes(args.cache_dir)
    if not indexes.get("global"):
        logger.info("Indexes not found. Building from corpus...")
        if not os.path.exists(args.corpus_path):
            logger.error("Corpus not found: %s", args.corpus_path)
            sys.exit(1)
        corpus_docs = load_corpus(args.corpus_path)
        build_faiss_indexes(
            args.model_dir,
            corpus_docs,
            use_pool_indexes=not args.no_pool,
            min_pool_docs=50,
            cache_dir=args.cache_dir,
        )
        indexes = load_faiss_indexes(args.cache_dir)

    if not indexes.get("global"):
        logger.error("Failed to load indexes from %s", args.cache_dir)
        sys.exit(1)

    # Load encoder
    if not os.path.exists(args.model_dir):
        logger.error("Model not found: %s. Run train_biencoder first.", args.model_dir)
        sys.exit(1)
    encoder = SentenceTransformer(args.model_dir)

    # Run retrieval for each query and track Recall@K
    hit_at: dict[int, int] = defaultdict(int)
    k_values = [1, 3, 5, 10]
    topc = max(args.topc, max(k_values))

    for i, row in enumerate(queries):
        query = row.get("query", "").strip()
        make = row.get("make", "")
        model = row.get("model", "")
        gold = (row.get("gold_campaign") or "").strip()
        year = row.get("year")

        if not query or not gold:
            logger.warning("Skipping row %d: missing query or gold_campaign", i + 1)
            continue

        top_campaigns = run_retrieval(
            query,
            make,
            model,
            year,
            indexes,
            encoder,
            dense_topk=args.dense_topk,
            keyword_topk=args.keyword_topk if args.hybrid else 0,
            alpha=args.alpha,
            use_hybrid=args.hybrid,
            use_pool_indexes=not args.no_pool,
            min_pool_docs=50,
        )

        top_k_list = top_campaigns[:topc]

        for k in k_values:
            if gold in top_k_list[:k]:
                hit_at[k] += 1

        if args.verbose:
            hit_1 = "YES" if gold in top_k_list[:1] else "NO"
            hit_3 = "YES" if gold in top_k_list[:3] else "NO"
            hit_5 = "YES" if gold in top_k_list[:5] else "NO"
            hit_10 = "YES" if gold in top_k_list[:10] else "NO"
            print(f"\nQuery: {query[:60]}{'...' if len(query) > 60 else ''}")
            print(f"Gold: {gold}")
            print(f"Top3: {top_k_list[:3]}")
            print(f"Hit@1: {hit_1}  Hit@3: {hit_3}  Hit@5: {hit_5}  Hit@10: {hit_10}")

    # Print final results
    n = len(queries)
    print()
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Total queries: {n}")
    for k in k_values:
        recall = hit_at[k] / n if n > 0 else 0.0
        print(f"Recall@{k}: {recall:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
