"""CLI: Demo retrieval with campaign aggregation."""

import argparse
import logging
import os
import sys
from collections import defaultdict

from sentence_transformers import SentenceTransformer

from .config import DATA_DIR, PROCESSED_DIR
from .rerank import rerank
from .retrieve import (
    build_faiss_indexes,
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


def _build_query_from_context(make: str, model: str, year: int | None, query_text: str) -> str:
    """Optionally prepend vehicle context to query for better retrieval."""
    if not query_text.strip():
        return query_text
    parts = []
    if make or model:
        parts.append(f"{make} {model}".strip())
    if year:
        parts.append(str(year))
    if parts:
        prefix = " ".join(parts) + ": "
        return prefix + query_text
    return query_text


def _aggregate_by_campaign_with_scores(
    results: list[tuple[dict, float, float, float]],
    make_norm: str | None,
) -> list[dict]:
    """
    Group by campaign_number, use combined_score for campaign_score.
    Each result is (doc, combined, dense_score, kw_norm).
    Evidence includes dense_score, keyword_score (kw_norm), combined.
    """
    if make_norm:
        results = [
            r for r in results
            if (r[0].get("make_norm") or "") == make_norm
        ]

    by_campaign: dict[str, list[tuple[dict, float, float, float]]] = defaultdict(list)
    for doc, combined, dense_score, kw_norm in results:
        cn = doc.get("campaign_number") or ""
        if not cn.strip():
            continue
        by_campaign[cn].append((doc, combined, dense_score, kw_norm))

    campaign_results = []
    for cn, items in by_campaign.items():
        combined_scores = [s[1] for s in items]
        campaign_score = sum(combined_scores) + 0.2 * max(combined_scores) + 0.5 * len(items)
        best = max(items, key=lambda x: x[1])
        best_doc, best_combined, best_dense, best_kw = best
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        evidence_snippets = []
        for doc, combined_s, dense_s, kw_s in sorted_items[:2]:
            text = (doc.get("text") or "")[:280]
            evidence_snippets.append({
                "doc_id": doc.get("doc_id", ""),
                "snippet": text,
                "dense_score": dense_s,
                "keyword_score": kw_s,
                "combined": combined_s,
            })
        campaign_results.append({
            "campaign_number": cn,
            "campaign_score": campaign_score,
            "best_doc": best_doc,
            "evidence_snippets": evidence_snippets,
        })

    campaign_results.sort(key=lambda x: x["campaign_score"], reverse=True)
    return campaign_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo retrieval: search recalls by make/model and aggregate by campaign"
    )
    parser.add_argument("--make", type=str, default="", help="Vehicle make (e.g. Ford)")
    parser.add_argument("--model", type=str, default="", help="Vehicle model (e.g. F-150)")
    parser.add_argument("--year", type=int, default=None, help="Vehicle year (e.g. 2017)")
    parser.add_argument("--query", type=str, default=None, help="Query text; if missing, read multiline from stdin")
    parser.add_argument("--topk", type=int, default=30, help="Number of docs to retrieve")
    parser.add_argument("--topc", type=int, default=3, help="Number of campaigns to show")
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
        "--alpha",
        type=float,
        default=0.15,
        help="Weight for keyword: combined = (1-alpha)*dense + alpha*kw_norm",
    )
    parser.add_argument(
        "--rerank-tokens",
        type=int,
        default=12,
        help="Max query tokens for rerank",
    )
    parser.add_argument(
        "--rerank-phrases",
        type=int,
        default=10,
        help="Max phrases for rerank",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable keyword reranking",
    )
    args = parser.parse_args()

    make_norm = normalize_make(args.make) if args.make else ""  # "Ford" -> "FORD"
    mkey = model_key(args.model) if args.model else ""  # "F-150" -> "f150"

    if args.query is not None:
        query_text = args.query
    else:
        logger.info("Reading query from stdin (multiline, Ctrl-D to finish)...")
        query_text = sys.stdin.read().strip()

    if not query_text:
        logger.error("No query text provided. Use --query or pipe to stdin.")
        sys.exit(1)

    # Optionally add vehicle context to query
    search_query = _build_query_from_context(args.make, args.model, args.year, query_text)

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
    model = SentenceTransformer(args.model_dir)

    # Search
    results = search(
        search_query,
        make_norm,
        mkey,
        indexes,
        model,
        top_k=args.topk,
        use_pool_indexes=not args.no_pool,
        min_pool_docs=50,
    )

    if not results:
        logger.info("No results found.")
        sys.exit(0)

    # Keyword rerank: TF-IDF-ish over candidates + phrase matching
    use_rerank = not args.no_rerank
    alpha = args.alpha
    max_tokens = args.rerank_tokens
    max_phrases = args.rerank_phrases

    if use_rerank:
        logger.info("Rerank: enabled=True alpha=%.2f tokens=%d phrases=%d", alpha, max_tokens, max_phrases)
        results_for_agg = rerank(
            results,
            query_text,
            alpha=alpha,
            max_tokens=max_tokens,
            max_phrases=max_phrases,
        )
    else:
        logger.info("Rerank: enabled=False")
        results_for_agg = [(doc, dense_score, dense_score, 0.0) for doc, dense_score in results]

    # Aggregate by campaign (use final_score)
    campaigns = _aggregate_by_campaign_with_scores(
        results_for_agg,
        make_norm=make_norm if make_norm else None,
    )

    # Print top campaigns
    for i, camp in enumerate(campaigns[: args.topc]):
        print()
        print(f"Campaign: {camp['campaign_number']} | Score: {camp['campaign_score']:.2f}")
        for j, ev in enumerate(camp["evidence_snippets"], 1):
            snippet = (ev.get("snippet") or "").replace("\n", " ")
            ds = ev.get("dense_score", 0)
            kw = ev.get("keyword_score", 0)
            comb = ev.get("combined", 0)
            print(f"  Evidence {j}: {ev.get('doc_id', '')} (dense={ds:.3f} kw={kw:.3f} combined={comb:.3f}) ... {snippet}...")
        if i == 0:
            best_text = (camp.get("best_doc", {}).get("text") or "")[:120].replace("\n", " ")
            print()
            print(f"Suggested recall match: Campaign {camp['campaign_number']} - {best_text}...")

    print()


if __name__ == "__main__":
    main()
