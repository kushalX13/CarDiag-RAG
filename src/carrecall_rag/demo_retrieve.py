"""CLI: Demo retrieval with campaign aggregation."""

import argparse
import logging
import math
import os
import re
import sys
from collections import defaultdict

from sentence_transformers import SentenceTransformer

from .config import DATA_DIR, PROCESSED_DIR
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

# Stopwords for keyword extraction (small set)
STOPWORDS = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one", "our", "out", "has", "his", "how", "its", "may", "new", "now", "old", "see", "way", "who", "did", "get", "got", "let", "put", "say", "too", "use"}

# Phrase boosts for common symptoms
PHRASE_BOOSTS = [
    "loss of power",
    "engine stalls",
    "stalls while driving",
    "shuts off",
    "won't start",
    "hard shift",
    "brake pedal",
    "steering",
]


def _normalize_text(text: str) -> str:
    """Lowercase, replace hyphens with spaces, collapse whitespace."""
    if not text:
        return ""
    t = text.lower().replace("-", " ")
    return " ".join(t.split())


def _extract_tokens(query: str) -> list[str]:
    """Split on non-alphanum, keep length>=3, drop stopwords."""
    norm = _normalize_text(query)
    tokens = re.findall(r"[a-z0-9]+", norm)
    return [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]


def _keyword_score(doc_text: str, query_tokens: list[str], phrase_boosts: list[str]) -> float:
    """
    Base keyword_score = (#token hits in doc text) / sqrt(len(tokens)+1)
    + 3 per phrase hit. Cap at 10.
    """
    if not doc_text:
        return 0.0
    norm_doc = _normalize_text(doc_text)
    doc_tokens = set(re.findall(r"[a-z0-9]+", norm_doc))

    token_hits = sum(1 for t in query_tokens if t in doc_tokens)
    base = token_hits / math.sqrt(len(query_tokens) + 1)

    phrase_hits = sum(3 for p in phrase_boosts if _normalize_text(p) in norm_doc)
    score = base + phrase_hits
    return min(score, 10.0)


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
    Group by campaign_number, use final_score for campaign_score.
    Each result is (doc, final_score, dense_score, keyword_score).
    Evidence includes dense_score and keyword_score.
    """
    if make_norm:
        results = [
            r for r in results
            if (r[0].get("make_norm") or "") == make_norm
        ]

    by_campaign: dict[str, list[tuple[dict, float, float, float]]] = defaultdict(list)
    for doc, final_score, dense_score, keyword_score in results:
        cn = doc.get("campaign_number") or ""
        if not cn.strip():
            continue
        by_campaign[cn].append((doc, final_score, dense_score, keyword_score))

    campaign_results = []
    for cn, items in by_campaign.items():
        final_scores = [s[1] for s in items]
        campaign_score = sum(final_scores) + 0.2 * max(final_scores) + 0.5 * len(items)
        best = max(items, key=lambda x: x[1])
        best_doc, best_final, best_dense, best_kw = best
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        evidence_snippets = []
        for doc, final_s, dense_s, kw_s in sorted_items[:2]:
            text = (doc.get("text") or "")[:280]
            evidence_snippets.append({
                "doc_id": doc.get("doc_id", ""),
                "snippet": text,
                "dense_score": dense_s,
                "keyword_score": kw_s,
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
        help="Weight for keyword score in final_score = dense_score + alpha * keyword_score",
    )
    parser.add_argument(
        "--no-keyword-rerank",
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

    # Keyword rerank: compute keyword_score, final_score, re-rank
    use_rerank = not args.no_keyword_rerank
    alpha = args.alpha

    if use_rerank:
        query_tokens = _extract_tokens(query_text)
        phrase_boosts = [p for p in PHRASE_BOOSTS if _normalize_text(p) in _normalize_text(query_text)]
        logger.info("Rerank: enabled=True alpha=%.2f tokens=%d phrases=%d", alpha, len(query_tokens), len(phrase_boosts))

        reranked = []
        for doc, dense_score in results:
            doc_text = doc.get("text", "")
            keyword_score = _keyword_score(doc_text, query_tokens, phrase_boosts)
            final_score = dense_score + alpha * keyword_score
            reranked.append((doc, final_score, dense_score, keyword_score))
        reranked.sort(key=lambda x: x[1], reverse=True)
        results_for_agg = reranked
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
            ks = ev.get("keyword_score", 0)
            print(f"  Evidence {j}: {ev.get('doc_id', '')} (dense={ds:.3f} kw={ks:.3f}) ... {snippet}...")
        if i == 0:
            best_text = (camp.get("best_doc", {}).get("text") or "")[:120].replace("\n", " ")
            print()
            print(f"Suggested recall match: Campaign {camp['campaign_number']} - {best_text}...")

    print()


if __name__ == "__main__":
    main()
