"""CLI: retrieval + campaign aggregation (dense/hybrid, optional rerank)."""

import argparse
import logging
import os
import sys
from collections import defaultdict

from sentence_transformers import SentenceTransformer

from .config import DATA_DIR, PROCESSED_DIR
from .rerank import NeuralReranker, build_hybrid_candidates, rerank_candidates
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


def _build_query_from_context(make: str, model: str, year: int | None, query_text: str) -> str:
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
    results: list[dict],
    make_norm: str | None,
    debug: bool = False,
) -> list[dict]:
    """Group by campaign_number; score = max(rerank_score or hybrid_score). Expects doc, *_score on each row."""
    if make_norm:
        results = [
            r for r in results
            if ((r.get("doc") or {}).get("make_norm") or "") == make_norm
        ]

    by_campaign: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        doc = row.get("doc") or {}
        cn = doc.get("campaign_number") or ""
        if not cn.strip():
            continue
        by_campaign[cn].append(row)

    campaign_results = []
    for cn, items in by_campaign.items():
        stage_scores = [x.get("rerank_score", x.get("hybrid_score", 0.0)) for x in items]
        if debug and cn in ("22V406000", "18V332000"):
            print("DEBUG cn", cn, "hits", len(items),
                  "best_stage", max(stage_scores),
                  "sum_stage", sum(stage_scores))
        campaign_score = max(stage_scores)
        best = max(items, key=lambda x: x.get("rerank_score", x.get("hybrid_score", 0.0)))
        best_doc = best.get("doc") or {}
        sorted_items = sorted(
            items,
            key=lambda x: x.get("rerank_score", x.get("hybrid_score", 0.0)),
            reverse=True,
        )
        evidence_snippets = []
        for item in sorted_items[:2]:
            doc = item.get("doc") or {}
            text = (doc.get("text") or "")[:280]
            evidence_snippets.append({
                "doc_id": doc.get("doc_id", ""),
                "snippet": text,
                "dense_score": item.get("dense_score", 0.0),
                "keyword_score": item.get("keyword_score", 0.0),
                "hybrid_score": item.get("hybrid_score", 0.0),
                "rerank_score": item.get("rerank_score", item.get("hybrid_score", 0.0)),
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
    parser.add_argument("--topk", type=int, default=30, help="Number of docs to retrieve (or dense_topk when --hybrid)")
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
        default=0.50,
        help="Stage-1 hybrid fusion weight: (1-alpha)*dense + alpha*keyword (default 0.50)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable stage-2 neural reranking over stage-1 candidates",
    )
    parser.add_argument(
        "--rerank-topn",
        type=int,
        default=50,
        help="Number of stage-1 candidates passed to reranker",
    )
    parser.add_argument(
        "--rerank-model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model name/path for neural reranking",
    )
    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=32,
        help="Batch size for neural reranking",
    )
    parser.add_argument(
        "--show-candidates",
        action="store_true",
        help="Print candidate doc IDs + snippets before stage-2 rerank",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Union of dense + keyword retrieval before rerank (dense_topk + keyword_topk)",
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
        default=100,
        help="Keyword (BM25) retrieval topk when --hybrid",
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

    # Search: dense only, or hybrid (dense + keyword fused with alpha)
    use_hybrid = args.hybrid
    dense_topk = max(args.dense_topk if use_hybrid else args.topk, args.rerank_topn)
    keyword_topk = max(args.keyword_topk, args.rerank_topn) if use_hybrid else 0

    dense_results = search(
        search_query,
        make_norm,
        mkey,
        indexes,
        model,
        top_k=dense_topk,
        use_pool_indexes=not args.no_pool,
        min_pool_docs=50,
    )

    keyword_results: list[tuple[dict, float]] = []
    if use_hybrid and keyword_topk > 0:
        keyword_results = keyword_search(
            query_text,
            make_norm,
            mkey,
            indexes,
            top_k=keyword_topk,
            use_pool_indexes=not args.no_pool,
            min_pool_docs=50,
        )
    candidates = build_hybrid_candidates(
        dense_results,
        keyword_results,
        alpha=args.alpha,
        top_n=args.rerank_topn,
    )
    logger.info(
        "Stage1 hybrid: dense=%d keyword=%d -> candidates=%d (top_n=%d, alpha=%.2f)",
        len(dense_results),
        len(keyword_results),
        len(candidates),
        args.rerank_topn,
        args.alpha,
    )

    if not candidates:
        logger.info("No results found.")
        sys.exit(0)

    # Debug: print candidates before stage-2 rerank.
    if args.show_candidates:
        print("\n--- Candidates (after stage1 hybrid, before stage2 rerank) ---")
        for i, item in enumerate(candidates[:30]):
            doc = item.get("doc") or {}
            cid = doc.get("campaign_number", "")
            did = doc.get("doc_id", "")
            snippet = (doc.get("text", "") or "")[:80].replace("\n", " ")
            hs = item.get("hybrid_score", 0.0)
            print(f"  {i+1}. {cid} | {did} | hybrid={hs:.3f} | {snippet}...")
        print("---\n")

    reranker = None
    if args.rerank:
        reranker = NeuralReranker(
            model_name=args.rerank_model,
            batch_size=args.rerank_batch_size,
        )
    ranked_docs = rerank_candidates(
        query_text,
        candidates,
        use_rerank=args.rerank,
        reranker=reranker,
        top_k=args.topk,
    )
    campaigns = _aggregate_by_campaign_with_scores(
        ranked_docs,
        make_norm=make_norm if make_norm else None,
        debug=args.show_candidates,
    )

    # Print top campaigns
    for i, camp in enumerate(campaigns[: args.topc]):
        print()
        print(f"Campaign: {camp['campaign_number']} | Score: {camp['campaign_score']:.2f}")
        for j, ev in enumerate(camp["evidence_snippets"], 1):
            snippet = (ev.get("snippet") or "").replace("\n", " ")
            ds = ev.get("dense_score", 0)
            kw = ev.get("keyword_score", 0)
            hs = ev.get("hybrid_score", 0)
            rs = ev.get("rerank_score", hs)
            print(f"  Evidence {j}: {ev.get('doc_id', '')} (dense={ds:.3f} kw={kw:.3f} hybrid={hs:.3f} rerank={rs:.3f}) ... {snippet}...")
        if i == 0:
            best_text = (camp.get("best_doc", {}).get("text") or "")[:120].replace("\n", " ")
            print()
            print(f"Suggested recall match: Campaign {camp['campaign_number']} - {best_text}...")

    print()


if __name__ == "__main__":
    main()
