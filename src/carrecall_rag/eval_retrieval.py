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

RetrievalMode = str  # "dense" | "keyword" | "hybrid"


def run_retrieval(
    query_text: str,
    make: str,
    model: str,
    year: int | None,
    indexes: dict,
    encoder: SentenceTransformer | None,
    *,
    mode: RetrievalMode = "hybrid",
    dense_topk: int = 100,
    keyword_topk: int = 150,
    alpha: float = 0.50,
    use_pool_indexes: bool = True,
    min_pool_docs: int = 50,
    rerank_tokens: int = 12,
    rerank_phrases: int = 10,
    return_score_details: bool = False,
) -> list[str] | tuple[list[str], list[dict]]:
    """
    Run retrieval pipeline: search (dense/keyword/hybrid) -> rerank -> aggregate.
    Returns list of campaign_numbers in rank order (top first).
    mode: "dense" | "keyword" | "hybrid"
    """
    make_norm = normalize_make(make) if make else ""
    mkey = model_key(model) if model else ""

    search_query = _build_query_from_context(make, model, year, query_text)

    # Dense search (skip for keyword-only)
    if mode == "keyword":
        results = []
    else:
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

    # Keyword search (for hybrid or keyword-only)
    if mode in ("hybrid", "keyword"):
        kw_results = keyword_search(
            query_text,
            make_norm,
            mkey,
            indexes,
            top_k=keyword_topk,
            use_pool_indexes=use_pool_indexes,
            min_pool_docs=min_pool_docs,
        )
        if mode == "keyword":
            results = kw_results
        else:
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

    # Rerank: alpha=0 for pure dense, alpha=1 for pure keyword, hybrid uses given alpha
    rerank_alpha = 0.0 if mode == "dense" else (1.0 if mode == "keyword" else alpha)
    results_for_agg = rerank(
        results,
        query_text,
        alpha=rerank_alpha,
        max_tokens=rerank_tokens,
        max_phrases=rerank_phrases,
        normalize_dense=(mode == "hybrid"),
    )

    # Aggregate by campaign
    campaign_results = _aggregate_by_campaign_with_scores(
        results_for_agg,
        make_norm=make_norm if make_norm else None,
    )
    campaigns = [c["campaign_number"] for c in campaign_results]

    if return_score_details:
        return campaigns, campaign_results
    return campaigns


def _parse_gold(row: dict) -> list[str]:
    """Extract gold campaign(s) from row. Supports gold_campaign or gold_campaigns."""
    golds = row.get("gold_campaigns")
    if golds is not None:
        return [str(g).strip() for g in (golds if isinstance(golds, list) else [golds]) if g]
    g = row.get("gold_campaign")
    return [g.strip()] if g and str(g).strip() else []


def _best_gold_rank(golds: list[str], campaigns: list[str]) -> tuple[int | None, str | None]:
    """1-based rank of first/best gold hit, and which gold. None if absent."""
    best_rank = None
    best_gold = None
    for gold in golds:
        for i, c in enumerate(campaigns):
            if c == gold:
                r = i + 1
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best_gold = gold
                break
    return best_rank, best_gold


def _gold_status(rank: int | None, top_k: int = 10) -> str:
    """Describe whether gold entered candidate set and where it ranked."""
    if rank is None:
        return "MISS (never in candidate set)"
    if rank <= top_k:
        return f"rank {rank}"
    return f"in candidates but ranked low (rank {rank})"


def _scores_for_campaign(campaign_result: dict) -> dict:
    """Extract dense, keyword, fused from campaign's best doc."""
    ev = (campaign_result.get("evidence_snippets") or [{}])[0]
    return {
        "dense": ev.get("dense_score", 0.0),
        "keyword": ev.get("keyword_score", 0.0),
        "fused": ev.get("combined", campaign_result.get("campaign_score", 0.0)),
    }


def _run_compare_table(
    queries: list[dict],
    indexes: dict,
    encoder: SentenceTransformer | None,
    args: argparse.Namespace,
    k_values: list[int],
) -> None:
    """Run dense, keyword, hybrid@0.3/0.5/0.7 and print compact comparison table."""
    configs = [
        ("dense", "dense", 0.0),
        ("keyword", "keyword", 0.0),
        ("hybrid@0.3", "hybrid", 0.3),
        ("hybrid@0.5", "hybrid", 0.5),
        ("hybrid@0.7", "hybrid", 0.7),
    ]
    results: dict[str, dict[int, int]] = {}
    valid_queries = [r for r in queries if r.get("query", "").strip() and _parse_gold(r)]
    n = len(valid_queries)
    if n == 0:
        logger.error("No valid queries for comparison")
        return

    for label, mode, alpha in configs:
        hit_at: dict[int, int] = defaultdict(int)
        for row in valid_queries:
            query = row.get("query", "").strip()
            make = row.get("make", "")
            model = row.get("model", "")
            golds = _parse_gold(row)
            year = row.get("year")
            if not query or not golds:
                continue
            out = run_retrieval(
                query, make, model, year, indexes, encoder,
                mode=mode, dense_topk=args.dense_topk, keyword_topk=args.keyword_topk,
                alpha=alpha, use_pool_indexes=not args.no_pool, min_pool_docs=50,
            )
            top10 = out[:10] if isinstance(out, list) else out[0][:10]
            for k in k_values:
                if any(g in top10[:k] for g in golds):
                    hit_at[k] += 1
        results[label] = hit_at

    print()
    print("=" * 70)
    print("Comparison: Recall@K by mode and alpha")
    print("=" * 70)
    header = f"{'Config':<14} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6}"
    print(header)
    print("-" * 40)
    for label in [c[0] for c in configs]:
        r = results[label]
        denom = n if n > 0 else 1
        line = f"{label:<14} {r[1]/denom:.2f}   {r[3]/denom:.2f}   {r[5]/denom:.2f}   {r[10]/denom:.2f}"
        print(line)
    print("=" * 70)


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
        "--output",
        type=str,
        default="eval/results/retrieval_debug.jsonl",
        help="JSONL file for per-query debug output",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dense", "keyword", "hybrid"],
        default="hybrid",
        help="Retrieval mode: dense, keyword, or hybrid",
    )
    parser.add_argument(
        "--dense-topk",
        type=int,
        default=100,
        help="Dense retrieval topk",
    )
    parser.add_argument(
        "--keyword-topk",
        type=int,
        default=150,
        help="Keyword (BM25) retrieval topk",
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
        default=0.50,
        help="Weight for keyword in hybrid: combined = (1-alpha)*dense + alpha*kw_norm",
    )
    parser.add_argument(
        "--alpha-list",
        type=str,
        default=None,
        help="Comma-separated alphas for sweep (e.g. 0.1,0.3,0.5,0.7). Only for mode=hybrid.",
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
        help="Extra verbose logging",
    )
    parser.add_argument(
        "--show-scores",
        action="store_true",
        help="Print dense, keyword, fused scores for gold and top competing campaigns",
    )
    parser.add_argument(
        "--compare-table",
        action="store_true",
        help="Run all modes + alpha sweep and print compact comparison table",
    )
    args = parser.parse_args()

    if not os.path.exists(args.eval_file):
        logger.error("Eval file not found: %s", args.eval_file)
        sys.exit(1)

    # Reduce log noise during eval unless verbose
    if not args.verbose:
        logging.getLogger("carrecall_rag").setLevel(logging.WARNING)

    # Alpha sweep: list of alphas to try (only for hybrid)
    if args.alpha_list:
        alpha_values = [float(x.strip()) for x in args.alpha_list.split(",")]
        if args.mode != "hybrid":
            logger.warning("--alpha-list ignored when mode != hybrid")
            alpha_values = [args.alpha]
    else:
        alpha_values = [args.alpha]

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
    logger.info("Mode: %s | Alpha(s): %s", args.mode, alpha_values)

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

    # Load encoder (not needed for keyword-only; required for --compare-table)
    encoder: SentenceTransformer | None = None
    if args.mode != "keyword" or args.compare_table:
        if not os.path.exists(args.model_dir):
            logger.error("Model not found: %s. Run train_biencoder first.", args.model_dir)
            sys.exit(1)
        encoder = SentenceTransformer(args.model_dir)

    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    k_values = [1, 3, 5, 10]
    topc = max(args.topc, max(k_values))

    # --compare-table: run all modes and alphas, print compact table
    if args.compare_table:
        _run_compare_table(
            queries=queries,
            indexes=indexes,
            encoder=encoder,
            args=args,
            k_values=k_values,
        )
        return

    # Run for each alpha (sweep) or single alpha
    for alpha in alpha_values:
        hit_at: dict[int, int] = defaultdict(int)
        miss_count = 0
        debug_rows: list[dict] = []

        for i, row in enumerate(queries):
            query = row.get("query", "").strip()
            make = row.get("make", "")
            model = row.get("model", "")
            golds = _parse_gold(row)
            year = row.get("year")

            if not query or not golds:
                logger.warning("Skipping row %d: missing query or gold_campaign", i + 1)
                continue

            out = run_retrieval(
                query,
                make,
                model,
                year,
                indexes,
                encoder,
                mode=args.mode,
                dense_topk=args.dense_topk,
                keyword_topk=args.keyword_topk,
                alpha=alpha,
                use_pool_indexes=not args.no_pool,
                min_pool_docs=50,
                return_score_details=args.show_scores,
            )
            campaign_results: list[dict] = []
            if args.show_scores:
                all_campaigns, campaign_results = out
            else:
                all_campaigns = out

            top10 = all_campaigns[:topc]
            rank, hit_gold = _best_gold_rank(golds, all_campaigns)
            status = _gold_status(rank, top_k=10)

            if rank is None:
                miss_count += 1
            for k in k_values:
                if any(g in top10[:k] for g in golds):
                    hit_at[k] += 1

            # Per-query debug
            camp_scores: dict[str, dict] = {}
            if args.show_scores and campaign_results:
                camp_scores = {c["campaign_number"]: _scores_for_campaign(c) for c in campaign_results}

            debug_row = {
                "query": query,
                "make": make,
                "model": model,
                "gold_campaign": golds[0] if len(golds) == 1 else golds,
                "top10_predicted": top10[:10],
                "gold_rank": rank,
                "gold_hit": hit_gold,
                "gold_status": status,
                "gold_in_candidates": rank is not None,
                "alpha": alpha,
                "mode": args.mode,
            }
            if camp_scores:
                debug_row["score_diagnostics"] = {
                    cn: camp_scores.get(cn, {}) for cn in golds + top10[:5]
                }
            debug_rows.append(debug_row)

            print()
            print(f"Query: {query}")
            print(f"Gold: {golds}")
            print(f"Top 10 predicted: {top10[:10]}")
            print(f"First correct hit: {rank if rank else 'MISS'}")
            print(f"Status: {status}")

            if camp_scores:
                to_show = list(golds)
                for c in top10[:5]:
                    if c not in to_show:
                        to_show.append(c)
                print("  Scores (dense, keyword, fused):")
                for cn in to_show:
                    s = camp_scores.get(cn, {"dense": 0, "keyword": 0, "fused": 0})
                    marker = " [GOLD]" if cn in golds else ""
                    print(f"    {cn}: dense={s['dense']:.4f} kw={s['keyword']:.4f} fused={s['fused']:.4f}{marker}")

        n = len(queries)

        # Summary section
        print()
        print("=" * 60)
        print(f"Summary (mode={args.mode}, alpha={alpha})")
        print("=" * 60)
        print(f"Total queries: {n}")
        print(f"Gold in top 1:  {hit_at[1]}")
        print(f"Gold in top 3:  {hit_at[3]}")
        print(f"Gold in top 5:  {hit_at[5]}")
        print(f"Gold in top 10: {hit_at[10]}")
        print(f"Total misses:   {miss_count}")
        for k in k_values:
            recall = hit_at[k] / n if n > 0 else 0.0
            print(f"Recall@{k}: {recall:.2f}")
        print("=" * 60)

        # Save debug JSONL (one file per alpha when sweeping)
        out_path = args.output
        if len(alpha_values) > 1:
            base, ext = os.path.splitext(args.output)
            out_path = f"{base}_alpha{alpha:.2f}{ext}"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in debug_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Wrote per-query debug to %s", out_path)


if __name__ == "__main__":
    main()
