"""Recall@K, MRR, per-query JSONL over labeled queries."""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict

from sentence_transformers import SentenceTransformer

from .config import DATA_DIR, PROCESSED_DIR
from .demo_retrieve import (
    _aggregate_by_campaign_with_scores,
    _build_query_from_context,
)
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
    rerank_topn: int = 50,
    use_rerank: bool = False,
    reranker: NeuralReranker | None = None,
    rerank_limit: int = 0,
    rerank_text_format: str = "full",
    return_score_details: bool = False,
) -> list[str] | tuple[list[str], list[dict], list[str], list[dict], list[dict]]:
    """Run search (dense/keyword/hybrid) -> optional rerank -> aggregate. Returns campaign_numbers in rank order."""
    make_norm = normalize_make(make) if make else ""
    mkey = model_key(model) if model else ""

    search_query = _build_query_from_context(make, model, year, query_text)

    if mode == "keyword":
        dense_results = []
    else:
        dense_results = search(
            search_query,
            make_norm,
            mkey,
            indexes,
            encoder,
            top_k=dense_topk,
            use_pool_indexes=use_pool_indexes,
            min_pool_docs=min_pool_docs,
        )

    keyword_results: list[tuple[dict, float]] = []
    if mode in ("hybrid", "keyword"):
        keyword_results = keyword_search(
            query_text,
            make_norm,
            mkey,
            indexes,
            top_k=keyword_topk,
            use_pool_indexes=use_pool_indexes,
            min_pool_docs=min_pool_docs,
        )

    stage1_alpha = alpha if mode == "hybrid" else (0.0 if mode == "dense" else 1.0)
    stage1_topn = rerank_topn if mode == "hybrid" else (dense_topk if mode == "dense" else keyword_topk)
    candidates = build_hybrid_candidates(
        dense_results,
        keyword_results,
        alpha=stage1_alpha,
        top_n=stage1_topn,
    )
    if not candidates:
        if return_score_details:
            return [], [], [], [], []
        return []

    # Save stage-1 ranking (before optional stage-2 rerank)
    pre_campaign_results = _aggregate_by_campaign_with_scores(
        candidates,
        make_norm=make_norm if make_norm else None,
    )
    pre_campaigns = [c["campaign_number"] for c in pre_campaign_results]

    ranked_docs = rerank_candidates(
        query_text,
        candidates,
        use_rerank=use_rerank,
        reranker=reranker,
        rerank_limit=rerank_limit,
        text_format=rerank_text_format,
    )

    # Aggregate by campaign after stage-2 rerank (or same order when disabled)
    campaign_results = _aggregate_by_campaign_with_scores(
        ranked_docs,
        make_norm=make_norm if make_norm else None,
    )
    campaigns = [c["campaign_number"] for c in campaign_results]

    if return_score_details:
        return campaigns, campaign_results, pre_campaigns, candidates, ranked_docs
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


def _reciprocal_rank(rank: int | None) -> float:
    """Reciprocal rank: 1/rank if gold found, else 0."""
    if rank is None or rank < 1:
        return 0.0
    return 1.0 / rank


def _gold_status(rank: int | None, top_k: int = 10) -> str:
    """Describe whether gold entered candidate set and where it ranked."""
    if rank is None:
        return "MISS (never in candidate set)"
    if rank <= top_k:
        return f"rank {rank}"
    return f"in candidates but ranked low (rank {rank})"


def _scores_for_campaign(campaign_result: dict) -> dict:
    """Extract dense, keyword, hybrid, rerank from campaign's best doc."""
    ev = (campaign_result.get("evidence_snippets") or [{}])[0]
    return {
        "dense": ev.get("dense_score", 0.0),
        "keyword": ev.get("keyword_score", 0.0),
        "hybrid": ev.get("hybrid_score", 0.0),
        "rerank": ev.get("rerank_score", campaign_result.get("campaign_score", 0.0)),
    }


def _rank_map(campaigns: list[str]) -> dict[str, int]:
    """Map campaign -> 1-based rank."""
    return {c: i + 1 for i, c in enumerate(campaigns)}


def _best_doc_by_campaign(rows: list[dict]) -> dict[str, dict]:
    """Pick best supporting doc per campaign using rerank_score (fallback hybrid)."""
    best: dict[str, dict] = {}
    for row in rows:
        cn = row.get("campaign_number", "")
        if not cn:
            continue
        score = row.get("rerank_score", row.get("hybrid_score", 0.0))
        if cn not in best:
            best[cn] = row
            continue
        prev = best[cn].get("rerank_score", best[cn].get("hybrid_score", 0.0))
        if score > prev:
            best[cn] = row
    return best


def _moved_above_gold(
    *,
    gold: str,
    pre_ranks: dict[str, int],
    post_ranks: dict[str, int],
) -> list[str]:
    """
    Campaigns that were below gold pre-rerank but moved above gold post-rerank.
    """
    pre_gold = pre_ranks.get(gold)
    post_gold = post_ranks.get(gold)
    if pre_gold is None or post_gold is None:
        return []

    moved = []
    for campaign, post_rank in post_ranks.items():
        pre_rank = pre_ranks.get(campaign, 10**9)
        if pre_rank > pre_gold and post_rank < post_gold:
            moved.append(campaign)
    moved.sort(key=lambda c: post_ranks[c])
    return moved


def _tokenize_terms(text: str) -> list[str]:
    """Simple lexical terms for overlap diagnostics."""
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in toks if len(t) >= 3]


def _term_overlap(query: str, candidate_text: str) -> dict[str, list[str] | int]:
    """Return overlap/missing query terms against candidate text."""
    q_terms = sorted(set(_tokenize_terms(query)))
    c_terms = set(_tokenize_terms(candidate_text))
    overlap = [t for t in q_terms if t in c_terms]
    missing = [t for t in q_terms if t not in c_terms]
    return {
        "query_terms": q_terms,
        "overlap_terms": overlap,
        "missing_terms": missing,
        "overlap_count": len(overlap),
        "missing_count": len(missing),
    }


def _run_compare_table(
    queries: list[dict],
    indexes: dict,
    encoder: SentenceTransformer | None,
    reranker: NeuralReranker | None,
    args: argparse.Namespace,
    k_values: list[int],
) -> None:
    """Compare baseline and rerank text construction variants."""
    rerank_limit_for_compare = args.rerank_limit if args.rerank_limit > 0 else args.rerank_topn
    configs = [
        ("no-rerank", False, "full"),
        ("rerank-full", True, "full"),
        ("rerank-compact", True, "compact"),
        ("rerank-smart", True, "smart"),
    ]
    results: dict[str, dict[int, int]] = {}
    valid_queries = [r for r in queries if r.get("query", "").strip() and _parse_gold(r)]
    n = len(valid_queries)
    if n == 0:
        logger.error("No valid queries for comparison")
        return

    for label, use_rerank, text_format in configs:
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
                mode="hybrid", dense_topk=args.dense_topk, keyword_topk=args.keyword_topk,
                alpha=args.alpha, use_pool_indexes=not args.no_pool, min_pool_docs=50,
                rerank_topn=max(args.rerank_topn, rerank_limit_for_compare),
                use_rerank=use_rerank,
                reranker=reranker,
                rerank_limit=rerank_limit_for_compare if use_rerank else 0,
                rerank_text_format=text_format,
            )
            top10 = out[:10]
            for k in k_values:
                if any(g in top10[:k] for g in golds):
                    hit_at[k] += 1
        results[label] = hit_at

    print()
    print("=" * 70)
    print("Comparison: no-rerank vs full/compact/smart rerank")
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
        help="Stage-1 hybrid fusion weight: (1-alpha)*dense + alpha*keyword",
    )
    parser.add_argument("--rerank", action="store_true", help="Enable stage-2 neural reranking")
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
        help="Cross-encoder model name/path for stage-2 reranking",
    )
    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=32,
        help="Batch size for neural reranking",
    )
    parser.add_argument(
        "--rerank-text-format",
        type=str,
        choices=["full", "compact", "smart"],
        default="full",
        help="Text construction for reranker: full, compact, or smart",
    )
    parser.add_argument(
        "--rerank-limit",
        type=int,
        default=0,
        help="Limit how many stage-1 candidates get reranked (0 = rerank all in topN)",
    )
    parser.add_argument(
        "--rerank-inspect-k",
        type=int,
        default=3,
        help="Number of promoted candidates to print with reranker score + input text preview",
    )
    parser.add_argument(
        "--save-rerank-input-text",
        action="store_true",
        help="Save full candidate text passed to reranker in debug JSONL (can be large)",
    )
    parser.add_argument(
        "--focus-query",
        type=str,
        default="",
        help="Only print deep diagnostics for queries containing this substring",
    )
    parser.add_argument(
        "--focus-campaigns",
        type=str,
        default="",
        help="Comma-separated campaigns to print full reranker input text for",
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
        help="Print dense/keyword/hybrid/rerank scores for gold and competing campaigns",
    )
    parser.add_argument(
        "--compare-table",
        action="store_true",
        help="Run no-rerank, rerank-full, rerank-compact, rerank-smart comparison",
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
    logger.info(
        "Mode: %s | Alpha(s): %s | Rerank: %s | topN: %d | limit: %d | text_format: %s",
        args.mode,
        alpha_values,
        args.rerank,
        args.rerank_topn,
        args.rerank_limit,
        args.rerank_text_format,
    )

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

    # Load encoder (not needed for keyword-only unless compare table is requested)
    encoder: SentenceTransformer | None = None
    if args.mode != "keyword" or args.compare_table:
        if not os.path.exists(args.model_dir):
            logger.error("Model not found: %s. Run train_biencoder first.", args.model_dir)
            sys.exit(1)
        encoder = SentenceTransformer(args.model_dir)

    # Load neural reranker when enabled (or for compare table's hybrid+rerank row).
    reranker: NeuralReranker | None = None
    if args.rerank or args.compare_table:
        reranker = NeuralReranker(
            model_name=args.rerank_model,
            batch_size=args.rerank_batch_size,
        )

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
            reranker=reranker,
            args=args,
            k_values=k_values,
        )
        return

    # Run for each alpha (sweep) or single alpha
    for alpha in alpha_values:
        hit_at: dict[int, int] = defaultdict(int)
        miss_count = 0
        first_correct_ranks: list[int] = []  # 1-based ranks for avg/median (excludes misses)
        reciprocal_ranks: list[float] = []   # for MRR (one per evaluated query, 0 if miss)
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
                rerank_topn=args.rerank_topn,
                use_rerank=args.rerank,
                reranker=reranker,
                rerank_limit=args.rerank_limit,
                rerank_text_format=args.rerank_text_format,
                return_score_details=True,
            )
            all_campaigns, campaign_results, pre_campaigns, pre_candidates, ranked_docs = out

            top10 = all_campaigns[:topc]
            rank, hit_gold = _best_gold_rank(golds, all_campaigns)
            status = _gold_status(rank, top_k=10)
            pre_rank, _ = _best_gold_rank(golds, pre_campaigns)
            pre_rank_map = _rank_map(pre_campaigns)
            post_rank_map = _rank_map(all_campaigns)

            if rank is None:
                miss_count += 1
            for k in k_values:
                if any(g in top10[:k] for g in golds):
                    hit_at[k] += 1

            rr = _reciprocal_rank(rank)
            reciprocal_ranks.append(rr)
            if rank is not None:
                first_correct_ranks.append(rank)

            hit_at_1 = 1 if (rank is not None and rank <= 1) else 0
            hit_at_3 = 1 if (rank is not None and rank <= 3) else 0
            hit_at_5 = 1 if (rank is not None and rank <= 5) else 0
            hit_at_10 = 1 if (rank is not None and rank <= 10) else 0

            # Per-query debug
            camp_scores: dict[str, dict] = {}
            if args.show_scores and campaign_results:
                camp_scores = {c["campaign_number"]: _scores_for_campaign(c) for c in campaign_results}

            best_doc_post = _best_doc_by_campaign(ranked_docs)
            best_doc_pre = _best_doc_by_campaign(pre_candidates)

            gold_deltas = []
            promoted_union: set[str] = set()
            for gold in golds:
                moved_above = _moved_above_gold(
                    gold=gold,
                    pre_ranks=pre_rank_map,
                    post_ranks=post_rank_map,
                )
                promoted_union.update(moved_above)
                gold_deltas.append({
                    "gold_campaign": gold,
                    "pre_rank": pre_rank_map.get(gold),
                    "post_rank": post_rank_map.get(gold),
                    "rank_delta": (
                        None
                        if pre_rank_map.get(gold) is None or post_rank_map.get(gold) is None
                        else post_rank_map[gold] - pre_rank_map[gold]
                    ),
                    "moved_above_gold": moved_above,
                })

            promoted_details = []
            for cn in sorted(promoted_union, key=lambda x: post_rank_map.get(x, 10**9)):
                post = best_doc_post.get(cn, {})
                pre = best_doc_pre.get(cn, {})
                post_text = post.get("rerank_input_text", "") or ""
                promoted_details.append({
                    "campaign_number": cn,
                    "pre_rank": pre_rank_map.get(cn),
                    "post_rank": post_rank_map.get(cn),
                    "dense_score": post.get("dense_score", pre.get("dense_score", 0.0)),
                    "keyword_score": post.get("keyword_score", pre.get("keyword_score", 0.0)),
                    "hybrid_score": post.get("hybrid_score", pre.get("hybrid_score", 0.0)),
                    "rerank_score": post.get("rerank_score", 0.0),
                    "doc_id": (post.get("doc") or {}).get("doc_id", ""),
                    "input_char_len": post.get("rerank_input_char_len", 0),
                    "input_text_preview": (post.get("rerank_input_text", "") or "")[:240],
                    "rerank_text_fields": post.get("rerank_text_fields", {}),
                    "query_term_overlap": _term_overlap(query, post_text),
                })

            gold_score_debug = []
            for gold in golds:
                post = best_doc_post.get(gold, {})
                pre = best_doc_pre.get(gold, {})
                post_text = post.get("rerank_input_text", "") or ""
                gold_score_debug.append({
                    "gold_campaign": gold,
                    "pre_rank": pre_rank_map.get(gold),
                    "post_rank": post_rank_map.get(gold),
                    "dense_score": post.get("dense_score", pre.get("dense_score", 0.0)),
                    "keyword_score": post.get("keyword_score", pre.get("keyword_score", 0.0)),
                    "hybrid_score": post.get("hybrid_score", pre.get("hybrid_score", 0.0)),
                    "rerank_score": post.get("rerank_score", 0.0),
                    "doc_id": (post.get("doc") or {}).get("doc_id", ""),
                    "input_char_len": post.get("rerank_input_char_len", 0),
                    "input_text_preview": (post.get("rerank_input_text", "") or "")[:240],
                    "rerank_text_fields": post.get("rerank_text_fields", {}),
                    "query_term_overlap": _term_overlap(query, post_text),
                })

            rerank_scope = args.rerank_limit if args.rerank_limit > 0 else args.rerank_topn
            scoped_rows = [r for r in ranked_docs if (r.get("stage1_rank") or 10**9) <= rerank_scope]
            input_lens = [r.get("rerank_input_char_len", 0) for r in scoped_rows]
            rerank_input_stats = {
                "rerank_scope": rerank_scope,
                "docs_scored": len(scoped_rows),
                "avg_char_len": (sum(input_lens) / len(input_lens)) if input_lens else 0.0,
                "max_char_len": max(input_lens) if input_lens else 0,
                "long_docs_over_2000": sum(1 for n in input_lens if n > 2000),
            }

            debug_row = {
                "query": query,
                "make": make,
                "model": model,
                "gold_campaigns": golds,
                "gold_campaign": golds[0] if len(golds) == 1 else golds,
                "predicted_topk": top10[: max(k_values)],
                "top10_predicted": top10[:10],
                "first_correct_rank": rank if rank is not None else "MISS",
                "gold_rank": rank,
                "reciprocal_rank": rr,
                "hit_at_1": hit_at_1,
                "hit_at_3": hit_at_3,
                "hit_at_5": hit_at_5,
                "hit_at_10": hit_at_10,
                "gold_hit": hit_gold,
                "gold_status": status,
                "gold_in_candidates": rank is not None,
                "alpha": alpha,
                "mode": args.mode,
                "rerank_enabled": args.rerank,
                "stage1_topn": args.rerank_topn,
                "rerank_limit": args.rerank_limit,
                "before_rerank_top10": pre_campaigns[:10],
                "after_rerank_top10": top10[:10],
                "gold_rank_before_rerank": pre_rank,
                "gold_rank_after_rerank": rank,
                "gold_deltas": gold_deltas,
                "promoted_above_gold": promoted_details,
                "gold_score_debug": gold_score_debug,
                "rerank_input_stats": rerank_input_stats,
            }
            if camp_scores:
                debug_row["score_diagnostics"] = {
                    cn: camp_scores.get(cn, {}) for cn in golds + top10[:5]
                }
            debug_row["reranker_inputs_top"] = [
                {
                    "campaign_number": item.get("campaign_number", ""),
                    "doc_id": (item.get("doc") or {}).get("doc_id", ""),
                    "stage1_rank": item.get("stage1_rank"),
                    "post_rerank_rank": item.get("post_rerank_rank"),
                    "dense_score": item.get("dense_score", 0.0),
                    "keyword_score": item.get("keyword_score", 0.0),
                    "hybrid_score": item.get("hybrid_score", 0.0),
                    "rerank_score": item.get("rerank_score", 0.0),
                    "input_char_len": item.get("rerank_input_char_len", 0),
                    "input_text_preview": (item.get("rerank_input_text", "") or "")[:240],
                    "rerank_text_fields": item.get("rerank_text_fields", {}),
                }
                for item in ranked_docs[: max(0, args.rerank_inspect_k)]
            ]
            if args.save_rerank_input_text:
                debug_row["reranker_inputs_all"] = [
                    {
                        "campaign_number": item.get("campaign_number", ""),
                        "doc_id": (item.get("doc") or {}).get("doc_id", ""),
                        "stage1_rank": item.get("stage1_rank"),
                        "post_rerank_rank": item.get("post_rerank_rank"),
                        "input_char_len": item.get("rerank_input_char_len", 0),
                        "rerank_text_fields": item.get("rerank_text_fields", {}),
                        "input_text": item.get("rerank_input_text", ""),
                    }
                    for item in ranked_docs[:rerank_scope]
                ]

            focus_campaigns = [
                c.strip()
                for c in (args.focus_campaigns or "").split(",")
                if c.strip()
            ]
            focus_match = not args.focus_query or args.focus_query.lower() in query.lower()
            if focus_match and focus_campaigns:
                focus_dump = {}
                for cn in focus_campaigns:
                    row = best_doc_post.get(cn) or best_doc_pre.get(cn) or {}
                    rtext = row.get("rerank_input_text", "") or ""
                    focus_dump[cn] = {
                        "campaign_number": cn,
                        "pre_rank": pre_rank_map.get(cn),
                        "post_rank": post_rank_map.get(cn),
                        "rerank_score": row.get("rerank_score", 0.0),
                        "doc_id": (row.get("doc") or {}).get("doc_id", ""),
                        "rerank_text_fields": row.get("rerank_text_fields", {}),
                        "query_term_overlap": _term_overlap(query, rtext),
                        "rerank_input_text": rtext,
                    }
                debug_row["focus_campaign_inputs"] = focus_dump
            debug_rows.append(debug_row)

            print()
            print(f"Query: {query}")
            print(f"Gold: {golds}")
            print(f"Top 10 predicted: {top10[:10]}")
            print(f"First correct hit: {rank if rank else 'MISS'}")
            print(f"Status: {status}")
            print("Gold rank changes:")
            for gd in gold_deltas:
                print(
                    f"  {gd['gold_campaign']}: pre={gd['pre_rank']} post={gd['post_rank']} "
                    f"delta={gd['rank_delta']} moved_above={gd['moved_above_gold'][:args.rerank_inspect_k]}"
                )

            if promoted_details:
                print("Promoted above gold after rerank:")
                for p in promoted_details[: max(0, args.rerank_inspect_k)]:
                    print(
                        f"  {p['campaign_number']} pre={p['pre_rank']} post={p['post_rank']} "
                        f"rerank={p['rerank_score']:.4f} len={p['input_char_len']}"
                    )

            if gold_score_debug:
                print("Gold reranker scores:")
                for gsd in gold_score_debug:
                    print(
                        f"  {gsd['gold_campaign']} pre={gsd['pre_rank']} post={gsd['post_rank']} "
                        f"rerank={gsd['rerank_score']:.4f} len={gsd['input_char_len']}"
                    )
                    qov = gsd.get("query_term_overlap", {})
                    print(
                        f"    overlap={qov.get('overlap_terms', [])[:8]} "
                        f"missing={qov.get('missing_terms', [])[:8]}"
                    )
            print(
                "Rerank input stats: "
                f"scored={rerank_input_stats['docs_scored']} "
                f"avg_len={rerank_input_stats['avg_char_len']:.1f} "
                f"max_len={rerank_input_stats['max_char_len']} "
                f"long>2000={rerank_input_stats['long_docs_over_2000']}"
            )
            if focus_match and focus_campaigns:
                print("Focused campaign reranker inputs:")
                for cn in focus_campaigns:
                    row = best_doc_post.get(cn) or best_doc_pre.get(cn) or {}
                    text = row.get("rerank_input_text", "") or ""
                    fields = row.get("rerank_text_fields", {})
                    qov = _term_overlap(query, text)
                    print(f"  {cn} doc_id={(row.get('doc') or {}).get('doc_id', '')} rerank={row.get('rerank_score', 0.0):.4f}")
                    print(f"    fields={fields}")
                    print(f"    overlap={qov.get('overlap_terms', [])}")
                    print(f"    missing={qov.get('missing_terms', [])}")
                    print(f"    text={text}")

            if camp_scores:
                to_show = list(golds)
                for c in top10[:5]:
                    if c not in to_show:
                        to_show.append(c)
                print("  Scores (dense, keyword, hybrid, rerank):")
                for cn in to_show:
                    s = camp_scores.get(cn, {"dense": 0, "keyword": 0, "hybrid": 0, "rerank": 0})
                    marker = " [GOLD]" if cn in golds else ""
                    print(
                        f"    {cn}: dense={s['dense']:.4f} kw={s['keyword']:.4f} "
                        f"hybrid={s['hybrid']:.4f} rerank={s['rerank']:.4f}{marker}"
                    )

        n = len(queries)
        evaluated_n = len(debug_rows)
        if evaluated_n == 0:
            logger.warning("No evaluated queries for this alpha; skipping summary.")
            continue

        mrr = sum(reciprocal_ranks) / evaluated_n
        avg_rank = sum(first_correct_ranks) / len(first_correct_ranks) if first_correct_ranks else float("nan")
        median_rank = float("nan")
        if first_correct_ranks:
            sorted_ranks = sorted(first_correct_ranks)
            mid = len(sorted_ranks) // 2
            median_rank = sorted_ranks[mid] if len(sorted_ranks) % 2 else (sorted_ranks[mid - 1] + sorted_ranks[mid]) / 2.0

        # Summary section
        print()
        print("=" * 60)
        print(
            f"Summary (mode={args.mode}, alpha={alpha}, "
            f"rerank={args.rerank}, rerank_topn={args.rerank_topn}, rerank_limit={args.rerank_limit})"
        )
        print("=" * 60)
        print(f"Total queries (in file): {n}")
        print(f"Evaluated queries:       {evaluated_n}")
        print(f"Gold in top 1:  {hit_at[1]}")
        print(f"Gold in top 3:  {hit_at[3]}")
        print(f"Gold in top 5:  {hit_at[5]}")
        print(f"Gold in top 10: {hit_at[10]}")
        print(f"Failure / miss count:    {miss_count}")
        for k in k_values:
            recall = hit_at[k] / evaluated_n
            print(f"Recall@{k}: {recall:.2f}")
        # In this setup each query has one (or more) gold label; hit = any gold in top-k.
        # So Hit@k and Recall@k are the same; we do not report them as separate metrics.
        print("  (Hit@k = Recall@k here: one gold set per query, hit = any gold in top-k)")
        print(f"MRR:                     {mrr:.4f}")
        if first_correct_ranks:
            print(f"Avg rank of 1st correct:   {avg_rank:.2f}")
            print(f"Median rank of 1st correct: {median_rank:.2f}")
        else:
            print("Avg rank of 1st correct:   N/A (no hits)")
            print("Median rank of 1st correct: N/A (no hits)")
        print("=" * 60)

        # CV-ready metrics (concise numbers for reporting)
        print()
        print("--- CV-ready metrics ---")
        print(f"  Recall@1:      {hit_at[1] / evaluated_n:.2f}")
        print(f"  Recall@10:     {hit_at[10] / evaluated_n:.2f}")
        print(f"  MRR:           {mrr:.4f}")
        print(f"  Queries (n):   {evaluated_n}")
        print("------------------------")

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
