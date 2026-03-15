"""CLI: End-to-end RAG demo — hybrid retrieval + grounded answer generation."""

import argparse
import logging
import os
import sys

from sentence_transformers import SentenceTransformer

from .config import DATA_DIR, PROCESSED_DIR
from .demo_retrieve import (
    _aggregate_by_campaign_with_scores,
    _build_query_from_context,
)
from .rag_answer import generate_rag_answer
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG demo: hybrid retrieval + grounded answer from top recalls"
    )
    parser.add_argument("--make", type=str, default="", help="Vehicle make (e.g. Jeep)")
    parser.add_argument("--model", type=str, default="", help="Vehicle model (e.g. Grand Cherokee)")
    parser.add_argument("--year", type=int, default=None, help="Vehicle year (e.g. 2019)")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query text; if missing, read multiline from stdin",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of top recalls to use for answer (default 3)",
    )
    parser.add_argument(
        "--dense-topk",
        type=int,
        default=100,
        help="Dense retrieval topk for hybrid",
    )
    parser.add_argument(
        "--keyword-topk",
        type=int,
        default=100,
        help="Keyword retrieval topk for hybrid",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.50,
        help="Stage-1 hybrid fusion weight: (1-alpha)*dense + alpha*keyword (default 0.50)",
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
    parser.add_argument("--rerank", action="store_true", help="Enable stage-2 neural reranking")
    parser.add_argument(
        "--rerank-topn",
        type=int,
        default=50,
        help="Number of stage-1 hybrid candidates passed to reranker",
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
        "--debug",
        action="store_true",
        help="Enable debug logs and retrieval internals",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default="",
        help="Optional file path to save final demo output",
    )
    args = parser.parse_args()

    if not args.debug:
        # Keep demo output clean and user-facing by default.
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("carrecall_rag").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)

    make_norm = normalize_make(args.make) if args.make else ""
    mkey = model_key(args.model) if args.model else ""
    vehicle_label = " ".join(x for x in [args.make.strip(), args.model.strip()] if x).strip()

    if args.query is not None:
        query_text = args.query
    else:
        logger.info("Reading query from stdin (multiline, Ctrl-D to finish)...")
        query_text = sys.stdin.read().strip()

    if not query_text:
        logger.error("No query text provided. Use --query or pipe to stdin.")
        sys.exit(1)

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

    if not os.path.exists(args.model_dir):
        logger.error("Model not found: %s. Run train_biencoder first.", args.model_dir)
        sys.exit(1)
    model = SentenceTransformer(args.model_dir)

    # Stage 1: Hybrid retrieval candidate generation (dense + keyword -> fused top N).
    dense_topk = max(args.dense_topk, args.rerank_topn)
    keyword_topk = max(args.keyword_topk, args.rerank_topn)
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
        answer = generate_rag_answer(query_text, [], top_k=args.topk, vehicle=vehicle_label)
        final_output = "\n" + answer
        print(final_output)
        if args.save_output:
            out_path = os.path.abspath(args.save_output)
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(final_output.rstrip() + "\n")
        sys.exit(0)

    # Stage 2: Optional neural reranking on stage-1 candidates.
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
        debug=args.debug,
    )

    # Generate grounded RAG answer
    answer = generate_rag_answer(query_text, campaigns, top_k=args.topk, vehicle=vehicle_label)
    final_output = "\n" + answer
    print(final_output)
    if args.save_output:
        out_path = os.path.abspath(args.save_output)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(final_output.rstrip() + "\n")


if __name__ == "__main__":
    main()
