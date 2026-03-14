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
        help="Hybrid weight: combined = (1-alpha)*dense + alpha*kw (default 0.50)",
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

    # Hybrid retrieval: dense + keyword
    results = search(
        search_query,
        make_norm,
        mkey,
        indexes,
        model,
        top_k=args.dense_topk,
        use_pool_indexes=not args.no_pool,
        min_pool_docs=50,
    )

    kw_results = keyword_search(
        query_text,
        make_norm,
        mkey,
        indexes,
        top_k=args.keyword_topk,
        use_pool_indexes=not args.no_pool,
        min_pool_docs=50,
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

    logger.info("Hybrid retrieval: %d candidates", len(results))

    if not results:
        logger.info("No results found.")
        answer = generate_rag_answer(query_text, [], top_k=args.topk)
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

    # Rerank with alpha=0.5 (hybrid)
    results_for_agg = rerank(
        results,
        query_text,
        alpha=args.alpha,
        max_tokens=args.rerank_tokens,
        max_phrases=args.rerank_phrases,
        normalize_dense=True,
    )

    campaigns = _aggregate_by_campaign_with_scores(
        results_for_agg,
        make_norm=make_norm if make_norm else None,
        debug=args.debug,
    )

    # Generate grounded RAG answer
    answer = generate_rag_answer(query_text, campaigns, top_k=args.topk)
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
