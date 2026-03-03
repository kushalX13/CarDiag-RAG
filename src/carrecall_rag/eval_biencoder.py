"""CLI: Evaluate bi-encoder on val_triples.jsonl (in-batch and full corpus retrieval)."""

import argparse
import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import DATA_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MIN_POOL_SIZE = 50


def get_device() -> str:
    """Return best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

BIENCODER_MODEL_DIR = os.path.join(DATA_DIR, "models", "biencoder")
VAL_TRIPLES_PATH = os.path.join(PROCESSED_DIR, "val_triples.jsonl")
CORPUS_MERGED_PATH = os.path.join(PROCESSED_DIR, "corpus_merged.jsonl")
METRICS_PATH = os.path.join(PROCESSED_DIR, "biencoder_metrics.json")
METRICS_FULL_CORPUS_PATH = os.path.join(PROCESSED_DIR, "biencoder_metrics_full_corpus.json")


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_recall_at_k(ranked_doc_ids: list[str], pos_doc_id: str, k_values: list[int]) -> dict[int, bool]:
    """Check if pos_doc_id is in top-k of ranked_doc_ids. Returns {k: hit}."""
    result = {}
    for k in k_values:
        top_k = ranked_doc_ids[:k]
        result[k] = pos_doc_id in top_k
    return result


def eval_in_batch(
    model: SentenceTransformer,
    val_triples: list[dict],
    k_values: list[int],
    device: str = "cpu",
) -> dict:
    """Evaluate on val_triples: rank (pos + negs) by dot product, report Recall@k."""
    recalls = {k: [] for k in k_values}
    for t in tqdm(val_triples, desc="In-batch eval"):
        query = t.get("query_text", "")
        pos = t.get("pos", {})
        negs = t.get("negs", [])
        pos_doc_id = pos.get("doc_id", "") if isinstance(pos, dict) else ""
        passages = [pos.get("text", "") if isinstance(pos, dict) else ""]
        doc_ids = [pos_doc_id]
        for n in negs:
            passages.append(n.get("text", "") if isinstance(n, dict) else "")
            doc_ids.append(n.get("doc_id", "") if isinstance(n, dict) else "")

        if not query or not passages:
            continue

        q_emb = model.encode([query], convert_to_numpy=True, device=device)
        p_embs = model.encode(passages, convert_to_numpy=True, device=device)
        scores = np.dot(q_emb, p_embs.T).squeeze(0)
        ranked_idx = np.argsort(-scores)
        ranked_doc_ids = [doc_ids[i] for i in ranked_idx]

        for k in k_values:
            hit = pos_doc_id in ranked_doc_ids[:k]
            recalls[k].append(hit)

    metrics = {}
    for k in k_values:
        r = np.mean(recalls[k]) if recalls[k] else 0.0
        metrics[f"recall_in_batch@{k}"] = float(r)
    return metrics


def eval_full_corpus(
    model: SentenceTransformer,
    val_triples: list[dict],
    corpus: list[dict],
    k_values: list[int],
    device: str = "cpu",
    batch_size: int = 128,
) -> dict:
    """Encode full corpus, build FAISS index, retrieve top-k per query, check if pos in top-k."""
    try:
        import faiss
    except ImportError:
        logger.warning("faiss-cpu not installed; skipping full corpus eval")
        return {}

    # Extract query_id, query_text, pos.doc_id from val_triples
    queries_data = []
    for t in val_triples:
        query_id = t.get("query_id", "")
        query_text = t.get("query_text", "")
        pos = t.get("pos", {})
        pos_doc_id = pos.get("doc_id", "") if isinstance(pos, dict) else ""
        if query_text and pos_doc_id:
            queries_data.append({"query_id": query_id, "query_text": query_text, "pos_doc_id": pos_doc_id})

    # Build doc_ids and passage_text aligned arrays from corpus
    doc_ids = []
    passage_texts = []
    for d in corpus:
        if d.get("doc_id") is not None:
            doc_ids.append(d["doc_id"])
            passage_texts.append(d.get("text", ""))

    logger.info("Encoding %d corpus passages in batches of %d...", len(passage_texts), batch_size)
    passage_embs = model.encode(
        passage_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        device=device,
        batch_size=batch_size,
    )
    passage_embs = np.ascontiguousarray(passage_embs.astype(np.float32))

    # IndexFlatIP + L2 normalize = cosine similarity
    d = passage_embs.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(passage_embs)
    index.add(passage_embs)

    max_k = max(k_values)
    recalls = {k: [] for k in k_values}

    # Encode all queries in batches
    query_texts = [q["query_text"] for q in queries_data]
    logger.info("Encoding %d val queries in batches of %d...", len(query_texts), batch_size)
    query_embs = model.encode(
        query_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        device=device,
        batch_size=batch_size,
    )
    query_embs = np.ascontiguousarray(query_embs.astype(np.float32))
    faiss.normalize_L2(query_embs)

    # Search for all queries
    distances, indices = index.search(query_embs, max_k)

    for i, q in enumerate(queries_data):
        pos_doc_id = q["pos_doc_id"]
        top_doc_ids = [doc_ids[idx] for idx in indices[i] if idx >= 0]
        for k in k_values:
            hit = pos_doc_id in top_doc_ids[:k]
            recalls[k].append(hit)

    metrics = {}
    for k in k_values:
        r = np.mean(recalls[k]) if recalls[k] else 0.0
        metrics[f"recall_full_corpus@{k}"] = float(r)
    return metrics


def eval_pool_filtered(
    model: SentenceTransformer,
    val_triples: list[dict],
    corpus: list[dict],
    k_values: list[int],
    device: str = "cpu",
    batch_size: int = 128,
) -> dict:
    """Pool-filtered eval: group corpus by (make_norm, model_key), build FAISS per pool, search per query."""
    try:
        import faiss
    except ImportError:
        logger.warning("faiss-cpu not installed; skipping pool-filtered eval")
        return {}

    # Group corpus by (make_norm, model_key)
    pools_make_model: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    pools_make: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for d in corpus:
        doc_id = d.get("doc_id")
        if doc_id is None:
            continue
        make_norm = d.get("make_norm") or ""
        model_key = d.get("model_key") or ""
        text = d.get("text", "")
        pools_make_model[(make_norm, model_key)].append((doc_id, text))
        pools_make[make_norm].append((doc_id, text))

    # Build FAISS index per pool
    pool_indexes: dict[tuple[str, str] | tuple[str], tuple] = {}

    def _build_index(doc_ids: list[str], texts: list[str]) -> tuple:
        if not texts:
            return None
        embs = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=device,
            batch_size=batch_size,
        )
        embs = np.ascontiguousarray(embs.astype(np.float32))
        faiss.normalize_L2(embs)
        d = embs.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embs)
        return (index, doc_ids)

    logger.info("Building FAISS indexes for %d (make,model) pools and make-only fallbacks...", len(pools_make_model))
    # Build indexes for (make, model) pools
    for (make, model_key), items in pools_make_model.items():
        if items:
            doc_ids, texts = zip(*items)
            pool_indexes[(make, model_key)] = _build_index(list(doc_ids), list(texts))
    # Build indexes for make-only pools (for fallback when make+model pool < 50)
    for make, items in pools_make.items():
        key = (make,)
        if key not in pool_indexes and items:
            doc_ids, texts = zip(*items)
            pool_indexes[key] = _build_index(list(doc_ids), list(texts))

    # Extract queries with make_norm, model_key, query_text, pos_doc_id
    queries_data = []
    for t in val_triples:
        query_text = t.get("query_text", "")
        make_norm = t.get("make_norm") or ""
        model_key = t.get("model_key") or ""
        pos = t.get("pos", {})
        pos_doc_id = pos.get("doc_id", "") if isinstance(pos, dict) else ""
        if query_text and pos_doc_id:
            queries_data.append({
                "query_text": query_text,
                "make_norm": make_norm,
                "model_key": model_key,
                "pos_doc_id": pos_doc_id,
            })

    max_k = max(k_values)
    recalls = {k: [] for k in k_values}

    # Encode all queries
    query_texts = [q["query_text"] for q in queries_data]
    query_embs = model.encode(
        query_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        device=device,
        batch_size=batch_size,
    )
    query_embs = np.ascontiguousarray(query_embs.astype(np.float32))
    faiss.normalize_L2(query_embs)

    for i, q in enumerate(queries_data):
        make_norm = q["make_norm"]
        model_key = q["model_key"]
        pos_doc_id = q["pos_doc_id"]
        pool_key = (make_norm, model_key)
        # If pool size < 50, fallback to make_norm-only pool
        if pool_key in pools_make_model and len(pools_make_model[pool_key]) >= MIN_POOL_SIZE:
            index_key = pool_key
        else:
            index_key = (make_norm,)
        if index_key not in pool_indexes or pool_indexes[index_key] is None:
            for k in k_values:
                recalls[k].append(False)
            continue
        index, doc_ids = pool_indexes[index_key]
        q_emb = query_embs[i : i + 1]
        _, indices = index.search(q_emb, min(max_k, len(doc_ids)))
        top_doc_ids = [doc_ids[idx] for idx in indices[0] if idx >= 0]
        for k in k_values:
            hit = pos_doc_id in top_doc_ids[:k]
            recalls[k].append(hit)

    metrics = {}
    for k in k_values:
        r = np.mean(recalls[k]) if recalls[k] else 0.0
        metrics[f"recall_pool_filtered@{k}"] = float(r)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate bi-encoder retriever")
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Recall@k values to compute",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=BIENCODER_MODEL_DIR,
        help="Path to saved biencoder model",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=VAL_TRIPLES_PATH,
        help="Path to val_triples.jsonl",
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        default=CORPUS_MERGED_PATH,
        help="Path to corpus_merged.jsonl for full corpus eval",
    )
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Run full corpus retrieval eval (FAISS)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=METRICS_PATH,
        help="Path to save metrics JSON",
    )
    parser.add_argument(
        "--max-val-triples",
        type=int,
        default=None,
        help="If set, only evaluate on the first N val triples",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for encoding (full corpus eval)",
    )
    parser.add_argument(
        "--no-pool-filter",
        action="store_true",
        help="Disable pool filtering (default: pool-filter enabled when --full-corpus)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        logger.error("Model not found at %s. Run train_biencoder first.", args.model_dir)
        return

    device = get_device()
    logger.info("Using device: %s", device)

    model = SentenceTransformer(args.model_dir)
    val_triples = load_jsonl(args.val_path)
    if args.max_val_triples is not None:
        val_triples = val_triples[: args.max_val_triples]
        logger.info("Limited to first %d val triples", len(val_triples))
    else:
        logger.info("Loaded %d val triples", len(val_triples))

    metrics = {}

    # In-batch eval (always)
    in_batch = eval_in_batch(model, val_triples, args.k, device=device)
    metrics.update(in_batch)
    for k, v in in_batch.items():
        logger.info("%s: %.4f", k, v)

    # Full corpus eval (optional)
    if args.full_corpus:
        if not os.path.exists(args.corpus_path):
            logger.warning("Corpus not found at %s; skipping full corpus eval", args.corpus_path)
        else:
            corpus = load_jsonl(args.corpus_path)
            full = eval_full_corpus(
                model, val_triples, corpus, args.k, device=device, batch_size=args.batch_size
            )
            metrics.update(full)
            for k, v in full.items():
                logger.info("%s: %.4f", k, v)

            # Pool-filtered eval (default when --full-corpus, disable with --no-pool-filter)
            pool_filter = not args.no_pool_filter
            if pool_filter:
                pool_metrics = eval_pool_filtered(
                    model, val_triples, corpus, args.k, device=device, batch_size=args.batch_size
                )
                metrics.update(pool_metrics)
                for k, v in pool_metrics.items():
                    logger.info("%s: %.4f", k, v)

            # Save full corpus metrics to dedicated file
            os.makedirs(os.path.dirname(METRICS_FULL_CORPUS_PATH) or ".", exist_ok=True)
            full_metrics = {
                k: v
                for k, v in metrics.items()
                if k.startswith("recall_full_corpus") or k.startswith("recall_pool_filtered")
            }
            with open(METRICS_FULL_CORPUS_PATH, "w", encoding="utf-8") as f:
                json.dump(full_metrics, f, indent=2)
            logger.info("Saved full corpus metrics to %s", METRICS_FULL_CORPUS_PATH)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", args.output)


if __name__ == "__main__":
    main()
