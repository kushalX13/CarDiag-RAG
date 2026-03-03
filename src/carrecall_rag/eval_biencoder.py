"""CLI: Evaluate bi-encoder on val_triples.jsonl (in-batch and full corpus retrieval)."""

import argparse
import json
import logging
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import DATA_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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
) -> dict:
    """Encode full corpus, build FAISS index, retrieve top-k per query, check if pos in top-k."""
    try:
        import faiss
    except ImportError:
        logger.warning("faiss-cpu not installed; skipping full corpus eval")
        return {}

    # Build doc_id -> text mapping
    doc_id_to_text = {d["doc_id"]: d.get("text", "") for d in corpus if d.get("doc_id")}
    doc_ids = list(doc_id_to_text.keys())
    texts = [doc_id_to_text[did] for did in doc_ids]

    logger.info("Encoding %d corpus passages...", len(texts))
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, device=device)
    embs = np.ascontiguousarray(embs.astype(np.float32))

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product = dot product for normalized vectors
    faiss.normalize_L2(embs)  # normalize for cosine ~ dot product
    index.add(embs)

    max_k = max(k_values)
    recalls = {k: [] for k in k_values}

    for t in tqdm(val_triples, desc="Full corpus eval"):
        query = t.get("query_text", "")
        pos = t.get("pos", {})
        pos_doc_id = pos.get("doc_id", "") if isinstance(pos, dict) else ""

        if not query or not pos_doc_id:
            continue

        q_emb = model.encode([query], convert_to_numpy=True, device=device).astype(np.float32)
        faiss.normalize_L2(q_emb)
        distances, indices = index.search(q_emb, max_k)
        top_doc_ids = [doc_ids[i] for i in indices[0]]

        for k in k_values:
            hit = pos_doc_id in top_doc_ids[:k]
            recalls[k].append(hit)

    metrics = {}
    for k in k_values:
        r = np.mean(recalls[k]) if recalls[k] else 0.0
        metrics[f"recall_full_corpus@{k}"] = float(r)
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
            full = eval_full_corpus(model, val_triples, corpus, args.k, device=device)
            metrics.update(full)
            for k, v in full.items():
                logger.info("%s: %.4f", k, v)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", args.output)


if __name__ == "__main__":
    main()
