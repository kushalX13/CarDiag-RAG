"""Retrieval module: FAISS indexes, search, and campaign aggregation."""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def load_corpus(path: str = "data/processed/corpus_merged.jsonl") -> list[dict]:
    """Load corpus from JSONL. Each doc has doc_id, campaign_number, make_norm, model_key, text, plus any fields."""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def build_faiss_indexes(
    model_dir: str,
    corpus_docs: list[dict],
    use_pool_indexes: bool = True,
    min_pool_docs: int = 50,
    cache_dir: str = "data/indexes/",
) -> None:
    """
    Build and save FAISS indexes:
    a) one global index (optional)
    b) per-(make_norm, model_key) pool indexes
    c) make-only fallback indexes

    Saves: faiss index file(s) and JSON mapping from index row -> doc_id (and campaign_number).
    Uses cosine similarity (L2-normalize embeddings, IndexFlatIP).
    """
    os.makedirs(cache_dir, exist_ok=True)
    model = SentenceTransformer(model_dir)

    # Encode all docs
    texts = [d.get("text", "") for d in corpus_docs]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    def _save_index(name: str, indices: list[int]) -> None:
        """Save FAISS index and mapping for given doc indices."""
        if not indices:
            return
        sub_emb = embeddings[indices]
        index = faiss.IndexFlatIP(sub_emb.shape[1])
        index.add(sub_emb)
        mapping = [corpus_docs[i] for i in indices]
        mapping_light = [
            {"doc_id": d.get("doc_id", ""), "campaign_number": d.get("campaign_number", "")}
            for d in mapping
        ]
        idx_path = os.path.join(cache_dir, f"{name}.faiss")
        map_path = os.path.join(cache_dir, f"{name}_mapping.json")
        faiss.write_index(index, idx_path)
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump({"mapping": mapping_light, "docs": mapping}, f, ensure_ascii=False)
        logger.info("Saved %s: %d docs", name, len(indices))

    # a) Global index
    global_indices = list(range(len(corpus_docs)))
    _save_index("global", global_indices)

    if not use_pool_indexes:
        return

    # b) Per (make_norm, model_key) pool indexes
    pool_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, doc in enumerate(corpus_docs):
        make_norm = doc.get("make_norm") or ""
        model_key = doc.get("model_key") or ""
        pool_groups[(make_norm, model_key)].append(i)

    for (make_norm, model_key), indices in pool_groups.items():
        if len(indices) >= min_pool_docs:
            safe_name = f"pool_{make_norm}_{model_key}".replace("/", "_").replace(" ", "_")
            _save_index(safe_name, indices)

    # c) Make-only fallback indexes
    make_groups: dict[str, list[int]] = defaultdict(list)
    for i, doc in enumerate(corpus_docs):
        make_norm = doc.get("make_norm") or ""
        if make_norm:
            make_groups[make_norm].append(i)

    for make_norm, indices in make_groups.items():
        safe_name = f"make_{make_norm}".replace("/", "_").replace(" ", "_")
        _save_index(safe_name, indices)


def load_faiss_indexes(cache_dir: str = "data/indexes/") -> dict:
    """
    Load FAISS indexes and mappings from cache_dir.
    Returns dict with keys: "global", "pools", "makes".
    Each index entry has: index (faiss.Index), mapping (list of {doc_id, campaign_number}), docs (full doc list).
    """
    result = {
        "global": None,
        "pools": {},  # (make_norm, model_key) -> {index, mapping, docs}
        "makes": {},  # make_norm -> {index, mapping, docs}
    }

    if not os.path.isdir(cache_dir):
        return result

    def _load_one(name: str) -> tuple | None:
        idx_path = os.path.join(cache_dir, f"{name}.faiss")
        map_path = os.path.join(cache_dir, f"{name}_mapping.json")
        if not os.path.exists(idx_path) or not os.path.exists(map_path):
            return None
        index = faiss.read_index(idx_path)
        with open(map_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mapping = data.get("mapping", [])
        docs = data.get("docs", [])
        return index, mapping, docs

    # Global
    g = _load_one("global")
    if g:
        result["global"] = {"index": g[0], "mapping": g[1], "docs": g[2]}

    # Pools and makes
    for fname in os.listdir(cache_dir):
        if not fname.endswith(".faiss"):
            continue
        name = fname[:-5]  # strip .faiss
        if name == "global":
            continue
        g = _load_one(name)
        if not g:
            continue
        if name.startswith("pool_"):
            parts = name[5:].rsplit("_", 1)  # pool_MAKE_modelkey
            if len(parts) == 2:
                make_norm, model_key = parts[0].replace("_", " "), parts[1]
                result["pools"][(make_norm, model_key)] = {
                    "index": g[0],
                    "mapping": g[1],
                    "docs": g[2],
                }
        elif name.startswith("make_"):
            make_norm = name[5:].replace("_", " ")
            result["makes"][make_norm] = {"index": g[0], "mapping": g[1], "docs": g[2]}

    return result


def search(
    query_text: str,
    make_norm: str,
    model_key: str,
    indexes: dict,
    model: SentenceTransformer,
    top_k: int = 30,
    use_pool_indexes: bool = True,
    min_pool_docs: int = 50,
) -> list[tuple[dict, float]]:
    """
    Search for relevant docs. Chooses index:
    - (make_norm, model_key) pool if exists and has >= min_pool_docs
    - else make-only fallback
    - else global

    Returns list of (doc, score) sorted by score desc.
    """
    pool_key = (make_norm, model_key)
    entry = None
    index_type = "global"

    if use_pool_indexes:
        pool_entry = indexes.get("pools", {}).get(pool_key)
        if pool_entry and len(pool_entry["docs"]) >= min_pool_docs:
            entry = pool_entry
            index_type = "pool"
        else:
            make_entry = indexes.get("makes", {}).get(make_norm)
            if make_entry:
                entry = make_entry
                index_type = "make"

    if entry is None:
        entry = indexes.get("global")
        if entry is None:
            return []

    faiss_index = entry["index"]
    docs = entry["docs"]

    q_emb = model.encode([query_text], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)
    scores, indices = faiss_index.search(q_emb, min(top_k, len(docs)))
    scores = scores[0]
    indices = indices[0]

    results = []
    for idx, score in zip(indices, scores):
        if idx < 0:
            continue
        if idx < len(docs):
            results.append((docs[idx], float(score)))
    return results


def aggregate_by_campaign(
    results: list[tuple[dict, float]],
) -> list[dict]:
    """
    Group by campaign_number and compute:
      campaign_score = sum(scores) + 0.2*max(scores) + 0.5*count

    For each campaign store:
      - campaign_number
      - campaign_score
      - best_doc (highest score)
      - evidence_snippets: top 2 docs (doc_id + first 280 chars)

    Sort campaigns by campaign_score desc.
    """
    by_campaign: dict[str, list[tuple[dict, float]]] = defaultdict(list)
    for doc, score in results:
        cn = doc.get("campaign_number") or ""
        if not cn.strip():
            continue
        by_campaign[cn].append((doc, score))

    campaign_results = []
    for cn, items in by_campaign.items():
        scores = [s for _, s in items]
        campaign_score = sum(scores) + 0.2 * max(scores) + 0.5 * len(items)
        best_doc, best_score = max(items, key=lambda x: x[1])
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        evidence_snippets = []
        for doc, _ in sorted_items[:2]:
            text = (doc.get("text") or "")[:280]
            evidence_snippets.append({"doc_id": doc.get("doc_id", ""), "snippet": text})
        campaign_results.append({
            "campaign_number": cn,
            "campaign_score": campaign_score,
            "best_doc": best_doc,
            "evidence_snippets": evidence_snippets,
        })

    campaign_results.sort(key=lambda x: x["campaign_score"], reverse=True)
    return campaign_results
