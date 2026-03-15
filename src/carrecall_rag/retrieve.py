"""FAISS + BM25 search, index build/load, campaign aggregation."""

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .utils import model_key as compute_model_key, normalize_make


def _bm25_tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    return re.findall(r"[a-z0-9]+", text)


METADATA_PREFIX_RE = re.compile(r"^\s*vehicle\s+description\s*:\s*", re.IGNORECASE)
LEADING_METADATA_LABEL_RE = re.compile(
    r"^\s*(passenger\s+vehicles|pickup\s+trucks|sport\s+utility\s+vehicles|"
    r"passenger\s+cars|mini\s+vans|multi[-\s]?purpose\s+passenger\s+vehicle[s]?|"
    r"multi[-\s]?purpose\s+vehicles?)\s*[\.:,-]?\s*",
    re.IGNORECASE,
)
QUERY_EXPANSIONS: list[tuple[re.Pattern[str], list[str]]] = [
    (
        re.compile(r"\borc\b", re.IGNORECASE),
        ["occupant restraint controller", "occupant restraint control module"],
    ),
    (
        re.compile(r"\bair\s*bag[s]?\b|\bairbag[s]?\b", re.IGNORECASE),
        ["supplemental restraint system", "occupant restraint system", "srs"],
    ),
    (
        re.compile(r"\bclock\s*spring\b", re.IGNORECASE),
        ["steering wheel clock spring", "steering column clock spring"],
    ),
]


def normalize_retrieval_query(
    query_text: str,
    *,
    apply_expansion: bool = True,
) -> str:
    """
    Normalize retrieval queries with lightweight domain cleanup.
    - lowercase + whitespace normalization
    - strips leading metadata labels ("VEHICLE DESCRIPTION: ...")
    - optional hand-written synonym expansion
    """
    raw = query_text or ""
    normalized = " ".join(raw.strip().lower().split())
    if not normalized:
        return ""

    normalized = METADATA_PREFIX_RE.sub("", normalized)
    normalized = LEADING_METADATA_LABEL_RE.sub("", normalized)
    normalized = normalized.strip(" .,:;-")
    if not normalized:
        normalized = " ".join(raw.strip().split()).lower()

    if not apply_expansion:
        return normalized

    expanded_phrases: list[str] = []
    existing = f" {normalized} "
    for pattern, phrases in QUERY_EXPANSIONS:
        if not pattern.search(normalized):
            continue
        for phrase in phrases:
            token = f" {phrase.lower()} "
            if token not in existing:
                expanded_phrases.append(phrase)
                existing += token

    if expanded_phrases:
        return f"{normalized} {' '.join(expanded_phrases)}".strip()
    return normalized

logger = logging.getLogger(__name__)


def load_corpus(path: str = "data/processed/corpus_merged.jsonl") -> list[dict]:
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
    """Build global + optional pool/make FAISS indexes; save .faiss + mapping JSON."""
    os.makedirs(cache_dir, exist_ok=True)
    model = SentenceTransformer(model_dir)

    # Encode all docs
    texts = [d.get("text", "") for d in corpus_docs]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    def _save_index(name: str, indices: list[int]) -> None:
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

    global_indices = list(range(len(corpus_docs)))
    _save_index("global", global_indices)

    if not use_pool_indexes:
        return

    pool_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, doc in enumerate(corpus_docs):
        make_norm = doc.get("make_norm") or ""
        model_key = doc.get("model_key") or ""
        pool_groups[(make_norm, model_key)].append(i)

    for (make_norm, mkey), indices in pool_groups.items():
        if len(indices) >= min_pool_docs:
            pool_key = f"pool_{make_norm}_{mkey}"
            _save_index(pool_key, indices)

    make_groups: dict[str, list[int]] = defaultdict(list)
    for i, doc in enumerate(corpus_docs):
        make_norm = doc.get("make_norm") or ""
        if make_norm:
            make_groups[make_norm].append(i)

    for make_norm, indices in make_groups.items():
        make_key = f"make_{make_norm}"
        _save_index(make_key, indices)


def load_faiss_indexes(cache_dir: str = "data/indexes/") -> dict:
    """Load .faiss indexes and _mapping.json from cache_dir. Keys: global, pools, makes."""
    cache_path = Path(cache_dir).resolve()

    indexes = {}
    mappings = {}

    if not cache_path.exists() or not cache_path.is_dir():
        logger.warning("Cache dir missing or not a directory: %s", cache_path)
        return {"global": None, "pools": {}, "makes": {}}

    faiss_files = list(cache_path.glob("*.faiss"))
    for faiss_path in faiss_files:
        key = faiss_path.stem  # e.g. "pool_FORD_f150", "global"
        map_path = cache_path / f"{key}_mapping.json"

        if not map_path.exists():
            logger.warning("Missing mapping for %s: %s", key, map_path)
            continue

        index = faiss.read_index(str(faiss_path))
        with open(map_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)

        indexes[key] = index
        mappings[key] = mapping_data

    result = {"global": None, "pools": {}, "makes": {}}
    for key in indexes:
        mapping_data = mappings[key]
        entry = {
            "index": indexes[key],
            "mapping": mapping_data.get("mapping", []),
            "docs": mapping_data.get("docs", []),
        }
        if key == "global":
            result["global"] = entry
        elif key.startswith("pool_"):
            result["pools"][key] = entry
        elif key.startswith("make_"):
            result["makes"][key] = entry

    return result


def select_index(
    make_norm: str,
    model_key: str,
    pools: dict,
    make_only: dict,
    global_entry: dict | None,
    min_pool_docs: int = 50,
) -> tuple[dict, str, str]:
    """Prefer pool → make → global when ntotal >= min_pool_docs. Returns (entry, index_name, index_type)."""
    pool_key = f"pool_{make_norm}_{model_key}"
    make_key = f"make_{make_norm}"

    if pool_key in pools:
        entry = pools[pool_key]
        ntotal = entry["index"].ntotal
        if ntotal >= min_pool_docs:
            return entry, pool_key, "pool"

    if make_key in make_only:
        entry = make_only[make_key]
        ntotal = entry["index"].ntotal
        if ntotal >= min_pool_docs:
            return entry, make_key, "make"

    if global_entry:
        return global_entry, "global", "global"

    return None, None, None


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
    """Dense search via select_index. Returns (doc, score) sorted by score desc."""
    make_norm = normalize_make(make_norm) if make_norm else ""
    model_key = compute_model_key(model_key) if model_key else ""

    if use_pool_indexes:
        entry, index_name, index_type = select_index(
            make_norm,
            model_key,
            indexes.get("pools", {}),
            indexes.get("makes", {}),
            indexes.get("global"),
            min_pool_docs=min_pool_docs,
        )
    else:
        entry = indexes.get("global")
        index_name = "global"
        index_type = "global"

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


def keyword_search(
    query_text: str,
    make_norm: str,
    model_key: str,
    indexes: dict,
    top_k: int = 100,
    use_pool_indexes: bool = True,
    min_pool_docs: int = 50,
) -> list[tuple[dict, float]]:
    """BM25 over same index as search. Returns [(doc, score), ...] desc."""
    make_norm = normalize_make(make_norm) if make_norm else ""
    model_key = compute_model_key(model_key) if model_key else ""

    if use_pool_indexes:
        entry, index_name, _ = select_index(
            make_norm,
            model_key,
            indexes.get("pools", {}),
            indexes.get("makes", {}),
            indexes.get("global"),
            min_pool_docs=min_pool_docs,
        )
    else:
        entry = indexes.get("global")

    if entry is None:
        return []

    docs = entry["docs"]
    if not docs:
        return []

    tokenized = [_bm25_tokenize(d.get("text", "")) for d in docs]
    bm25 = BM25Okapi(tokenized)
    query_tokens = _bm25_tokenize(query_text)
    scores = bm25.get_scores(query_tokens)
    indexed = list(zip(docs, scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [(d, float(s)) for d, s in indexed[:top_k]]


def aggregate_by_campaign(
    results: list[tuple[dict, float]],
    make_norm: str | None = None,
) -> list[dict]:
    """Group by campaign_number; score = sum + 0.2*max + 0.5*count. Filter by make_norm if set. Sorted desc."""
    if make_norm:
        results = [(doc, s) for doc, s in results if (doc.get("make_norm") or "") == make_norm]

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
