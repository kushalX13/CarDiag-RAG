"""Hybrid candidate fusion + optional neural reranking."""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


def _minmax_normalize(scores: list[float]) -> list[float]:
    """Map a score list to [0,1] with safe degenerate handling."""
    if not scores:
        return []
    s_min = min(scores)
    s_max = max(scores)
    span = s_max - s_min
    if span <= 1e-9:
        return [0.0 for _ in scores]
    return [(s - s_min) / span for s in scores]


def build_hybrid_candidates(
    dense_results: list[tuple[dict, float]],
    keyword_results: list[tuple[dict, float]],
    *,
    alpha: float = 0.50,
    top_n: int = 50,
) -> list[dict]:
    """
    Build stage-1 candidates from dense + keyword retrieval.

    Returns a list of candidate dicts sorted by `hybrid_score` desc:
    {
      "doc": doc,
      "doc_id": ...,
      "campaign_number": ...,
      "dense_raw": ...,
      "keyword_raw": ...,
      "dense_score": ...,
      "keyword_score": ...,
      "hybrid_score": ...
    }
    """
    merged: dict[str, dict] = {}

    for doc, dense_raw in dense_results:
        doc_id = doc.get("doc_id", "").strip()
        if not doc_id:
            continue
        merged[doc_id] = {
            "doc": doc,
            "doc_id": doc_id,
            "campaign_number": doc.get("campaign_number", ""),
            "dense_raw": float(dense_raw),
            "keyword_raw": 0.0,
        }

    for doc, keyword_raw in keyword_results:
        doc_id = doc.get("doc_id", "").strip()
        if not doc_id:
            continue
        if doc_id not in merged:
            merged[doc_id] = {
                "doc": doc,
                "doc_id": doc_id,
                "campaign_number": doc.get("campaign_number", ""),
                "dense_raw": 0.0,
                "keyword_raw": float(keyword_raw),
            }
        else:
            merged[doc_id]["keyword_raw"] = float(keyword_raw)

    if not merged:
        return []

    items = list(merged.values())
    dense_norm = _minmax_normalize([x["dense_raw"] for x in items])
    keyword_norm = _minmax_normalize([x["keyword_raw"] for x in items])

    for i, item in enumerate(items):
        d = dense_norm[i]
        k = keyword_norm[i]
        item["dense_score"] = d
        item["keyword_score"] = k
        item["hybrid_score"] = (1 - alpha) * d + alpha * k

    items.sort(key=lambda x: x["hybrid_score"], reverse=True)
    if top_n > 0:
        return items[:top_n]
    return items


class NeuralReranker:
    """Thin wrapper around a sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        device: str | None = None,
        max_length: int = 256,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._model: CrossEncoder | None = None

    def _load(self) -> CrossEncoder:
        if self._model is None:
            logger.info("Loading neural reranker: %s", self.model_name)
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
            )
        return self._model

    def score(self, query: str, docs: list[str]) -> list[float]:
        """Score query-doc pairs. Higher means more relevant."""
        if not docs:
            return []
        model = self._load()
        pairs = [[query, d] for d in docs]
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]


def rerank_candidates(
    query: str,
    candidates: list[dict],
    *,
    use_rerank: bool,
    reranker: NeuralReranker | None,
    rerank_limit: int = 0,
    top_k: int | None = None,
) -> list[dict]:
    """
    Stage-2 reranking for stage-1 hybrid candidates.

    If rerank is disabled, this keeps stage-1 order and sets rerank_score=hybrid_score.
    """
    if not candidates:
        return []

    ranked = [dict(c) for c in candidates]
    for i, c in enumerate(ranked):
        c["stage1_rank"] = i + 1
        text = (c.get("doc") or {}).get("text", "") or ""
        c["rerank_input_char_len"] = len(text)

    if use_rerank and reranker is not None:
        limit = len(ranked)
        if rerank_limit > 0:
            limit = min(limit, rerank_limit)

        to_rerank = ranked[:limit]
        texts = [((c.get("doc") or {}).get("text", "") or "") for c in to_rerank]
        rerank_scores = reranker.score(query, texts)
        for i, score in enumerate(rerank_scores):
            to_rerank[i]["rerank_score"] = score
            to_rerank[i]["rerank_input_text"] = texts[i]

        for c in ranked[limit:]:
            c["rerank_score"] = c.get("hybrid_score", 0.0)
            c["rerank_input_text"] = ""

        ranked = sorted(to_rerank, key=lambda x: x.get("rerank_score", float("-inf")), reverse=True) + ranked[limit:]
    else:
        for c in ranked:
            c["rerank_score"] = c.get("hybrid_score", 0.0)
            c["rerank_input_text"] = ""

    for i, c in enumerate(ranked):
        c["post_rerank_rank"] = i + 1

    if top_k is not None and top_k > 0:
        return ranked[:top_k]
    return ranked


def rerank(
    results: list[tuple[dict, float]],
    query: str,
    alpha: float = 0.15,
    max_tokens: int = 12,
    max_phrases: int = 10,
    normalize_dense: bool = False,
) -> list[tuple[dict, float, float, float]]:
    """
    Backward-compatible adapter for older callers.

    This now delegates to stage-1 hybrid fusion and returns:
    [(doc, combined, dense_score_norm, keyword_score_norm), ...]
    """
    del query, max_tokens, max_phrases, normalize_dense
    candidates = build_hybrid_candidates(results, [], alpha=alpha, top_n=0)
    return [
        (
            c["doc"],
            c["hybrid_score"],
            c["dense_score"],
            c["keyword_score"],
        )
        for c in candidates
    ]
