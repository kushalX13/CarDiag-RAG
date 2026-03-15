"""Hybrid candidate fusion + optional neural reranking."""

from __future__ import annotations

import logging
import re

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

IMPORTANT_ISSUE_TERMS = {
    "airbag", "airbags", "orc", "clock", "spring", "warning", "light",
    "disable", "deployment", "restraint", "steering", "module", "stall",
    "brake", "fire", "crash", "injury", "loss", "power",
}

BOILERPLATE_PATTERNS = [
    r"\bis recalling\b",
    r"\bowners may contact\b",
    r"\bnhtsa campaign\b",
    r"\bthis remedy is\b",
    r"\bdealers will\b",
    r"\bvehicles included in this recall\b",
    r"\bif your vehicle\b",
    r"\bfor additional information\b",
]


def _first_sentence(text: str, max_chars: int = 180) -> str:
    """Best-effort first sentence extraction."""
    text = (text or "").strip()
    if not text:
        return ""
    m = re.search(r"[.!?]\s", text)
    if m:
        return text[: m.start() + 1][:max_chars].strip()
    return text[:max_chars].strip()


def _extract_consequence(text: str, max_chars: int = 220) -> str:
    """Extract a likely consequence phrase from free text."""
    t = (text or "").strip()
    if not t:
        return ""
    patterns = [
        r"(?:may|can|could)\s+(?:disable|increase|cause|result in|lead to)[^.]{0,220}",
        r"(?:increasing|resulting in|leading to)[^.]{0,220}",
    ]
    tl = t.lower()
    for p in patterns:
        m = re.search(p, tl)
        if m:
            return t[m.start() : m.end()][:max_chars].strip(" ,.;")
    return ""


def _tokenize(text: str) -> list[str]:
    """Tokenize text to lowercase alphanumeric terms."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _is_boilerplate_sentence(sentence: str) -> bool:
    """Heuristic boilerplate detector for low-signal recall prose."""
    s = (sentence or "").strip().lower()
    if not s:
        return True
    if len(s) < 18:
        return True
    if re.search(r"\b(vin|my \d{4}-\d{4}|[a-z]{2,}-\d{2,})\b", s):
        return True
    for pat in BOILERPLATE_PATTERNS:
        if re.search(pat, s):
            return True
    return False


def _is_informative_title(title: str) -> bool:
    """Reject generic/non-informative titles."""
    t = " ".join((title or "").split()).strip()
    if len(t) < 12:
        return False
    tl = t.lower()
    generic = [
        "safety recall", "recall notice", "important safety information",
        "manufacturer recall", "customer advisory", "campaign",
    ]
    if any(g in tl for g in generic):
        return False
    if _is_boilerplate_sentence(t):
        return False
    return True


def _split_sentences(text: str) -> list[str]:
    """Split by sentence punctuation/newlines."""
    if not text:
        return []
    raw = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in raw if s and s.strip()]


def _sentence_relevance_score(sentence: str, query_terms: set[str]) -> float:
    """Score sentence by query overlap + issue vocab overlap."""
    toks = set(_tokenize(sentence))
    if not toks:
        return 0.0
    overlap = len(toks & query_terms)
    issue_overlap = len(toks & IMPORTANT_ISSUE_TERMS)
    starts_with_issue = 1.0 if any(sentence.lower().startswith(x) for x in ("the ", "a ", "an ")) else 0.0
    return (2.0 * overlap) + (1.5 * issue_overlap) + starts_with_issue


def _top_relevant_sentences(text: str, query: str, top_k: int = 3) -> tuple[list[str], int]:
    """Return top relevant non-boilerplate sentences and count of removed boilerplate."""
    q_terms = set(_tokenize(query)) | IMPORTANT_ISSUE_TERMS
    sents = _split_sentences(text)
    filtered = []
    removed = 0
    for s in sents:
        if _is_boilerplate_sentence(s):
            removed += 1
            continue
        score = _sentence_relevance_score(s, q_terms)
        if score > 0:
            filtered.append((s, score))
    filtered.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in filtered[:top_k]], removed


def _best_defect_text(doc: dict, text: str, query: str) -> tuple[str, str]:
    """Pick defect/summary from explicit fields or relevant sentence."""
    for key in ("defect", "summary", "description"):
        val = (doc.get(key) or "").strip()
        if val and not _is_boilerplate_sentence(val):
            return val[:220], key
    top_sents, _ = _top_relevant_sentences(text, query, top_k=1)
    if top_sents:
        return top_sents[0][:220], "relevant_sentence"
    return "", ""


def _chunk_selection_text(doc: dict) -> str:
    """Text used for chunk-level query relevance scoring."""
    parts = [
        doc.get("title", ""),
        doc.get("summary", ""),
        doc.get("subject", ""),
        doc.get("component", ""),
        doc.get("consequence", ""),
        doc.get("text", ""),
    ]
    return " ".join((p or "").strip() for p in parts if p).strip()


def _phrase_hits(query: str, text: str) -> list[str]:
    """Important phrase matches to bias chunk selection."""
    q = (query or "").lower()
    t = (text or "").lower()
    phrases = [
        "clock spring",
        "warning light",
        "air bag",
        "airbag",
        "orc",
        "cruise control",
        "disable air bags",
    ]
    hits = []
    for p in phrases:
        if p in q and p in t:
            hits.append(p)
    return hits


def _select_campaign_evidence_chunks(
    rows: list[dict],
    query: str,
    *,
    max_chunks: int = 3,
) -> dict[str, dict]:
    """
    Score all chunks within each campaign and select top query-relevant evidence.
    """
    by_campaign: dict[str, list[dict]] = {}
    q_terms = set(_tokenize(query))
    for row in rows:
        campaign = row.get("campaign_number", "") or ""
        if not campaign:
            continue
        by_campaign.setdefault(campaign, []).append(row)

    selected: dict[str, dict] = {}
    for campaign, campaign_rows in by_campaign.items():
        scored = []
        for row in campaign_rows:
            doc = row.get("doc") or {}
            text_for_score = _chunk_selection_text(doc)
            toks = set(_tokenize(text_for_score))
            overlap_terms = sorted(toks & q_terms)
            issue_terms = sorted(toks & IMPORTANT_ISSUE_TERMS)
            phrase_terms = _phrase_hits(query, text_for_score)
            base = float(row.get("hybrid_score", 0.0))
            score = (3.0 * len(overlap_terms)) + (2.0 * len(issue_terms)) + (4.0 * len(phrase_terms)) + (0.25 * base)
            scored.append({
                "row": row,
                "score": score,
                "overlap_terms": overlap_terms,
                "issue_terms": issue_terms,
                "phrase_hits": phrase_terms,
                "hybrid_score": base,
            })

        scored.sort(key=lambda x: (x["score"], x["hybrid_score"]), reverse=True)
        chosen = scored[: max(1, max_chunks)]
        if chosen and chosen[0]["score"] <= 0:
            chosen = scored[:1]

        selected[campaign] = {
            "chosen_rows": [x["row"] for x in chosen],
            "chosen_chunk_ids": [((x["row"].get("doc") or {}).get("doc_id", "")) for x in chosen],
            "selection_debug": [
                {
                    "doc_id": (x["row"].get("doc") or {}).get("doc_id", ""),
                    "score": x["score"],
                    "hybrid_score": x["hybrid_score"],
                    "overlap_terms": x["overlap_terms"],
                    "important_terms": x["issue_terms"],
                    "phrase_hits": x["phrase_hits"],
                }
                for x in chosen
            ],
        }
    return selected


def _build_rerank_text(
    doc: dict,
    *,
    query: str = "",
    text_format: str = "full",
    campaign_context: dict | None = None,
) -> tuple[str, dict]:
    """
    Build reranker input text and expose exact source fields used.

    text_format:
      - full: doc.text
      - compact: title + component + consequence
      - smart: campaign + informative fields + top relevant sentences
    """
    context_rows = (campaign_context or {}).get("chosen_rows") or []
    if text_format == "smart" and context_rows:
        context_text = "\n".join(_chunk_selection_text((r.get("doc") or {})) for r in context_rows)
        text = context_text.strip()
    else:
        text = (doc.get("text") or "").strip()
    component = (doc.get("component") or "").strip()
    consequence = (doc.get("consequence") or "").strip()
    campaign_number = (doc.get("campaign_number") or "").strip()
    title_raw = (doc.get("title") or doc.get("summary") or doc.get("subject") or "").strip()
    title = " ".join(title_raw.split()).strip()
    title_source = "field" if title else ""
    if not title:
        title = _first_sentence(text)
        title_source = "text_first_sentence" if title else ""
    if not _is_informative_title(title):
        title = ""
        title_source = ""

    if not consequence:
        consequence = _extract_consequence(text)
        consequence_source = "text_extract" if consequence else ""
    else:
        consequence_source = "field"

    defect_text, defect_source = _best_defect_text(doc, text, query)
    query_overlap_terms = sorted(set(_tokenize(query)) & set(_tokenize(text)))
    top_sents, removed_boilerplate = _top_relevant_sentences(text, query, top_k=3)

    fields = {
        "text_format": text_format,
        "campaign_number": campaign_number,
        "title": title,
        "title_source": title_source,
        "component": component,
        "defect_summary": defect_text,
        "defect_source": defect_source,
        "consequence": consequence,
        "consequence_source": consequence_source,
        "text_len": len(text),
        "query_overlap_terms": query_overlap_terms,
        "relevant_sentences": top_sents,
        "excluded_boilerplate_count": removed_boilerplate,
        "selected_fields": [],
        "selected_chunk_ids": (campaign_context or {}).get("chosen_chunk_ids", []),
        "selection_debug": (campaign_context or {}).get("selection_debug", []),
    }

    if text_format == "compact":
        parts = []
        if title:
            parts.append(f"Title: {title}")
            fields["selected_fields"].append("title")
        if component:
            parts.append(f"Component: {component}")
            fields["selected_fields"].append("component")
        if consequence:
            parts.append(f"Consequence: {consequence}")
            fields["selected_fields"].append("consequence")
        compact = " | ".join(parts).strip()
        if compact:
            return compact, fields
        return text, fields

    if text_format == "smart":
        if context_rows:
            # Use campaign-level evidence-selected chunks instead of arbitrary chunk text.
            context_doc0 = (context_rows[0].get("doc") or {})
            if not component:
                component = (context_doc0.get("component") or "").strip()
            if not title:
                context_title = (
                    context_doc0.get("title")
                    or context_doc0.get("summary")
                    or context_doc0.get("subject")
                    or ""
                )
                context_title = " ".join(str(context_title).split()).strip()
                if _is_informative_title(context_title):
                    title = context_title
                    title_source = "campaign_selected_chunk"
            if not consequence:
                consequence = _extract_consequence(text)
                consequence_source = "campaign_selected_chunk_extract" if consequence else consequence_source
            defect_text, defect_source = _best_defect_text(context_doc0, text, query)

        fields["title"] = title
        fields["title_source"] = title_source
        fields["component"] = component
        fields["defect_summary"] = defect_text
        fields["defect_source"] = defect_source
        fields["consequence"] = consequence
        fields["consequence_source"] = consequence_source

        lines: list[str] = []
        if campaign_number:
            lines.append(f"Campaign: {campaign_number}")
            fields["selected_fields"].append("campaign_number")
        if title:
            lines.append(f"Title: {title}")
            fields["selected_fields"].append("title")
        if component:
            lines.append(f"Component: {component}")
            fields["selected_fields"].append("component")
        if defect_text:
            lines.append(f"Defect: {defect_text}")
            fields["selected_fields"].append("defect_summary")
        if consequence:
            lines.append(f"Risk: {consequence}")
            fields["selected_fields"].append("consequence")
        for sent in top_sents:
            if len(lines) >= 5:
                break
            if sent not in " ".join(lines):
                lines.append(f"Evidence: {sent}")
                fields["selected_fields"].append("relevant_sentence")
        if len(lines) < 2 and text:
            lines.append(_first_sentence(text, max_chars=260))
            fields["selected_fields"].append("fallback_text")

        smart = "\n".join(lines[:5]).strip()
        if len(smart) > 900:
            smart = smart[:900].rsplit(" ", 1)[0]
        if smart:
            return smart, fields
        return text, fields

    return text, fields


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
    text_format: str = "full",
    top_k: int | None = None,
) -> list[dict]:
    """
    Stage-2 reranking for stage-1 hybrid candidates.

    If rerank is disabled, this keeps stage-1 order and sets rerank_score=hybrid_score.
    """
    if not candidates:
        return []

    campaign_contexts: dict[str, dict] = {}
    if text_format == "smart":
        campaign_contexts = _select_campaign_evidence_chunks(candidates, query, max_chunks=3)

    ranked = [dict(c) for c in candidates]
    for i, c in enumerate(ranked):
        c["stage1_rank"] = i + 1
        campaign = c.get("campaign_number", "") or ((c.get("doc") or {}).get("campaign_number", "") or "")
        campaign_ctx = campaign_contexts.get(campaign)
        text, fields = _build_rerank_text(
            c.get("doc") or {},
            query=query,
            text_format=text_format,
            campaign_context=campaign_ctx,
        )
        c["rerank_input_char_len"] = len(text)
        c["rerank_text_fields"] = fields
        c["rerank_input_text"] = text

    if use_rerank and reranker is not None:
        limit = len(ranked)
        if rerank_limit > 0:
            limit = min(limit, rerank_limit)

        to_rerank = ranked[:limit]
        texts = [c.get("rerank_input_text", "") for c in to_rerank]
        rerank_scores = reranker.score(query, texts)
        for i, score in enumerate(rerank_scores):
            to_rerank[i]["rerank_score"] = score

        for c in ranked[limit:]:
            c["rerank_score"] = c.get("hybrid_score", 0.0)

        ranked = sorted(to_rerank, key=lambda x: x.get("rerank_score", float("-inf")), reverse=True) + ranked[limit:]
    else:
        for c in ranked:
            c["rerank_score"] = c.get("hybrid_score", 0.0)

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
