"""RAG answer generation: grounded explanations from retrieved recall documents.

Supports multiple backends:
- template: deterministic extraction from retrieved docs (no LLM)
- llm_local: future local model
- llm_api: future API-based model
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

# -----------------------------------------------------------------------------
# Prompt templates (for future LLM backends)
# -----------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are a vehicle recall assistant. Given a user query about a vehicle issue and retrieved NHTSA recall documents, produce a grounded explanation. Use ONLY information from the retrieved recalls. Do not hallucinate or add information not present in the context."""

RAG_USER_PROMPT_TEMPLATE = """User query: {query}

Retrieved recall documents (top {n}):
{context}

Produce a response with these sections:
1. Top Recall Candidates: numbered list of campaign_number — short title
2. Grounded Summary: 2-5 sentences based only on retrieved recalls
3. Safety Risk: brief risk summary
4. Suggested Next Step: brief action (e.g., inspect recall, dealer check, NHTSA lookup)"""


# -----------------------------------------------------------------------------
# Answer backend protocol (for swapping template / local LLM / API)
# -----------------------------------------------------------------------------


@runtime_checkable
class AnswerBackend(Protocol):
    """Protocol for answer generation backends."""

    def generate(self, query: str, retrieved_docs: list[dict], top_k: int) -> str:
        """Generate a grounded answer from query and retrieved docs."""
        ...


# -----------------------------------------------------------------------------
# Helpers: extract structured info from campaign docs
# -----------------------------------------------------------------------------


def _extract_short_title(campaign: dict, max_len: int = 80) -> str:
    """Extract a short title from campaign best_doc or evidence."""
    best = campaign.get("best_doc") or {}
    component = (best.get("component") or "").strip()
    text = (best.get("text") or "").strip()

    if component:
        return (component[:max_len] + "..." if len(component) > max_len else component)
    if text:
        # Use first sentence or first max_len chars
        first_sent = text.split(".")[0].strip()
        if first_sent:
            return (first_sent[:max_len] + "..." if len(first_sent) > max_len else first_sent)

    ev = (campaign.get("evidence_snippets") or [{}])[0]
    snippet = (ev.get("snippet") or "").strip()
    if snippet:
        return (snippet[:max_len] + "..." if len(snippet) > max_len else snippet)
    return "Recall campaign"


def _build_context_from_campaigns(campaigns: list[dict], top_k: int) -> str:
    """Build context string from top campaigns for LLM prompt."""
    lines = []
    for i, c in enumerate(campaigns[:top_k], 1):
        cn = c.get("campaign_number", "")
        title = _extract_short_title(c)
        evs = c.get("evidence_snippets") or []
        snippets = [e.get("snippet", "") for e in evs[:2] if e.get("snippet")]
        body = " | ".join(s[:200] for s in snippets) if snippets else ""
        lines.append(f"{i}. Campaign {cn}: {title}\n   {body}")
    return "\n\n".join(lines)


def _extract_safety_keywords(text: str) -> list[str]:
    """Extract safety-related phrases from recall text."""
    keywords = []
    lower = (text or "").lower()
    patterns = [
        r"loss of (?:power|control|steering|braking)",
        r"engine (?:stall|stalls|failure)",
        r"fire\b",
        r"crash|collision",
        r"injury|injuries",
        r"death|fatal",
        r"air ?bag",
        r"brake (?:failure|loss)",
        r"fuel (?:leak|starvation)",
        r"rollaway",
        r"may (?:result in|cause|lead to) ([^.]{5,60})",
    ]
    for p in patterns:
        for m in re.finditer(p, lower, re.IGNORECASE):
            keywords.append(m.group(0).strip())
    return keywords[:5]  # Limit


def _first_substantive_sentence(text: str, min_len: int = 30) -> str | None:
    """Extract first sentence, avoiding false splits on '3.0 L' or 'No. 1'."""
    if not text or len(text.strip()) < min_len:
        return None
    # Split on sentence boundaries: ". " followed by capital (avoids "3.0" engine sizes)
    parts = re.split(r"\.\s+(?=[A-Z])", text.strip())
    for p in parts:
        p = p.strip()
        if p and len(p) >= min_len:
            return p
    # Fallback: first segment before ". " (period-space)
    first = re.split(r"\.\s+", text.strip(), maxsplit=1)[0].strip()
    return first if len(first) >= min_len else None


def _build_grounded_summary(campaigns: list[dict], top_k: int) -> str:
    """Build 2-5 sentence summary from retrieved evidence (template-based)."""
    sentences = []
    seen_campaigns: set[str] = set()

    for c in campaigns[:top_k]:
        cn = c.get("campaign_number", "")
        if cn in seen_campaigns:
            continue
        seen_campaigns.add(cn)

        evs = c.get("evidence_snippets") or []
        best = c.get("best_doc") or {}
        texts = [best.get("text", "")] + [e.get("snippet", "") for e in evs]
        combined = " ".join(t for t in texts if t).strip()

        if not combined:
            continue

        first = _first_substantive_sentence(combined)
        if first:
            sentences.append(f"Campaign {cn} addresses: {first}.")
        if len(sentences) >= 4:
            break

    if not sentences:
        return "No detailed recall information was retrieved for this query."
    return " ".join(sentences)


def _build_safety_risk(campaigns: list[dict], top_k: int) -> str:
    """Extract safety risk summary from retrieved recalls."""
    all_text = []
    for c in campaigns[:top_k]:
        evs = c.get("evidence_snippets") or []
        best = c.get("best_doc") or {}
        all_text.append(best.get("text", ""))
        for e in evs:
            all_text.append(e.get("snippet", ""))
    combined = " ".join(t for t in all_text if t)
    keywords = _extract_safety_keywords(combined)

    if keywords:
        return "Potential risks may include: " + "; ".join(keywords[:3]) + "."
    return "Review the recalled component details above for specific safety implications."


def _suggest_next_step(campaigns: list[dict]) -> str:
    """Suggest next step based on retrieved recalls."""
    if not campaigns:
        return "No recalls found. Consider checking NHTSA.gov for your vehicle's VIN."
    return "Contact your dealer to verify if your vehicle is affected, or look up your VIN at NHTSA.gov/recalls."


# -----------------------------------------------------------------------------
# Template backend (deterministic, no LLM)
# -----------------------------------------------------------------------------


class TemplateAnswerBackend:
    """Deterministic template-based answer generation. No LLM required."""

    def generate(self, query: str, retrieved_docs: list[dict], top_k: int = 3) -> str:
        """Generate structured answer from retrieved campaign docs."""
        top = retrieved_docs[:top_k]
        if not top:
            return _format_output(
                query=query,
                candidates=[],
                summary="No recall candidates were retrieved for this query.",
                safety_risk="Unable to assess risk without retrieved recalls.",
                next_step="Try broadening your search or check NHTSA.gov directly.",
            )

        candidates = [
            (c.get("campaign_number", ""), _extract_short_title(c))
            for c in top
        ]
        summary = _build_grounded_summary(top, top_k)
        safety_risk = _build_safety_risk(top, top_k)
        next_step = _suggest_next_step(top)

        return _format_output(
            query=query,
            candidates=candidates,
            summary=summary,
            safety_risk=safety_risk,
            next_step=next_step,
        )


def _format_output(
    query: str,
    candidates: list[tuple[str, str]],
    summary: str,
    safety_risk: str,
    next_step: str,
) -> str:
    """Format the final RAG output."""
    lines = [
        "Possible Recall-Related Issue",
        "",
        "Query:",
        query,
        "",
        "Top Recall Candidates:",
    ]
    for i, (cn, title) in enumerate(candidates, 1):
        lines.append(f"  {i}. {cn} — {title}")
    lines.extend([
        "",
        "Grounded Summary:",
        summary,
        "",
        "Safety Risk:",
        safety_risk,
        "",
        "Suggested Next Step:",
        next_step,
        "",
    ])
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

# Default backend: template (works without LLM)
_DEFAULT_BACKEND: AnswerBackend = TemplateAnswerBackend()


def generate_rag_answer(
    query: str,
    retrieved_docs: list[dict],
    top_k: int = 3,
    backend: AnswerBackend | None = None,
) -> str:
    """
    Generate a grounded RAG answer from the user query and retrieved recall docs.

    Args:
        query: User's vehicle issue query.
        retrieved_docs: List of campaign dicts from retrieval (each with
            campaign_number, best_doc, evidence_snippets).
        top_k: Number of top recalls to use as context (default 3).
        backend: Answer backend (template, llm_local, llm_api). Default: template.

    Returns:
        Formatted string with candidates, grounded summary, safety risk, next step.
    """
    backend = backend or _DEFAULT_BACKEND
    return backend.generate(query, retrieved_docs, top_k)


def set_default_backend(backend: AnswerBackend) -> None:
    """Set the default answer backend (for swapping template/LLM)."""
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = backend
