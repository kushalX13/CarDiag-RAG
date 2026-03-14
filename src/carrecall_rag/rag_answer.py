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

# Boilerplate patterns to strip (manufacturer recall preamble)
_BOILERPLATE_PATTERNS = [
    r"^[A-Za-z\s\-]+(?:\([^)]+\))?\s+is recalling certain\s+",
    r"^[A-Za-z\s\-]+(?:\([^)]+\))?\s+is recalling\s+certain\s+",
    r"\d{4}-\d{4}\s+[A-Za-z\s\-]+(?:,\s*(?:and\s+)?\d{4}-\d{4}\s+[A-Za-z\s\-]+)*\s+vehicles?\s+(?:equipped with\s+)?",
    r"\d{4}-\d{4}\s+[A-Za-z\s\-]+\s+vehicles?\s+(?:equipped with\s+)?",
    r"certain\s+\d{4}-\d{4}\s+[^.]{10,120}?(?:vehicles?|models?)\s+(?:equipped with\s+)?",
]


def _strip_boilerplate(text: str) -> str:
    """Remove manufacturer preamble and vehicle-list-heavy phrasing."""
    if not text or not text.strip():
        return ""
    t = text.strip()
    for pat in _BOILERPLATE_PATTERNS:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _truncate(text: str, max_len: int = 120) -> str:
    """Truncate to max_len, break at word boundary."""
    if not text or len(text) <= max_len:
        return (text or "").strip()
    cut = text[:max_len].rsplit(" ", 1)[0]
    return cut + "..." if len(cut) < len(text) else cut


def _extract_issue_and_consequence(text: str) -> str:
    """Extract what failed and what can happen, stripping boilerplate."""
    t = _strip_boilerplate(text)
    if not t:
        return ""

    # Defect/component phrases (what failed)
    defect_patterns = [
        r"(?:the |a )?(?:high pressure fuel pump|fuel pump|HPFP)[^.]{0,90}",
        r"(?:the |a )?(?:brake master cylinder)[^.]{0,80}",
        r"(?:failure of|defect in|malfunction of)\s+([^.]{10,70})",
        r"(?:the |a )?([\w\s]+(?:pump|sensor|module|latch|actuator|cable)[^.]{0,50})",
        r"([\w\s]+(?:may fail|may introduce|may leak|can fail)[^.]{0,80})",
    ]
    for pat in defect_patterns:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            phrase = m.group(1).strip() if m.lastindex and m.lastindex >= 1 else m.group(0).strip()
            if len(phrase) > 15:
                phrase = re.sub(r"^\s*(?:the |a )\s*", "", phrase, flags=re.IGNORECASE)
                return _truncate(phrase, 100)

    # Consequence phrases (what can happen)
    consequence_patterns = [
        r"(?:may|could|can)\s+(?:result in|cause|lead to)\s+([^.]{10,80})",
        r"(?:resulting in|which may cause)\s+([^.]{10,80})",
        r"(?:engine stall|loss of power|fuel starvation|fire|crash|injury|rollaway)[^.]*",
    ]
    for pat in consequence_patterns:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            phrase = m.group(1).strip() if m.lastindex else m.group(0).strip()
            return _truncate(phrase, 100)

    # Fallback: first substantive clause after boilerplate
    first = re.split(r"\.\s+(?=[A-Z])", t)[0] if t else ""
    return _truncate(first, 100)


def _extract_short_title(campaign: dict, max_len: int = 60) -> str:
    """Extract a short user-facing title from campaign (component or issue)."""
    best = campaign.get("best_doc") or {}
    component = (best.get("component") or "").strip()

    if component:
        return _truncate(component, max_len)

    evs = campaign.get("evidence_snippets") or []
    texts = [best.get("text", "")] + [e.get("snippet", "") for e in evs[:2]]
    combined = " ".join(t for t in texts if t).strip()
    issue = _extract_issue_and_consequence(combined)
    if issue:
        return _truncate(issue, max_len)

    # Last resort: first few words
    first = re.split(r"\.\s+", combined)[0] if combined else ""
    stripped = _strip_boilerplate(first)
    return _truncate(stripped or first, max_len) if stripped or first else "Recall campaign"


def _extract_short_reason(campaign: dict, max_len: int = 50) -> str:
    """One short phrase: why this recall is relevant (for Other Candidates)."""
    title = _extract_short_title(campaign, max_len)
    if title and title != "Recall campaign":
        return title
    best = campaign.get("best_doc") or {}
    evs = campaign.get("evidence_snippets") or [{}]
    snippet = (evs[0].get("snippet", "") if evs else "") or best.get("text", "")
    return _truncate(_extract_issue_and_consequence(snippet), max_len) or "related recall"


def _extract_safety_phrases(text: str) -> list[str]:
    """Extract safety-related phrases (for deduplication)."""
    phrases = []
    lower = (text or "").lower()
    patterns = [
        (r"loss of (?:power|control|steering|braking|propulsion)", "loss of power/control"),
        (r"engine (?:stall|stalls|failure)", "engine stall"),
        (r"fuel (?:starvation|leak)", "fuel starvation or leak"),
        (r"\bfire\b", "fire"),
        (r"crash|collision", "crash"),
        (r"injury|injuries", "injury"),
        (r"death|fatal", "death or serious injury"),
        (r"air ?bag", "airbag malfunction"),
        (r"brake (?:failure|loss|fluid leak|function)", "brake failure or fluid leak"),
        (r"rollaway", "vehicle rollaway"),
    ]
    for pat, norm in patterns:
        if re.search(pat, lower):
            phrases.append(norm)
    return phrases


def _deduplicate_risks(phrases: list[str]) -> list[str]:
    """Deduplicate and normalize risk phrases."""
    seen: set[str] = set()
    out = []
    for p in phrases:
        key = p.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out


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


def _why_it_matches(query: str, campaign: dict) -> str:
    """
    Build 2-3 sentence plain-English explanation for the best match.
    Covers: what failed, what can happen, why it matches the query.
    """
    best = campaign.get("best_doc") or {}
    evs = campaign.get("evidence_snippets") or []
    texts = [best.get("text", "")] + [e.get("snippet", "") for e in evs]
    combined = " ".join(t for t in texts if t).strip()
    if not combined:
        return "No detailed information available for this recall."

    issue = _extract_issue_and_consequence(combined)
    if not issue:
        issue = _extract_short_title(campaign, 80)

    # Query keywords that appear in text (why it matches)
    query_words = set(re.findall(r"[a-z0-9]{3,}", (query or "").lower()))
    stop = {"the", "and", "for", "may", "can", "with", "into", "from"}
    text_lower = combined.lower()
    matching = [w for w in query_words if w in text_lower and w not in stop]
    match_phrase = f" Your query mentions {', '.join(sorted(matching)[:5])}, which appear in this recall." if matching else ""

    return f"This recall involves {issue}. It matches your description because the defect and consequences align with what you reported.{match_phrase}"


def _build_safety_risk_deduplicated(campaigns: list[dict], top_k: int) -> str:
    """Extract and deduplicate safety risks from retrieved recalls."""
    all_phrases: list[str] = []
    for c in campaigns[:top_k]:
        evs = c.get("evidence_snippets") or []
        best = c.get("best_doc") or {}
        texts = [best.get("text", "")] + [e.get("snippet", "") for e in evs]
        for t in texts:
            all_phrases.extend(_extract_safety_phrases(t))

    unique = _deduplicate_risks(all_phrases)
    if unique:
        if len(unique) == 1:
            return "Potential risks include " + unique[0] + "."
        return "Potential risks include " + ", ".join(unique[:-1]) + ", and " + unique[-1] + "."
    return "Review the recalled component details for specific safety implications."


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
                best_match=None,
                why_it_matches="No recall candidates were retrieved for this query.",
                other_candidates=[],
                safety_risk="Unable to assess risk without retrieved recalls.",
                next_step="Try broadening your search or check NHTSA.gov directly.",
            )

        best = top[0]
        best_cn = best.get("campaign_number", "")
        best_title = _extract_short_title(best)
        why = _why_it_matches(query, best)
        other = [
            (c.get("campaign_number", ""), _extract_short_reason(c))
            for c in top[1:]
        ]
        safety_risk = _build_safety_risk_deduplicated(top, top_k)
        next_step = _suggest_next_step(top)

        return _format_output(
            query=query,
            best_match=(best_cn, best_title),
            why_it_matches=why,
            other_candidates=other,
            safety_risk=safety_risk,
            next_step=next_step,
        )


def _format_output(
    query: str,
    best_match: tuple[str, str] | None,
    why_it_matches: str,
    other_candidates: list[tuple[str, str]],
    safety_risk: str,
    next_step: str,
) -> str:
    """Format the final RAG output (user-facing)."""
    lines = [
        "Possible Recall-Related Issue",
        "",
        "Query:",
        query,
        "",
    ]
    if best_match:
        cn, title = best_match
        lines.extend([
            "Best Match:",
            f"{cn} — {title}",
            "",
            "Why It Matches:",
            why_it_matches,
            "",
        ])
    else:
        lines.extend([
            "Why It Matches:",
            why_it_matches,
            "",
        ])

    if other_candidates:
        lines.append("Other Relevant Recall Candidates:")
        for cn, reason in other_candidates:
            lines.append(f"  - {cn} — {reason}")
        lines.append("")

    lines.extend([
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
