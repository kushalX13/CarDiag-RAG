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

    def generate(self, query: str, retrieved_docs: list[dict], top_k: int, vehicle: str = "") -> str:
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


def _clean_phrase(text: str) -> str:
    """Clean extracted fragments into sentence-ready phrases."""
    s = re.sub(r"\s+", " ", (text or "")).strip(" ,;:-")
    # Drop dangling clause starters often caused by snippet clipping.
    s = re.sub(r"\b(?:and|or|with|which|that|because|while|when|either|the|a)\s*$", "", s, flags=re.IGNORECASE)
    # Drop very short trailing token likely clipped from source (e.g., 'fu').
    m = re.search(r"\b([a-zA-Z]{1,2})$", s)
    if m and not s.endswith(("3.0", "2.0")):
        s = re.sub(r"\b[a-zA-Z]{1,2}$", "", s).strip(" ,;:-")
    return s


def _is_noisy_defect_phrase(text: str) -> bool:
    """Detect low-quality extracted defect phrases and force safer fallback."""
    t = (text or "").strip().lower()
    if not t:
        return True
    noisy_starts = ("and ", "or ", "which ", "that ", "this ", "dealers ")
    noisy_terms = ("dealers will", "inspect and replace", "is recalling", "certain ")
    if t.startswith(noisy_starts):
        return True
    return any(x in t for x in noisy_terms)


def _extract_defect_and_consequence(text: str) -> tuple[str, str]:
    """Extract (defect, consequence) separately for grammatical sentence building."""
    t = _strip_boilerplate(text)
    defect, consequence = "", ""

    # Defect: component or failure mode
    if re.search(r"high pressure fuel pump|HPFP|fuel pump", t, re.IGNORECASE):
        defect = "high pressure fuel pump (HPFP)"
    elif re.search(r"brake master cylinder", t, re.IGNORECASE):
        defect = "brake master cylinder"
    else:
        m = re.search(r"([\w\s]+(?:pump|sensor|module|latch|actuator|cable|cylinder))", t, re.IGNORECASE)
        if m:
            defect = re.sub(r"^\s*(?:the |a )\s*", "", m.group(1).strip(), flags=re.IGNORECASE)
            defect = _clean_phrase(defect)

    # Consequence: what can happen
    for pat in [
        r"(?:resulting in|which may cause|may result in)\s+([^.]{10,90})",
        r"(?:may|could|can)\s+(?:result in|cause|lead to)\s+([^.]{10,90})",
        r"(fuel starvation|engine stall|loss of power|loss of propulsion|debris into the fuel system)[^.]*",
        r"(reduced brake function|reduced braking ability)[^.]*",
    ]:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            consequence = m.group(1).strip() if m.lastindex else m.group(0).strip()
            consequence = re.sub(r"\s+", " ", consequence)
            # Clean common fragments for grammatical completeness
            consequence = re.sub(r"\band extend the distance required to stop\b", " and extended stopping distance", consequence, flags=re.IGNORECASE)
            consequence = re.sub(r",?\s*increasing the[^.]*\.*$", ", which may increase crash risk", consequence, flags=re.IGNORECASE)
            consequence = re.sub(r",?\s*increasing the\s+", ", which may increase ", consequence, flags=re.IGNORECASE)
            consequence = re.sub(r"\s+", " ", consequence).strip()
            consequence = _clean_phrase(_truncate(consequence, 90))
            if len(consequence) > 8:
                break

    return (defect, consequence)


def _extract_issue_and_consequence(text: str) -> str:
    """Extract what failed and what can happen (combined, for titles)."""
    defect, consequence = _extract_defect_and_consequence(text)
    if defect and consequence:
        return f"{defect}, which can lead to {consequence}"
    return defect or consequence or ""


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


def _extract_candidate_specific_desc(campaign: dict) -> str:
    """Extract a short, grounded description from the candidate's own evidence."""
    best = campaign.get("best_doc") or {}
    component = (best.get("component") or "").strip()
    evs = campaign.get("evidence_snippets") or []
    texts = [best.get("text", "")] + [e.get("snippet", "") for e in evs[:2]]
    combined = " ".join(t for t in texts if t).strip()

    def _clean_desc(s: str, max_len: int = 45) -> str:
        """Truncate at word boundary; avoid trailing fragments like 'either the'."""
        s = (s or "").strip()
        if len(s) <= max_len:
            return s
        cut = s[:max_len].rsplit(" ", 1)[0]
        # Drop trailing incomplete phrases
        cut = re.sub(r",?\s+(?:either|which|that|the|a)\s*$", "", cut).strip()
        return cut + "..." if len(cut) < len(s) else cut

    # Prefer component when it's descriptive (not generic)
    if component and len(component) > 5 and component.lower() not in {"unknown", "other"}:
        defect, consequence = _extract_defect_and_consequence(combined)
        # Only add consequence if it's a clean phrase (not "either the...", "which may...")
        if consequence and not re.match(r"^(?:either|which|that|the|a)\s", consequence, re.IGNORECASE):
            if "engine stall" in consequence.lower():
                return f"{_clean_desc(component, 45)}; may cause engine stall"
            if "stall" in consequence.lower():
                return f"{_clean_desc(component, 45)}; can lead to stalling"
            if len(consequence) < 50:
                return f"{_clean_desc(component, 38)}; {_clean_desc(consequence, 38)}"
        return _clean_desc(component, 55)

    # Fallback: extract defect/consequence from text
    defect, consequence = _extract_defect_and_consequence(combined)
    if defect and consequence:
        return f"{_clean_desc(defect, 35)}; {_clean_desc(consequence, 35)}"
    if defect:
        return _clean_desc(defect, 55)
    if consequence:
        return _clean_desc(consequence, 55)

    # Last resort: title or first substantive phrase
    title = _extract_short_title(campaign, 55)
    if title and title != "Recall campaign":
        return title
    first = re.split(r"\.\s+", combined)[0] if combined else ""
    stripped = _strip_boilerplate(first)
    return _clean_desc(stripped or first, 50) if stripped or first else "related recall"


def _extract_relevance_note(
    campaign: dict, best_campaign: dict, query: str
) -> str:
    """
    Short relevance note for secondary candidate.
    Uses candidate-specific description from its own evidence; adds confidence qualifier when weak.
    """
    best_parts = [(best_campaign.get("best_doc") or {}).get("text", "")]
    best_parts.extend(e.get("snippet", "") for e in (best_campaign.get("evidence_snippets") or []))
    best_text = " ".join(best_parts).lower()

    camp_parts = [(campaign.get("best_doc") or {}).get("text", "")]
    camp_parts.extend(e.get("snippet", "") for e in (campaign.get("evidence_snippets") or []))
    camp_text = " ".join(camp_parts).lower()
    q = (query or "").lower()

    # Query keyword overlap: this candidate vs best match
    q_words = set(re.findall(r"[a-z0-9]{3,}", q)) - {"the", "and", "for", "may", "can", "with", "into"}
    overlap_best = sum(1 for w in q_words if w in best_text)
    overlap_this = sum(1 for w in q_words if w in camp_text)

    # Candidate-specific description from its own evidence (never generic "engine-related" unless evidence supports it)
    specific = _extract_candidate_specific_desc(campaign)

    # Score relevance for filtering and note style.
    relevance = _secondary_relevance_score(query, best_text, camp_text, overlap_best, overlap_this)

    if relevance < 0.30:
        return f"lower-confidence match; less directly related secondary candidate; {specific}"
    if relevance < 0.55:
        return f"less directly related secondary candidate; {specific}"
    return specific


def _secondary_relevance_score(
    query: str,
    best_text: str,
    camp_text: str,
    overlap_best: int,
    overlap_this: int,
) -> float:
    """Heuristic relevance score for supporting candidates."""
    q = (query or "").lower()

    # Base overlap ratio.
    overlap_ratio = overlap_this / max(1, len(set(re.findall(r"[a-z0-9]{3,}", q))))
    relative_to_best = overlap_this / max(1, overlap_best)

    # Shared issue-type signals (component/consequence).
    issue_groups = [
        {"fuel", "pump", "hpfp", "starvation", "diesel"},
        {"brake", "booster", "cylinder", "fluid", "stopping"},
        {"transmission", "park", "rollaway", "shift", "cable"},
        {"airbag", "orc", "clock", "spring"},
    ]
    shared_issue = 0.0
    for g in issue_groups:
        if any(t in q for t in g) and any(t in camp_text for t in g):
            shared_issue = 1.0
            break

    # Penalty for likely noisy candidates (e.g., lighting in brake/fuel query).
    noise_terms = {"lamp", "lighting", "headlight", "tail lamp", "wiper", "seat trim"}
    noisy = any(t in camp_text for t in noise_terms) and not any(t in q for t in noise_terms)
    noise_penalty = 0.25 if noisy else 0.0

    score = 0.45 * overlap_ratio + 0.35 * min(1.0, relative_to_best) + 0.20 * shared_issue - noise_penalty
    return max(0.0, min(1.0, score))


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


def _build_natural_connection(query: str, defect: str, consequence: str, text: str) -> str:
    """Build natural-language connection between query and recall (no debug phrasing)."""
    q = (query or "").lower()
    t = (text or "").lower()

    # Semantic concept pairs: query phrase -> recall concept
    if any(w in q for w in ["hpfp", "fuel pump", "high pressure"]) and any(w in t for w in ["fuel pump", "hpfp"]):
        if any(w in t for w in ["starvation", "stall"]) and any(w in q for w in ["starvation", "stall", "failure"]):
            return "This closely matches your query because it directly connects HPFP failure with fuel starvation and engine stall."
        return "This closely matches your query because it addresses the high pressure fuel pump defect you described."
    if any(w in q for w in ["brake", "cylinder", "booster", "leak"]) and any(w in t for w in ["brake", "cylinder", "booster", "leak"]):
        return "This closely matches your query because it directly addresses brake fluid leak into the brake booster."
    if any(w in q for w in ["stall", "engine"]) and any(w in t for w in ["stall", "engine"]):
        return "This closely matches your query because it addresses engine stalling or loss of power."
    if defect and consequence:
        return f"This closely matches your query because the defect ({defect}) and its consequences align with what you reported."
    return "This closely matches your query because the recalled issue aligns with your description."


def _why_it_matches(query: str, campaign: dict) -> str:
    """
    Build 2-3 sentence plain-English explanation for the best match.
    Grammatically complete; no debug phrasing.
    """
    best = campaign.get("best_doc") or {}
    evs = campaign.get("evidence_snippets") or []
    texts = [best.get("text", "")] + [e.get("snippet", "") for e in evs]
    combined = " ".join(t for t in texts if t).strip()
    if not combined:
        return "No detailed information available for this recall."

    defect, consequence = _extract_defect_and_consequence(combined)
    if _is_noisy_defect_phrase(defect):
        defect = ""
    if not defect:
        defect = _extract_short_title(campaign, 60)

    # Build grammatically complete sentences (avoid raw fragments)
    lead = defect
    if defect and not defect.lower().startswith(("the ", "a ")):
        lead = f"The {defect}"

    if defect and consequence:
        # Avoid "can lead to may result in..." — consequence should be noun phrase
        cons_clean = re.sub(r"^(?:may|could|can)\s+(?:result in|cause|lead to)\s+", "", consequence, flags=re.IGNORECASE).strip()
        cons_clean = _clean_phrase(cons_clean)
        first = f"{lead} may fail, resulting in {cons_clean}."
    elif defect:
        first = f"{lead} may be defective."
    else:
        first = "This recall addresses a known defect."

    connection = _build_natural_connection(query, defect, consequence, combined)
    return f"{first} {connection}"


def _build_safety_risk_narrative(campaigns: list[dict], top_k: int) -> str:
    """Build natural safety risk summary (not a raw phrase list)."""
    all_phrases: list[str] = []
    for c in campaigns[:top_k]:
        evs = c.get("evidence_snippets") or []
        best = c.get("best_doc") or {}
        texts = [best.get("text", "")] + [e.get("snippet", "") for e in evs]
        for t in texts:
            all_phrases.extend(_extract_safety_phrases(t))

    unique = _deduplicate_risks(all_phrases)
    if not unique:
        return "Review the recalled component details for specific safety implications."

    # Build natural narrative based on risk types
    engine_related = [p for p in unique if "engine stall" in p or "fuel" in p.lower() or "propulsion" in p.lower()]
    brake_related = [p for p in unique if "brake" in p.lower()]
    crash_included = any("crash" in p.lower() for p in unique)
    other = [p for p in unique if p not in engine_related and p not in brake_related and "crash" not in p.lower()]

    parts = []
    if engine_related:
        parts.append("engine stall and sudden loss of propulsion while driving")
    if brake_related:
        parts.append("reduced braking ability and increased stopping distance")
    for p in other:
        if p not in ["engine stall", "loss of power/control"]:
            parts.append(p.lower())

    if not parts:
        parts = [p.lower() for p in unique[:2]]

    primary = parts[0] if parts else ""
    if crash_included and "crash" not in primary:
        return f"Potential risks include {primary}, which may increase crash risk."
    if len(parts) == 1:
        return f"Potential risks include {primary}."
    return f"Potential risks include {primary} and {parts[1]}, which may increase crash risk."


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

    def generate(
        self,
        query: str,
        retrieved_docs: list[dict],
        top_k: int = 3,
        vehicle: str = "",
    ) -> str:
        """Generate structured answer from retrieved campaign docs."""
        top = retrieved_docs[:top_k]
        if not top:
            return _format_output(
                query=query,
                vehicle=vehicle,
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
        other_scored = []
        for c in top[1:]:
            best_parts = [(best.get("best_doc") or {}).get("text", "")]
            best_parts.extend(e.get("snippet", "") for e in (best.get("evidence_snippets") or []))
            best_text = " ".join(best_parts).lower()
            camp_parts = [(c.get("best_doc") or {}).get("text", "")]
            camp_parts.extend(e.get("snippet", "") for e in (c.get("evidence_snippets") or []))
            camp_text = " ".join(camp_parts).lower()
            q_words = set(re.findall(r"[a-z0-9]{3,}", (query or "").lower())) - {"the", "and", "for", "may", "can", "with", "into"}
            overlap_best = sum(1 for w in q_words if w in best_text)
            overlap_this = sum(1 for w in q_words if w in camp_text)
            score = _secondary_relevance_score(query, best_text, camp_text, overlap_best, overlap_this)
            other_scored.append(
                (
                    score,
                    c.get("campaign_number", ""),
                    _extract_short_title(c),
                )
            )

        # Keep only reasonably related supports; allow at most one weak/noisy candidate last.
        strong = [x for x in other_scored if x[0] >= 0.45]
        weak = [x for x in other_scored if x[0] < 0.45]
        strong.sort(key=lambda x: x[0], reverse=True)
        weak.sort(key=lambda x: x[0], reverse=True)

        selected = strong[:2]
        if not selected and weak:
            selected = [weak[0]]

        other = []
        for score, cn, title in selected:
            confidence = "lower-confidence" if score < 0.55 else "relevant"
            other.append((cn, title, confidence))
        safety_risk = _build_safety_risk_narrative(top, top_k)
        next_step = _suggest_next_step(top)

        return _format_output(
            query=query,
            vehicle=vehicle,
            best_match=(best_cn, best_title),
            why_it_matches=why,
            other_candidates=other,
            safety_risk=safety_risk,
            next_step=next_step,
        )


def _format_output(
    query: str,
    vehicle: str,
    best_match: tuple[str, str] | None,
    why_it_matches: str,
    other_candidates: list[tuple[str, str, str]],
    safety_risk: str,
    next_step: str,
) -> str:
    """Format the final RAG output (user-facing)."""
    vehicle_line = vehicle.strip() if vehicle and vehicle.strip() else "Not specified"
    lines = [
        "Possible Recall-Related Issue",
        "",
        "Vehicle:",
        vehicle_line,
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
            "Why it matches:",
            why_it_matches,
            "",
        ])
    else:
        lines.extend([
            "Why it matches:",
            why_it_matches,
            "",
        ])

    if other_candidates:
        lines.append("Other relevant recall candidates:")
        for cn, title, confidence in other_candidates:
            suffix = " (lower-confidence)" if confidence == "lower-confidence" else ""
            lines.append(f"- {cn} — {title}{suffix}")
        lines.append("")

    lines.extend([
        "Potential safety risk:",
        safety_risk,
        "",
        "Recommended next step:",
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
    vehicle: str = "",
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
    return backend.generate(query, retrieved_docs, top_k, vehicle=vehicle)


def set_default_backend(backend: AnswerBackend) -> None:
    """Set the default answer backend (for swapping template/LLM)."""
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = backend
