"""Lightweight keyword reranker: TF-IDF-ish over candidates + phrase matching."""

import logging
import math
import re
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

# English + domain stopwords
STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was",
    "one", "our", "out", "has", "his", "how", "its", "may", "new", "now", "old", "see",
    "way", "who", "did", "get", "got", "let", "put", "say", "too", "use", "any", "own",
    "ford", "motor", "company", "vehicle", "vehicles", "recall", "recalls", "certain",
    "model", "models", "year", "years", "equipped", "campaign", "f150", "f-150",
    "pickup", "truck", "trucks", "honda", "toyota", "nhtsa", "dealer", "dealers",
}

# Default symptom phrases with boosts
DEFAULT_PHRASES = [
    ("loss of power", 2.0),
    ("engine stalls", 2.0),
    ("stalls while driving", 2.0),
    ("shuts off", 1.5),
    ("engine shuts off", 2.0),
    ("power loss", 2.0),
    ("cut out", 1.5),
    ("stalling", 1.5),
]


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. f-150 -> f150."""
    if not text:
        return ""
    t = text.lower().replace("-", "").replace("'", " ")
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def _tokenize(text: str, min_len: int = 3) -> list[str]:
    """Extract tokens, drop stopwords, min length."""
    norm = _normalize(text)
    tokens = re.findall(r"[a-z0-9]+", norm)
    return [t for t in tokens if len(t) >= min_len and t not in STOPWORDS]


def _extract_phrases(query: str, max_tokens: int = 12, max_phrases: int = 10) -> list[tuple[str, float]]:
    """
    Extract phrases: default symptom phrases (if in query) + 2-grams and 3-grams.
    Returns [(phrase, boost), ...] deduplicated.
    """
    norm_query = _normalize(query)
    tokens = _tokenize(query)[:max_tokens]
    seen = set()
    phrases = []

    # Default symptom phrases
    for phrase, boost in DEFAULT_PHRASES:
        norm_phrase = _normalize(phrase)
        if norm_phrase in norm_query and norm_phrase not in seen:
            phrases.append((norm_phrase, boost))
            seen.add(norm_phrase)

    # Auto 2-grams and 3-grams
    for n in [3, 2]:
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i : i + n])
            if gram not in seen and len(phrases) < max_phrases:
                boost = 1.5 if n == 3 else 1.0
                phrases.append((gram, boost))
                seen.add(gram)

    return phrases[:max_phrases]


def rerank(
    results: list[tuple[dict, float]],
    query: str,
    alpha: float = 0.15,
    max_tokens: int = 12,
    max_phrases: int = 10,
    normalize_dense: bool = False,
) -> list[tuple[dict, float, float, float]]:
    """
    Rerank (doc, dense_score) by combined = (1-alpha)*dense + alpha*kw_norm.
    If normalize_dense: scale dense to [0,1] so keyword-only docs (dense=0) can compete.
    Returns [(doc, combined, dense_score, kw_norm), ...] sorted by combined desc.
    """
    if not results:
        return []

    query_tokens = _tokenize(query)[:max_tokens]
    phrases = _extract_phrases(query, max_tokens=max_tokens, max_phrases=max_phrases)

    # Build doc texts and tokenize
    doc_texts = [r[0].get("text", "") for r in results]
    N = len(doc_texts)

    # Document frequency over candidates
    df: dict[str, int] = defaultdict(int)
    doc_token_counts: list[dict[str, int]] = []
    for text in doc_texts:
        norm = _normalize(text)
        counts = Counter(re.findall(r"[a-z0-9]+", norm))
        doc_token_counts.append(counts)
        for t in set(counts.keys()):
            df[t] += 1

    # IDF: log((N+1)/(df+1)) + 1
    idf = {}
    for t in set(query_tokens) | {p.split()[0] for p, _ in phrases if p}:
        idf[t] = math.log((N + 1) / (df.get(t, 0) + 1)) + 1

    # Per-doc keyword score
    kw_scores = []
    for i, (doc, dense_score) in enumerate(results):
        text = doc_texts[i]
        norm_text = _normalize(text)
        counts = doc_token_counts[i]

        # Token score: sum(idf * min(3, tf)) for query tokens
        token_score = 0.0
        for t in query_tokens:
            tf = min(3, counts.get(t, 0))
            token_score += idf.get(t, 1.0) * tf

        # Phrase score
        phrase_score = 0.0
        for phrase, boost in phrases:
            if phrase in norm_text:
                phrase_score += boost

        kw = token_score + phrase_score
        kw_scores.append((doc, dense_score, kw))

    # Min-max normalize kw to [0, 1]
    kw_vals = [k for _, _, k in kw_scores]
    kw_min = min(kw_vals)
    kw_max = max(kw_vals)
    eps = 1e-9
    span = max(kw_max - kw_min, eps)

    if kw_max - kw_min < 1e-6:
        logger.warning("kw has no variance; check tokenization/stopwords/phrases")

    # Debug: top 5 kw values
    sorted_by_kw = sorted(kw_scores, key=lambda x: x[2], reverse=True)
    top5_kw = [f"{x[2]:.4f}" for x in sorted_by_kw[:5]]
    logger.info("kw top5: %s  |  min=%.4f max=%.4f", top5_kw, kw_min, kw_max)

    # Optionally normalize dense for hybrid (keyword-only docs have dense=0)
    dense_vals = [d for _, d, _ in kw_scores]
    dense_min, dense_max = min(dense_vals), max(dense_vals)
    dense_span = max(dense_max - dense_min, eps)

    # Combine and sort
    output = []
    for doc, dense_score, kw in kw_scores:
        kw_norm = (kw - kw_min) / span
        dense_norm = (dense_score - dense_min) / dense_span if normalize_dense else dense_score
        combined = (1 - alpha) * dense_norm + alpha * kw_norm
        output.append((doc, combined, dense_score, kw_norm))

    output.sort(key=lambda x: x[1], reverse=True)
    return output
