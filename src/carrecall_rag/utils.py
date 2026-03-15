"""Normalize make/model, chunk text, JSONL write."""

import hashlib
import json
import re


def safe_slug(text: str) -> str:
    """Normalize text for filenames: lowercase, replace spaces/slashes with underscores."""
    s = text.lower().strip()
    s = re.sub(r"[\s/]+", "_", s)
    s = re.sub(r"[^\w\-_.]", "", s)
    return s or "unknown"


def jsonl_write(path: str, rows: list[dict]) -> None:
    """Write rows as JSONL to path."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunk_text(text: str, max_words: int = 250, overlap_words: int = 50) -> list[str]:
    """Split text into overlapping chunks by word count."""
    text = normalize_whitespace(text)
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, max_words - overlap_words)
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += step
    return chunks


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace to single space and strip."""
    if not text or not isinstance(text, str):
        return ""
    return " ".join(text.split())


def extract_best_text_fields(record: dict, field_candidates: list[str]) -> str:
    """Join existing non-empty string fields from record. Returns concatenated text."""
    parts = []
    for key in field_candidates:
        val = record.get(key)
        if val is not None and isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return " ".join(parts) if parts else ""


MAKE_SYNONYMS = {
    "MERCEDESBENZ": "MERCEDES-BENZ",
    "MERCEDES BENZ": "MERCEDES-BENZ",
    "MERCEDES": "MERCEDES-BENZ",
}


def normalize_make(s: str) -> str:
    """Uppercase, collapse spaces/hyphens, map synonyms (e.g. MERCEDES BENZ -> MERCEDES-BENZ)."""
    if not s or not isinstance(s, str):
        return ""
    t = s.upper().strip()
    t = re.sub(r"[\s\-]+", "", t)  # collapse spaces and hyphens
    return MAKE_SYNONYMS.get(t, t)


def normalize_model(s: str) -> str:
    """Uppercase, replace hyphens with spaces, collapse spaces, strip trim words (4DR, 2DR, etc)."""
    if not s or not isinstance(s, str):
        return ""
    t = s.upper().strip()
    t = t.replace("-", " ")
    t = " ".join(t.split())  # collapse spaces
    trim_words = {"4DR", "2DR", "4D", "2D", "SEDAN", "COUPE", "HATCHBACK", "WAGON", "SUV", "PICKUP"}
    words = t.split()
    words = [w for w in words if w not in trim_words]
    return " ".join(words).strip()


def model_key(s: str) -> str:
    """Alphanumeric-only lowercase for matching."""
    if not s or not isinstance(s, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", normalize_model(s).lower())
