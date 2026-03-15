"""
Generate candidate evaluation queries from the recall corpus.

Reads data/processed/corpus_merged.jsonl, groups docs by campaign_number,
and produces query variants (shortened/paraphrased from recall text) so that
all gold_campaign IDs exist in the corpus. Output is JSONL with fields:
query, make, model, gold_campaign (or gold_campaigns).

Deterministic and reproducible. Use --max-queries to cap total (default ~100).

Clean mode (--clean): prefer defect/consequence/symptom text; exclude metadata
phrases like "VEHICLE DESCRIPTION", "PASSENGER VEHICLES"; generate 4-6
realistic symptom queries per campaign. Output to eval/recall_queries_clean_100.jsonl.
"""

import argparse
import json
import logging
import os
import re
from collections import defaultdict

from .config import PROCESSED_DIR
from .utils import normalize_whitespace

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CORPUS_PATH = os.path.join(PROCESSED_DIR, "corpus_merged.jsonl")

# Metadata phrases to exclude from query generation (noise from NHTSA format)
METADATA_PREFIXES = (
    r"vehicle\s+description\s*:",
    r"passenger\s+vehicles\s*\.?",
    r"pickup\s+trucks\s*\.?",
    r"passenger\s+cars\s*\.?",
    r"sport\s+utility\s+vehicles\s*\.?",
    r"mini\s+vans\s*\.?",
    r"certain\s+model\s+year\s+\d{4}",
)
# Symptom/defect keywords: text containing these is preferred for queries
DEFECT_KEYWORDS = (
    "failure", "fail", "leak", "leaking", "stall", "stalls", "brake", "fire", "crash",
    "may not", "could cause", "result in", "loss of", "power", "steering", "separat",
    "recall", "defect", "risk", "injury", "damage", "stuck", "pedal", "fuel", "engine",
    "transmission", "airbag", "deploy", "wheel", "lock", "accelerator",
)


def _is_metadata_heavy(text: str) -> bool:
    """True if text is dominated by metadata (e.g. VEHICLE DESCRIPTION: ...)."""
    if not text or len(text.strip()) < 15:
        return True
    t = (text.strip() + " ").lower()[:350]
    for pat in METADATA_PREFIXES:
        if re.search(pat, t):
            # If most of the first 200 chars is this kind of phrase, skip
            if re.match(r"^[\s\w:.,]+$", t[:200]) and len(t.split()) < 25:
                return True
    return False


def _has_defect_content(text: str) -> bool:
    """True if text contains defect/symptom-related keywords."""
    if not text:
        return False
    tl = (text or "").lower()
    return any(kw in tl for kw in DEFECT_KEYWORDS)


def _first_sentence(text: str, max_chars: int = 200) -> str:
    """Take first sentence or prefix up to max_chars. Normalize whitespace."""
    text = normalize_whitespace(text or "")
    if not text:
        return ""
    for sep in (". ", ".\n", "."):
        idx = text.find(sep)
        if idx != -1:
            out = text[: idx + 1].strip()
            return out[:max_chars] if len(out) > max_chars else out
    return text[:max_chars].strip()


def _short_phrase(text: str, max_words: int = 14) -> str:
    """First max_words words, normalized. Good for keyword-style query."""
    text = normalize_whitespace(text or "")
    words = text.split()[:max_words]
    return " ".join(words).strip()


def _tokenize_for_dedup(text: str) -> str:
    """Normalize for deduplication: lowercase, alphanumeric tokens."""
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return " ".join(tokens)


def _extract_sentences(text: str, max_sentences: int = 4) -> list[str]:
    """Split into sentences (by . ? !), return up to max_sentences non-empty."""
    text = normalize_whitespace(text or "")
    if not text:
        return []
    parts = re.split(r"[.!?]+\s+", text)
    out = [p.strip() for p in parts if len(p.strip()) >= 20]
    return out[:max_sentences]


def load_corpus(path: str) -> list[dict]:
    """Load corpus JSONL. Return list of doc dicts."""
    if not os.path.exists(path):
        return []
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def group_by_campaign(docs: list[dict]) -> dict[str, list[dict]]:
    """Group corpus docs by campaign_number. Each doc has make, model, text."""
    by_campaign: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        cn = (d.get("campaign_number") or "").strip()
        if not cn or not (d.get("text") or "").strip():
            continue
        by_campaign[cn].append(d)
    return dict(by_campaign)


def generate_query_variants(texts: list[str], max_variants: int = 3) -> list[str]:
    """
    From a list of recall chunk texts, produce up to max_variants distinct
    query-like strings (first sentence or short phrase). Deduplicated.
    """
    seen: set[str] = set()
    variants: list[str] = []
    for t in texts:
        if not t or len(t.strip()) < 20:
            continue
        s1 = _first_sentence(t, max_chars=180)
        if s1 and _tokenize_for_dedup(s1) not in seen:
            seen.add(_tokenize_for_dedup(s1))
            variants.append(s1)
            if len(variants) >= max_variants:
                return variants
        s2 = _short_phrase(t, max_words=16)
        if s2 and _tokenize_for_dedup(s2) not in seen and s2 != s1:
            seen.add(_tokenize_for_dedup(s2))
            variants.append(s2)
            if len(variants) >= max_variants:
                return variants
    return variants


def generate_clean_query_variants(texts: list[str], max_variants: int = 6) -> list[str]:
    """
    From recall chunk texts, produce up to max_variants defect/symptom-focused
    queries. Excludes metadata-heavy text (VEHICLE DESCRIPTION, etc.);
    prefers text containing defect/symptom keywords.
    """
    seen: set[str] = set()
    variants: list[str] = []
    # Prefer defect-containing chunks first
    defect_first = sorted(
        texts,
        key=lambda t: (0 if _has_defect_content(t) else 1, _is_metadata_heavy(t), -len(t or "")),
    )
    for t in defect_first:
        if not t or len(t.strip()) < 20:
            continue
        if _is_metadata_heavy(t) and not _has_defect_content(t):
            continue
        # Use first sentence, short phrase, or individual sentences
        s1 = _first_sentence(t, max_chars=180)
        if s1 and _tokenize_for_dedup(s1) not in seen and not _is_metadata_heavy(s1):
            seen.add(_tokenize_for_dedup(s1))
            variants.append(s1)
            if len(variants) >= max_variants:
                return variants
        for sent in _extract_sentences(t, max_sentences=3):
            if _is_metadata_heavy(sent) or not _has_defect_content(sent):
                continue
            sent = sent[:160].strip()
            if sent and _tokenize_for_dedup(sent) not in seen:
                seen.add(_tokenize_for_dedup(sent))
                variants.append(sent)
                if len(variants) >= max_variants:
                    return variants
        s2 = _short_phrase(t, max_words=14)
        if s2 and _tokenize_for_dedup(s2) not in seen and s2 != s1 and not _is_metadata_heavy(s2):
            seen.add(_tokenize_for_dedup(s2))
            variants.append(s2)
            if len(variants) >= max_variants:
                return variants
    return variants


def make_display_name(raw: str, field: str) -> str:
    """Title-case for make/model display (e.g. HONDA -> Honda, F-150 unchanged)."""
    if not raw:
        return raw
    s = raw.strip().title()
    # Preserve F-150 style
    if field == "model" and re.match(r"^[A-Z0-9]+-[0-9]+$", raw.strip().upper()):
        return raw.strip()
    return s


def build_candidates(
    corpus_path: str,
    max_queries: int = 100,
    max_per_campaign: int = 3,
    seed_queries_path: str | None = None,
    use_clean: bool = False,
) -> list[dict]:
    """
    Build list of candidate eval rows: {"query", "make", "model", "gold_campaign"}.
    Uses only campaigns present in the corpus. Optionally prepend seed queries
    (e.g. from recall_queries.jsonl) if their gold_campaign exists in corpus.
    If use_clean=True, use defect-focused query generation (exclude metadata-heavy text).
    """
    docs = load_corpus(corpus_path)
    if not docs:
        logger.warning("Corpus empty or missing at %s. Return empty list.", corpus_path)
        return []

    by_campaign = group_by_campaign(docs)
    campaign_ids = sorted(by_campaign.keys())
    logger.info("Corpus has %d docs, %d campaigns", len(docs), len(campaign_ids))

    out: list[dict] = []
    if seed_queries_path and os.path.exists(seed_queries_path):
        with open(seed_queries_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                gold = row.get("gold_campaign") or (row.get("gold_campaigns") or [None])[0]
                if gold and gold in by_campaign:
                    out.append({
                        "query": (row.get("query") or "").strip(),
                        "make": row.get("make", ""),
                        "model": row.get("model", ""),
                        "gold_campaign": gold,
                    })
        logger.info("Seeded %d queries from %s", len(out), seed_queries_path)

    remaining = max_queries - len(out)
    if remaining <= 0:
        return out[:max_queries]

    gen_fn = generate_clean_query_variants if use_clean else generate_query_variants
    per_cap = min(max_per_campaign, remaining)

    for cn in campaign_ids:
        if len(out) >= max_queries:
            break
        group = by_campaign[cn]
        first = group[0]
        make = make_display_name(first.get("make") or "", "make")
        model = make_display_name(first.get("model") or "", "model")
        texts = [g.get("text", "").strip() for g in group if g.get("text")]
        variants = gen_fn(texts, max_variants=per_cap)
        for q in variants:
            if len(out) >= max_queries:
                break
            out.append({"query": q, "make": make, "model": model, "gold_campaign": cn})

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate candidate eval queries from corpus (campaign-grouped)"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=DEFAULT_CORPUS_PATH,
        help="Path to corpus_merged.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/recall_queries_100.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=100,
        help="Target number of queries (default 100)",
    )
    parser.add_argument(
        "--max-per-campaign",
        type=int,
        default=3,
        help="Max query variants per campaign (default 3)",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="",
        help="Optional path to seed JSONL (e.g. eval/recall_queries.jsonl); seeds prepended if gold in corpus",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Generate defect/symptom-focused queries only; exclude metadata (VEHICLE DESCRIPTION, etc.); 4-6 per campaign",
    )
    args = parser.parse_args()

    # When --clean, default output to clean_100 and more variants per campaign
    if args.clean and args.output == "eval/recall_queries_100.jsonl":
        args.output = "eval/recall_queries_clean_100.jsonl"
    if args.clean and args.max_per_campaign == 3:
        args.max_per_campaign = 5

    seed_path = args.seed.strip() or None
    candidates = build_candidates(
        corpus_path=args.corpus,
        max_queries=args.max_queries,
        max_per_campaign=args.max_per_campaign,
        seed_queries_path=seed_path,
        use_clean=args.clean,
    )

    if not candidates:
        logger.warning("No candidates generated (corpus missing or empty). Create corpus with: python -m carrecall_rag.build_corpus")
        # Optional: copy existing small eval file so output exists and is valid JSONL
        fallback = "eval/recall_queries.jsonl"
        if os.path.exists(fallback):
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(fallback, "r", encoding="utf-8") as fin:
                with open(args.output, "w", encoding="utf-8") as fout:
                    fout.write(fin.read())
            logger.info("Copied %s to %s. Re-run after building corpus to generate ~100 queries.", fallback, args.output)
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in candidates:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d queries to %s", len(candidates), args.output)


if __name__ == "__main__":
    main()
