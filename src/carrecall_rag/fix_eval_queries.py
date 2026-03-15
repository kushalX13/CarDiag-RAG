"""Create a cleaner eval JSONL by rewriting/dropping metadata-only queries."""

import argparse
import json
import re
from collections import defaultdict


GARBAGE_QUERY_RE = re.compile(
    r"^\s*vehicle\s+description\s*:\s*"
    r"(passenger\s+vehicles|pickup\s+trucks|sport\s+utility\s+vehicles)\s*\.?\s*$",
    re.IGNORECASE,
)
LEADING_METADATA_RE = re.compile(r"^\s*vehicle\s+description\s*:\s*", re.IGNORECASE)
INLINE_METADATA_RE = re.compile(r"\bvehicle\s+description\b\s*:?", re.IGNORECASE)
LEADING_CATEGORY_RE = re.compile(
    r"^\s*(passenger\s+vehicles|pickup\s+trucks|sport\s+utility\s+vehicles)\s*[\.:,-]?\s*",
    re.IGNORECASE,
)

DEFECT_HINTS = (
    "may",
    "could",
    "failure",
    "defect",
    "leak",
    "stall",
    "warning",
    "air bag",
    "airbag",
    "brake",
    "fuel",
    "fire",
    "crash",
    "deploy",
    "disable",
    "rollaway",
    "separate",
    "injury",
)
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "these",
    "from",
    "into",
    "when",
    "while",
    "are",
    "was",
    "were",
    "may",
    "could",
}


def _norm(s: str) -> str:
    return " ".join((s or "").strip().split())


def _strip_leading_metadata(query: str) -> str:
    q = _norm(query)
    q = LEADING_METADATA_RE.sub("", q).strip(" .,:;-")
    q = INLINE_METADATA_RE.sub("", q).strip(" .,:;-")
    q = LEADING_CATEGORY_RE.sub("", q).strip(" .,:;-")
    return _norm(q)


def _is_garbage_query(query: str) -> bool:
    return bool(GARBAGE_QUERY_RE.match(_norm(query)))


def _is_meaningful_query(query: str) -> bool:
    q = _strip_leading_metadata(query).lower()
    if len(q) < 20:
        return False
    return any(h in q for h in DEFECT_HINTS)


def _extract_gold(row: dict) -> str:
    gold = row.get("gold_campaign")
    if gold:
        return str(gold).strip()
    golds = row.get("gold_campaigns") or []
    if isinstance(golds, list) and golds:
        return str(golds[0]).strip()
    return ""


def _tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(t) > 2 and t not in STOPWORDS
    }


def _token_overlap(a: str, b: str) -> int:
    return len(_tokens(a) & _tokens(b))


def _load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pick_rewrite(candidates: list[str]) -> str | None:
    if not candidates:
        return None
    # Prefer richer symptom phrasing if available.
    ranked = sorted(
        candidates,
        key=lambda q: (0 if _is_meaningful_query(q) else 1, -len(_strip_leading_metadata(q))),
    )
    return _strip_leading_metadata(ranked[0])


def _campaign_rewrite_seed(texts: list[str]) -> str | None:
    if not texts:
        return None
    ranked = sorted(texts, key=lambda t: (-len(t), t))
    for text in ranked:
        clean = _strip_leading_metadata(text)
        if not clean:
            continue
        parts = re.split(r"(?<=[.!?])\s+", clean)
        for sentence in parts:
            sentence = _norm(sentence).strip(" .,:;-")
            if len(sentence) < 20:
                continue
            if "is recalling certain" in sentence.lower():
                continue
            if _is_meaningful_query(sentence):
                return sentence[:180]
    best = _strip_leading_metadata(ranked[0])
    return best[:180] if best else None


def _load_campaign_texts(corpus_path: str) -> dict[str, list[str]]:
    by_gold: dict[str, list[str]] = defaultdict(list)
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                gold = str(row.get("campaign_number") or "").strip()
                text = str(row.get("text") or "").strip()
                if gold and text:
                    by_gold[gold].append(text)
    except FileNotFoundError:
        return {}
    return dict(by_gold)


def build_fixed_eval(rows: list[dict], campaign_texts: dict[str, list[str]] | None = None) -> tuple[list[dict], dict]:
    campaign_texts = campaign_texts or {}
    campaign_seeds: dict[str, str] = {}
    for gold, texts in campaign_texts.items():
        seed = _campaign_rewrite_seed(texts)
        if seed:
            campaign_seeds[gold] = seed

    by_gold_meaningful: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        q = row.get("query", "")
        gold = _extract_gold(row)
        if not gold:
            continue
        if _is_garbage_query(q):
            continue
        cleaned = _strip_leading_metadata(q)
        if _is_meaningful_query(cleaned):
            by_gold_meaningful[gold].append(cleaned)

    out: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    rewritten = 0
    dropped = 0
    mismatch_rewritten = 0

    for row in rows:
        gold = _extract_gold(row)
        q = row.get("query", "")
        make = row.get("make", "")
        model = row.get("model", "")
        if not gold:
            continue

        if _is_garbage_query(q):
            replacement = _pick_rewrite(by_gold_meaningful.get(gold, []))
            if replacement:
                rewritten += 1
                q_final = replacement
            else:
                dropped += 1
                continue
        else:
            q_final = _strip_leading_metadata(q)
            if len(q_final) < 8:
                dropped += 1
                continue
            # Repair obviously mismatched query/gold pairs using campaign text.
            campaign_seed = campaign_seeds.get(gold, "")
            if campaign_seed and _token_overlap(q_final, campaign_seed) < 2:
                q_final = campaign_seed
                mismatch_rewritten += 1

        key = (_norm(q_final), _norm(make), _norm(model), gold)
        if key in seen:
            continue
        seen.add(key)

        out_row = dict(row)
        out_row["query"] = q_final
        out.append(out_row)

    stats = {
        "input_rows": len(rows),
        "output_rows": len(out),
        "rewritten_garbage_rows": rewritten,
        "rewritten_mismatch_rows": mismatch_rewritten,
        "dropped_rows": dropped,
    }
    return out, stats


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix metadata-only eval queries safely")
    parser.add_argument(
        "--input",
        type=str,
        default="eval/recall_queries_100.jsonl",
        help="Input eval JSONL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/recall_queries_100_fixed.jsonl",
        help="Output fixed eval JSONL",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/processed/corpus_merged.jsonl",
        help="Corpus JSONL used for campaign-aware query rewriting",
    )
    args = parser.parse_args()

    rows = _load_jsonl(args.input)
    campaign_texts = _load_campaign_texts(args.corpus)
    fixed, stats = build_fixed_eval(rows, campaign_texts=campaign_texts)
    _write_jsonl(args.output, fixed)

    print(
        "Wrote {output} | input={input_rows} output={output_rows} rewritten={rewritten_garbage_rows} mismatch_rewritten={rewritten_mismatch_rows} dropped={dropped_rows}".format(
            output=args.output,
            **stats,
        )
    )


if __name__ == "__main__":
    main()
