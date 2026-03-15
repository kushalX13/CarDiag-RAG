#!/usr/bin/env python3
"""
Analyze retrieval failures from retrieval_debug.jsonl.

Reads per-query debug JSONL, extracts failures (miss or gold not in top-10),
and produces a markdown report and optional CSV under eval/results/.
Heuristic categories: symptom_wording_mismatch, weak_lexical_overlap,
ambiguous_symptom, likely_pool_index_issue.

Run from project root: python scripts/analyze_failures.py [--input path] [--output-dir eval/results]
"""

import argparse
import json
import os
import sys


def _is_failure(row: dict, top_k: int = 10) -> bool:
    """True if gold not in top-K (miss or ranked below K)."""
    rank = row.get("gold_rank")
    if rank is None or rank == "MISS":
        return True
    if isinstance(rank, int) and rank > top_k:
        return True
    return False


def _categorize(row: dict) -> str:
    """
    Heuristic category for failure. Uses gold_score_debug[0].query_term_overlap
    and gold_in_candidates when available.
    """
    in_candidates = row.get("gold_in_candidates", False)
    rank = row.get("gold_rank")

    if not in_candidates or rank == "MISS":
        return "likely_pool_index_issue"

    overlap = None
    gsd = row.get("gold_score_debug") or []
    if gsd and isinstance(gsd, list) and gsd:
        overlap = (gsd[0] or {}).get("query_term_overlap") or {}
    missing_count = (overlap or {}).get("missing_count", 0)
    overlap_count = (overlap or {}).get("overlap_count", 0)
    query_terms = (overlap or {}).get("query_terms") or []
    n_terms = len(query_terms) if query_terms else 0

    if n_terms > 0 and missing_count >= n_terms * 0.6:
        return "symptom_wording_mismatch"
    if overlap_count <= 2 and n_terms >= 4:
        return "weak_lexical_overlap"
    if in_candidates and isinstance(rank, int) and rank > 5:
        return "ambiguous_symptom"
    return "other"


def load_debug(path: str) -> list[dict]:
    """Load JSONL debug file."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze retrieval failures from debug JSONL")
    parser.add_argument(
        "--input",
        type=str,
        default="eval/results/retrieval_debug.jsonl",
        help="Path to retrieval_debug.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval/results",
        help="Directory for failure_analysis.md and failure_analysis.csv",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Consider failure when gold not in top-K (default 10)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50,
        help="Max failure rows to include in report (default 50)",
    )
    args = parser.parse_args()

    rows = load_debug(args.input)
    if not rows:
        print("No rows in %s (or file missing). Run eval first to generate debug JSONL." % args.input, file=sys.stderr)
        sys.exit(1)

    failures = [r for r in rows if _is_failure(r, args.top_k)]
    for r in failures:
        r["_category"] = _categorize(r)

    os.makedirs(args.output_dir, exist_ok=True)
    md_path = os.path.join(args.output_dir, "failure_analysis.md")
    csv_path = os.path.join(args.output_dir, "failure_analysis.csv")

    # Markdown report
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Retrieval failure analysis\n\n")
        f.write("Source: `%s`  \n" % args.input)
        f.write("Total queries: %d  \n" % len(rows))
        f.write("Failures (gold not in top-%d): %d  \n\n" % (args.top_k, len(failures)))
        f.write("## Summary by category (heuristic)\n\n")
        from collections import Counter
        cats = Counter(r["_category"] for r in failures)
        f.write("| Category | Count |\n|----------|-------|\n")
        for cat, count in cats.most_common():
            f.write("| %s | %d |\n" % (cat, count))
        f.write("\n## Top missed queries\n\n")
        f.write("| Query | Make | Model | Gold | First correct rank | Top 5 returned | Category |\n")
        f.write("|-------|------|-------|------|--------------------|----------------|----------|\n")
        for r in failures[: args.max_rows]:
            query = (r.get("query") or "")[:60] + ("..." if len(r.get("query") or "") > 60 else "")
            make = r.get("make", "")
            model = r.get("model", "")
            gold = r.get("gold_campaign") or (r.get("gold_campaigns") or ["—"])[0]
            rank = r.get("first_correct_rank") if r.get("gold_rank") is not None else "MISS"
            top5 = (r.get("top10_predicted") or r.get("predicted_topk") or [])[:5]
            top5_s = ", ".join(str(c) for c in top5)
            cat = r.get("_category", "other")
            f.write("| %s | %s | %s | %s | %s | %s | %s |\n" % (
                query.replace("|", "\\|"), make, model, gold, rank, top5_s[:50], cat
            ))
        if len(failures) > args.max_rows:
            f.write("\n*... and %d more failures.*\n" % (len(failures) - args.max_rows))
    print("Wrote %s" % md_path)

    # CSV
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("query,make,model,gold_campaign,first_correct_rank,top10_returned,category\n")
        for r in failures:
            query = (r.get("query") or "").replace('"', '""')
            top10 = r.get("top10_predicted") or r.get("predicted_topk") or []
            f.write('"%s","%s","%s","%s","%s","%s","%s"\n' % (
                query,
                r.get("make", ""),
                r.get("model", ""),
                r.get("gold_campaign") or (r.get("gold_campaigns") or [""])[0],
                r.get("first_correct_rank") if r.get("gold_rank") is not None else "MISS",
                ";".join(str(c) for c in top10[:10]),
                r.get("_category", "other"),
            ))
    print("Wrote %s" % csv_path)


if __name__ == "__main__":
    main()
