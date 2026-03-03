"""CLI: Download NHTSA data and build complaints.jsonl + corpus.jsonl."""

import argparse
import csv
import hashlib
import json
import logging
import os
from tqdm import tqdm

from .config import MODELS, PROCESSED_DIR, RAW_COMPLAINTS_DIR, RAW_RECALLS_DIR
from .nhtsa_api import fetch_complaints, fetch_recalls
from .utils import (
    chunk_text,
    extract_best_text_fields,
    jsonl_write,
    model_key,
    normalize_make,
    normalize_model,
    normalize_whitespace,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESOLUTION_REPORT_PATH = os.path.join(PROCESSED_DIR, "model_resolution_report.json")
PULL_REPORT_PATH = os.path.join(PROCESSED_DIR, "pull_report.csv")

# Recall text field candidates (NHTSA API)
RECALL_TEXT_FIELDS = ["Summary", "Consequence", "Remedy", "Notes", "Defect", "Description"]

# Complaint narrative field candidates
COMPLAINT_TEXT_FIELDS = ["summary", "Summary", "narrative", "Narrative"]


def _query_id(make: str, model: str, year: int, text: str) -> str:
    """Stable hash for query_id."""
    raw = f"{make}|{model}|{year}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _load_raw_complaints(make: str, model: str, year: int) -> list[dict]:
    """Load complaints from cached raw file if it exists."""
    from .utils import safe_slug
    slug = safe_slug(f"{make}_{model}_{year}")
    path = os.path.join(RAW_COMPLAINTS_DIR, f"{slug}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results = data.get("results")
            return results if isinstance(results, list) else []
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _load_resolution_table(use_resolver: bool) -> dict[tuple[str, str, int], dict]:
    """Load or generate model resolution table. Returns dict keyed by (make, model, year)."""
    if not use_resolver:
        return {}
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if not os.path.exists(RESOLUTION_REPORT_PATH):
        logger.info("Generating model resolution report (run model_resolver if needed)...")
        from .model_resolver import build_resolution_table
        table = build_resolution_table(MODELS)
        with open(RESOLUTION_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(table, f, indent=2, ensure_ascii=False)
    else:
        with open(RESOLUTION_REPORT_PATH, "r", encoding="utf-8") as f:
            table = json.load(f)
    lookup = {}
    for r in table:
        key = (r["make_in"], r["model_in"], r["year"])
        lookup[key] = r
    return lookup


def _filter_global_to_targets(global_rows: list[dict]) -> list[dict]:
    """Filter corpus_global chunks to target make/model (and year if available)."""
    filtered = []
    year_approx_count = 0
    for row in global_rows:
        make = (row.get("make") or "").upper()
        model = (row.get("model") or "").upper().replace("-", " ")
        year = row.get("year")

        for m in MODELS:
            tmake = m["make"].upper()
            tmodel = m["model"].upper().replace("-", " ")
            if make != tmake:
                continue
            # Model: empty matches (broad), or fuzzy match
            if model and tmodel not in model and model not in tmodel:
                continue
            # Year: if present, check range; if absent, include (approximate)
            if year is not None:
                if m["year_start"] <= year <= m["year_end"]:
                    filtered.append(row)
                    break
                # year outside range - skip
            else:
                filtered.append(row)
                year_approx_count += 1
                break
    if year_approx_count:
        logger.info("Year filtering approximate: %d global chunks have no year info", year_approx_count)
    return filtered


def _load_raw_recalls(make: str, model: str, year: int) -> list[dict]:
    """Load recalls from cached raw file if it exists."""
    from .utils import safe_slug
    slug = safe_slug(f"{make}_{model}_{year}")
    path = os.path.join(RAW_RECALLS_DIR, f"{slug}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results = data.get("results")
            return results if isinstance(results, list) else []
        except (json.JSONDecodeError, OSError):
            pass
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Build complaints and corpus from NHTSA data")
    parser.add_argument("--from-cache", action="store_true", help="Skip API calls, use only cached raw files")
    parser.add_argument("--no-use-resolver", dest="use_resolver", action="store_false", default=True,
                        help="Disable model resolver (use config make/model as-is)")
    args = parser.parse_args()

    resolution = _load_resolution_table(args.use_resolver)

    complaints_rows: list[dict] = []
    corpus_rows: list[dict] = []
    per_model_counts: dict[str, dict[str, int]] = {}
    pull_report_rows: list[dict] = []

    for m in tqdm(MODELS, desc="Models"):
        make, model = m["make"], m["model"]
        key = f"{make} {model}"
        per_model_counts[key] = {"complaints": 0, "chunks": 0}

        for year in range(m["year_start"], m["year_end"] + 1):
            res = resolution.get((make, model, year), {})
            # Use resolved make/model for each endpoint separately; only if not None
            complaints_make = res.get("make_used_complaints") if res.get("make_used_complaints") is not None else make
            complaints_model = res.get("model_used_complaints") if res.get("model_used_complaints") is not None else model
            recalls_make = res.get("make_used_recalls") if res.get("make_used_recalls") is not None else make
            recalls_model = res.get("model_used_recalls")  # None if resolver failed - do NOT substitute

            # If recalls_model is None, do NOT substitute; log FAIL and skip recalls
            if recalls_model is None:
                logger.info("Resolver FAIL: %s %s %d -> recalls_model=None, skipping recalls", make, model, year)
                recalls_make = make  # for report

            if complaints_make != make or complaints_model != model or recalls_make != make or recalls_model != model:
                logger.info(
                    "Resolver: %s %s %d -> complaints=(%s,%s) recalls=(%s,%s)",
                    make, model, year, complaints_make, complaints_model, recalls_make, recalls_model,
                )

            complaints_err = None
            recalls_err = None

            def _get_complaints() -> list[dict]:
                nonlocal complaints_err
                if args.from_cache:
                    out = _load_raw_complaints(complaints_make, complaints_model, year)
                    return out
                try:
                    out = fetch_complaints(complaints_make, complaints_model, year)
                    return out
                except Exception as e:
                    complaints_err = str(e)
                    logger.warning("Complaints fetch failed for %s %s %d: %s", make, model, year, e)
                    return _load_raw_complaints(complaints_make, complaints_model, year)

            def _get_recalls() -> list[dict]:
                nonlocal recalls_err
                if recalls_model is None:
                    return []
                if args.from_cache:
                    out = _load_raw_recalls(recalls_make, recalls_model, year)
                    return out
                try:
                    out = fetch_recalls(recalls_make, recalls_model, year)
                    return out
                except Exception as e:
                    recalls_err = str(e)
                    logger.warning("Recalls fetch failed for %s %s %d: %s", make, model, year, e)
                    return _load_raw_recalls(recalls_make, recalls_model, year)

            complaints = _get_complaints()
            recalls = _get_recalls()

            complaints_status = "ok" if complaints else "empty"
            recalls_status = "ok" if recalls else "empty"
            error_msg = "; ".join(
                s for s in [
                    f"complaints: {complaints_err}" if complaints_err else None,
                    f"recalls: {recalls_err}" if recalls_err else None,
                ] if s
            ) or ""

            pull_report_rows.append({
                "make_in": make,
                "model_in": model,
                "year": year,
                "make_used_complaints": complaints_make,
                "model_used_complaints": complaints_model,
                "complaints_status": complaints_status,
                "complaints_count": len(complaints),
                "make_used_recalls": recalls_make,
                "model_used_recalls": recalls_model,
                "recalls_status": recalls_status,
                "recalls_count": len(recalls),
                "error_message": error_msg,
            })

            for c in complaints:
                complaint_text = extract_best_text_fields(c, COMPLAINT_TEXT_FIELDS)
                if not complaint_text or len(complaint_text.strip()) < 10:
                    continue
                qid = _query_id(make, model, year, complaint_text)
                complaints_rows.append({
                    "query_id": qid,
                    "make": make,
                    "model": model,
                    "year": year,
                    "make_norm": normalize_make(make),
                    "model_norm": normalize_model(model),
                    "model_key": model_key(model),
                    "complaint_text": complaint_text,
                    "raw_source": "nhtsa_complaints",
                })
                per_model_counts[key]["complaints"] += 1

            for rec in recalls:
                doc_text = extract_best_text_fields(rec, RECALL_TEXT_FIELDS)
                if not doc_text:
                    continue
                campaign = rec.get("NHTSACampaignNumber") or rec.get("CampaignNumber") or "unknown"
                component = rec.get("Component") or ""
                chunks = chunk_text(doc_text, max_words=250, overlap_words=50)

                for idx, chunk in enumerate(chunks):
                    word_count = len(chunk.split())
                    if word_count < 40:
                        continue
                    doc_id = f"recall_{campaign}_{idx}"
                    corpus_rows.append({
                        "doc_id": doc_id,
                        "make": make,
                        "model": model,
                        "year": year,
                        "make_norm": normalize_make(make),
                        "model_norm": normalize_model(model),
                        "model_key": model_key(model),
                        "campaign_number": campaign,
                        "component": component,
                        "text": chunk,
                        "raw_source": "nhtsa_recalls",
                    })
                    per_model_counts[key]["chunks"] += 1

    # Ensure output dirs exist
    import os
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RAW_COMPLAINTS_DIR, exist_ok=True)
    os.makedirs(RAW_RECALLS_DIR, exist_ok=True)

    complaints_path = os.path.join(PROCESSED_DIR, "complaints.jsonl")
    corpus_path = os.path.join(PROCESSED_DIR, "corpus.jsonl")
    corpus_global_path = os.path.join(PROCESSED_DIR, "corpus_global.jsonl")
    corpus_merged_path = os.path.join(PROCESSED_DIR, "corpus_merged.jsonl")

    jsonl_write(complaints_path, complaints_rows)
    jsonl_write(corpus_path, corpus_rows)

    # Merge with corpus_global if present (filter by target make/model, approximate year)
    merged = list(corpus_rows)
    if os.path.exists(corpus_global_path):
        global_rows: list[dict] = []
        with open(corpus_global_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    global_rows.append(json.loads(line))
        filtered = _filter_global_to_targets(global_rows)
        merged = list(corpus_rows) + filtered
        logger.info("Merged: %d from corpus + %d from corpus_global (filtered) -> %d total",
                    len(corpus_rows), len(filtered), len(merged))
    jsonl_write(corpus_merged_path, merged)
    logger.info("Wrote %s", corpus_merged_path)

    # Write pull report
    if pull_report_rows:
        with open(PULL_REPORT_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "make_in", "model_in", "year",
                "make_used_complaints", "model_used_complaints", "complaints_status", "complaints_count",
                "make_used_recalls", "model_used_recalls", "recalls_status", "recalls_count",
                "error_message",
            ])
            writer.writeheader()
            writer.writerows(pull_report_rows)
        logger.info("Wrote %s", PULL_REPORT_PATH)

    # Print counts
    logger.info("--- Summary ---")
    logger.info("Total complaints kept: %d", len(complaints_rows))
    logger.info("Total recall chunks produced: %d", len(corpus_rows))
    logger.info("Per make/model:")
    for k, v in per_model_counts.items():
        logger.info("  %s: complaints=%d, chunks=%d", k, v["complaints"], v["chunks"])


if __name__ == "__main__":
    main()
