"""CLI: Build corpus_global.jsonl from NHTSA global recalls."""

import json
import logging
import os

from tqdm import tqdm

from .config import PROCESSED_DIR, RAW_RECALLS_GLOBAL_DIR
from .global_recalls import download_global_recalls
from .utils import chunk_text, jsonl_write, model_key, normalize_make, normalize_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

GLOBAL_RAW_PATH = os.path.join(RAW_RECALLS_GLOBAL_DIR, "recalls_global.jsonl")
CORPUS_GLOBAL_PATH = os.path.join(PROCESSED_DIR, "corpus_global.jsonl")

TEXT_FIELDS = ["summary", "description", "consequence", "remedy"]


def _doc_text(record: dict) -> str:
    """Concatenate text fields for chunking."""
    parts = []
    for key in TEXT_FIELDS:
        val = record.get(key)
        if val and isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return " ".join(parts) if parts else ""


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build corpus_global.jsonl from NHTSA global recalls")
    parser.add_argument("--max-records", type=int, default=None, help="Limit records to fetch (for testing)")
    args = parser.parse_args()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RAW_RECALLS_GLOBAL_DIR, exist_ok=True)

    if not os.path.exists(GLOBAL_RAW_PATH):
        logger.info("Global recalls not found, downloading...")
        download_global_recalls(max_records=args.max_records)
    else:
        logger.info("Loading global recalls from %s", GLOBAL_RAW_PATH)

    records: list[dict] = []
    skipped_count = 0
    with open(GLOBAL_RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                skipped_count += 1
    logger.info("Loaded %d records, skipped %d lines", len(records), skipped_count)

    corpus_rows: list[dict] = []
    for rec in tqdm(records, desc="Chunking global recalls"):
        doc_text = _doc_text(rec)
        if not doc_text or len(doc_text.strip()) < 40:
            continue

        campaign = rec.get("campaign_number") or "unknown"
        make = rec.get("make", "")
        model = rec.get("model", "")
        years = rec.get("years") or []
        component = rec.get("component", "")

        chunks = chunk_text(doc_text, max_words=250, overlap_words=50)
        for idx, chunk in enumerate(chunks):
            word_count = len(chunk.split())
            if word_count < 40:
                continue
            doc_id = f"recall_global_{campaign}_{idx}"
            year = years[0] if years else None
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
                "raw_source": "nhtsa_recalls_global",
            })

    jsonl_write(CORPUS_GLOBAL_PATH, corpus_rows)
    logger.info("Wrote %d chunks to %s", len(corpus_rows), CORPUS_GLOBAL_PATH)


if __name__ == "__main__":
    main()
