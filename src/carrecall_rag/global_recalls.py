"""Download broad recall dataset from NHTSA /recalls API (not per-vehicle)."""

import json
import logging
import os
import re
import time

import requests

from .config import RAW_RECALLS_GLOBAL_DIR

logger = logging.getLogger(__name__)

NHTSA_RECALLS_GLOBAL_URL = "https://api.nhtsa.gov/recalls"

# Map manufacturer names to canonical make
MAKE_ALIASES = {
    "honda": "HONDA",
    "american honda": "HONDA",
    "toyota": "TOYOTA",
    "toyota motor": "TOYOTA",
    "ford": "FORD",
    "ford motor": "FORD",
    "jeep": "JEEP",
    "chrysler": "JEEP",  # FCA
    "fca": "JEEP",
    "stellantis": "JEEP",
    "bmw": "BMW",
    "mercedes": "MERCEDES-BENZ",
    "mercedes-benz": "MERCEDES-BENZ",
    "mercedes benz": "MERCEDES-BENZ",
    "daimler": "MERCEDES-BENZ",
}


def _normalize_make(manufacturer: str) -> str:
    """Map manufacturer name to canonical make."""
    if not manufacturer:
        return ""
    s = manufacturer.lower().strip()
    for alias, canonical in MAKE_ALIASES.items():
        if alias in s:
            return canonical
    return manufacturer.strip().upper()


def _extract_years(text: str) -> list[int]:
    """Extract year mentions from text. Returns list of years found."""
    if not text:
        return []
    years = set()
    # Match 20XX or 19XX
    for m in re.finditer(r"\b(19|20)(\d{2})\b", text):
        years.add(int(m.group(1) + m.group(2)))
    return sorted(years)


def _extract_model_hint(description: str, manufacturer: str) -> str:
    """Extract model hint from description. Approximate."""
    if not description:
        return ""
    # Common pattern: "certain 2021-2026 F-150" or "certain 2025 Telluride"
    # Look for model-like words after "certain" and year
    desc = description
    # Remove manufacturer prefix
    for mfr in [manufacturer, "recalling", "certain"]:
        desc = re.sub(re.escape(mfr), "", desc, flags=re.I)
    # Try to find model names (capitalized words, possibly with hyphen/numbers)
    models = ["CIVIC", "CAMRY", "F-150", "F150", "GRAND CHEROKEE", "CHEROKEE", "3 SERIES", "C-CLASS"]
    desc_upper = desc.upper()
    for model in models:
        if model.replace("-", " ") in desc_upper or model in desc_upper:
            return model
    return ""


def fetch_global_recalls(page_size: int = 100, max_records: int | None = None) -> list[dict]:
    """Fetch all recalls from NHTSA /recalls API. Returns list of uniform records."""
    all_records = []
    offset = 0
    total = None

    while True:
        try:
            resp = requests.get(
                NHTSA_RECALLS_GLOBAL_URL,
                params={"format": "json", "max": page_size, "offset": offset},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("Fetch failed at offset %d: %s", offset, e)
            break

        meta = data.get("meta", {})
        pagination = meta.get("pagination", {})
        total = pagination.get("total", 0)
        results = data.get("results", [])

        if not results:
            break

        for r in results:
            manufacturer = r.get("manufacturerName", "")
            make = _normalize_make(manufacturer)
            description = r.get("description", "") or ""
            consequence = r.get("consequence", "") or ""
            corrective = r.get("correctiveAction", "") or ""
            subject = r.get("subject", "") or ""
            campaign = r.get("campaignId", "") or r.get("nhtsaCampaignNumber", "") or ""

            doc_text = " ".join(
                x for x in [subject, description, consequence, corrective] if x
            ).strip()

            years = _extract_years(doc_text)
            model_hint = _extract_model_hint(description, manufacturer)

            record = {
                "campaign_number": campaign,
                "make": make or manufacturer,
                "model": model_hint,
                "years": years,
                "component": subject[:200] if subject else "",
                "summary": description[:500] if description else "",
                "description": description,
                "consequence": consequence,
                "remedy": corrective,
                "raw_source": "nhtsa_recalls_global",
            }
            all_records.append(record)

        offset += len(results)
        if max_records and len(all_records) >= max_records:
            all_records = all_records[:max_records]
            break
        if offset >= total:
            break

        time.sleep(0.2)  # rate limit

    return all_records


def download_global_recalls(
    out_dir: str | None = None,
    max_records: int | None = None,
) -> list[dict]:
    """Download global recalls and save raw payload. Returns parsed records."""
    out_dir = out_dir or RAW_RECALLS_GLOBAL_DIR
    os.makedirs(out_dir, exist_ok=True)

    logger.info("Fetching global recalls from NHTSA API...")
    records = fetch_global_recalls(page_size=100, max_records=max_records)

    out_path = os.path.join(out_dir, "recalls_global.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Saved %d records to %s", len(records), out_path)
    return records
