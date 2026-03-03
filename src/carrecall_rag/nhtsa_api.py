"""NHTSA API client for complaints and recalls. Uses requests, no scraping."""

import json
import logging
import os
import time

import requests

from .config import NHTSA_COMPLAINTS_URL, NHTSA_RECALLS_URL, RAW_COMPLAINTS_DIR, RAW_RECALLS_DIR
from .utils import safe_slug

logger = logging.getLogger(__name__)


def fetch_json(url: str, params: dict, timeout: int = 30, retries: int = 3) -> dict | None:
    """Fetch JSON from URL with exponential backoff on failure."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            wait = 2 ** attempt
            logger.warning("Request failed (attempt %d/%d): %s. Retrying in %ds.", attempt + 1, retries, e, wait)
            time.sleep(wait)
    logger.error("All %d retries failed for %s", retries, url)
    return None


def _save_raw_response(data: dict, out_path: str) -> None:
    """Save raw API response to JSON file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_cached(path: str) -> dict | None:
    """Load cached JSON if file exists."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load cached %s: %s", path, e)
    return None


def fetch_complaints(make: str, model: str, year: int) -> list[dict]:
    """Fetch complaints for a vehicle. Saves raw response to data/raw/complaints/."""
    slug = safe_slug(f"{make}_{model}_{year}")
    out_path = os.path.join(RAW_COMPLAINTS_DIR, f"{slug}.json")
    params = {"make": make, "model": model, "modelYear": str(year)}
    data = fetch_json(NHTSA_COMPLAINTS_URL, params)
    if data is None:
        data = _load_cached(out_path)
    if data is None:
        return []
    _save_raw_response(data, out_path)
    results = data.get("results")
    return results if isinstance(results, list) else []


def fetch_recalls(make: str, model: str, year: int) -> list[dict]:
    """Fetch recalls for a vehicle. Saves raw response to data/raw/recalls/."""
    slug = safe_slug(f"{make}_{model}_{year}")
    out_path = os.path.join(RAW_RECALLS_DIR, f"{slug}.json")
    params = {"make": make, "model": model, "modelYear": str(year)}
    data = fetch_json(NHTSA_RECALLS_URL, params)
    if data is None:
        data = _load_cached(out_path)
    if data is None:
        return []
    _save_raw_response(data, out_path)
    results = data.get("results")
    return results if isinstance(results, list) else []
