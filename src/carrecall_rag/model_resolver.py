"""Canonical make/model resolution using NHTSA products/vehicle endpoints.
Milestone 0.7: make resolution + strict model-family gating."""

import json
import logging
import os
import re
from difflib import get_close_matches

import requests

from .config import MODELS, PROCESSED_DIR

logger = logging.getLogger(__name__)

NHTSA_MAKES_URL = "https://api.nhtsa.gov/products/vehicle/makes"
NHTSA_MODELS_URL = "https://api.nhtsa.gov/products/vehicle/models"
NHTSA_COMPLAINTS_URL = "https://api.nhtsa.gov/complaints/complaintsByVehicle"
NHTSA_RECALLS_URL = "https://api.nhtsa.gov/recalls/recallsByVehicle"


def norm(s: str) -> str:
    """Lowercase, remove all non-alphanumeric."""
    if not s or not isinstance(s, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def model_family_ok(desired_model: str, candidate_model: str) -> bool:
    """Strict model-family gating. Returns True if candidate is in same family as desired."""
    d = norm(desired_model)
    c = norm(candidate_model)
    if not d:
        return True
    # CIVIC => require "civic" in candidate
    if d == "civic":
        return "civic" in c
    # CAMRY => require "camry"
    if d == "camry":
        return "camry" in c
    # GRAND CHEROKEE => require "grand" and "cherokee"
    if d == "grandcherokee":
        return "grand" in c and "cherokee" in c
    # F-150 => require one of ["f150","f-150","f 150"]
    if d == "f150":
        return "f150" in c
    # 3 SERIES => require ("3" and "series") or "3series" or 3xx pattern (320, 328, 335)
    if d == "3series":
        return (
            ("3" in c and "series" in c)
            or "3series" in c
            or (re.match(r"^3\d{2}", c) is not None)  # 320i, 328i, 335i
        )
    # C-CLASS => require "cclass" in candidate (excludes CLA-CLASS)
    if d == "cclass":
        return "cclass" in c
    # Unknown: default True
    return True


def probe_endpoint(
    endpoint_type: str, make: str, model: str, year: int
) -> tuple[int, int]:
    """Probe NHTSA endpoint. Returns (status_code, n_results)."""
    if endpoint_type == "complaints":
        url = NHTSA_COMPLAINTS_URL
    elif endpoint_type == "recalls":
        url = NHTSA_RECALLS_URL
    else:
        return (0, 0)
    params = {"make": make, "model": model, "modelYear": str(year)}
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            return (resp.status_code, 0)
        data = resp.json()
        results = data.get("results") or data.get("Results")
        n = len(results) if isinstance(results, list) else 0
        return (200, n)
    except Exception:
        return (0, 0)


def get_makes_for_year(issue_type: str, year: int) -> list[str]:
    """Fetch make names from NHTSA for a year. issue_type: 'c' complaints, 'r' recalls."""
    params = {"modelYear": str(year), "issueType": issue_type}
    try:
        resp = requests.get(NHTSA_MAKES_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Failed to fetch makes for %d %s: %s", year, issue_type, e)
        return []
    results = data.get("results") or data.get("Results")
    if not isinstance(results, list):
        return []
    makes = []
    seen = set()
    for r in results:
        m = r.get("make") if isinstance(r, dict) else None
        if m and isinstance(m, str) and m.strip():
            key = m.strip().upper()
            if key not in seen:
                seen.add(key)
                makes.append(m.strip())
    return makes


def resolve_make(
    desired_make: str, year: int, desired_model: str, endpoint_type: str
) -> tuple[str | None, int, int]:
    """Resolve make for endpoint. Returns (resolved_make, status_code, n_results).
    Validates by probing with desired_model."""
    makes_c = get_makes_for_year(endpoint_type, year)
    nd = norm(desired_make)
    candidates = [desired_make.strip()] if desired_make.strip() else []
    for m in makes_c:
        if m not in candidates and norm(m) == nd:
            candidates.append(m)
    for m in makes_c:
        if m in candidates:
            continue
        nm = norm(m)
        if nd in nm or nm in nd:
            candidates.append(m)
    suggested = get_close_matches(desired_make, makes_c, n=10, cutoff=0.0)
    for s in suggested:
        if s not in candidates:
            candidates.append(s)

    for make_cand in candidates[:15]:
        status_code, n_results = probe_endpoint(endpoint_type, make_cand, desired_model, year)
        if status_code == 200:
            return (make_cand, status_code, n_results)
    # Return first failed probe for reporting
    if candidates:
        sc, n = probe_endpoint(endpoint_type, candidates[0], desired_model, year)
        return (None, sc, n)
    return (None, 0, 0)


def get_models_for_make_year(make: str, year: int, issue_type: str) -> list[str]:
    """Fetch model names from NHTSA for a make/year. issue_type: 'c' complaints, 'r' recalls."""
    params = {"modelYear": str(year), "make": make, "issueType": issue_type}
    try:
        resp = requests.get(NHTSA_MODELS_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Failed to fetch models for %s %d %s: %s", make, year, issue_type, e)
        return []
    results = data.get("results") or data.get("Results")
    if not isinstance(results, list):
        return []
    models = []
    seen = set()
    for r in results:
        m = r.get("model") if isinstance(r, dict) else None
        if m and isinstance(m, str) and m.strip():
            key = m.strip().upper()
            if key not in seen:
                seen.add(key)
                models.append(m.strip())
    return models


def resolve_model(make: str, desired_model: str, year: int) -> dict:
    """Resolve make+model for complaints and recalls. Uses model_family_ok gating."""
    complaint_models = get_models_for_make_year(make, year, "c")
    recall_models = get_models_for_make_year(make, year, "r")
    all_candidates_pool = list(dict.fromkeys(complaint_models + recall_models))
    nd = norm(desired_model)

    # Build model candidate list (exact, contains, difflib) - will filter by model_family_ok
    model_candidates = []
    if desired_model.strip():
        model_candidates.append(desired_model.strip())
    for c in all_candidates_pool:
        if c not in model_candidates and norm(c) == nd:
            model_candidates.append(c)
    for c in all_candidates_pool:
        if c in model_candidates:
            continue
        nc = norm(c)
        if nd in nc or nc in nd:
            model_candidates.append(c)
    suggested = get_close_matches(desired_model, all_candidates_pool, n=10, cutoff=0.0)
    for s in suggested:
        if s not in model_candidates:
            model_candidates.append(s)

    # Get make candidates
    makes_c = get_makes_for_year("c", year)
    makes_r = get_makes_for_year("r", year)
    nd_make = norm(make)
    make_candidates = [make.strip()] if make.strip() else []
    for m in makes_c + makes_r:
        if m not in make_candidates and norm(m) == nd_make:
            make_candidates.append(m)
    for m in makes_c + makes_r:
        if m in make_candidates:
            continue
        nm = norm(m)
        if nd_make in nm or nm in nd_make:
            make_candidates.append(m)
    make_suggested = get_close_matches(make, list(dict.fromkeys(makes_c + makes_r)), n=10, cutoff=0.0)
    for s in make_suggested:
        if s not in make_candidates:
            make_candidates.append(s)

    # Complaints: try (make, model) pairs - make first, then model, with model_family_ok gating
    complaints_make = None
    complaints_model = None
    complaints_status_code = None
    complaints_n_results = 0
    complaints_candidates_tried = []
    family_ok_blocked_complaints = False

    for make_cand in make_candidates[:10]:
        for model_cand in model_candidates[:15]:
            if not model_family_ok(desired_model, model_cand):
                family_ok_blocked_complaints = True
                continue
            complaints_candidates_tried.append((make_cand, model_cand))
            if len(complaints_candidates_tried) > 15:
                complaints_candidates_tried = complaints_candidates_tried[:15]
            status_code, n_results = probe_endpoint("complaints", make_cand, model_cand, year)
            if complaints_status_code is None:
                complaints_status_code = status_code
                complaints_n_results = n_results
            if status_code == 200:
                complaints_make = make_cand
                complaints_model = model_cand
                complaints_status_code = status_code
                complaints_n_results = n_results
                break
        if complaints_model is not None:
            break

    complaints_candidates_tried = [f"{m}/{mo}" for m, mo in complaints_candidates_tried[:15]]

    # Recalls: same logic
    recalls_make = None
    recalls_model = None
    recalls_status_code = None
    recalls_n_results = 0
    recalls_candidates_tried = []
    family_ok_blocked_recalls = False

    for make_cand in make_candidates[:10]:
        for model_cand in model_candidates[:15]:
            if not model_family_ok(desired_model, model_cand):
                family_ok_blocked_recalls = True
                continue
            recalls_candidates_tried.append((make_cand, model_cand))
            if len(recalls_candidates_tried) > 15:
                recalls_candidates_tried = recalls_candidates_tried[:15]
            status_code, n_results = probe_endpoint("recalls", make_cand, model_cand, year)
            if recalls_status_code is None:
                recalls_status_code = status_code
                recalls_n_results = n_results
            if status_code == 200:
                recalls_make = make_cand
                recalls_model = model_cand
                recalls_status_code = status_code
                recalls_n_results = n_results
                break
        if recalls_model is not None:
            break

    recalls_candidates_tried = [f"{m}/{mo}" for m, mo in recalls_candidates_tried[:15]]

    # Status FAIL if recalls_model is None (including when model_family_ok blocked all)
    status = "OK" if recalls_model is not None else "FAIL"
    if recalls_model is None and (family_ok_blocked_recalls or family_ok_blocked_complaints):
        status = "FAIL"

    return {
        "make_in": make,
        "model_in": desired_model,
        "year": year,
        "make_used_complaints": complaints_make,
        "model_used_complaints": complaints_model,
        "make_used_recalls": recalls_make,
        "model_used_recalls": recalls_model,
        "status": status,
        "complaints_status_code": complaints_status_code,
        "complaints_n_results": complaints_n_results,
        "recalls_status_code": recalls_status_code,
        "recalls_n_results": recalls_n_results,
        "complaints_candidates_tried": complaints_candidates_tried,
        "recalls_candidates_tried": recalls_candidates_tried,
        # Backwards compat
        "complaints_model": complaints_model,
        "recalls_model": recalls_model,
    }


def build_resolution_table(models_config: list[dict]) -> list[dict]:
    """Build resolution table for all (make, model, year) in config."""
    from tqdm import tqdm
    table = []
    entries = [
        (m["make"], m["model"], year)
        for m in models_config
        for year in range(m["year_start"], m["year_end"] + 1)
    ]
    for make, model, year in tqdm(entries, desc="Resolving models"):
        table.append(resolve_model(make, model, year))
    return table


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    table = build_resolution_table(MODELS)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    report_path = os.path.join(PROCESSED_DIR, "model_resolution_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)

    logger.info("Wrote %s", report_path)

    ok_count = sum(1 for r in table if r.get("recalls_model") is not None)
    fail_count = sum(1 for r in table if r.get("recalls_model") is None)

    logger.info("--- Summary ---")
    logger.info("OK (recalls_model set): %d  FAIL (recalls_model None): %d", ok_count, fail_count)

    if fail_count > 0:
        fails = [r for r in table if r.get("recalls_model") is None]
        top_10 = fails[:10]
        logger.info("Top 10 FAIL items:")
        for r in top_10:
            logger.info(
                "  %s %s %d -> recalls_status=%s n=%s tried=%s",
                r["make_in"],
                r["model_in"],
                r["year"],
                r.get("recalls_status_code"),
                r.get("recalls_n_results"),
                r.get("recalls_candidates_tried", [])[:3],
            )


if __name__ == "__main__":
    main()
