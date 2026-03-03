# CarRecall-RAG

Milestone 0: Download NHTSA complaints + recalls, build evidence corpus, run BM25 baseline retrieval sanity check.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Commands

0. **Model resolution** (optional; build_corpus auto-generates if missing):

```bash
python -m carrecall_rag.model_resolver
```

Writes `data/processed/model_resolution_report.json` and prints OK/NO_MATCH summary.

1. **Build corpus** (downloads from NHTSA APIs, builds complaints.jsonl + corpus.jsonl):

```bash
python -m carrecall_rag.build_corpus
```

Uses model resolver by default to resolve canonical make/model (e.g. F-150 → F-150 SUPERCAB). Disable with `--no-use-resolver`.

To build from cached raw files only (skip API calls, for quick local testing):

```bash
python -m carrecall_rag.build_corpus --from-cache
```

2. **BM25 sanity check** (sample random complaints, retrieve top-K chunks):

```bash
python -m carrecall_rag.bm25_sanity --n 30 --k 10
```

With custom seed:

```bash
python -m carrecall_rag.bm25_sanity --n 30 --k 10 --seed 13
```

## Output Artifacts

- `data/raw/complaints/` — raw JSON API responses (complaints)
- `data/raw/recalls/` — raw JSON API responses (recalls)
- `data/processed/complaints.jsonl` — complaint rows with query_id, complaint_text
- `data/processed/corpus.jsonl` — recall chunk rows for retrieval
- `data/processed/bm25_sanity_results.jsonl` — BM25 retrieval results

## Data Source

NHTSA public APIs only (no scraping). Uses `requests` to call:

- `https://api.nhtsa.gov/complaints/complaintsByVehicle`
- `https://api.nhtsa.gov/recalls/recallsByVehicle`
