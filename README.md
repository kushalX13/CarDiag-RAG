# CarDiag-RAG

[GitHub](https://github.com/kushalX13/CarDiag-RAG)

**CarDiag-RAG** is a hybrid retrieval system that maps free-text vehicle symptom descriptions—plus make/model/year context—to likely NHTSA recall campaigns. Symptom-to-recall matching is treated as a **ranking problem**: the system returns a ranked list of recall campaigns with optional grounded explanations and is evaluated with retrieval metrics (Recall@K, MRR). The pipeline combines dense retrieval (SentenceTransformer + FAISS), BM25 keyword retrieval, score fusion, campaign-level aggregation, and an optional cross-encoder reranker.

---

## Results snapshot

### Small benchmark (`eval/recall_queries.jsonl`)

| Metric      | Value   |
|------------|---------|
| Recall@1   | 0.90    |
| Recall@10  | 1.00    |
| MRR        | 0.95    |
| Eval set   | 10 queries |

### Extended fixed benchmark (`eval/recall_queries_100_fixed.jsonl`)

| Metric      | Value   |
|------------|---------|
| Recall@1   | 0.94    |
| Recall@10  | 1.00    |
| MRR        | 0.9671  |
| Eval set   | 81 queries |

Baseline: hybrid retrieval (α=0.5), no rerank. Test set: `eval/recall_queries.jsonl`.
Both benchmark sets are versioned in git:
- `eval/recall_queries.jsonl` (original 10-query benchmark)
- `eval/recall_queries_100_fixed.jsonl` (extended cleaned benchmark derived from the original ~100-query generation pipeline)

**Reproducibility** — Python 3.10+. From project root (with corpus and indexes in place):

```bash
pip install -e . && ./scripts/run_eval.sh
```

The script prints Recall@1, Recall@10, and MRR in the **CV-ready metrics** block at the end.

**Not in the repo (regenerate after clone or when moving off a machine):** The `data/` directory is gitignored. You need:

| What | Where | How to create |
|------|--------|----------------|
| Processed corpus | `data/processed/corpus_merged.jsonl` | `python -m carrecall_rag.build_corpus` (fetches NHTSA; optional: `build_corpus_global` then re-run merge step in build_corpus) |
| FAISS + BM25 indexes | `data/indexes/*.faiss`, `*_mapping.json` | Built automatically by demo/eval if missing, using corpus + model |
| Encoder model | `data/models/biencoder/` | Train: `python -m carrecall_rag.train_biencoder` (needs `data/processed/train_triples.jsonl` from `build_triples`). Or copy a SentenceTransformer model (e.g. `sentence-transformers/all-MiniLM-L6-v2`) into this path. |

So after a fresh clone: install deps, then run `build_corpus` (and optionally `build_corpus_global`), put a model in `data/models/biencoder/`, then run demo or eval; indexes will be built on first run if absent.

---

## Why this problem is hard

- **Noisy, informal user input** — Drivers and technicians describe symptoms in free text (“fuel starvation,” “brake fluid leaking,” “won’t stay in park”). Wording is inconsistent and often lacks technical terms.
- **Formal, technical recall text** — NHTSA recall descriptions are legal and structured. Lexical overlap with casual complaints is low; semantic matching is required.
- **One-to-many and many-to-one** — A single symptom can match several campaigns; similar campaigns share overlapping language. The task is ranking and retrieval, not a single-label classification.
- **Evaluation** — We measure whether the correct campaign appears in the top-K and how high it ranks (Recall@K, MRR), not precision/F1 on a fixed label set.

---

## Architecture

End-to-end flow (plain text):

```
User query + vehicle metadata (make, model, year)
  → Index selection (pool by make+model → make-only → global fallback)
  → Dense retrieval (SentenceTransformer + FAISS, top-K)
  → BM25 keyword retrieval (same pool, top-K)
  → Score fusion: (1 − α)·dense + α·keyword (default α = 0.5)
  → Campaign aggregation (group by campaign_number, keep best doc score per campaign)
  → Optional: cross-encoder rerank on top-N candidates
  → Ranked list of recall campaigns (+ optional template-based answer)
```

Corpus: `data/processed/corpus_merged.jsonl` (doc_id, campaign_number, make_norm, model_key, text). FAISS and BM25 indexes are built over it; pool and make-only indexes are used when doc count ≥ `min_pool_docs`.

---

## Example: input and output

**Input**

| Field  | Value |
|--------|--------|
| Make   | Jeep |
| Model  | Grand Cherokee |
| Query  | fuel starvation HPFP failure |

**Illustrative output** (from the demo). Full example: [demo_outputs/hpfp.txt](demo_outputs/hpfp.txt).

- **Best match:** 22V406000 — High Pressure Fuel Pump Failure  
- **Why it matches:** The high pressure fuel pump (HPFP) may fail, resulting in fuel starvation. This closely matches your query because it directly connects HPFP failure with fuel starvation and engine stall.  
- **Other candidates:** e.g. Crankshaft Position Sensor–related campaign (lower-confidence).  
- **Safety risk:** Engine stall and sudden loss of propulsion while driving, which may increase crash risk.  
- **Suggested next step:** Contact your dealer or look up your VIN at NHTSA.gov/recalls.

The demo uses template-based answer generation from the top retrieved campaigns (no LLM).

---

## Quick start

From project root after `pip install -e .`:

```bash
# Run demo (one query → ranked recalls + short answer)
carrecall-demo --make Jeep --model "Grand Cherokee" --query "fuel starvation HPFP failure"

# Run full evaluation (Recall@K, MRR over labeled set)
./scripts/run_eval.sh
```

---

## Commands

| What | Command |
|------|---------|
| **Demo** (hybrid) | `carrecall-demo --make Jeep --model "Grand Cherokee" --query "fuel starvation HPFP failure"` |
| **Demo + rerank** | Add `--rerank --rerank-topn 50` to the above |
| **Eval** (small 10-query benchmark) | `python -m carrecall_rag.eval --eval-file eval/recall_queries.jsonl --output eval/results/retrieval_debug.jsonl --mode hybrid --alpha 0.5 --dense-topk 100 --keyword-topk 150 --topc 10` |
| **Eval** (script) | `./scripts/run_eval.sh` |
| **Compare rerank formats** | `python -m carrecall_rag.eval --eval-file eval/recall_queries.jsonl --compare-table --dense-topk 100 --keyword-topk 150 --rerank-topn 50` |
| **Compare methods** (dense/keyword/hybrid/rerank) | `python -m carrecall_rag.eval --eval-file eval/recall_queries.jsonl --compare-methods --compare-methods-output eval/results/comparison_methods.md` |
| **Eval** (extended fixed benchmark) | `python -m carrecall_rag.eval --eval-file eval/recall_queries_100_fixed.jsonl --output eval/results/retrieval_debug_100_fixed.jsonl` |
| **Eval** (hard benchmark) | `python -m carrecall_rag.eval --eval-file eval/recall_queries_hard.jsonl --output eval/results/retrieval_debug_hard.jsonl` |
| **Failure analysis** | `python scripts/analyze_failures.py --input eval/results/retrieval_debug.jsonl --output-dir eval/results` |

Copy-paste:

```bash
# Demo
carrecall-demo --make Jeep --model "Grand Cherokee" --query "fuel starvation HPFP failure"

# Demo with neural reranker
carrecall-demo --make Jeep --model "Grand Cherokee" --query "fuel starvation HPFP failure" --rerank --rerank-topn 50

# Evaluation (writes eval/results/retrieval_debug.jsonl)
python -m carrecall_rag.eval \
  --eval-file eval/recall_queries.jsonl \
  --output eval/results/retrieval_debug.jsonl \
  --mode hybrid --alpha 0.5 --dense-topk 100 --keyword-topk 150 --topc 10

# Compare rerank text formats (no-rerank, full, compact, smart)
python -m carrecall_rag.eval --eval-file eval/recall_queries.jsonl --compare-table \
  --dense-topk 100 --keyword-topk 150 --rerank-topn 50

# Compare retrieval methods (dense, keyword, hybrid, hybrid+rerank) — writes eval/results/comparison_methods.md
python -m carrecall_rag.eval --eval-file eval/recall_queries.jsonl --compare-methods

# Run evaluation on the extended fixed benchmark
python -m carrecall_rag.eval --eval-file eval/recall_queries_100_fixed.jsonl --output eval/results/retrieval_debug_100_fixed.jsonl

# Run evaluation on the hard benchmark
python -m carrecall_rag.eval --eval-file eval/recall_queries_hard.jsonl --output eval/results/retrieval_debug_hard.jsonl

# Analyze failures from debug JSONL (after running eval)
python scripts/analyze_failures.py --input eval/results/retrieval_debug.jsonl
```

---

## Evaluation

- **Metrics:** Recall@1, @3, @5, @10 · MRR · avg/median rank · miss count  
- **Test sets:**  
  - **Small (default):** `eval/recall_queries.jsonl` — 10 labeled queries.  
  - **Extended fixed:** `eval/recall_queries_100_fixed.jsonl` — cleaned extended benchmark (in git).  
  - **Hard:** `eval/recall_queries_hard.jsonl` — challenging paraphrased complaint-style benchmark (in git).  
- **Fields per query:** `query`, `make`, `model`, `gold_campaign` or `gold_campaigns`.  
- **Output:** Per-query debug JSONL and logs under `eval/results/`.

| Metric   | Meaning |
|----------|---------|
| Recall@K | Fraction of queries where a gold campaign appears in top-K |
| MRR      | Mean of 1/rank for first correct (0 if none in list) |

**Baseline** (hybrid α=0.5, no rerank, small set): **Recall@1 0.90** · **Recall@10 1.00** · **MRR 0.95** (n=10).

**Extended fixed benchmark:** evaluate directly:

```bash
python -m carrecall_rag.eval --eval-file eval/recall_queries_100_fixed.jsonl --output eval/results/retrieval_debug_100_fixed.jsonl
```

**Hard benchmark:** evaluate directly:

```bash
python -m carrecall_rag.eval --eval-file eval/recall_queries_hard.jsonl --output eval/results/retrieval_debug_hard.jsonl
```

**Method comparison** (dense vs keyword vs hybrid vs hybrid+rerank) with Recall@K, MRR, avg/median rank, miss count:

```bash
python -m carrecall_rag.eval --eval-file eval/recall_queries.jsonl --compare-methods --compare-methods-output eval/results/comparison_methods.md
```

**Failure / error analysis:** After running eval, inspect missed or low-ranked queries:

```bash
python scripts/analyze_failures.py --input eval/results/retrieval_debug.jsonl --output-dir eval/results
```

This produces `eval/results/failure_analysis.md` and `failure_analysis.csv` with failures, gold campaign, top returned campaigns, first correct rank, and heuristic categories (e.g. symptom wording mismatch, weak lexical overlap).

---

## System design highlights

- **Hierarchical index fallback** — Prefer pool index (make+model); if too few docs, use make-only; else global. Keeps retrieval focused on the relevant vehicle slice when possible.
- **Hybrid retrieval** — Dense (semantic) + BM25 (lexical) with configurable fusion (default α=0.5). More robust than dense-only to wording mismatch.
- **Campaign-level aggregation** — Results are grouped by `campaign_number`; each campaign is scored from its best-matching document. Output is a ranked list of campaigns, not raw document hits.
- **Optional reranking** — A cross-encoder can rerank the top-N stage-1 candidates. Experimental; not part of the reported baseline.

---

## Known limitations

- **Small eval set** — Current labeled set has 10 queries. Metrics are indicative but not statistically broad.
- **Symptom–recall wording gap** — User phrasing often differs from official recall language; retrieval quality depends on semantic coverage of the encoder and keyword match.
- **No guarantee of recall coverage** — Some complaints may not map to any NHTSA recall (e.g. not yet investigated or outside recall scope).
- **Ambiguous symptoms** — Queries that match several conditions can correctly retrieve multiple plausible campaigns; the system ranks them but does not choose a single “correct” one.

---

## Pipeline (reference)

1. Query = user text + make/model/year.  
2. Index choice: pool (make+model) → make-only → global when doc count ≥ `min_pool_docs`.  
3. Dense: same encoder as index, FAISS top-K (cosine).  
4. BM25: same corpus/pool, top-K.  
5. Fusion: `(1 − α)·dense + α·keyword`, default α = 0.5. Top-N candidates.  
6. Aggregate by `campaign_number`; keep best doc score per campaign.  
7. Optional: cross-encoder reranks top-N (experimental).

---

## Layout

- **`src/carrecall_rag/`** — config, retrieve (FAISS+BM25), rerank (fusion + optional reranker), demo (CLI), answer (template output), eval, generate_eval_queries, build_corpus, nhtsa, utils.  
- **`eval/`** — recall_queries.jsonl, recall_queries_100_fixed.jsonl, recall_queries_hard.jsonl, results/.
- **`scripts/`** — run_eval.sh, spot_checks.sh, analyze_failures.py.

---

## Data

NHTSA public APIs (no scraping):

- `https://api.nhtsa.gov/complaints/complaintsByVehicle`
- `https://api.nhtsa.gov/recalls/recallsByVehicle`

---

## Install

```bash
git clone https://github.com/kushalX13/CarDiag-RAG
cd CarDiag-RAG
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run demo or eval from project root.
