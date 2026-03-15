# CarDiag-RAG

[GitHub](https://github.com/kushalX13/CarDiag-RAG)

**CarDiag-RAG** is a hybrid retrieval system that maps free-text vehicle symptom descriptions—plus make/model/year context—to likely NHTSA recall campaigns. It treats symptom-to-recall matching as a **ranking problem**: given a user query, the system returns a ranked list of recall campaigns with optional grounded explanations, and is evaluated with retrieval metrics (Recall@K, MRR), not classification accuracy. The pipeline combines dense retrieval (SentenceTransformer + FAISS), BM25 keyword retrieval, score fusion, campaign-level aggregation, and an optional cross-encoder reranker.

---

## Results snapshot

| Metric      | Value   |
|------------|---------|
| Recall@1   | 0.90    |
| Recall@10  | 1.00    |
| MRR        | 0.95    |
| Eval set   | 10 queries |

Baseline: hybrid retrieval (α=0.5), no rerank. Test set: `eval/recall_queries.jsonl`.

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

**Illustrative output** (from the demo; format matches `demo_outputs/hpfp.txt`):

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
| **Eval** (full) | `python -m carrecall_rag.eval --eval-file eval/recall_queries.jsonl --output eval/results/retrieval_debug.jsonl --mode hybrid --alpha 0.5 --dense-topk 100 --keyword-topk 150 --topc 10` |
| **Eval** (script) | `./scripts/run_eval.sh` |
| **Compare modes** | `python -m carrecall_rag.eval --eval-file eval/recall_queries.jsonl --compare-table --dense-topk 100 --keyword-topk 150 --rerank-topn 50` |

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

# Compare dense vs keyword vs hybrid vs hybrid+rerank
python -m carrecall_rag.eval --eval-file eval/recall_queries.jsonl --compare-table \
  --dense-topk 100 --keyword-topk 150 --rerank-topn 50
```

---

## Evaluation

- **Metrics:** Recall@1, @3, @5, @10 · MRR · avg/median rank · miss count  
- **Test set:** `eval/recall_queries.jsonl` (fields: `query`, `make`, `model`, `gold_campaign` or `gold_campaigns`)  
- **Output:** Per-query JSONL and logs under `eval/results/`

| Metric   | Meaning |
|----------|---------|
| Recall@K | Fraction of queries where a gold campaign appears in top-K |
| MRR      | Mean of 1/rank for first correct (0 if none in list) |

**Baseline** (hybrid α=0.5, no rerank): **Recall@1 0.90** · **Recall@10 1.00** · **MRR 0.95** (n=10).

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

- **`src/carrecall_rag/`** — config, retrieve (FAISS+BM25), rerank (fusion + optional reranker), demo (CLI), answer (template output), eval, build_corpus, nhtsa, utils.  
- **`eval/`** — recall_queries.jsonl, results/.  
- **`scripts/`** — run_eval.sh, spot_checks.sh.

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
