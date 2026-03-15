# CarDiag-RAG

[GitHub](https://github.com/kushalX13/CarDiag-RAG)

Maps symptom descriptions (e.g. "brake fluid leaking from master cylinder") to NHTSA recall campaigns. Hybrid retrieval: dense (SentenceTransformer/FAISS) + BM25, optional cross-encoder rerank. Evaluated with Recall@K and MRR, not classification metrics.

## Pipeline

1. Query = user text + make/model/year.
2. Index choice: pool (make+model) → make-only → global, when doc count ≥ `min_pool_docs`.
3. Dense: same encoder as index, FAISS top-K (cosine).
4. BM25: same corpus/pool, top-K.
5. Fusion: `(1 - α) * dense + α * keyword`, default α = 0.5. Top-N candidates.
6. Aggregate by `campaign_number`; keep best doc score per campaign.
7. Optional: cross-encoder reranks top-N (experimental).

Corpus: `data/processed/corpus_merged.jsonl` (doc_id, campaign_number, make_norm, model_key, text). FAISS and BM25 built over it; pool/make indexes when enough docs.

## Eval

- **Metrics:** Recall@1/@3/@5/@10, MRR, avg/median rank, miss count.
- **Test set:** `eval/recall_queries.jsonl` — `query`, `make`, `model`, `gold_campaign`(s).
- From project root: `pip install -e .` then run below (or `./scripts/run_eval.sh`).

| Metric   | Description |
|----------|-------------|
| Recall@K | Fraction of queries with gold in top-K |
| MRR      | Mean of 1/rank for first correct (0 if miss) |

Baseline (hybrid α=0.5, no rerank) on current eval set: Recall@1 0.90, Recall@10 1.00, MRR 0.95 (n=10). See `eval/results/` for per-query output.

## Commands

**Demo (hybrid):**
```bash
carrecall-demo --make Jeep --model "Grand Cherokee" --query "fuel starvation HPFP failure"
```

**With rerank:**
```bash
carrecall-demo --make Jeep --model "Grand Cherokee" --query "fuel starvation HPFP failure" --rerank --rerank-topn 50
```

**Eval:**
```bash
python -m carrecall_rag.eval \
  --eval-file eval/recall_queries.jsonl \
  --output eval/results/retrieval_debug.jsonl \
  --mode hybrid --alpha 0.5 --dense-topk 100 --keyword-topk 150 --topc 10
```

**Compare modes:**
```bash
python -m carrecall_rag.eval --eval-file eval/recall_queries.jsonl --compare-table --dense-topk 100 --keyword-topk 150 --rerank-topn 50
```

## Layout

- `src/carrecall_rag/`: config, retrieve (FAISS+BM25), rerank (fusion + optional reranker), demo (CLI), answer (template output), eval, build_corpus, nhtsa, utils.
- `eval/`: recall_queries.jsonl, results/.
- `scripts/`: run_eval.sh, spot_checks.sh.

## Data

NHTSA APIs: `api.nhtsa.gov/complaints/complaintsByVehicle`, `api.nhtsa.gov/recalls/recallsByVehicle`.

## Install

```bash
git clone https://github.com/kushalX13/CarDiag-RAG
cd CarDiag-RAG
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run demo or eval from project root.
