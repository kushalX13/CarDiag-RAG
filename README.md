# CarDiag-RAG

CarDiag-RAG is a multi-stage retrieval system that maps natural-language vehicle failure symptoms to relevant NHTSA recall campaigns.  
The system uses stage-1 hybrid retrieval (FAISS + SentenceTransformer + BM25) for candidate generation, optional stage-2 neural reranking for final ordering, and grounded answer generation.

## Architecture

```text
           ┌─────────────┐
           │ User Query  │
           └──────┬──────┘
                  │
        ┌─────────▼─────────┐
        │ Hybrid Retrieval  │
        │ FAISS + BM25      │
        └─────────┬─────────┘
                  │
       ┌──────────▼──────────┐
       │ Stage-1 Candidates   │
       │ (Top N, default 50)  │
       └──────────┬──────────┘
                  │
       ┌──────────▼──────────┐
       │ Neural Reranker      │
       │ (optional stage-2)   │
       └──────────┬──────────┘
                  │
       ┌──────────▼──────────┐
       │ RAG Explanation      │
       └──────────┬──────────┘
                  │
           ┌──────▼──────┐
           │ Final Answer │
           └──────────────┘
```

## Key Features

- Two-stage retrieval: hybrid candidate generation + optional neural reranking
- FAISS index pools with make/model-aware retrieval
- BM25 lexical retrieval for exact symptom language
- Cross-encoder reranker for stronger top-ranked precision
- Campaign-level aggregation from recall text chunks
- Retrieval evaluation harness with Recall@K and before/after rerank diagnostics
- Grounded answer generation from top-ranked recall evidence
- Debug mode for retrieval internals and score diagnostics

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Demo (CLI)

After editable install, run a baseline hybrid-only demo:

```bash
carrecall-demo \
  --make Jeep \
  --model "Grand Cherokee" \
  --query "fuel starvation HPFP failure"
```

Run hybrid + neural rerank:

```bash
carrecall-demo \
  --make Jeep \
  --model "Grand Cherokee" \
  --query "fuel starvation HPFP failure" \
  --rerank \
  --rerank-topn 50
```

Equivalent module invocation (hybrid + rerank):

```bash
python -m carrecall_rag.demo_rag \
  --make Jeep \
  --model "Grand Cherokee" \
  --query "fuel starvation HPFP failure" \
  --rerank \
  --rerank-topn 50
```

Enable debug details (stage-1 and stage-2 internals):

```bash
carrecall-demo --make Jeep --model "Grand Cherokee" --query "fuel starvation HPFP failure" --debug
```

## Demo Example

From `demo_outputs/hpfp.txt`:

```text
Query:
fuel starvation HPFP failure

Best Match:
22V406000 — High Pressure Fuel Pump Failure

Why It Matches:
The high pressure fuel pump (HPFP) may fail, resulting in fuel starvation. This closely matches your query because it directly connects HPFP failure with fuel starvation and engine stall.

Safety Risk:
Potential risks include engine stall and sudden loss of propulsion while driving, which may increase crash risk.

Suggested Next Step:
Contact your dealer to verify if your vehicle is affected, or look up your VIN at NHTSA.gov/recalls.
```

## Evaluation

Run evaluation for **hybrid-only**:

```bash
python -m carrecall_rag.eval_retrieval \
  --eval-file eval/recall_queries.jsonl \
  --mode hybrid \
  --alpha 0.5 \
  --dense-topk 100 \
  --keyword-topk 150 \
  --rerank-topn 50
```

Run evaluation for **hybrid + rerank**:

```bash
python -m carrecall_rag.eval_retrieval \
  --eval-file eval/recall_queries.jsonl \
  --mode hybrid \
  --alpha 0.5 \
  --dense-topk 100 \
  --keyword-topk 150 \
  --rerank \
  --rerank-topn 50
```

Run side-by-side comparison table (dense, keyword, hybrid-only, hybrid+rerank):

```bash
python -m carrecall_rag.eval_retrieval \
  --eval-file eval/recall_queries.jsonl \
  --compare-table \
  --dense-topk 100 \
  --keyword-topk 150 \
  --rerank-topn 50
```

Per-query debug output (`--output`) includes before/after stage-2 ranking fields:

- `before_rerank_top10`
- `after_rerank_top10`
- `gold_rank_before_rerank`
- `gold_rank_after_rerank`

Example comparison format:

| Config | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| dense | 0.30 | 0.40 | 0.40 | 0.40 |
| keyword | 0.70 | 0.80 | 0.90 | 0.90 |
| hybrid-only | 0.xx | 0.xx | 0.xx | 0.xx |
| hybrid+rerank | **0.xx** | **0.xx** | **0.xx** | **0.xx** |

Use your local eval run to populate exact numbers.

## Project Structure

```text
CarDiag-RAG
├── src/carrecall_rag/
│   ├── retrieve.py          # FAISS + BM25 retrieval
│   ├── rerank.py            # stage-1 hybrid fusion + stage-2 neural rerank
│   ├── demo_retrieve.py     # retrieval-only demo
│   ├── demo_rag.py          # full retrieval + grounded answer CLI
│   ├── rag_answer.py        # grounded template answer generation
│   └── eval_retrieval.py    # Recall@K evaluation harness
├── eval/
│   └── recall_queries.jsonl # labeled retrieval test set
└── demo_outputs/
    └── hpfp.txt             # saved demo output
```

## Data Sources

NHTSA public APIs (no scraping):

- `https://api.nhtsa.gov/complaints/complaintsByVehicle`
- `https://api.nhtsa.gov/recalls/recallsByVehicle`

## Future Work

- Integrate an LLM reasoning layer for richer explanation synthesis
- Add VIN lookup workflow to move from symptom-level to vehicle-specific recall checks
- Expand evidence to complaints + service bulletins + investigations
- Build a simple web UI (Streamlit/Gradio/FastAPI) for interactive diagnostics
