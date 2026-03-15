#!/bin/bash
# Run retrieval evaluation: Recall@K over labeled test set.
# Usage: ./scripts/run_eval.sh [--verbose] [--mode dense|keyword|hybrid]
#        ./scripts/run_eval.sh --alpha-list 0.1,0.3,0.5,0.7
# Run from project root.

cd "$(dirname "$0")/.." && export PYTHONPATH=src

EVAL_FILE="${EVAL_FILE:-eval/recall_queries.jsonl}"
OUTPUT="${OUTPUT:-eval/results/retrieval_debug.jsonl}"
DENSE_TOPK=400
KW_TOPK=400
TOPC=10
ALPHA=0.80
MODE="${MODE:-hybrid}"

python -m carrecall_rag.eval \
  --eval-file "$EVAL_FILE" \
  --output "$OUTPUT" \
  --dense-topk $DENSE_TOPK \
  --keyword-topk $KW_TOPK \
  --topc $TOPC \
  --alpha $ALPHA \
  --mode "$MODE" \
  "$@"
