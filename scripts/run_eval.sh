#!/bin/bash
# Run retrieval evaluation: Recall@K over labeled test set.
# Usage: ./scripts/run_eval.sh [--verbose]
# Run from project root.

cd "$(dirname "$0")/.." && export PYTHONPATH=src

EVAL_FILE="${EVAL_FILE:-data/eval/recall_queries.jsonl}"
DENSE_TOPK=100
KW_TOPK=150
TOPC=10
ALPHA=0.30

VERBOSE=""
[[ "$1" == "--verbose" ]] && VERBOSE="--verbose"

python -m carrecall_rag.eval_retrieval \
  --eval-file "$EVAL_FILE" \
  --dense-topk $DENSE_TOPK \
  --keyword-topk $KW_TOPK \
  --topc $TOPC \
  --alpha $ALPHA \
  --hybrid \
  $VERBOSE
