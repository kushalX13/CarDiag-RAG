#!/bin/bash
# Run rerank tests on cluster. Execute from project root.

set -e
# Run from project root: ./scripts/run_rerank_tests.sh
cd "$(dirname "$0")/.." && export PYTHONPATH=src

echo "=== A) Baseline (alpha=0.15) ==="
python -m carrecall_rag.demo_retrieve \
  --make "Ford" --model "F-150" \
  --query "engine stalls while driving and loss of power" \
  --topk 30 --topc 3 --alpha 0.15

echo ""
echo "=== B1) Alpha sensitivity: alpha=0.05 ==="
python -m carrecall_rag.demo_retrieve \
  --make "Ford" --model "F-150" \
  --query "engine stalls while driving and loss of power" \
  --topk 30 --topc 3 --alpha 0.05

echo ""
echo "=== B2) Alpha sensitivity: alpha=0.50 ==="
python -m carrecall_rag.demo_retrieve \
  --make "Ford" --model "F-150" \
  --query "engine stalls while driving and loss of power" \
  --topk 30 --topc 3 --alpha 0.50

echo ""
echo "=== C1) Query-dependent: door latch freezes ==="
python -m carrecall_rag.demo_retrieve \
  --make "Ford" --model "F-150" \
  --query "door latch freezes" \
  --topk 30 --topc 3 --alpha 0.15

echo ""
echo "=== C2) Query-dependent: brake fluid leak ==="
python -m carrecall_rag.demo_retrieve \
  --make "Ford" --model "F-150" \
  --query "brake fluid leak" \
  --topk 30 --topc 3 --alpha 0.15
