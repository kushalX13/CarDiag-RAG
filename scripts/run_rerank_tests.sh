#!/bin/bash
# Run rerank tests. Execute from project root.

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
echo "=== C1) Query-dependent: door latch freezes (with --show-candidates to debug) ==="
python -m carrecall_rag.demo_retrieve \
  --make "Ford" --model "F-150" \
  --query "door latch freezes" \
  --topk 30 --topc 3 --alpha 0.15 --show-candidates

echo ""
echo "=== C1b) door latch - topk=200 (retrieve all in pool) ==="
python -m carrecall_rag.demo_retrieve \
  --make "Ford" --model "F-150" \
  --query "door latch freezes" \
  --topk 200 --topc 3 --alpha 0.15 --show-candidates

echo ""
echo "=== C1c) door latch - global index (--no-pool) for broader dense recall ==="
python -m carrecall_rag.demo_retrieve \
  --make "Ford" --model "F-150" \
  --query "door latch freezes" \
  --topk 200 --topc 3 --alpha 0.15 --no-pool --show-candidates

echo ""
echo "=== C2) Query-dependent: brake fluid leak ==="
python -m carrecall_rag.demo_retrieve \
  --make "Ford" --model "F-150" \
  --query "brake fluid leak" \
  --topk 30 --topc 3 --alpha 0.15
