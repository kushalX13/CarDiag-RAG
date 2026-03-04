#!/bin/bash
# Spot checks: alpha=0.30, hybrid. Run from project root.
# Uses || true so one failure doesn't kill the whole run.

cd "$(dirname "$0")/.." && export PYTHONPATH=src

echo "=== 1) F-150: brake fluid leak ==="
python -m carrecall_rag.demo_retrieve --make "Ford" --model "F-150" --query "brake fluid leak" --topk 100 --topc 3 --alpha 0.30 --hybrid || true

echo ""
echo "=== 2) Grand Cherokee: stalling while driving ==="
python -m carrecall_rag.demo_retrieve --make "Jeep" --model "Grand Cherokee" --query "stalling while driving" --topk 100 --topc 3 --alpha 0.30 --hybrid || true

echo ""
echo "=== 3) Grand Cherokee: airbag warning light ==="
python -m carrecall_rag.demo_retrieve --make "Jeep" --model "Grand Cherokee" --query "airbag warning light" --topk 100 --topc 3 --alpha 0.30 --hybrid || true

echo ""
echo "=== 4) Camry: transmission shudder and hard shift ==="
python -m carrecall_rag.demo_retrieve --make "Toyota" --model "Camry" --query "transmission shudder and hard shift" --topk 100 --topc 3 --alpha 0.30 --hybrid || true

echo ""
echo "=== 5) Civic: engine stalls and loss of power ==="
python -m carrecall_rag.demo_retrieve --make "Honda" --model "Civic" --query "engine stalls and loss of power" --topk 100 --topc 3 --alpha 0.30 --hybrid || true

echo ""
echo "=== Done ==="
