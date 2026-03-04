#!/bin/bash
# Spot checks: alpha=0.30, hybrid. Run from project root.
# Uses || true so one failure doesn't kill the whole run.

cd "$(dirname "$0")/.." && export PYTHONPATH=src

TOPC=3
DENSE_TOPK=100
KW_TOPK=200   # bump keyword coverage for spot-check stability
ALPHA=0.30

COMMON_ARGS="--topc $TOPC --dense-topk $DENSE_TOPK --keyword-topk $KW_TOPK --alpha $ALPHA --hybrid"

echo "=== 1) F-150: brake master cylinder leak into brake booster (expect: 20V332000) ==="
python -m carrecall_rag.demo_retrieve \
  --make "Ford" --model "F-150" \
  --query "brake master cylinder may leak brake fluid into the brake booster" \
  $COMMON_ARGS --show-candidates || true

echo ""
echo "=== 2) Grand Cherokee: fuel starvation / HPFP failure (expect: 22V406000) ==="
python -m carrecall_rag.demo_retrieve \
  --make "Jeep" --model "Grand Cherokee" \
  --query "high pressure fuel pump failure may introduce debris into fuel system resulting in fuel starvation engine stall" \
  $COMMON_ARGS --show-candidates || true

echo ""
echo "=== 3) Grand Cherokee: airbag warning light / ORC module / clock spring ==="
python -m carrecall_rag.demo_retrieve \
  --make "Jeep" --model "Grand Cherokee" \
  --query "airbag warning light ORC module clock spring may disable air bags" \
  $COMMON_ARGS --show-candidates || true

echo ""
echo "=== 4) Camry: shift control cable / not in park / rollaway (expect: 14V414000-ish) ==="
python -m carrecall_rag.demo_retrieve \
  --make "Toyota" --model "Camry" \
  --query "transmission may not be in PARK despite selecting park vehicle rollaway shift control cable" \
  $COMMON_ARGS --show-candidates || true

echo ""
echo "=== 5) Civic: piston assembly / stall (expect: 16V074000-ish) ==="
python -m carrecall_rag.demo_retrieve \
  --make "Honda" --model "Civic" \
  --query "piston assemblies may have been manufactured without a piston pin engine stall loss of power" \
  $COMMON_ARGS --show-candidates || true

echo ""
echo "=== Done ==="