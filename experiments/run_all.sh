#!/usr/bin/env bash
# Run the full grid. Defaults to the offline mock; pass a backend spec to go live.
#
#   ./experiments/run_all.sh
#   ./experiments/run_all.sh anthropic:<model-id>
#
# Mock runs verify the harness and the L3/L4 architectural claim. They are not
# evidence for the behavioural predictions - see README, "What the mock backend
# can and cannot show".
set -euo pipefail

BACKEND="${1:-mock}"
SCENES="${2:-40}"
cd "$(dirname "$0")/.."

echo "backend=$BACKEND scenes=$SCENES"
for e in e1_leak_probe e2_attribution e3_instrument_symmetry e4_externalization; do
  echo
  echo "=== $e ==="
  python3 "experiments/$e.py" --backend "$BACKEND" --scenes "$SCENES" \
      --out "results/${e%%_*}.json"
done
echo
echo "results written to results/"
