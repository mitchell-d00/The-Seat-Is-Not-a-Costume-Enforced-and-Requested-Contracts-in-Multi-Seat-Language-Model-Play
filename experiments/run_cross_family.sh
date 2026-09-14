#!/usr/bin/env bash
# Run the full grid across several model families (paper 9.5).
#
#   ./experiments/run_cross_family.sh openai:<model> anthropic:<model> gemini:<model>
#
# Results land in results/<provider>_<model>/. Nothing is merged across
# families: if a ladder effect appears on one model and not another, that is a
# finding about deployment practice and must stay visible in the output tree.
#
# Caching is on by default and keyed per provider+model. Reruns and re-analysis
# are then free; a cached run is one run replayed, not a replication.
set -euo pipefail

cd "$(dirname "$0")/.."
SCENES="${SCENES:-40}"
TURNS="${TURNS:-24}"

if [ $# -eq 0 ]; then
  echo "usage: $0 <backend-spec> [<backend-spec> ...]" >&2
  echo "example: $0 openai:<model-id> anthropic:<model-id>" >&2
  exit 2
fi

echo "preflight"
python3 experiments/preflight.py --backends "$@" || {
  echo "preflight failed; fix before spending a full grid" >&2; exit 1; }

for spec in "$@"; do
  slug="$(echo "$spec" | tr ':/' '__')"
  outdir="results/$slug"
  mkdir -p "$outdir"
  echo
  echo "################ $spec ################"
  for e in e1_leak_probe e2_attribution e3_instrument_symmetry e4_externalization; do
    echo "--- $e ---"
    python3 "experiments/$e.py" --backend "$spec" --scenes "$SCENES" \
        --turns "$TURNS" --cache ".cache/$slug" \
        --out "$outdir/${e%%_*}.json"
  done
done

echo
echo "comparing families"
python3 experiments/compare_families.py results/
