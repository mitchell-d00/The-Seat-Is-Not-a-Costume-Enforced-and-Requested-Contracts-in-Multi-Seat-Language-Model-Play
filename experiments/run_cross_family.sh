#!/usr/bin/env bash
# Run the full grid across several model families (paper 9.5).
#
#   JUDGE=<provider:model> ./experiments/run_cross_family.sh openai:<model> anthropic:<model>
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
# Model used for paraphrase, attribution, leak grading and stance (E1-E3).
# Should be a different model from the ones under test. Required: the stand-in
# judges bias E2 toward the hypothesis, and E1-E3 refuse a live run without one.
JUDGE="${JUDGE:-$(python3 -c "import json;print(json.load(open('experiments/configs/judge.json')).get('model') or '')")}"

if [ -z "$JUDGE" ]; then
  echo "no judge: set \"model\" in experiments/configs/judge.json, or JUDGE=<provider:model>" >&2
  exit 2
fi

if [ $# -eq 0 ]; then
  echo "usage: $0 <backend-spec> [<backend-spec> ...]" >&2
  echo "example: $0 openai:<model-id> anthropic:<model-id>" >&2
  exit 2
fi

echo "judge check"
python3 experiments/judge_check.py --judge-backend "$JUDGE" || {
  echo "judge failed its known-answer check; fix before spending a grid" >&2; exit 1; }
python3 experiments/freeze_prereg.py --check || \
  echo "[warning] not running under a final, matching lock: results are exploratory" >&2

echo "preflight"
python3 experiments/preflight.py --backends "$@" "$JUDGE" || {
  echo "preflight failed; fix before spending a full grid" >&2; exit 1; }

for spec in "$@"; do
  slug="$(echo "$spec" | tr ':/' '__')"
  outdir="results/$slug"
  mkdir -p "$outdir"
  echo
  echo "################ $spec ################"
  for e in e1_leak_probe e2_attribution e3_instrument_symmetry; do
    echo "--- $e ---"
    python3 "experiments/$e.py" --backend "$spec" --scenes "$SCENES" \
        --turns "$TURNS" --cache ".cache/$slug" --judge-backend "$JUDGE" \
        --out "$outdir/${e%%_*}.json"
  done
  echo "--- e4_externalization ---"
  python3 experiments/e4_externalization.py --backend "$spec" --scenes "$SCENES" \
      --turns "$TURNS" --cache ".cache/$slug" --out "$outdir/e4.json"
done

echo
echo "comparing families"
python3 experiments/compare_families.py results/
