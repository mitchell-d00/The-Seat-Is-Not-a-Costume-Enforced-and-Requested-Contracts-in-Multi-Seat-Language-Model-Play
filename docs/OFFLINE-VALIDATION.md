# Offline harness validation

## Scope and provenance

This validation used seatkit 0.2.1 on Python 3.12, starting from commit
`feccf96c0f2ec20b780dce3240b0efc065ab83b3`. The checkout stored the library in
`sec/`, while package discovery, test imports, experiment scripts and the draft
preregistration lock expected `src/`. Renaming `sec/` to `src/` restored the
expected layout without modifying the Python source contents.

The backend was `mock`, the judge was the offline `rule` panel, and each of
E1–E4 ran with **40 scenes and 24 turns**. No live API calls were made.

## Verified outcomes

- All **149 tests passed**.
- Mock preflight passed, including prior verification.
- The rule judge passed all six calibration checks, with fingerprint
  `b43528062af73504` and zero unparseable replies.
- All four experiment scripts completed and wrote their result files.
- E1 reported **zero routing bleed** in every cell and **zero elicited and
  volunteered leakage** at L3, L4 and the independently implemented ceiling.
- The restored source tree matched the existing draft lock. The lock remains
  draft, and all results are recorded as exploratory.

The new operational finding is that the `sec/` directory name prevented the
repository's documented layout from working. The offline outcomes confirm that
restoring `src/` makes the existing tests and experiment harness runnable.
They do not establish any new empirical claim about live language models.

## Interpretation and outstanding decisions

The mock's leakage ordering is stipulated by its constructor, and it has no
prior bleed. Its numeric rates are regression checks, not evidence for the
paper's predictions. The rule judge is a stand-in, not a qualified live judge.

The run also preserves the already documented limits:

- E3 records `P6_L1_identical_by_construction`: below L3, the private and shared
  buffer conditions do not differ in visibility.
- P8 remains undecided because its equivalence margin is unset.
- E4 does not implement the paper's eight-turn continuation; convention
  survival is scored by presence in the reseeding context.

No scientific decisions, judge settings or preregistration lock were changed.

## Reproduction

From the repository root, with Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python experiments/preflight.py --backends mock
python experiments/judge_check.py --judge-backend rule

for e in e1_leak_probe e2_attribution e3_instrument_symmetry; do
  python "experiments/$e.py" --backend mock --judge-backend rule \
    --scenes 40 --turns 24 --cache .cache/offline-validation \
    --out "results/offline-validation/${e%%_*}.json"
done
python experiments/e4_externalization.py --backend mock \
  --scenes 40 --turns 24 --cache .cache/offline-validation \
  --out results/offline-validation/e4.json
python experiments/freeze_prereg.py --check
```

The final command is expected to exit 1 while the lock is draft. It should
report no changed files after this layout correction. Do not finalize the lock
until the live judge, P6 scope and P8 equivalence margin are selected.

[`results/offline-validation/`](../results/offline-validation/) includes E1–E4
JSON outputs, the E1 audit JSONL, execution logs and a standalone run guide.
[`results/judge_checks/`](../results/judge_checks/) includes the rule calibration
record. Response caches are excluded from version control. A cached rerun is a
replay, not an independent replication.
