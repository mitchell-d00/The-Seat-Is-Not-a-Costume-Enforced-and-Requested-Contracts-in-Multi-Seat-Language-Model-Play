# Review fixes (v0.2.0, with the v0.2.1 judge and lock)

What changed after an external review of v0.1.0, why, and what is still open.
Each item says where the code and the paper disagreed; in every case the code
was brought up to what the paper already claimed, except where noted.

## Fixed

**The ceiling condition did not exist.** The paper (§9) says every experiment
runs against a floor and a ceiling and reports position between them.
`metrics.normalize` existed but nothing called it, and there was no
separate-process arm anywhere. P11 (prior bleed dominance) was untestable.

- New `seatkit/ceiling.py`: `SeparatedBench` runs each seat with its own
  transcript, its own backend object (via a factory), and its own private
  instrument. Only the literal text of utterances crosses between seats. It
  never calls `ladder.assemble_view`, so it is an independent implementation.
- E1, E2, E3, and E4 all run a ceiling cell. E1–E3 report every ladder cell
  as a position between L0 and the ceiling. E1 and E3 report P11 directly.
- The ceiling uses its own cache directories (`<cache>/ceiling/<seat>`). With a
  shared cache it would replay L3's samples and equal L3 by construction.

*One consequence the paper now states (§9).* For a stateless API, an ideal L3
projection and the ceiling hand the model the same string.
`tests/test_ceiling.py` asserts this prompt by prompt, which makes the ceiling a
differential test of `ladder.py`. A second test deliberately breaks the
projection and checks that the ceiling catches it. The implication is that P2
(L3/L4 indistinguishable from the prior baseline), on stateless APIs, largely
tests implementation correctness plus sampling noise. The deployments where
L3 and the ceiling genuinely diverge are those with shared server-side state:
thread APIs, provider memory, and shared retrieval.

**E4 printed a fixed conclusion.** Lines 71–77 printed "A transcript tail
degrades constraint survival at EVERY level…" regardless of the data. That text
is gone. `verdicts()` builds every finding from the run's numbers, each P9/P10
claim carries a scene-paired sign test, and mock runs are tagged "harness check,
not evidence". At n=10 on the mock, the old "prediction 2: supported" line
does not survive a paired test.

**E4 did not measure what §9.4 defines as constraint survival.** The paper
says it is the post-reseed leak rate from E1's probe. The code only checked
whether the reseeding context contained the fact. E4 now puts E1's probe
question to the reseeded seat (primary measure) and keeps the context check as
`constraint_survival_context` (the mechanism).

**E2's live path still used the regex stand-in.** New `seatkit/judges.py` adds:

- `ModelParaphraser`, the separate-model register-normalizing pass;
- `ModelAttributionJudge`, a held-out model reader working leave-one-out with
  anonymized, per-call reshuffled speaker labels;
- `ModelLeakGrader`, a paraphrase-equivalence grader for E1;
- `ModelStanceJudge` with `incompatibility_by_turn`, the positional-disagreement
  judge for E3.

The old regex, lexical-centroid, and lexical-similarity versions remain as
labelled stand-ins (`is_standin = True`). E1–E3 now **refuse** a live scene
backend without `--judge-backend` unless `--allow-standin-judge` is passed. The
judge is resolved before the scene backend, so a misconfigured run fails before
any client exists. Unparseable judge replies are counted and given a documented
fallback, never dropped.

**E1 had no paraphrase grading or human audit.** With `--judge-backend`, E1
grades every non-exact response, reports exact and graded rates side by side,
and writes `results/e1_audit.jsonl`: a deterministic 20% sample with an empty
`human` column. `experiments/grader_agreement.py` reads the labelled file and
reports Cohen's kappa (model vs human, exact vs human) plus missed and false
leaks. Kappa is NaN on an all-negative sample, not 1.0.

**E3 printed no warning when nothing converged.** It now warns when at least
80% of scenes are censored in every cell, and when the disagreement measure is
the lexical stand-in. Half-life summaries use a restricted mean with
never-converged scenes censored at T+1, always alongside `n_never_converged`.

**P8 (contract swap) had no code path.** E3 has an `L3_private_swapped` cell:
each voice keeps its persona and takes the other seat's needs and facts. The
instrument replay log is built from the unswapped pair, so only contracts move.
P8 is a prediction of *no change*, which needs an equivalence margin; pass it as
`--p8-margin`. Without one, P8 is reported but not decided.

**Preregistration had no timestamps.** New `experiments/freeze_prereg.py`
writes `docs/PREREGISTRATION.lock` with SHA-256 hashes of the preregistration,
the protocol, the config, and all source and experiment files, plus the UTC time
and git commit. Every results file records whether the tree matched the lock
and lists any file that changed since. This is self-attested; the paper (§9.5)
now says to deposit the digest with a third party for an independent timestamp.
**No lock ships in this archive**: freezing is the author's act, done once, before
data collection.

**Parameter substitutions were not recorded.** §9.5 says they are. When a
provider rejected `temperature` or `seed`, the adapter dropped it silently.
Drops are now recorded on each `Completion.dropped_params`, survive the cache,
and are counted in `backends.SUBSTITUTIONS`. That counter is written to every
results file, with a warning if temperature was ever dropped.

## Bugs found along the way

**The mock was not reproducible across processes, and the cache never hit on
rerun.** Seeds were built with `hash(seat_id)` and `hash((fact_id, t))`.
Python salts those per process (PYTHONHASHSEED), so every new run drew different
seeds. That broke the README's "only the mock is reproducible", and because
`CachedBackend` keys on seed, every rerun of a live grid was a full cache miss
and re-billed every call. All seeds now go through `_seeding.stable_int`
(CRC32). `tests/test_seeding.py` runs scenes in fresh interpreters with
different hash seeds; it fails on v0.1.0 and passes now.

**Routing was checked on the wrong view.** `Bench` ran the routing-bleed check
on the view *before* instrument output was added, so a mis-routed instrument
span on the final turn could go unrecorded. It now checks the view that is sent.

**E2 regression axes could misalign.** `xs` took every scene and `ys` only
scenes with judgements, so a single empty scene would shift every pair. Pairs
are now appended together.

**Counterbalancing existed only in E1.** All experiments now build scenes
through `experiments/_common.build_pair`, which counterbalances fact pools
across seats on odd scenes.

**`tests/test_ladder.py` overclaimed.** Its docstring said a failure meant the
enforced kill condition had fired. A failure there is a routing bug; the kill
condition is behavioural and only E1 against the ceiling can test it. The
README's "the claim is a unit test" is reworded the same way.

## Also added

- The E1 L1 temperature sweep (`--sweep`), which the protocol listed but E1
  did not run.
- P5 at L0, turns ≥ 16, in E2: late-turn surface and content accuracy with sign
  tests.
- `compare_families.py` shows the ceiling column and positions, and flags runs
  that are unfrozen, ungraded, or missing a ceiling.
- `run_cross_family.sh` requires `JUDGE=<provider:model>`. `run_all.sh`
  accepts it, and is safe under macOS's bash 3.2 with `set -u`.
- 43 new tests (86 → 129). Every new code path is covered offline, including
  the live-judge paths, which use fake backends.

## v0.2.1: a reusable judge and lock

**Judge.** E1–E3 now share one `JudgePanel` (`seatkit/judges.py`), built from
`experiments/configs/judge.json`. It bundles the paraphraser, attribution reader,
leak grader, and stance judge on one model, with one config. Its fingerprint
is a hash of the four prompts, the prompt version, the model, and the settings,
and it is written into every results file with call counts and the unparseable
rate.

- `experiments/judge_check.py` qualifies a judge on `configs/judge_gold.json`:
  12 grader items, 8 stance items, 12 attribution turns, and 6 paraphrase items,
  all clear cases. It applies the thresholds in `judge.json` and saves the result
  under `results/judge_checks/<fingerprint>.json`. E1–E3 record whether a
  passing check exists for their judge and warn on live runs when one doesn't.
  Constant-answer judges fail it (tested).
- `RuleJudgeBackend` (`--judge-backend rule`) answers the same prompt formats
  with rules, so the full judge path runs offline. It passes the gold check and
  is still a stand-in: refused for live data, and marked as such.
- `run_cross_family.sh` reads the judge from `judge.json`, runs the judge check,
  and checks the lock before spending a grid.

**Lock.** `freeze_prereg.py` now has three modes. `--draft` writes a
baseline lock even with decisions unset. `--force` writes a final lock and
refuses while anything is unset. `--check` verifies the tree and exits non-zero
on any change. A draft lock is recorded as exploratory, not frozen. Re-freezing
keeps the old lock as a superseded file. The lock now also covers `judge.json`,
`judge_gold.json`, and `prereg.json`. The decisions a final lock requires live
in `experiments/configs/prereg.json`; E3 reads the P8 margin from there.

**This archive ships a draft lock**, made after every other file in it. A final
lock needs three things only the author can supply: the P8 margin, the P6-at-L1
choice, and the judge model.

20 new tests (129 → 149).

## Still open (decisions for the author, not bugs)

1. **The L1 half of P6 is untestable as implemented.** Below L3,
   `ladder.visibility_for` shows every span to every seat, so `L1_private` and
   `L1_shared` are identical by construction. E3 detects this and says so in
   its output. P6 predicts a difference at L1, so "private instrument at L1"
   needs a definition. The natural one, consistent with the ladder, is
   *requested* privacy: output stays in the shared window with a prohibition
   paragraph naming whose instrument it is. Whether that is the intended
   meaning is a design call.
2. **E4 phase 2 is partial.** §9.4 specifies an 8-turn continuation between
   reseeded seats and probing for each of n_c = 6 established conventions.
   Conventions are still scored by presence in the reseeding context, and there
   is no continuation. Both are recorded under `measures.not_implemented` in
   every E4 results file.
3. **The P8 equivalence margin** goes in `experiments/configs/prereg.json`
   before the final lock, as does the P6-at-L1 choice (item 1).
4. **E2's second judge task** (assigning each forbidden-knowledge clause to the
   seat that held it, §9.2) is not implemented; only turn attribution is.
5. **The mock cannot exercise E4's post-reseed probe.** The mock only uses
   values written in its setup format (`… is 18.4 ppm.`), not values echoed in
   conversation, so its post-reseed leak rate is always zero. A live model has no
   such restriction. The context measure still exercises the mechanism offline.

## Paper

Two passages changed; the argument and the design are otherwise untouched.

- **§9, reference conditions.** A new paragraph states the stateless-API
  property of the ceiling and what it implies for P2.
- **§9.5, preregistration.** "With timestamps" is replaced by what actually
  ships: a freeze script with content hashes, plus third-party deposit of the
  digest.

The `.txt` and `.pdf` were regenerated from the `.md` with the settings that
reproduce the v0.1.0 PDF page for page (all 20 pages text-identical when built
from the v0.1.0 source):

    pandoc the-seat-is-not-a-costume.md -t plain --wrap=none -o the-seat-is-not-a-costume.txt
    pandoc the-seat-is-not-a-costume.md -o the-seat-is-not-a-costume.pdf \
      --pdf-engine=xelatex --toc -V papersize=letter -V fontsize=11pt \
      -V geometry:margin=1.1in -V mainfont="DejaVu Serif" \
      -V monofont="DejaVu Sans Mono" -V mathfont="Latin Modern Math"
