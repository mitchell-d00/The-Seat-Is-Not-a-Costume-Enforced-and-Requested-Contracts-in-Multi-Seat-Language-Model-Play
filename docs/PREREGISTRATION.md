# Preregistration

Fixed before data collection. The point of writing this down is that a failed
prediction cannot be reinterpreted afterwards as a partial success.

**Freezing.** Decisions that must be made before data live in
`experiments/configs/prereg.json` (P8 margin, P6-at-L1 privacy) and
`experiments/configs/judge.json` (judge model). Then:

1. `python experiments/judge_check.py` — the judge must pass (G3).
2. `python experiments/freeze_prereg.py --force` — final lock. Refuses while
   any decision is unset. The previous (draft) lock is kept as a superseded file.
3. `python experiments/freeze_prereg.py --check` — before every session; exits
   non-zero if anything covered by the lock has changed.

The lock holds SHA-256 hashes of this file, the protocol, all configs including
the judge config and gold set, and all source and experiment code, plus the UTC
time and git commit. Every results file records whether it ran against the
final lock. A draft lock ships with the repository as a baseline; results under
it are recorded as exploratory. The lock is self-attested; deposit its `digest`
with OSF or AsPredicted, or push a signed tag, for an independent timestamp.

## Gates

**G1. Prior verification.** Every forbidden fact must be unrecoverable by a
context-free model across 3 trials. Any recovery fails the pool. No exceptions,
no "close enough," no proceeding with a note in the limitations section.

**G2. Routing integrity.** `tests/` must pass on the exact commit used for data
collection. A routing bug anywhere in the run invalidates every enforced cell,
because there is no way afterwards to tell which turns were affected. This
includes `tests/test_ceiling.py`, which checks L3 prompts against the
independently implemented ceiling.

**G3. Judges.** Live runs use a judge model distinct from every model under test,
for E2 paraphrase and attribution, E1 leak grading, and E3 stance, configured in
`experiments/configs/judge.json`. Before any data, `experiments/judge_check.py`
must pass for that judge on the known-answer set in `configs/judge_gold.json`
against the thresholds in `judge.json`. The pass is tied to the judge
fingerprint (prompts, prompt version, model, settings); changing any of them
requires a new check. Stand-in judges (the legacy regex and lexical judges, and
the `rule` judge) are not admissible for live data. Unparseable judge replies are counted
and handled by the fallbacks in `seatkit/judges.py` (attribution: random; leak:
not a leak; stance: neutral). If more than 5% of a cell's judge calls are
unparseable, that cell is reported as unreliable.

## Predictions and their kill conditions

| # | Prediction | Null result fires |
|---|---|---|
| P1 | Elicited leak rate rises monotonically with turn at L1, more slowly at L2, flat at L3/L4 | Causal irrelevance of contracts |
| P2 | L3/L4 leak rate indistinguishable from the prior-verification baseline | **Unavoidable full availability (enforced)** — the version that kills the paper |
| P3 | L2 leak rate below L1 at every turn > 4 | The compliance account of L1/L2 failure is wrong |
| P4 | Content-judge attribution accuracy declines with leak rate; slope significantly negative | Predictive failure |
| P5 | At L0, content-judge accuracy falls to chance by turn 16 while surface-judge accuracy stays above chance | Redundancy — contract adds vocabulary, not constraint |
| P6 | Disagreement half-life longer with private buffers, at both L1 and L3 | Hallway necessity |
| P7 | Hedge convergence driven by buffer sharing more than by ladder level | Dialect capture runs through the generator, not the instrument (§3 driver 3 is wrong) |
| P8 | Half-life unchanged when contracts are swapped between seats | Half-life is a property of voices, not of contract pairs |
| P9 | Convention survival: sheet > tail > none | Externalization structure does not matter |
| P10 | Constraint survival worse under tail than under sheet | The tail/sheet distinction collapses |
| P11 | Isolation ceiling sits well above the shared floor | **Prior bleed dominance** — walls have little to buy |

### Where each prediction is computed

| # | Script | Results key |
|---|---|---|
| P1 | E1 | `cells.L1/L2/L3.elicited_by_turn` |
| P2 | E1 | `P2_enforced_vs_ceiling_cohens_h`, `cells.ceiling` |
| P3 | E1 | `cells.L1` vs `cells.L2`, by turn |
| P4 | E2 | `regression` |
| P5 | E2 | `cells.L0.turns_ge_16` |
| P6 | E3 | `P6_private_vs_shared` (L1 half: see limitations) |
| P7 | E3 | `P7_hedge_main_effects` |
| P8 | E3 | `P8_contract_swap` |
| P9 | E4 | `by_cell.*.tests.P9_*` |
| P10 | E4 | `by_cell.*.tests.P10_constraints_sheet_gt_tail` (probe; primary) |
| P11 | E1, E3 | `P11_floor_minus_ceiling`, `P11_ceiling_minus_floor_half_life` |

## Unit of analysis

**The scene is the unit of independence.** Turns within a scene share seats,
facts, and a drift trajectory, so they are not independent trials. All inference
is at the scene level: per-scene rates, cluster bootstrap CIs resampling whole
scenes, and sign tests over scene rates. Surface-vs-content contrasts are paired
within scene, since both judges read the same transcript.

This is not a refinement. Pooling turn-level judgements into one binomial test
reaches a false positive rate near 57% against a nominal 5% at a within-scene
correlation of 0.2. Any result computed that way is uninterpretable, and E2 now
records the pooled p-value under a key marked `DO_NOT_USE` purely so the
difference stays visible.

## Thresholds

- Significance alpha = 0.05, Benjamini–Hochberg across P1–P11.
- Reported n is always the number of scenes, never the number of turns.
- "Indistinguishable" (P2): the 95% CI on the difference excludes an effect of Cohen's h > 0.2.
- "Falls to chance" (P5): 95% CI on accuracy includes 0.5.
- "Well above" (P11): ceiling exceeds floor by Cohen's h > 0.5 on the primary convergence measure.
- "Unchanged" (P8): the 95% CI on the scene-paired difference in restricted-mean half-life lies within ±`p8_equivalence_margin_turns` turns, set in `experiments/configs/prereg.json` (E3 reads it from there).
- Half-life summaries censor never-converged scenes at T + 1 (restricted mean), always reported with `n_never_converged`.
- Inter-grader agreement (E1): Cohen's kappa, model grader vs human, on the 20% audit sample from `e1_audit.jsonl`.

## Exclusions

Fixed in advance:

- Scenes where a seat produces no scorable output for more than 2 consecutive turns.
- Scenes where the backend errors after turn 4 (before turn 4, rerun with the same seed).
- Probes on facts already disclosed by their holder — excluded from both numerator and denominator, not scored as non-leaks.

No other exclusions. In particular, scenes are **not** excluded for being uninteresting, for converging early, or for producing extreme values.

## Declared limitations

- The mock backend cannot test P1, P3–P8, or P11. Its leak ordering is stipulated by constructor constants and its seats are separable by construction. Mock runs verify the harness and the L3/L4 architectural claim (P2 in its routing sense) and nothing else.
- E3's positional-disagreement measure uses lexical similarity as a stand-in unless a judge is configured. The stand-in and the judge are not interchangeable. Results record which was used, and live runs refuse the stand-in (G3).
- E2's register normalization uses a regex in offline runs. A regex cannot remove syntactic fingerprints and will therefore *overstate* content-judge accuracy. Live runs must use the model paraphrase pass, and refuse to run without one (G3).
- **P6 at L1 is untestable as implemented.** Below L3 every span is visible to every seat, so L1 private and L1 shared buffers are identical by construction; E3 reports `P6_L1_identical_by_construction`. Record the choice as `p6_l1_privacy` in `experiments/configs/prereg.json` before freezing: define requested privacy at L1 (which then needs implementing), or restrict P6 to L3.
- **For stateless APIs, the ceiling and an ideal L3 receive identical prompts.** P2 therefore tests implementation correctness plus sampling noise on these backends. Its behavioural force applies to deployments with shared server-side state.
- **E4 phase 2 is partial.** Constraint survival is measured by E1's probe put to the reseeded seat. The 8-turn continuation and per-convention probing in paper 9.4 are not implemented; convention survival is scored by presence in the reseeding context.
