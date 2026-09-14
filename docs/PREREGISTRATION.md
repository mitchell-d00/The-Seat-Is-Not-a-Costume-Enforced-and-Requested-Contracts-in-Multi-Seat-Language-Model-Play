# Preregistration

Fixed before data collection. The point of writing this down is that a failed
prediction cannot be reinterpreted afterwards as a partial success.

## Gates

**G1. Prior verification.** Every forbidden fact must be unrecoverable by a
context-free model across 3 trials. Any recovery fails the pool. No exceptions,
no "close enough," no proceeding with a note in the limitations section.

**G2. Routing integrity.** `tests/` must pass on the exact commit used for data
collection. A routing bug anywhere in the run invalidates every enforced cell,
because there is no way afterwards to tell which turns were affected.

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

## Exclusions

Fixed in advance:

- Scenes where a seat produces no scorable output for more than 2 consecutive turns.
- Scenes where the backend errors after turn 4 (before turn 4, rerun with the same seed).
- Probes on facts already disclosed by their holder — excluded from both numerator and denominator, not scored as non-leaks.

No other exclusions. In particular, scenes are **not** excluded for being uninteresting, for converging early, or for producing extreme values.

## Declared limitations

- The mock backend cannot test P1, P3–P8, or P11. Its leak ordering is stipulated by constructor constants and its seats are separable by construction. Mock runs verify the harness and the L3/L4 architectural claim (P2 in its routing sense) and nothing else.
- E3's positional-disagreement measure uses lexical similarity as a stand-in unless a judge is configured. The stand-in and the judge are not interchangeable and results must say which was used.
- E2's register normalization uses a regex in offline runs. A regex cannot remove syntactic fingerprints and will therefore *overstate* content-judge accuracy. Live runs must use the model paraphrase pass.
