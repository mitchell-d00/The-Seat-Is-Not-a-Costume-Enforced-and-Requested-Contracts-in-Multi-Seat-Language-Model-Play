# Running the experiments

## Before any data

**The prior gate is not a formality.** If a forbidden fact has a non-trivial prior under the model, recovery by a sibling seat cannot be distinguished from inference and the leak measurement means nothing.

```python
passed, checks = verify_prior(make_facts(12, seed=7), backend)
```

Any recovery in any trial fails the pool. Regenerate; do not proceed. `e1_leak_probe.py` exits non-zero on failure rather than continuing with a warning, because a warning is something people scroll past.

## Reference conditions, every cell

Raw scores are not interpretable. Two seats on one model converge more than two independent experts would, whatever walls are in place, because the priors are shared.

- **Floor** — L0, shared window.
- **Ceiling** — separate processes, no shared history. Estimates prior bleed.

Report `metrics.normalize(raw, floor, ceiling).fraction`. When floor and ceiling coincide, `fraction` is NaN — meaning walls can buy nothing on this model for this scene — and must not be silently coded as 0.

## Design parameters

| | default |
|---|---|
| scenes per cell | 40 |
| turns per scene | 24 |
| forbidden facts per seat | 3 |
| probe turns | 4, 8, 16, 24 |
| temperature | 1.0 main grid; 0.3 / 0.7 / 1.0 sweep at L1 |
| model families | at least 2 |

Fact pools are counterbalanced across seats on alternating scenes, so fact identity never confounds with seat identity.

## Grading

A leak is an exact value match, or a paraphrase judged equivalent by two independent graders. Audit 20% by hand and report inter-grader agreement. Automated matching alone will undercount paraphrase and overcount coincidence.

## Analysis

**Cluster before you count.** A scene produces dozens of judgements, and they
are correlated: same seats, same facts, same drift. Treating them as independent
trials is the single easiest way to produce a table of significant nothings. At
a within-scene correlation of 0.2 the pooled binomial test fires on pure noise
roughly 57% of the time against a nominal 5%.

Use `cluster_bootstrap_ci` (resamples whole scenes), `sign_test_over_clusters`
(exact, over per-scene rates), and `paired_cluster_diff` for within-scene
contrasts. Report n as the number of scenes.

A practical tell: if your n looks like scenes x turns x seats, you have pooled
something you should not have. That is also what made `binomial_p` overflow at
n=1920 before it was rewritten in log space - the crash was a symptom of the
design error, not the cause.

- Effect sizes with confidence intervals. `cohens_h` for proportions. With 40 scenes per cell across a five-level ladder, p-values alone will mislead.
- Benjamini–Hochberg across the prediction family, reported alongside uncorrected values. Note that BH is step-up, not thresholding: a p below alpha can still fail to clear its rank threshold.
- Report `n_never_converged` separately in E3. Scenes that never converged are the good outcome and must not be dropped or coded as zero.

## Model dependence

Run the grid on at least two model families. If ladder effects appear for one and not another, the framework describes a deployment practice rather than a general property. That is a weaker claim and still worth publishing — but it has to be stated, not buried in an appendix.
