# seat-contracts

Reference implementation and experimental harness for **"The Seat Is Not a Costume: Enforced and Requested Contracts in Multi-Seat Language Model Play"** (M. D. McPhetridge).

The paper is in [`paper/`](paper/) as Markdown, PDF, and plain text.

## What this is for

The library exists to make one distinction executable.

A prohibition written in a shared context window is a **request**. The forbidden tokens are sitting right there; the model honours the prohibition only by complying, and compliance degrades under load — most sharply at the moment the forbidden fact becomes relevant, which is exactly what the wall was built for.

A prohibition implemented in view assembly is **enforced**. The forbidden tokens are not in the seat's context at all. There is no prohibition to comply with, because there is nothing to suppress.

These are not two strengths of one dial. They are different objects with different failure modes. Most multi-persona work is requested, most of its failures are compliance failures, and most of its proposed fixes are more paragraphs.

The architectural half of the claim is a unit test:

```python
view = assemble_view(transcript, geo_seat, Level.REQUESTED, turn=1)
assert view.contains(forbidden_fact.value)      # the wall is a paragraph

view = assemble_view(transcript, geo_seat, Level.PARTITIONED, turn=1)
assert not view.contains(forbidden_fact.value)  # the wall is a projection
```

See `tests/test_ladder.py`. That test verifies the implementation, not the behaviour: a failure there is a routing bug in `ladder.py`. Whether a live model at L3 actually leaks no more than the prior-bleed floor is an empirical question, and only E1 against the ceiling cell can answer it.

## The enforcement ladder

| Level | Name | Admissible state is… | Wall lives in | Characteristic failure |
|---|---|---|---|---|
| L0 | Costume | unconstrained | nothing | collapse into one voice |
| L1 | Requested | window minus a prohibition | a paragraph, stated once | compliance decay with turn count |
| L2 | Reminded | window minus a re-injected prohibition | a paragraph at the context tail | slower decay; decay under topical pressure |
| L3 | Partitioned | a filtered view, assembled per call | the assembly code | routing error |
| L4 | Isolated | L3 plus private instrument buffers and a declared bench | assembly code and router | routing error; bench capture |

The jump that matters is **L2 → L3**. Below it, ignorance is a behaviour. At and above it, ignorance is a property of the input.

## Install

```bash
git clone <this repo> && cd seat-contracts
pip install -e .            # core, no dependencies
pip install -e ".[live]"    # + openai, anthropic, google-genai
pytest                      # 149 tests, no API key, about two seconds
```

Python 3.10+. The core library has no dependencies. Provider SDKs are needed only for live runs, `pytest` only for tests. Every test runs offline; provider adapters are covered by injecting fake clients.

## Quick start

```python
from seatkit import (Bench, Instrument, Level, MockBackend, SceneConfig,
                     facing_pair, leak_rates, make_facts)

# Low-prior facts. If a forbidden fact is something the model would produce
# anyway, leakage cannot be distinguished from inference.
a_facts = make_facts(3, seed=1, prefix="a")
b_facts = make_facts(3, seed=2, prefix="b")

# Each seat's held facts are the other's forbidden facts.
geo, bio = facing_pair("geo", "bio", a_facts, b_facts,
                       a_extra={"persona": "a geologist"},
                       b_extra={"persona": "a paleobiologist"})

backend = MockBackend()
instruments = {s.seat_id: Instrument(host=s, backend=backend) for s in (geo, bio)}
bench = Bench([geo, bio], backend, instruments)

res = bench.run(SceneConfig(turns=24, level=Level.PARTITIONED))
print(leak_rates(res.scoring_probes()).as_dict())
print("routing bleed:", res.routing_bleed_count)   # must be 0
```

## Experiments

```bash
python experiments/e1_leak_probe.py          --backend mock --scenes 40
python experiments/e2_attribution.py         --backend mock --scenes 40
python experiments/e3_instrument_symmetry.py --backend mock --scenes 40
python experiments/e4_externalization.py     --backend mock --scenes 40
```

Live runs take any provider spec, and the experimental logic does not change. E1–E3 also need a **judge**: one model, from a family not under test, that does the paraphrase pass, attribution, leak grading, and stance scoring.

### The judge

The judge is configured once, in `experiments/configs/judge.json`: the model, temperature, settings, and pass thresholds. Its prompts live in `src/seatkit/judges.py` under a version number. Together these produce a **fingerprint** that every results file records, so two runs used the same judge only if their fingerprints match.

Before a judge scores data it has to pass `experiments/judge_check.py`, which runs it on the known-answer items in `configs/judge_gold.json` (clear leaks and non-leaks, clear stances, two distinct voices, hedged sentences with numbers). The pass is saved under `results/judge_checks/<fingerprint>.json`, and E1–E3 report whether a passing check exists for the judge they used. A judge that always gives the same answer fails, and that's tested.

`--judge-backend rule` selects an offline rule-based judge that speaks the same prompt formats. It exercises the whole judge path without a key and passes the gold check, but it is a stand-in and is refused for live data.

### The lock

`docs/PREREGISTRATION.lock` records hashes of the protocol, every config (including the judge and its gold set), and all code. The repository ships a **draft** lock as a baseline. Results under it are marked exploratory, and `freeze_prereg.py --check` lists everything that has changed since.

```bash
# 1. decide: experiments/configs/prereg.json (P8 margin, P6 at L1)
#            experiments/configs/judge.json  ("model": "<provider:model>")
python experiments/judge_check.py                 # 2. qualify the judge
python experiments/freeze_prereg.py --force       # 3. final lock (refuses while anything is unset)
python experiments/freeze_prereg.py --check       # 4. before every session
python experiments/preflight.py --backends openai:<model> anthropic:<model>
./experiments/run_cross_family.sh openai:<model> anthropic:<model>
python experiments/compare_families.py results/
# after hand-labelling the 'human' column of each e1_audit.jsonl:
python experiments/grader_agreement.py results/<family>/e1_audit.jsonl
```

`run_cross_family.sh` takes the judge from `judge.json`; `JUDGE=<provider:model>` overrides it. E1–E3 refuse a live backend without a model judge. The offline stand-ins (a hedge-stripping regex, a bag-of-words judge, lexical similarity) bias E2 toward the paper's own hypothesis, so `--allow-standin-judge` is required to run them live, and the results file says so.

Every results file records the seatkit version, the judges used, any provider parameter that was dropped, and whether the code and protocol matched `docs/PREREGISTRATION.lock`.

| Spec | Key | SDK |
|---|---|---|
| `mock` | — | — |
| `openai:<model-id>` | `OPENAI_API_KEY` | `openai` |
| `anthropic:<model-id>` | `ANTHROPIC_API_KEY` | `anthropic` |
| `grok:<model-id>` (alias `xai:`) | `XAI_API_KEY` | `openai` |
| `gemini:<model-id>` (alias `google:`) | `GEMINI_API_KEY` | `google-genai` |

`pip install -e ".[live]"` gets all four. Add `--cache .cache/<run>` to make reruns and re-analysis free. Details and caveats in [`docs/backends.md`](docs/backends.md).

**Two things that quietly corrupt a cross-family table.** `prompt_tokens` is provider-reported and tokenizer-dependent, so E1's leak-vs-context-length curve uses `prompt_words` instead — computed identically everywhere. And only the mock is reproducible: seeds are absent on some APIs and best-effort on others, so cross-run variation is part of the measurement.

| | Question | The confound it fixes |
|---|---|---|
| **E1** | Does forbidden content get used? | Gives bleed a measure that never mentions attribution, so E2 stops being circular. |
| **E2** | Can a held-out reader recover who held what? | Two judges — one sees style, one sees only content. If style works and content doesn't, the costume was doing the work. |
| **E3** | Does instrument privacy slow disagreement decay? | A replay log makes instrument output byte-identical across arms, so only routing differs. The original fetch-vs-critic comparison measured injected content, not walls. |
| **E4** | Does externalization structure matter? | Three-way baseline (none / tail / sheet). A transcript tail *is* externalization, just the lossy kind. |

## Three kinds of bleed

- **Compliance bleed** — content was present and was used. L0–L2. Rises with context length. A behaviour.
- **Routing bleed** — content was presented to a seat whose visibility set excluded it. L3–L4. A defect in `ladder.py`. Deterministic, unit-testable, fixed once.
- **Prior bleed** — content was never presented and the seat produced it anyway, because the weights are shared. Not a wall failure. A property of the model.

Prior bleed sets a floor no wall can cross, which is why every experiment runs against two reference conditions and reports results as position between them:

- **Floor:** L0, shared window. The worst case.
- **Ceiling:** two seats in genuinely separate processes, no shared history (`seatkit.ceiling.SeparatedBench`). Estimates prior bleed and bounds what any wall can achieve.

`metrics.normalize` computes the position. When floor and ceiling coincide it returns NaN rather than a number, because "walls can buy nothing here" must not read as zero.

**What the ceiling is, for stateless APIs.** Every provider adapter here is stateless, and for a stateless API an ideal L3 projection and the ceiling hand the model the *same prompt*. `SeparatedBench` never calls the projection code, so `tests/test_ceiling.py` uses that equality as a differential test of `ladder.py`. The ceiling's empirical jobs are to estimate prior bleed and to give the floor-to-ceiling span (P11). An L3 cell that differs from the ceiling by more than sampling noise indicates an assembly defect. L3 and the ceiling genuinely diverge only when seats share server-side state: thread APIs, provider memory, or a shared retrieval layer.

## What the mock backend can and cannot show

`MockBackend` is deterministic, runs offline, and needs no key. It uses a forbidden fact only when that fact's value appears in the prompt it was handed, and never invents one. Prior bleed is therefore exactly zero, which means **any nonzero L3/L4 leak in a mock run is a routing bug in `ladder.py` and nothing else**. That is what makes it a useful regression harness.

It cannot do the other half. Its L0 > L1 > L2 leak ordering is *stipulated by three constants in the constructor*, not discovered. Its two seats draw from disjoint canned vocabularies, so E2's judge sits at ceiling and E3's seats almost never converge. It only uses values written in its setup format, so E4's post-reseed probe never leaks on it. The scripts print explicit warnings in these cases rather than reporting the numbers as findings, and every mock results file carries a notice saying so.

It is reproducible across processes: seeds are derived with CRC32, not Python's per-process salted `hash()`. That also makes the response cache hit on reruns.

**No number produced by the mock belongs in a results table.** Paper §9.5 requires a live backend on at least two model families — which is what the provider adapters are for. `preflight.py` will tell you how many distinct families you actually have keys for, and `compare_families.py` refuses to pool them, because pooling averages away exactly the disagreement the comparison exists to find.

## Repository layout

```
paper/          the paper (md, pdf, txt)
src/seatkit/
  ladder.py     the admissibility operator. the module the paper is about
  ceiling.py    the ceiling reference: seats in separate processes, independent of ladder.py
  judges.py     the judge panel: paraphrase, attribution, leak grading, stance; fingerprint; rule stand-in
  contracts.py  five clauses: needs, synergies, forbidden, instrument, externalize
  transcript.py provenance-tagged spans; without tags there is nothing to filter on
  bench.py      scene orchestration, disclosure tracking, bench-capture check
  instruments.py private assistants and buffer routing
  probes.py     nonce fact generation, the prior gate, forked leak probes
  backends.py   mock, openai, anthropic, grok, gemini, disk cache
  metrics.py    leak rates, attribution, convergence, half-life, BH, effect sizes
  artifacts.py  seat sheets and the three reseeding conditions
experiments/    E1–E4, preflight, cross-family runner, family comparison,
                judge_check.py, freeze_prereg.py, grader_agreement.py, _common.py (shared setup)
  configs/      default.yaml, judge.json, judge_gold.json, prereg.json (decisions to make)
tests/          149 tests. test_ladder.py is the architectural claim; test_ceiling.py checks it independently
docs/           enforcement ladder, protocol, backends, preregistration, changelogs
```

## The scene is the unit of independence

A scene produces dozens of attribution judgements and they are not independent: same seats, same facts, one drift trajectory. Pooling them into a single binomial test rejects a true null about **57% of the time** at a within-scene correlation of 0.2, against a nominal 5%.

So inference is at the scene level — `cluster_bootstrap_ci` resamples whole scenes, `sign_test_over_clusters` works on per-scene rates, and `paired_cluster_diff` pairs the surface-vs-content contrast within scene because both judges read the same transcript. Reported n is always scenes.

A practical tell: if your n looks like `scenes × turns × seats`, something got pooled that shouldn't have been. That is also what made `binomial_p` overflow — `comb(1920, 960)` is ~1e576 and exceeds float range — so the crash was a symptom of the design error rather than a separate bug. It now computes in log space via `lgamma`, and E2 records the pooled p-value under a key marked `DO_NOT_USE` so the gap stays visible.

## Two things worth knowing before relying on this

**Isolation relocates the hallway; it does not abolish it.** L4 introduces a bench that legitimately sees everything. Bench capture — the arbiter starting to summarize or adjudicate — is the residual risk, and it lives in the one component that has to be trusted. `Bench.assert_transport_only` checks it, and runs at the end of every scene.

**Enforcement handles ignorance well and motivation badly.** You cannot construct "this seat wants to protect its claim" by filtering a transcript; you can only construct "this seat has no access to the considerations that would make abandoning it attractive." The needs clause stays requested all the way up the ladder.

## Status

Reference implementation for a single-author paper. The predictions in §9 have not been run against live models; the harness is what makes running them possible. Negative results are mapped to kill conditions in `docs/PREREGISTRATION.md` in advance, so a failed prediction cannot be reinterpreted afterwards as a partial success. Freeze the preregistration with `experiments/freeze_prereg.py` before collecting data.

Known gaps between the harness and §9 are listed in [`docs/CHANGES-review-fixes.md`](docs/CHANGES-review-fixes.md#still-open-decisions-for-the-author-not-bugs). The main ones: the L1 half of P6 is untestable as implemented (nothing can be private below L3), E4 phase 2 has the leak probe but not the 8-turn continuation, and the P8 equivalence margin is not yet set.

## Licence

MIT. See `LICENSE`.
