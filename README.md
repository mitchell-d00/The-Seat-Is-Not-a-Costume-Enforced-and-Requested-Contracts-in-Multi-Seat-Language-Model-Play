# seat-contracts

Reference implementation and experimental harness for **"The Seat Is Not a Costume: Enforced and Requested Contracts in Multi-Seat Language Model Play"** (M. D. McPhetridge).

The paper is in [`paper/`](paper/) as Markdown, PDF, and plain text.

## What this is for

The library exists to make one distinction executable.

A prohibition written in a shared context window is a **request**. The forbidden tokens are sitting right there; the model honours the prohibition only by complying, and compliance degrades under load — most sharply at the moment the forbidden fact becomes relevant, which is exactly what the wall was built for.

A prohibition implemented in view assembly is **enforced**. The forbidden tokens are not in the seat's context at all. There is no prohibition to comply with, because there is nothing to suppress.

These are not two strengths of one dial. They are different objects with different failure modes. Most multi-persona work is requested, most of its failures are compliance failures, and most of its proposed fixes are more paragraphs.

The claim is a unit test:

```python
view = assemble_view(transcript, geo_seat, Level.REQUESTED, turn=1)
assert view.contains(forbidden_fact.value)      # the wall is a paragraph

view = assemble_view(transcript, geo_seat, Level.PARTITIONED, turn=1)
assert not view.contains(forbidden_fact.value)  # the wall is a projection
```

See `tests/test_ladder.py`.

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
pytest                      # 86 tests, no API key, under a second
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

Live runs take any provider spec, and the experimental logic does not change:

```bash
python experiments/preflight.py --backends openai:<model> anthropic:<model> gemini:<model>
./experiments/run_cross_family.sh openai:<model> anthropic:<model> grok:<model>
python experiments/compare_families.py results/
```

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
- **Ceiling:** two seats in genuinely separate processes, no shared history. Estimates prior bleed and bounds what any wall can achieve.

`metrics.normalize` enforces this. When floor and ceiling coincide it returns NaN rather than a number, because "walls can buy nothing here" must not read as zero.

## What the mock backend can and cannot show

`MockBackend` is deterministic, runs offline, and needs no key. It uses a forbidden fact only when that fact's value appears in the prompt it was handed, and never invents one. Prior bleed is therefore exactly zero, which means **any nonzero L3/L4 leak in a mock run is a routing bug in `ladder.py` and nothing else**. That is what makes it a useful regression harness.

It cannot do the other half. Its L0 > L1 > L2 leak ordering is *stipulated by three constants in the constructor*, not discovered. Its two seats draw from disjoint canned vocabularies, so E2's judge sits at ceiling and E3's seats never converge — the scripts print explicit warnings when this happens rather than reporting the numbers as findings.

**No number produced by the mock belongs in a results table.** Paper §9.5 requires a live backend on at least two model families — which is what the provider adapters are for. `preflight.py` will tell you how many distinct families you actually have keys for, and `compare_families.py` refuses to pool them, because pooling averages away exactly the disagreement the comparison exists to find.

## Repository layout

```
paper/          the paper (md, pdf, txt)
src/seatkit/
  ladder.py     the admissibility operator. the module the paper is about
  contracts.py  five clauses: needs, synergies, forbidden, instrument, externalize
  transcript.py provenance-tagged spans; without tags there is nothing to filter on
  bench.py      scene orchestration, disclosure tracking, bench-capture check
  instruments.py private assistants and buffer routing
  probes.py     nonce fact generation, the prior gate, forked leak probes
  backends.py   mock, openai, anthropic, grok, gemini, disk cache
  metrics.py    leak rates, attribution, convergence, half-life, BH, effect sizes
  artifacts.py  seat sheets and the three reseeding conditions
experiments/    E1–E4, preflight, cross-family runner, family comparison
tests/          86 tests. test_ladder.py is the paper's central claim
docs/           enforcement ladder, protocol, backends, preregistration
```

## The scene is the unit of independence

A scene produces dozens of attribution judgements and they are not independent: same seats, same facts, one drift trajectory. Pooling them into a single binomial test rejects a true null about **57% of the time** at a within-scene correlation of 0.2, against a nominal 5%.

So inference is at the scene level — `cluster_bootstrap_ci` resamples whole scenes, `sign_test_over_clusters` works on per-scene rates, and `paired_cluster_diff` pairs the surface-vs-content contrast within scene because both judges read the same transcript. Reported n is always scenes.

A practical tell: if your n looks like `scenes × turns × seats`, something got pooled that shouldn't have been. That is also what made `binomial_p` overflow — `comb(1920, 960)` is ~1e576 and exceeds float range — so the crash was a symptom of the design error rather than a separate bug. It now computes in log space via `lgamma`, and E2 records the pooled p-value under a key marked `DO_NOT_USE` so the gap stays visible.

## Two things worth knowing before relying on this

**Isolation relocates the hallway; it does not abolish it.** L4 introduces a bench that legitimately sees everything. Bench capture — the arbiter starting to summarize or adjudicate — is the residual risk, and it lives in the one component that has to be trusted. `Bench.assert_transport_only` checks it, and runs at the end of every scene.

**Enforcement handles ignorance well and motivation badly.** You cannot construct "this seat wants to protect its claim" by filtering a transcript; you can only construct "this seat has no access to the considerations that would make abandoning it attractive." The needs clause stays requested all the way up the ladder.

## Status

Reference implementation for a single-author paper. The predictions in §9 have not been run against live models; the harness is what makes running them possible. Negative results are mapped to kill conditions in `docs/PREREGISTRATION.md` in advance, so a failed prediction cannot be reinterpreted afterwards as a partial success.

## Licence

MIT. See `LICENSE`.
