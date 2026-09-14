# The enforcement ladder

Implementation notes for `seatkit.ladder`. The paper's §5 is the argument; this is what it takes to build.

## The operator

`assemble_view(transcript, seat, level, turn)` is the admissibility operator `alpha(S_t, C_k) -> A_k,t`.

The contract enters **once**, inside `alpha`, and is not passed to the generator as a separate argument. That is the formal content of the whole distinction:

- At L0–L2, `alpha` is the identity map. The contract is smuggled back in as text inside `A_k,t`. The constraint is *content*, and a model may ignore content.
- At L3–L4, `alpha` is a real projection. What it excludes is not available to be ignored. The constraint is *structure*.

A formalism that passes `C_k` to the generator alongside `A_k,t` has erased the only distinction worth drawing.

## What L3 requires

**1. Provenance on every span.** `Span` carries `origin`, `author`, and `visible_to`. There is no default visibility; construction requires one. A span that arrives without provenance cannot be governed, which is why retrieval is the hard case.

**2. Per-seat view assembly.** Build each call's context from spans whose `visible_to` contains this seat. Forbidden spans are **absent, not redacted** — a redaction marker is itself information about what was removed. `test_enforced_view_has_no_redaction_marker` checks this.

**3. A declared transfer protocol.** Spoken content crosses; nothing else does. This is where most implementations quietly fail: it is easy to filter a transcript by speaker, and harder to notice that a seat's instrument produced 400 tokens of analysis, the seat summarized one line aloud, and the turn object forwarded all of it.

## Disclosure is not leakage

A seat saying its **own** held fact out loud is a legitimate transfer through the one channel designed to carry content across seats. It is not a wall failure.

The library tracks this. `Bench` records disclosed fact ids; `ProbeResult.counts` excludes them from both numerator and denominator of the leak rate.

Getting this wrong in either direction ruins the measurement:

- Count disclosure as leakage, and well-designed scenes look broken in proportion to how talkative the holders were.
- Ignore disclosure entirely, and a sibling repeating something said openly ten turns ago is scored as a wall failure.

The routing check in `assemble_view` reflects the same asymmetry: only `SETUP` and `INSTRUMENT` spans count. A forbidden fact arriving through a spoken span is the system working.

## Prohibition text is omitted above L2

At L3/L4 the seat never sees `Contract.prohibition_text()`. Naming a prohibition leaks the existence of the thing prohibited — "do not speculate about the assay" tells a seat there is an assay. Since the content is already absent, the sentence buys nothing and costs information.

## The visibility matrix

`visibility_for(author, origin, all_seats, level, instrument_shares_buffer)`:

| origin | L0–L2 | L3–L4, private buffer | L3–L4, shared buffer |
|---|---|---|---|
| `SETUP` (held facts) | all seats | holder only | holder only |
| `SEAT` (spoken) | all seats | all seats | all seats |
| `TRANSFER` | all seats | all seats | all seats |
| `INSTRUMENT` | all seats | host only | all seats, tagged `hallway` |
| `BENCH` | all seats | all seats | all seats |

Two flags can make an instrument shared: `InstrumentRights.writes_to_shared` (a contract clause) and `Instrument.shares_buffer` (a deployment fact). They are OR-ed and kept separate on purpose. Conflating a clause with a deployment fact is the error the entire paper is about.

## Where isolation stops

**Bench capture.** L4 introduces a component that legitimately sees everything. If it starts summarizing, adjudicating, or smoothing, it is a hallway with a gavel. `Bench.assert_transport_only` runs after every scene and raises on any bench-authored span.

**Motivation.** Enforcement removes alternatives; it cannot install a want. The needs clause (`Contract.needs`) is requested at every rung, including L4. Any account of the ladder that claims otherwise is overselling it.

**Reseeding.** Enforcement during a session does not survive an undisciplined handoff out of it. A raw transcript tail discards the visibility tags the live system was maintaining, and E4 finds constraint survival degrading under tail reseeding *even when phase 1 ran at L3*. The wall has to be rebuilt on the way out, not just held during.
