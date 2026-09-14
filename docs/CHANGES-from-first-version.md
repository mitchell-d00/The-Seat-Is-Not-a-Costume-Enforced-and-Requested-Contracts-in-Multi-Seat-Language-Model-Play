# What changed, and why

This version restructures the argument around one distinction and rebuilds the
empirical section. Notes for anyone who read the earlier text.

## The structural change

The earlier version argued that seats are constraint bundles and that bleed is
the default, then listed "unavoidable full availability" as a kill condition —
without noticing that, for the single-window case it spent most of its length
discussing, that condition was close to already satisfied. A paragraph asking a
model not to use tokens it can see competes against every one of those tokens,
and §3 and §6 had already argued the paragraph loses.

The enforcement ladder is the response. It converts a near-fatal admission into
the paper's main finding: full availability *is* unavoidable in a shared window,
which is the argument for moving walls out of the prompt and into the serving
layer. The kill condition now exists in two versions, one of which the author
expects to fire and one of which would end the framework.

## What got added

- **Three kinds of bleed** (§6), separating compliance from routing from prior.
- **Prior bleed and the isolation ceiling.** A shared generator means seats are
  correlated even under perfect isolation. Every experiment now needs a ceiling
  control, and results are reported as position between floor and ceiling. The
  earlier version had no way to say how much distinctness a wall could buy.
- **Bench capture** (§5.4). Isolation relocates the hallway to the one component
  that must be trusted. Saying so is better than implying the wall is complete.
- **A worked failure** (§6.1). The earlier version had no example anywhere.
- **Related work** (§11), with an explicit note that the citations are unverified.
- **The motivation asymmetry** (§8). Enforcement removes alternatives; it cannot
  install a want. The needs clause stays requested at every rung.

## What got fixed

| Earlier | Problem | Now |
|---|---|---|
| Label-stripped attribution as the definition of seats holding *and* the prediction | Circular | E1 measures bleed by direct probe; E2 regresses attribution on it |
| Fetch-only vs critic-only instruments | Content confound — a critic emits critical sentences | Identical replay log; only visibility varies |
| Baseline: "a fresh costume given the last page of transcript" | A tail *is* externalization, just lossy | Three-way: none / tail / sheet |
| "A seat is a contract" | The paper's own conflation error | A seat is a trajectory under contract; the pair |
| `U = F(M, A, E, C)` | Passing the contract to the generator erases the distinction | `A = alpha(S, C)`; the contract enters once, inside the operator |
| §5 and §6 both about coupling | Duplication | Merged; §6.1 carries the example |
| "Grok-bio" / "Grok-geo" | Vendor-specific leftover | Generic seat ids |

## What survived unchanged

The five clauses. Costume collapse as the unconstrained attractor. The refusal to
treat fluency as occupancy. The instance-thesis relationship. Those were right.
