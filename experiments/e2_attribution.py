"""E2 - Label-stripped attribution, made non-circular (paper 9.2).

Two judges. The surface judge sees raw text and may use style. The content judge
sees text after a register-normalizing paraphrase pass and may use only what was
claimed. If attribution succeeds for the surface judge and fails for the content
judge, the costume was doing the work and the contract was not. That contrast is
the paper's most direct test of its own thesis.

The primary analysis is a regression of content-judge accuracy on E1 leak rate:
two independently measured quantities, which is what the original prediction
lacked.

Usage:
    python experiments/e2_attribution.py --backend mock --scenes 40
"""
from __future__ import annotations

import argparse, json, random, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seatkit import (Bench, Instrument, Level, Origin, SceneConfig,
                     attribution_score, facing_pair, get_backend, leak_rates,
                     make_facts)
from seatkit.metrics import _content_vector, binomial_p, cosine

LEVELS = [Level.COSTUME, Level.REQUESTED, Level.REMINDED,
          Level.PARTITIONED, Level.ISOLATED]

HEDGE_STRIP = re.compile(
    r"\b(arguably|roughly|somewhat|perhaps|broadly speaking|that said|"
    r"it may be|to some extent|in fairness|it's worth noting)\b,?\s*",
    re.IGNORECASE)


def normalize_register(text: str) -> str:
    """Stand-in for the paraphrase pass that strips style but keeps content.

    In a live run this is a separate model instructed to preserve propositional
    content and neutralize register. Here it removes hedges and lowercases, which
    is enough to demonstrate the contrast; a real run must use the model pass,
    because a regex cannot remove syntactic fingerprints and will therefore
    overstate content-judge accuracy.
    """
    return HEDGE_STRIP.sub("", text).lower().strip()


def judge(spans, seats, content_only: bool, rng):
    """Leave-one-out nearest-centroid attribution over content words.

    For each turn, build each seat's centroid from that seat's *other* turns and
    assign the turn to the nearest. This is what a held-out reader does: decide
    which speaker a turn belongs with, given the rest of the transcript. Leaving
    the turn itself out of its own centroid is what keeps the measure honest.

    The surface judge sees raw text. The content judge sees register-normalized
    text, so it can use claims but not style. The gap between them is the
    measurement: style differences without knowledge differences mean the
    costume was doing the work.
    """
    docs = [((normalize_register(sp.text) if content_only else sp.text.lower()),
             sp.author) for sp in spans]
    vecs = [_content_vector(t) for t, _ in docs]
    ids = [s.seat_id for s in seats]

    totals = {sid: Counter() for sid in ids}
    for (_, author), v in zip(docs, vecs):
        totals[author].update(v)

    out = []
    for i, (_, author) in enumerate(docs):
        scores = {}
        for sid in ids:
            cent = totals[sid].copy()
            if author == sid:
                cent.subtract(vecs[i])          # leave one out
                cent = Counter({k: v for k, v in cent.items() if v > 0})
            scores[sid] = cosine(vecs[i], cent)
        best = max(scores.values())
        winners = [sid for sid, v in scores.items() if v == best]
        pred = rng.choice(winners) if len(winners) > 1 else winners[0]
        out.append((pred, author))
    return out


def run_cell(level, backend, scenes, turns, rng):
    surface, content, probes = [], [], []
    for i in range(scenes):
        a, b = make_facts(3, seed=1000 + i, prefix="a"), make_facts(3, seed=5000 + i, prefix="b")
        geo, bio = facing_pair(
            "geo", "bio", a, b,
            a_extra={"persona": "a geologist", "needs": ("protect the rapid-event reading",)},
            b_extra={"persona": "a paleobiologist", "needs": ("protect the quiet-interval reading",)},
        )
        instruments = {s.seat_id: Instrument(host=s, backend=backend) for s in (geo, bio)}
        res = Bench([geo, bio], backend, instruments).run(
            SceneConfig(turns=turns, level=level, seed=i))
        spoken = [s for s in res.transcript if s.origin is Origin.SEAT]
        surface += judge(spoken, [geo, bio], content_only=False, rng=rng)
        content += judge(spoken, [geo, bio], content_only=True, rng=rng)
        probes += res.scoring_probes()
    return surface, content, leak_rates(probes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--turns", type=int, default=24)
    ap.add_argument("--out", default="results/e2.json")
    args = ap.parse_args()

    backend = get_backend(args.backend)
    rng = random.Random(0)
    out = {"backend": args.backend, "scenes": args.scenes, "cells": {}}
    xs, ys = [], []

    for level in LEVELS:
        surf, cont, rates = run_cell(level, backend, args.scenes, args.turns, rng)
        s_sc, c_sc = attribution_score(surf), attribution_score(cont)
        out["cells"][level.label] = {
            "surface_accuracy": round(s_sc.accuracy, 4),
            "content_accuracy": round(c_sc.accuracy, 4),
            "content_above_chance": round(c_sc.above_chance(), 4),
            "content_p_vs_chance": round(
                binomial_p(int(c_sc.accuracy * c_sc.n), c_sc.n, 0.5), 6),
            "elicited_leak_rate": round(rates.elicited, 4),
            "n_judgements": c_sc.n,
        }
        xs.append(rates.elicited)
        ys.append(c_sc.accuracy)
        print(f"{level.label:>3} surface={s_sc.accuracy:.3f} "
              f"content={c_sc.accuracy:.3f} leak={rates.elicited:.3f}")

    # Primary analysis: content-judge accuracy regressed on leak rate.
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) if den else float("nan")
    out["regression"] = {
        "slope_content_accuracy_on_leak_rate": round(slope, 4) if slope == slope else None,
        "n_cells": n,
        "note": "Negative slope is the prediction. A null slope falsifies the "
                "framework's central empirical claim.",
    }
    print(f"\nslope(content accuracy ~ leak rate) = {out['regression']['slope_content_accuracy_on_leak_rate']}")

    # Ceiling check. The mock's two seats draw from disjoint canned vocabularies,
    # so they are separable by construction and never genuinely converge. When
    # every cell sits near 1.0 the experiment has not been run, whatever the
    # numbers say. Flag it rather than tuning the mock until the table looks
    # right, which would be fabricating the result.
    accs = [c["content_accuracy"] for c in out["cells"].values()]
    if min(accs) > 0.95:
        out["ceiling_warning"] = (
            "All cells above 0.95. Judge is at ceiling and cannot discriminate. "
            "Expected with --backend mock; E2 requires a live backend."
        )
        print("\n[warning] " + out["ceiling_warning"])
    if min(accs) < 0.55 and max(accs) < 0.6:
        out["floor_warning"] = (
            "All cells near chance. Judge has no signal to recover; check that "
            "seats are producing knowledge-differentiated content at all."
        )
        print("\n[warning] " + out["floor_warning"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
