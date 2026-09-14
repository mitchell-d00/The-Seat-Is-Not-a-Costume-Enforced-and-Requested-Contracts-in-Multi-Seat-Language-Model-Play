"""E3 - Instrument symmetry with the content confound removed (paper 9.3).

The original prediction compared fetch-only against critic-only instruments.
That comparison is uninterpretable: a critic instrument emits critical
sentences, so of course the seats diverge. That is a fact about injected
content, not about walls.

The fix is a replay log. Both arms receive byte-identical instrument output.
Only visibility differs. Any divergence is attributable to routing and to
nothing else.

Usage:
    python experiments/e3_instrument_symmetry.py --backend mock --scenes 40
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seatkit import (Bench, family_of, Instrument, Level, SceneConfig, build_replay_log,
                     disagreement_half_life, facing_pair, get_backend,
                     hedge_index, lexical_convergence, make_facts)

CELLS = [(lvl, shared) for lvl in (Level.REQUESTED, Level.PARTITIONED)
         for shared in (True, False)]


def run_cell(level, shared_buffer, backend, scenes, turns):
    conv, hedges, halflives = [], [], []
    for i in range(scenes):
        a, b = make_facts(3, seed=1000 + i, prefix="a"), make_facts(3, seed=5000 + i, prefix="b")
        geo, bio = facing_pair(
            "geo", "bio", a, b,
            a_extra={"persona": "a geologist", "needs": ("protect the rapid-event reading",)},
            b_extra={"persona": "a paleobiologist", "needs": ("protect the quiet-interval reading",)},
        )
        # Identical instrument content in both arms. This is the control.
        logs = {s.seat_id: build_replay_log(backend, s, turns + 1, seed=i)
                for s in (geo, bio)}
        instruments = {
            s.seat_id: Instrument(host=s, backend=backend,
                                  shares_buffer=shared_buffer,
                                  replay_log=logs[s.seat_id])
            for s in (geo, bio)
        }
        res = Bench([geo, bio], backend, instruments).run(
            SceneConfig(turns=turns, level=level, seed=i,
                        shared_instrument_buffer=shared_buffer))

        lc = lexical_convergence(res.transcript, "geo", "bio")
        if lc:
            conv.append(lc)
        hg = hedge_index(res.transcript, "geo")
        if hg:
            hedges.append(mean(hg.values()))

        # Proxy for positional disagreement: 1 - lexical similarity. A live run
        # replaces this with a judge scoring compatibility on the scene's
        # designated contested proposition; similarity is a stand-in only, and
        # conflating the two would overstate what has been measured.
        incompat = {t: 1.0 - v for t, v in lc.items()}
        halflives.append(disagreement_half_life(incompat))
    return conv, hedges, halflives


def curve(convs):
    turns = sorted({t for c in convs for t in c})
    return {str(t): round(mean([c[t] for c in convs if t in c]), 4) for t in turns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock",
                    help="mock | openai:<model> | anthropic:<model> | "
                         "grok:<model> | gemini:<model>")
    ap.add_argument("--cache", default=None,
                    help="directory for the response cache; omit to disable")
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--turns", type=int, default=24)
    ap.add_argument("--out", default="results/e3.json")
    args = ap.parse_args()

    backend = get_backend(args.backend, cache=args.cache or False)
    out = {"backend": args.backend, "family": family_of(args.backend),
           "scenes": args.scenes, "cells": {}}

    for level, shared in CELLS:
        conv, hedges, hl = run_cell(level, shared, backend, args.scenes, args.turns)
        name = f"{level.label}_{'shared' if shared else 'private'}"
        resolved = [h for h in hl if h is not None]
        out["cells"][name] = {
            "convergence_curve": curve(conv),
            "mean_hedge_index": round(mean(hedges), 4) if hedges else None,
            "median_half_life": sorted(resolved)[len(resolved) // 2] if resolved else None,
            "n_never_converged": sum(1 for h in hl if h is None),
            "n_scenes": args.scenes,
        }
        c = out["cells"][name]
        print(f"{name:<18} half_life={c['median_half_life']} "
              f"never_converged={c['n_never_converged']}/{args.scenes} "
              f"hedge={c['mean_hedge_index']}")

    print("\nPrediction 1: half-life longer with private buffers, at both levels.")
    print("Prediction 2: hedge convergence driven by buffer sharing more than by level.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
