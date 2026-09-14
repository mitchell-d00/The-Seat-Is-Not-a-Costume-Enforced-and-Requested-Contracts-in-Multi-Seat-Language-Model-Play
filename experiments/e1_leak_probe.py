"""E1 - Leak probe. Measures bleed independently of attribution (paper 9.1).

The original version of this argument defined seats as holding if attribution
succeeded, then predicted attribution would succeed when seats held. E1 exists
so that bleed has a measure that does not mention attribution at all.

Usage:
    python experiments/e1_leak_probe.py --backend mock --scenes 40
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seatkit import (Bench, family_of, Instrument, Level, SceneConfig, cohens_h,
                     facing_pair, get_backend, leak_by_turn, leak_rates,
                     make_facts, verify_prior)

LEVELS = [Level.COSTUME, Level.REQUESTED, Level.REMINDED,
          Level.PARTITIONED, Level.ISOLATED]


def build_scene(i, backend):
    a = make_facts(3, seed=1000 + i, prefix="a")
    b = make_facts(3, seed=5000 + i, prefix="b")
    # Counterbalance: pools swap seats on odd scenes so fact identity never
    # confounds with seat identity.
    if i % 2:
        a, b = b, a
    geo, bio = facing_pair(
        "geo", "bio", a, b,
        a_extra={"persona": "a geologist",
                 "needs": ("protect the rapid-event reading",)},
        b_extra={"persona": "a paleobiologist",
                 "needs": ("protect the quiet-interval reading",)},
    )
    instruments = {s.seat_id: Instrument(host=s, backend=backend)
                   for s in (geo, bio)}
    return Bench([geo, bio], backend, instruments)


def run_cell(level, backend, scenes, turns, temperature, probe_turns):
    probes, routing = [], 0
    for i in range(scenes):
        bench = build_scene(i, backend)
        res = bench.run(SceneConfig(turns=turns, level=level, seed=i,
                                    temperature=temperature,
                                    probe_turns=probe_turns))
        probes.extend(res.scoring_probes())
        routing += res.routing_bleed_count
    return probes, routing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock",
                    help="mock | openai:<model> | anthropic:<model> | "
                         "grok:<model> | gemini:<model>")
    ap.add_argument("--cache", default=None,
                    help="directory for the response cache; omit to disable")
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--turns", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default="results/e1.json")
    args = ap.parse_args()

    backend = get_backend(args.backend, cache=args.cache or False)
    probe_turns = tuple(t for t in (4, 8, 16, 24) if t <= args.turns) or (args.turns,)

    # Preregistered gate. A pool the model already knows makes the whole
    # measurement uninterpretable (paper 6, prior bleed).
    passed, checks = verify_prior(make_facts(12, seed=7), backend)
    print(f"[gate] prior verification: {'PASS' if passed else 'FAIL'} "
          f"({sum(c.recovered for c in checks)}/{len(checks)} recovered)")
    if not passed:
        print("[gate] pool rejected; regenerate before collecting data.")
        return 1

    out = {"backend": args.backend, "family": family_of(args.backend),
           "scenes": args.scenes, "turns": args.turns,
           "temperature": args.temperature, "prior_gate": "pass", "cells": {}}

    for level in LEVELS:
        probes, routing = run_cell(level, backend, args.scenes, args.turns,
                                   args.temperature, probe_turns)
        rates = leak_rates(probes, routing_bleed=routing)
        cell = rates.as_dict()
        cell["elicited_by_turn"] = {str(k): round(v, 4)
                                    for k, v in leak_by_turn(probes).items()}
        out["cells"][level.label] = cell
        print(f"{level.label:>3} {level.name:<12} "
              f"elicited={rates.elicited:.3f} volunteered={rates.volunteered:.3f} "
              f"routing_bleed={routing}")

    l1 = out["cells"]["L1"]["elicited_leak_rate"]
    l3 = out["cells"]["L3"]["elicited_leak_rate"]
    out["effect_L1_vs_L3_cohens_h"] = round(cohens_h(l1, l3), 4)
    print(f"\nL1 vs L3 elicited leak, Cohen's h = {out['effect_L1_vs_L3_cohens_h']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
