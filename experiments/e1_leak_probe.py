"""E1 - Leak probe. Measures bleed independently of attribution (paper 9.1).

The original version of this argument defined seats as holding if attribution
succeeded, then predicted attribution would succeed when seats held. E1 exists
so that bleed has a measure that does not mention attribution at all.

Cells: L0-L4 plus the ceiling (seats in separate processes; ceiling.py). Every
cell is reported raw and as position between floor (L0) and ceiling.

Scoring: exact value match always. With --judge-backend, a model grader also
checks every non-exact response for a paraphrased leak, and a 20% audit sample
is written for human labelling; experiments/grader_agreement.py turns the
labelled file into inter-grader agreement (paper 9.1, Scoring).

Usage:
    python experiments/e1_leak_probe.py --backend mock --scenes 40
    python experiments/e1_leak_probe.py --backend anthropic:<model> \\
        --judge-backend openai:<model> --cache .cache/run1
"""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (add_common_args, build_pair, ceiling_bench, header,  # noqa: E402
                     judge_record, ladder_bench, resolve_judge, write)

from seatkit import (CEILING, Level, SceneConfig, cohens_h, get_backend,  # noqa: E402
                     grade_probes, leak_by_turn, leak_rates, make_facts,
                     normalize, verify_prior)
from seatkit.metrics import cluster_bootstrap_ci  # noqa: E402

LEVELS = [Level.COSTUME, Level.REQUESTED, Level.REMINDED,
          Level.PARTITIONED, Level.ISOLATED]
AUDIT_FRACTION = 0.20


def run_cell(cell, args, backend, temperature, probe_turns, grader):
    """Returns (scoring probes, per-scene elicited clusters, routing bleed)."""
    probes, clusters_exact, clusters_graded, routing = [], [], [], 0
    for i in range(args.scenes):
        geo, bio = build_pair(i)
        cfg = SceneConfig(turns=args.turns, seed=i, temperature=temperature,
                          probe_turns=probe_turns,
                          level=Level.PARTITIONED if cell == CEILING else cell)
        bench = (ceiling_bench((geo, bio), args.backend, args.cache)
                 if cell == CEILING else ladder_bench((geo, bio), backend))
        res = bench.run(cfg)
        scene = res.scoring_probes()
        if grader is not None:
            facts = {f.fact_id: f for s in (geo, bio) for f in s.contract.holds}
            scene = grade_probes(scene, facts, grader)
        probes.extend((i, p) for p in scene)
        el = [p for p in scene if p.kind == "elicited"]
        clusters_exact.append([p.leaked for p in el])
        clusters_graded.append([p.leaked_any for p in el])
        routing += res.routing_bleed_count
    return probes, clusters_exact, clusters_graded, routing


def summarize(label, tagged, cl_exact, cl_graded, routing, graded_on):
    probes = [p for _, p in tagged]
    rates = leak_rates(probes, routing_bleed=routing)
    cell = rates.as_dict()
    pt, lo, hi = cluster_bootstrap_ci(cl_exact, seed=0)
    cell["elicited_ci95_scene_bootstrap"] = [round(lo, 4), round(hi, 4)]
    cell["elicited_by_turn"] = {str(k): round(v, 4)
                                for k, v in leak_by_turn(probes).items()}
    if graded_on:
        g = leak_rates(probes, routing_bleed=routing, graded=True)
        gpt, glo, ghi = cluster_bootstrap_ci(cl_graded, seed=0)
        cell["graded"] = {"elicited_leak_rate": round(g.elicited, 4),
                          "volunteered_leak_rate": round(g.volunteered, 4),
                          "elicited_ci95_scene_bootstrap": [round(glo, 4), round(ghi, 4)],
                          "n_paraphrase_only": sum(1 for p in probes
                                                   if p.graded and not p.leaked)}
    return cell


def write_audit(all_tagged, facts_lookup, path, seed=0):
    """Deterministic 20% sample of scoring probes for human labelling."""
    rng = random.Random(seed)
    rows = []
    for cell, tagged in all_tagged.items():
        for scene, p in tagged:
            if rng.random() < AUDIT_FRACTION:
                f = facts_lookup[(scene, p.fact_id)]
                rows.append({"cell": cell, "scene": scene, "turn": p.turn,
                             "seat": p.seat_id, "kind": p.kind, "fact_id": p.fact_id,
                             "fact": f.statement(), "response": p.response,
                             "exact": p.leaked, "model": p.graded if p.graded is not None
                             else p.leaked, "human": None})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} audit rows to {path} (fill 'human' with true/false, "
          f"then run experiments/grader_agreement.py)")


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap, judges=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sweep", action="store_true",
                    help="also run the secondary L1 temperature sweep (0.3/0.7/1.0)")
    ap.add_argument("--out", default="results/e1.json")
    args = ap.parse_args()

    # Judges first: a misconfigured run should fail before any client exists.
    panel = resolve_judge(args)
    backend = get_backend(args.backend, cache=args.cache or False)
    grader = panel.grader if panel is not None else None
    probe_turns = tuple(t for t in (4, 8, 16, 24) if t <= args.turns) or (args.turns,)

    # Preregistered gate. A pool the model already knows makes the whole
    # measurement uninterpretable (paper 6, prior bleed).
    passed, checks = verify_prior(make_facts(12, seed=7), backend)
    print(f"[gate] prior verification: {'PASS' if passed else 'FAIL'} "
          f"({sum(c.recovered for c in checks)}/{len(checks)} recovered)")
    if not passed:
        print("[gate] pool rejected; regenerate before collecting data.")
        return 1

    out = header(args, "E1")
    out.update({"temperature": args.temperature, "prior_gate": "pass",
                "leak_scoring": grader.name if grader else "exact-match-only",
                "cells": {}})
    facts_lookup = {}
    for i in range(args.scenes):
        for s in build_pair(i):
            for f in s.contract.holds:
                facts_lookup[(i, f.fact_id)] = f

    all_tagged = {}
    for cell in LEVELS + [CEILING]:
        label = cell if cell == CEILING else cell.label
        tagged, ce, cg, routing = run_cell(cell, args, backend, args.temperature,
                                           probe_turns, grader)
        all_tagged[label] = tagged
        out["cells"][label] = summarize(label, tagged, ce, cg, routing, grader is not None)
        c = out["cells"][label]
        name = "CEILING" if cell == CEILING else cell.name
        print(f"{label:>7} {name:<12} elicited={c['elicited_leak_rate']:.3f} "
              f"{c['elicited_ci95_scene_bootstrap']} "
              f"volunteered={c['volunteered_leak_rate']:.3f} routing_bleed={routing}")

    cells = out["cells"]
    floor, ceil = cells["L0"]["elicited_leak_rate"], cells[CEILING]["elicited_leak_rate"]
    out["positions_elicited"] = {
        lbl: normalize(cells[lbl]["elicited_leak_rate"], floor, ceil).as_dict()
        for lbl in ("L1", "L2", "L3", "L4")}
    out["P2_enforced_vs_ceiling_cohens_h"] = {
        lbl: round(cohens_h(cells[lbl]["elicited_leak_rate"], ceil), 4)
        for lbl in ("L3", "L4")}
    out["P11_floor_minus_ceiling"] = {
        "elicited_span": round(floor - ceil, 4),
        "cohens_h": round(cohens_h(floor, ceil), 4),
        "ci_floor": cells["L0"]["elicited_ci95_scene_bootstrap"],
        "ci_ceiling": cells[CEILING]["elicited_ci95_scene_bootstrap"],
        "note": "Prior bleed dominance fires if this span is near zero.",
    }
    out["effect_L1_vs_L3_cohens_h"] = round(
        cohens_h(cells["L1"]["elicited_leak_rate"], cells["L3"]["elicited_leak_rate"]), 4)

    print("\nposition between floor (L0) and ceiling, elicited leak:")
    for lbl, pos in out["positions_elicited"].items():
        print(f"  {lbl}: {pos['position']}")
    print(f"floor - ceiling span = {out['P11_floor_minus_ceiling']['elicited_span']}")
    if floor == 0 and ceil == 0:
        print("[warning] floor and ceiling both zero: positions are undefined (NaN). "
              "Nothing leaked anywhere, so the ladder cannot be read from this run.")

    if args.sweep:
        out["L1_temperature_sweep"] = {}
        for t in (0.3, 0.7, 1.0):
            tagged, *_ = run_cell(Level.REQUESTED, args, backend, t, probe_turns, None)
            r = leak_rates([p for _, p in tagged])
            out["L1_temperature_sweep"][str(t)] = round(r.elicited, 4)
        print(f"L1 sweep (exact): {out['L1_temperature_sweep']}")

    out["judge"] = judge_record(panel, args)
    if grader is not None:
        out["grader_unparseable"] = grader.unparseable
        write_audit(all_tagged, facts_lookup,
                    str(Path(args.out).with_name(Path(args.out).stem + "_audit.jsonl")))

    write(out, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
