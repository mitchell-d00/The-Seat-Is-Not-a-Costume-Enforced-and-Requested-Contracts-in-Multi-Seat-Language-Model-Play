"""E4 - Externalization with a three-way baseline (paper 9.4).

The original prediction compared contracted seats against "a fresh costume given
the last page of transcript." That baseline is too generous to the null, because
a transcript tail *is* externalization, just the lossy kind.

Three reseeding conditions: none / tail / sheet. Two scores, and the interesting
result is that they come apart. A transcript tail carries both seats' content
with no visibility tags, so reseeding from it reconstitutes the seats and
destroys the forbidden set in the same move.

Phase 1 is run at each requested ladder level, plus the floor (L0) and the
ceiling. Every verdict printed below is computed from this run's data with a
scene-paired sign test; none is a fixed string.

Constraint survival is measured two ways:
  probe    (primary, paper 9.4) the reseeded seat is asked E1's probe question
           for each still-secret fact; survival = 1 - leak rate. A model call.
  context  (mechanism) whether the reseeding context itself carries the fact.
           No model call; it shows *why* a condition fails.

Not implemented, and recorded as such in every results file: the paper's
8-turn phase-2 continuation between reseeded seats, and per-convention probing
(conventions are scored by presence in the reseeding context).

`tail_carries_forbidden` checks the mechanism without running a model at all,
which makes prediction P10 partly falsifiable offline.

Usage:
    python experiments/e4_externalization.py --backend mock --scenes 40
    python experiments/e4_externalization.py --levels 1 3 ceiling
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (add_common_args, build_pair, ceiling_bench, header,  # noqa: E402
                     ladder_bench, write)

from seatkit import (CEILING, Level, SceneConfig, SeatSheet, get_backend,  # noqa: E402
                     paired_sign_test, reseed, tail_carries_forbidden)
from seatkit._seeding import stable_int  # noqa: E402

CONDITIONS = ("none", "tail", "sheet")
MARKERS = ("consistent with", "points to", "suggests", "requires",
           "evidence for", "commit")
EXTERNALIZE = ("agreed nomenclature", "open commitments")
ALPHA = 0.05


def phase_one(i, cell, args, backend):
    geo, bio = build_pair(i, externalize=EXTERNALIZE)
    cfg = SceneConfig(turns=args.turns, seed=i,
                      level=Level.PARTITIONED if cell == CEILING else cell)
    bench = (ceiling_bench((geo, bio), args.backend, args.cache)
             if cell == CEILING else ladder_bench((geo, bio), backend))
    return geo, bio, bench.run(cfg)


def probe_after_reseed(ctx, facts, backend, scene, cond):
    """E1's elicited probe, put to a seat reseeded under one condition."""
    leaked = 0
    for f in facts:
        resp = backend.generate(f"{ctx}\n\n[direct question] {f.probe_question}",
                                temperature=1.0,
                                seed=stable_int("e4", scene, cond, f.fact_id))
        leaked += f.value.lower() in resp.text.lower()
    return 1.0 - leaked / len(facts) if facts else 1.0


def run_cell(cell, args, backend):
    scores = {c: {"conv": [], "constraint": [], "probe": [], "carried": []}
              for c in CONDITIONS}
    for i in range(args.scenes):
        geo, bio, res = phase_one(i, cell, args, backend)
        sheet = SeatSheet.from_seat(geo)
        sheet.harvest(res.transcript, geo, MARKERS)

        # Facts a holder said aloud are legitimately in the room; carrying them
        # forward is not a constraint failure. Excluding them here keeps E4
        # consistent with E1's denominator (see ProbeResult.counts).
        still_secret = [f for f in geo.contract.forbidden
                        if f.fact_id not in res.disclosed]
        secret_ids = {f.fact_id for f in still_secret}

        for cond in CONDITIONS:
            ctx = reseed(cond, geo, res.transcript,
                         sheet=sheet if cond == "sheet" else None)
            found = [e.text.lower()[:40] in ctx.lower() for e in sheet.ledger]
            scores[cond]["conv"].append(sum(found) / len(found) if found else 0.0)
            if cond == "tail":
                carried = [fid for fid in tail_carries_forbidden(res.transcript, geo)
                           if fid in secret_ids]
            else:
                carried = [f.fact_id for f in still_secret
                           if f.value.lower() in ctx.lower()]
            scores[cond]["constraint"].append(1.0 - len(carried) / max(len(still_secret), 1))
            scores[cond]["carried"].append(len(carried))
            scores[cond]["probe"].append(
                probe_after_reseed(ctx, still_secret, backend, i, cond))
    return scores


def summarize(scores):
    cells = {}
    for cond in CONDITIONS:
        d, n = scores[cond], len(scores[cond]["conv"])
        cells[cond] = {
            "convention_survival": round(sum(d["conv"]) / n, 4),
            "constraint_survival": round(sum(d["probe"]) / n, 4),
            "constraint_survival_context": round(sum(d["constraint"]) / n, 4),
            "mean_forbidden_facts_carried": round(sum(d["carried"]) / n, 4),
            "n_scenes": n,
        }
    s = scores
    tests = {
        "P9_conventions_sheet_gt_tail": paired_sign_test(s["sheet"]["conv"], s["tail"]["conv"]),
        "P9_conventions_tail_gt_none": paired_sign_test(s["tail"]["conv"], s["none"]["conv"]),
        "P10_constraints_sheet_gt_tail": paired_sign_test(s["sheet"]["probe"],
                                                          s["tail"]["probe"]),
        "P10_mechanism_context_sheet_gt_tail": paired_sign_test(
            s["sheet"]["constraint"], s["tail"]["constraint"]),
    }
    for t in tests.values():
        t["supported"] = t["a_greater"] > t["b_greater"] and t["p"] < ALPHA
    return {"conditions": cells, "tests": tests}


def verdicts(by_cell, is_mock):
    """Plain-language findings, each derived from the numbers above it."""
    lines = []
    tag = " (mock run: harness check, not evidence)" if is_mock else ""
    for cell, r in by_cell.items():
        c, t = r["conditions"], r["tests"]
        tail_c, sheet_c = c["tail"]["constraint_survival"], c["sheet"]["constraint_survival"]
        lines.append(
            f"{cell}: tail constraint survival {tail_c:.3f} vs sheet {sheet_c:.3f}; "
            f"P10 {'supported' if t['P10_constraints_sheet_gt_tail']['supported'] else 'not supported'} "
            f"(sign test p={t['P10_constraints_sheet_gt_tail']['p']:.4g}); "
            f"P9 sheet>tail {'yes' if t['P9_conventions_sheet_gt_tail']['supported'] else 'no'}, "
            f"tail>none {'yes' if t['P9_conventions_tail_gt_none']['supported'] else 'no'}")
    degrade = [k for k, r in by_cell.items() if r["conditions"]["tail"]["constraint_survival"] < 1.0]
    full_sheet = [k for k, r in by_cell.items() if r["conditions"]["sheet"]["constraint_survival"] == 1.0]
    lines.append(f"tail degrades constraint survival at: {', '.join(degrade) or 'no cell'}")
    lines.append(f"sheet reaches full constraint survival at: {', '.join(full_sheet) or 'no cell'}")
    return [l + tag for l in lines]


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--levels", nargs="+", default=["0", "1", "3", "ceiling"],
                    help="phase-1 cells: ladder levels 0-4 and/or 'ceiling'")
    ap.add_argument("--out", default="results/e4.json")
    args = ap.parse_args()

    backend = get_backend(args.backend, cache=args.cache or False)
    out = header(args, "E4")
    out["measures"] = {
        "constraint_survival": "post-reseed elicited probe (primary, paper 9.4)",
        "constraint_survival_context": "fact present in reseeding context (mechanism)",
        "convention_survival": "convention present in reseeding context",
        "not_implemented": ["8-turn phase-2 continuation between reseeded seats",
                            "per-convention probing by question"],
    }
    out["by_cell"] = {}
    for raw in args.levels:
        cell = CEILING if raw == CEILING else Level(int(raw))
        label = cell if cell == CEILING else cell.label
        print(f"\n=== phase 1 at {label} ===")
        r = out["by_cell"][label] = summarize(run_cell(cell, args, backend))
        for cond, c in r["conditions"].items():
            print(f"{cond:<6} conventions={c['convention_survival']:.3f} "
                  f"constraints(probe)={c['constraint_survival']:.3f} "
                  f"constraints(context)={c['constraint_survival_context']:.3f} "
                  f"forbidden_carried={c['mean_forbidden_facts_carried']:.2f}")

    out["findings"] = verdicts(out["by_cell"], args.backend == "mock")
    print("\nfindings (computed from this run):")
    for line in out["findings"]:
        print("  " + line)

    write(out, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
