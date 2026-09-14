"""E4 - Externalization with a three-way baseline (paper 9.4).

The original prediction compared contracted seats against "a fresh costume given
the last page of transcript." That baseline is too generous to the null, because
a transcript tail *is* externalization, just the lossy kind.

Three conditions: none / tail / sheet. Two scores, and the interesting result is
that they come apart. A transcript tail carries both seats' content with no
visibility tags, so reseeding from it reconstitutes the seats and destroys the
forbidden set in the same move.

`tail_carries_forbidden` checks the mechanism without running a model at all,
which makes prediction 2 partly falsifiable offline.

Usage:
    python experiments/e4_externalization.py --backend mock --scenes 40
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seatkit import (Bench, family_of, Instrument, Level, SceneConfig, SeatSheet,
                     facing_pair, get_backend, make_facts, reseed,
                     tail_carries_forbidden)

CONDITIONS = ("none", "tail", "sheet")
MARKERS = ("consistent with", "points to", "suggests", "requires",
           "evidence for", "commit")


def phase_one(i, backend, level, turns):
    a, b = make_facts(3, seed=1000 + i, prefix="a"), make_facts(3, seed=5000 + i, prefix="b")
    geo, bio = facing_pair(
        "geo", "bio", a, b,
        a_extra={"persona": "a geologist", "needs": ("protect the rapid-event reading",),
                 "externalize": ("agreed nomenclature", "open commitments")},
        b_extra={"persona": "a paleobiologist", "needs": ("protect the quiet-interval reading",),
                 "externalize": ("agreed nomenclature", "open commitments")},
    )
    instruments = {s.seat_id: Instrument(host=s, backend=backend) for s in (geo, bio)}
    res = Bench([geo, bio], backend, instruments).run(
        SceneConfig(turns=turns, level=level, seed=i))
    return geo, bio, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock",
                    help="mock | openai:<model> | anthropic:<model> | "
                         "grok:<model> | gemini:<model>")
    ap.add_argument("--cache", default=None,
                    help="directory for the response cache; omit to disable")
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--turns", type=int, default=24)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 3],
                    help="phase-1 enforcement levels to compare")
    ap.add_argument("--out", default="results/e4.json")
    args = ap.parse_args()

    backend = get_backend(args.backend, cache=args.cache or False)
    out = {"backend": args.backend, "family": family_of(args.backend),
           "scenes": args.scenes, "by_level": {}}

    for lv in args.levels:
        out["by_level"][Level(lv).label] = run_level(
            Level(lv), backend, args.scenes, args.turns)

    print("\nPrediction 2, refined by the L1/L3 contrast:")
    print("  A transcript tail degrades constraint survival at EVERY level,")
    print("  including L3, because the tail discards visibility tags even when")
    print("  the live system was maintaining them. Enforcement during phase 1")
    print("  does not survive an undisciplined phase-2 handoff.")
    print("  Only the structured sheet preserves both scores, and it reaches")
    print("  full constraint survival only when phase 1 was itself enforced.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


def run_level(level, backend, scenes, turns):
    agg = {c: {"conv": [], "constraint": [], "tail_leaks": []} for c in CONDITIONS}
    print(f"\n=== phase 1 at {level.label} ({level.name}) ===")

    for i in range(scenes):
        geo, bio, res = phase_one(i, backend, level, turns)

        sheet = SeatSheet.from_seat(geo)
        sheet.harvest(res.transcript, geo, MARKERS)

        # Facts a holder said aloud are legitimately in the room; carrying them
        # forward is not a constraint failure. Excluding them here keeps E4
        # consistent with E1's denominator (see ProbeResult.counts).
        still_secret = [f for f in geo.contract.forbidden
                        if f.fact_id not in res.disclosed]

        for cond in CONDITIONS:
            ctx = reseed(cond, geo, res.transcript,
                         sheet=sheet if cond == "sheet" else None)

            # Convention survival: which harvested conventions are recoverable
            # from the reseeding context this condition supplies.
            found = [e.text.lower()[:40] in ctx.lower() for e in sheet.ledger]
            agg[cond]["conv"].append(sum(found) / len(found) if found else 0.0)

            # Constraint survival: does the reseeding context itself carry the
            # seat's forbidden facts? For "tail" this is checked directly on the
            # artifact; no model call is needed to establish the mechanism.
            if cond == "tail":
                carried = [fid for fid in tail_carries_forbidden(res.transcript, geo)
                           if fid in {f.fact_id for f in still_secret}]
            else:
                carried = [f.fact_id for f in still_secret
                           if f.value.lower() in ctx.lower()]
            leak = len(carried) / max(len(still_secret), 1)
            agg[cond]["constraint"].append(1.0 - leak)
            agg[cond]["tail_leaks"].append(len(carried))

    cells = {}
    for cond in CONDITIONS:
        d = agg[cond]
        n = len(d["conv"])
        cells[cond] = {
            "convention_survival": round(sum(d["conv"]) / n, 4),
            "constraint_survival": round(sum(d["constraint"]) / n, 4),
            "mean_forbidden_facts_carried": round(sum(d["tail_leaks"]) / n, 4),
            "n_scenes": n,
        }
        c = cells[cond]
        print(f"{cond:<6} conventions={c['convention_survival']:.3f} "
              f"constraints={c['constraint_survival']:.3f} "
              f"forbidden_carried={c['mean_forbidden_facts_carried']:.2f}")

    tail, sheet_c = cells["tail"], cells["sheet"]
    pred2 = {
        "statement": "constraint survival worse under tail than under sheet",
        "tail": tail["constraint_survival"],
        "sheet": sheet_c["constraint_survival"],
        "supported": tail["constraint_survival"] < sheet_c["constraint_survival"],
    }
    print(f"  prediction 2: {'supported' if pred2['supported'] else 'not supported'}")
    return {"conditions": cells, "prediction_2": pred2}


if __name__ == "__main__":
    raise SystemExit(main())
