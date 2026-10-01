"""E3 - Instrument symmetry with the content confound removed (paper 9.3).

The original prediction compared fetch-only against critic-only instruments.
That comparison is uninterpretable: a critic instrument emits critical
sentences, so of course the seats diverge. That is a fact about injected
content, not about walls.

The fix is a replay log. Every arm of a scene receives byte-identical
instrument output. Only visibility differs.

Cells
  L0_shared            floor: one window, so instruments cannot be private
  L1_shared/private    requested walls, buffer sharing manipulated
  L3_shared/private    enforced walls, buffer sharing manipulated
  L3_private_swapped   P8: contracts exchanged between the two voices
  ceiling              separate processes, private instruments by construction

Disagreement
  --judge-backend      a model scores each turn's stance on the contested
                       proposition; incompatibility is |stance_a - stance_b|/2
  omitted (mock only)  stand-in: 1 - lexical similarity, which measures shared
                       vocabulary, not shared position

Usage:
    python experiments/e3_instrument_symmetry.py --backend mock --scenes 40
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (add_common_args, build_pair, ceiling_bench, decisions,  # noqa: E402
                     header, judge_record, ladder_bench, resolve_judge, write)

from seatkit import (CEILING, Instrument, Level, Origin, SceneConfig,  # noqa: E402
                     build_replay_log, disagreement_half_life, get_backend,
                     hedge_index, lexical_convergence, normalize, paired_sign_test)
from seatkit.judges import incompatibility_by_turn  # noqa: E402
from seatkit.metrics import bootstrap_mean_ci, censored_half_life  # noqa: E402

# (name, level, shared_buffer, swap_contracts)
CELLS = [
    ("L0_shared", Level.COSTUME, True, False),
    ("L1_shared", Level.REQUESTED, True, False),
    ("L1_private", Level.REQUESTED, False, False),
    ("L3_shared", Level.PARTITIONED, True, False),
    ("L3_private", Level.PARTITIONED, False, False),
    ("L3_private_swapped", Level.PARTITIONED, False, True),
    (CEILING, None, False, False),
]


def run_cell(name, level, shared, swap, args, backend, stance, window=3):
    per_scene = []
    for i in range(args.scenes):
        geo, bio = build_pair(i, swap_contracts=swap)
        # Identical instrument content in every arm of scene i. The replay log
        # is built from the *unswapped* pair so swapping changes contracts only.
        ref_geo, ref_bio = build_pair(i)
        logs = {"geo": build_replay_log(backend, ref_geo, args.turns + 1, seed=i),
                "bio": build_replay_log(backend, ref_bio, args.turns + 1, seed=i)}
        instruments = {s.seat_id: Instrument(host=s, backend=backend,
                                             shares_buffer=shared,
                                             replay_log=logs[s.seat_id])
                       for s in (geo, bio)}
        cfg = SceneConfig(turns=args.turns, seed=i,
                          level=level if level is not None else Level.PARTITIONED,
                          shared_instrument_buffer=shared)
        if name == CEILING:
            res = ceiling_bench((geo, bio), args.backend, args.cache,
                                instruments=instruments).run(cfg)
        else:
            res = ladder_bench((geo, bio), backend, instruments).run(cfg)

        if stance is not None:
            spoken = [sp for sp in res.transcript if sp.origin is Origin.SEAT]
            incompat = incompatibility_by_turn(spoken, "geo", "bio", stance,
                                               cfg.contested_proposition, window)
        else:
            incompat = {t: 1.0 - v for t, v in
                        lexical_convergence(res.transcript, "geo", "bio").items()}
        hl = disagreement_half_life(incompat)
        hedges = [v for sid in ("geo", "bio")
                  for v in hedge_index(res.transcript, sid).values()]
        per_scene.append({"half_life": hl,
                          "censored": censored_half_life(hl, args.turns),
                          "incompat": incompat,
                          "hedge": mean(hedges) if hedges else 0.0})
    return per_scene


def summarize(per_scene, turns):
    cens = [s["censored"] for s in per_scene]
    resolved = sorted(s["half_life"] for s in per_scene if s["half_life"] is not None)
    rm, lo, hi = bootstrap_mean_ci(cens, seed=0)
    curve_turns = sorted({t for s in per_scene for t in s["incompat"]})
    return {
        "restricted_mean_half_life": round(rm, 3),
        "restricted_mean_ci95": [round(lo, 3), round(hi, 3)],
        "censoring": f"never-converged coded as {turns + 1} (turns + 1)",
        "median_resolved_half_life": resolved[len(resolved) // 2] if resolved else None,
        "n_never_converged": sum(1 for s in per_scene if s["half_life"] is None),
        "n_scenes": len(per_scene),
        "mean_hedge_index": round(mean(s["hedge"] for s in per_scene), 4),
        "incompatibility_curve": {
            str(t): round(mean(s["incompat"][t] for s in per_scene if t in s["incompat"]), 4)
            for t in curve_turns},
    }


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap, judges=True)
    ap.add_argument("--p8-margin", type=float, default=None,
                    help="equivalence margin in turns for P8; defaults to "
                         "p8_equivalence_margin_turns in configs/prereg.json")
    ap.add_argument("--out", default="results/e3.json")
    args = ap.parse_args()

    # Judges first: a misconfigured run should fail before any client exists.
    panel = resolve_judge(args)
    backend = get_backend(args.backend, cache=args.cache or False)
    stance = panel.stance if panel is not None else None
    window = panel.config.stance_window if panel is not None else 3
    if args.p8_margin is None:
        args.p8_margin = decisions().get("p8_equivalence_margin_turns")

    out = header(args, "E3")
    out["disagreement_measure"] = (stance.name if stance
                                   else "lexical-similarity-standin")
    out["cells"] = {}
    raw = {}
    for name, level, shared, swap in CELLS:
        raw[name] = run_cell(name, level, shared, swap, args, backend, stance, window)
        c = out["cells"][name] = summarize(raw[name], args.turns)
        print(f"{name:<19} rmean_half_life={c['restricted_mean_half_life']:<7} "
              f"never_converged={c['n_never_converged']}/{args.scenes} "
              f"hedge={c['mean_hedge_index']}")

    cens = {k: [s["censored"] for s in v] for k, v in raw.items()}
    hedge = {k: [s["hedge"] for s in v] for k, v in raw.items()}
    cells = out["cells"]

    out["P6_private_vs_shared"] = {
        lvl: paired_sign_test(cens[f"{lvl}_private"], cens[f"{lvl}_shared"])
        for lvl in ("L1", "L3")}
    buffer_effect = mean(mean(hedge[f"{l}_shared"]) - mean(hedge[f"{l}_private"])
                         for l in ("L1", "L3"))
    level_effect = mean(mean(hedge[f"L1_{b}"]) - mean(hedge[f"L3_{b}"])
                        for b in ("shared", "private"))
    out["P7_hedge_main_effects"] = {
        "buffer_sharing (shared - private)": round(buffer_effect, 4),
        "ladder_level (L1 - L3)": round(level_effect, 4),
        "prediction": "|buffer_sharing| > |ladder_level|",
        "holds_in_this_run": abs(buffer_effect) > abs(level_effect),
    }
    diffs = [a - b for a, b in zip(cens["L3_private_swapped"], cens["L3_private"])]
    d, dlo, dhi = bootstrap_mean_ci(diffs, seed=0)
    p8 = {"mean_diff_swapped_minus_original": round(d, 3),
          "ci95": [round(dlo, 3), round(dhi, 3)],
          "sign_test": paired_sign_test(cens["L3_private_swapped"], cens["L3_private"])}
    if args.p8_margin is not None:
        p8["equivalence_margin"] = args.p8_margin
        p8["equivalent_within_margin"] = (-args.p8_margin < dlo and dhi < args.p8_margin)
    else:
        p8["equivalence_margin"] = None
        p8["note"] = "no margin supplied; P8 is reported, not decided"
    out["P8_contract_swap"] = p8

    fl, ce = cells["L0_shared"]["restricted_mean_half_life"], cells[CEILING]["restricted_mean_half_life"]
    out["positions_half_life"] = {
        k: normalize(cells[k]["restricted_mean_half_life"], fl, ce).as_dict()
        for k in ("L1_shared", "L1_private", "L3_shared", "L3_private")}
    out["P11_ceiling_minus_floor_half_life"] = round(ce - fl, 3)

    print(f"\nP6 private vs shared (paired sign test p): "
          f"L1={out['P6_private_vs_shared']['L1']['p']:.4f} "
          f"L3={out['P6_private_vs_shared']['L3']['p']:.4f}")
    print(f"P7 hedge effects: buffer={buffer_effect:.4f} level={level_effect:.4f}")
    print(f"P8 swap diff = {p8['mean_diff_swapped_minus_original']} {p8['ci95']}")
    print(f"P11 ceiling - floor (restricted mean half-life) = "
          f"{out['P11_ceiling_minus_floor_half_life']}")

    warnings = []
    if all(c["n_never_converged"] >= 0.8 * args.scenes for c in cells.values()):
        warnings.append("At least 80% of scenes never converged in every cell. "
                        "Half-lives are dominated by censoring, so P6, P8 and P11 "
                        "cannot be read from this run. Expected with --backend "
                        "mock, whose seats draw from disjoint canned vocabularies.")
    # Below L3 every span is visible to every seat (ladder.visibility_for), so a
    # "private" instrument at L1 is the shared one under another name. The two
    # L1 arms are identical by construction and P6-at-L1 tests nothing until
    # requested privacy is given an implementation.
    if cens["L1_private"] == cens["L1_shared"] and hedge["L1_private"] == hedge["L1_shared"]:
        out["P6_L1_identical_by_construction"] = True
        warnings.append("L1_private and L1_shared are identical: below L3 nothing "
                        "can be hidden, so the L1 half of P6 is untestable as "
                        "implemented. See docs/CHANGES-review-fixes.md.")
    if stance is None:
        warnings.append("Stand-in disagreement measure (lexical similarity). It "
                        "tracks shared vocabulary, not shared position.")
    for w in warnings:
        print("[warning] " + w)
    out["warnings"] = warnings
    out["judge"] = judge_record(panel, args)
    if stance is not None:
        out["judge_unparseable"] = stance.unparseable

    write(out, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
