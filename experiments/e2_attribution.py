"""E2 - Label-stripped attribution, made non-circular (paper 9.2).

Two judges. The surface judge sees raw text and may use style. The content judge
sees text after a register-normalizing paraphrase pass by a separate model and
may use only what was claimed. If attribution succeeds for the surface judge and
fails for the content judge, the costume was doing the work and the contract was
not. That contrast is the paper's most direct test of its own thesis.

The primary analysis is a regression of content-judge accuracy on E1 leak rate:
two independently measured quantities, which is what the original prediction
lacked.

Judges (seatkit.judges):
  --judge-backend <provider:model>  model paraphrase + model attribution reader
  omitted, with --backend mock      regex paraphrase + lexical centroid stand-ins

A live --backend with stand-in judges is refused unless --allow-standin-judge
is passed, because the regex paraphraser keeps syntax and word choice and so
overstates content-judge accuracy - a bias in the hypothesis's favour.

Usage:
    python experiments/e2_attribution.py --backend mock --scenes 40
    python experiments/e2_attribution.py --backend anthropic:<model> \\
        --judge-backend openai:<model> --cache .cache/run1
"""
from __future__ import annotations

import argparse, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (add_common_args, build_pair, ceiling_bench, header,  # noqa: E402
                     judge_record, ladder_bench, resolve_judge, write)

from seatkit import (CEILING, Level, Origin, SceneConfig, get_backend,  # noqa: E402
                     leak_rates, normalize)
from seatkit.judges import LexicalCentroidJudge, RegexParaphraser  # noqa: E402
from seatkit.metrics import (binomial_p, cluster_bootstrap_ci,  # noqa: E402
                             paired_cluster_diff, sign_test_over_clusters)

LEVELS = [Level.COSTUME, Level.REQUESTED, Level.REMINDED,
          Level.PARTITIONED, Level.ISOLATED]
LATE_TURN = 16   # P5: at L0, content accuracy at chance by this turn


def run_cell(cell, args, backend, judge, paraphrase, rng):
    """Return per-scene results. The scene is the unit of independence.

    Flattening turns into one list was the original mistake: 40 scenes of 48
    turns is not 1920 independent trials. Turns inside a scene share seats,
    facts and a drift trajectory, and pooling them inflates significance badly
    (see docs/protocol.md).
    """
    per_scene = []
    for i in range(args.scenes):
        geo, bio = build_pair(i)
        cfg = SceneConfig(turns=args.turns, seed=i,
                          level=Level.PARTITIONED if cell == CEILING else cell)
        bench = (ceiling_bench((geo, bio), args.backend, args.cache)
                 if cell == CEILING else ladder_bench((geo, bio), backend))
        res = bench.run(cfg)
        spoken = [sp for sp in res.transcript if sp.origin is Origin.SEAT]
        authors = [sp.author for sp in spoken]
        turns = [sp.turn for sp in spoken]
        ids = ("geo", "bio")

        surf = [p == t for p, t in judge.judge([sp.text for sp in spoken],
                                               authors, ids, rng)]
        cont = [p == t for p, t in judge.judge([paraphrase(sp.text) for sp in spoken],
                                               authors, ids, rng)]
        per_scene.append({
            "surface": surf, "content": cont,
            "surface_late": [c for c, t in zip(surf, turns) if t >= LATE_TURN],
            "content_late": [c for c, t in zip(cont, turns) if t >= LATE_TURN],
            "leak": leak_rates(res.scoring_probes()).elicited,
        })
    return per_scene


def summarize(per_scene):
    surf_cl = [s["surface"] for s in per_scene]
    cont_cl = [s["content"] for s in per_scene]
    point, lo, hi = cluster_bootstrap_ci(cont_cl, seed=0)
    s_point, s_lo, s_hi = cluster_bootstrap_ci(surf_cl, seed=0)
    above, n_used, sign_p = sign_test_over_clusters(cont_cl, 0.5)
    paired = paired_cluster_diff(surf_cl, cont_cl, seed=0)
    n_turns = sum(len(c) for c in cont_cl)
    late = {}
    for which in ("surface", "content"):
        cl = [s[f"{which}_late"] for s in per_scene if s[f"{which}_late"]]
        pt, l, h = cluster_bootstrap_ci(cl, seed=0) if cl else (float("nan"),) * 3
        a, u, p = sign_test_over_clusters(cl, 0.5) if cl else (0, 0, 1.0)
        late[which] = {"accuracy": round(pt, 4), "ci95": [round(l, 4), round(h, 4)],
                       "sign_test": {"scenes_above_chance": a, "n": u, "p": round(p, 6)}}
    return {
        "n_scenes": len(per_scene),
        "n_turn_judgements": n_turns,
        "surface_accuracy": round(s_point, 4),
        "surface_ci95": [round(s_lo, 4), round(s_hi, 4)],
        "content_accuracy": round(point, 4),
        "content_ci95": [round(lo, 4), round(hi, 4)],
        "content_sign_test": {"scenes_above_chance": above, "n": n_used,
                              "p": round(sign_p, 6)},
        "surface_minus_content": {k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in paired.items()},
        f"turns_ge_{LATE_TURN}": late,
        "elicited_leak_rate": round(sum(s["leak"] for s in per_scene) / len(per_scene), 4),
        # Retained only to show what the pooled test would have claimed.
        # Not used for any inference.
        "naive_pooled_p_DO_NOT_USE": round(
            binomial_p(round(point * n_turns), n_turns, 0.5), 8),
    }


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap, judges=True)
    ap.add_argument("--out", default="results/e2.json")
    args = ap.parse_args()

    # Judges first: a misconfigured run should fail before any client exists.
    panel = resolve_judge(args)
    backend = get_backend(args.backend, cache=args.cache or False)
    if panel is None:
        judge, paraphrase = LexicalCentroidJudge(), RegexParaphraser()
    else:
        judge, paraphrase = panel.attribution, panel.paraphrase
    rng = random.Random(0)

    out = header(args, "E2")
    out["judges"] = {"attribution": judge.name, "paraphrase": paraphrase.name,
                     "standin": judge.is_standin or paraphrase.is_standin}
    out["cells"] = {}
    xs, ys = [], []

    for cell in LEVELS + [CEILING]:
        label = cell if cell == CEILING else cell.label
        per_scene = run_cell(cell, args, backend, judge, paraphrase, rng)
        c = out["cells"][label] = summarize(per_scene)
        # Scene-level points for the primary regression; ladder cells only.
        # Pairs are kept together, so a scene with no judgements drops out of
        # both axes rather than misaligning them.
        if cell != CEILING:
            for s in per_scene:
                if s["content"]:
                    xs.append(s["leak"])
                    ys.append(sum(s["content"]) / len(s["content"]))
        print(f"{label:>7} surface={c['surface_accuracy']:.3f} "
              f"content={c['content_accuracy']:.3f} {c['content_ci95']} "
              f"sign_p={c['content_sign_test']['p']:.4f} "
              f"leak={c['elicited_leak_rate']:.3f}")

    cells = out["cells"]
    out["positions_content_accuracy"] = {
        lbl: normalize(cells[lbl]["content_accuracy"], cells["L0"]["content_accuracy"],
                       cells[CEILING]["content_accuracy"]).as_dict()
        for lbl in ("L1", "L2", "L3", "L4")}

    # Primary analysis: content accuracy on leak rate, at the SCENE level.
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) if den else float("nan")
    out["regression"] = {
        "slope_content_accuracy_on_leak_rate": round(slope, 4) if slope == slope else None,
        "n_points": n,
        "unit": "scene",
        "note": "Negative slope is the prediction. A null slope falsifies the "
                "framework's central empirical claim.",
    }
    print(f"\nslope(content accuracy ~ leak rate) = "
          f"{out['regression']['slope_content_accuracy_on_leak_rate']} over {n} scenes")

    late0 = cells["L0"][f"turns_ge_{LATE_TURN}"]
    print(f"P5 at L0, turns >= {LATE_TURN}: surface={late0['surface']['accuracy']} "
          f"content={late0['content']['accuracy']}")

    out["judge_unparseable"] = judge.unparseable + paraphrase.unparseable
    out["judge"] = judge_record(panel, args)
    if out["judges"]["standin"]:
        out["standin_warning"] = ("Stand-in judges. The regex paraphraser keeps "
                                  "syntax and word choice, so content accuracy is "
                                  "overstated. Not valid for the E2 contrast.")
        print("\n[warning] " + out["standin_warning"])

    # Ceiling check. When every cell sits near 1.0 the experiment has not been
    # run, whatever the numbers say. Flag it rather than tuning the mock until
    # the table looks right, which would be fabricating the result.
    accs = [c["content_accuracy"] for c in cells.values()]
    if min(accs) > 0.95:
        out["ceiling_warning"] = (
            "All cells above 0.95. Judge is at ceiling and cannot discriminate. "
            "Expected with --backend mock; E2 requires a live backend.")
        print("[warning] " + out["ceiling_warning"])
    if max(accs) < 0.6:
        out["floor_warning"] = (
            "All cells near chance. Judge has no signal to recover; check that "
            "seats are producing knowledge-differentiated content at all.")
        print("[warning] " + out["floor_warning"])

    write(out, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
