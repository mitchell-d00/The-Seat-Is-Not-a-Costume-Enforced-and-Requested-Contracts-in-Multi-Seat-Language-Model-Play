"""Qualify a judge before it scores any data (preregistration G3).

Runs the judge panel on the known-answer items in configs/judge_gold.json and
compares each task with the thresholds in configs/judge.json:

  grader       leak / no-leak on clear paraphrases and clear non-leaks
  stance       affirm / deny / neutral on the contested proposition
  attribution  leave-one-out speaker attribution on two distinct voices
  paraphrase   every number kept, hedges removed

The result is written to results/judge_checks/<fingerprint>.json. E1-E3 look
there for a passing check matching their judge's fingerprint and record what
they find, so a results file always says whether its judge was qualified.

    python experiments/judge_check.py --judge-backend openai:<model>
    python experiments/judge_check.py --judge-backend rule     # offline stand-in
Exit status is 0 on pass, 1 on fail.
"""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CHECKS_DIR, GOLD, JUDGE_CONFIG  # noqa: E402

from seatkit import Fact  # noqa: E402
from seatkit.judges import HEDGE_STRIP, JudgeConfig, build_panel, _NUM  # noqa: E402


def as_fact(statement: str) -> Fact:
    """Wrap a gold statement so the grader prompt renders it verbatim."""
    class _F(Fact):
        def statement(self):  # noqa: D401
            return statement
    return _F("gold", "", "", "", "")


def run_check(panel, gold: dict) -> dict:
    th = panel.config.thresholds
    g = gold["grader"]
    grader_ok = [panel.grader.grade(as_fact(x["fact"]), x["response"]) == x["leak"] for x in g]
    st = gold["stance"]
    stance_ok = [panel.stance.stance(x["utterance"], x["proposition"]) == x["stance"] for x in st]
    at = gold["attribution"]
    pairs = panel.attribution.judge(at["texts"], at["authors"], ("geo", "bio"),
                                    random.Random(0))
    attr_ok = [p == t for p, t in pairs]
    kept, hedged = [], []
    for text in gold["paraphrase"]:
        out = panel.paraphrase(text)
        kept.append(all(n in out for n in _NUM.findall(text)))
        hedged.append(HEDGE_STRIP.search(out) is None)

    rate = lambda xs: round(sum(xs) / len(xs), 4) if xs else 0.0  # noqa: E731
    scores = {
        "grader_accuracy": rate(grader_ok),
        "stance_accuracy": rate(stance_ok),
        "attribution_accuracy": rate(attr_ok),
        "paraphrase_numbers_kept": rate(kept),
        "paraphrase_hedges_removed": rate(hedged),
    }
    usage = panel.usage()
    checks = {k: v >= th[k] for k, v in scores.items()}
    checks["max_unparseable_rate"] = usage["unparseable_rate"] <= th["max_unparseable_rate"]
    failures = {
        "grader": [x for x, ok in zip(g, grader_ok) if not ok],
        "stance": [x for x, ok in zip(st, stance_ok) if not ok],
    }
    return {**panel.describe(), "scores": scores, "unparseable_rate": usage["unparseable_rate"],
            "thresholds": th, "checks": checks, "passed": all(checks.values()),
            "failures": failures, "n_items": {"grader": len(g), "stance": len(st),
                                             "attribution": len(at["texts"]),
                                             "paraphrase": len(gold["paraphrase"])}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-backend", default=None,
                    help="provider:model, or 'rule' for the offline stand-in; "
                         "defaults to `model` in configs/judge.json")
    ap.add_argument("--judge-config", default=str(JUDGE_CONFIG))
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    cfg = JudgeConfig.load(args.judge_config)
    spec = args.judge_backend or cfg.model
    if spec is None:
        print("no judge: set `model` in configs/judge.json or pass --judge-backend")
        return 2
    panel = build_panel(spec, cfg, args.cache)
    if panel is None:
        print(f"{spec!r} selects the legacy stand-ins, which have no panel to check; "
              "use 'rule' or a provider:model spec")
        return 2

    res = run_check(panel, json.loads(Path(GOLD).read_text()))
    res["spec"] = spec
    for k, v in res["scores"].items():
        print(f"{k:<28}{v:>7.3f}  (min {res['thresholds'][k]:.2f})  "
              f"{'ok' if res['checks'][k] else 'FAIL'}")
    print(f"{'unparseable_rate':<28}{res['unparseable_rate']:>7.3f}  "
          f"(max {res['thresholds']['max_unparseable_rate']:.2f})  "
          f"{'ok' if res['checks']['max_unparseable_rate'] else 'FAIL'}")
    print(f"\njudge {spec} fingerprint {res['fingerprint']}: "
          f"{'PASSED' if res['passed'] else 'FAILED'}"
          + ("  [stand-in: not valid for live data]" if res["standin"] else ""))

    CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    dest = CHECKS_DIR / f"{res['fingerprint']}.json"
    dest.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"wrote {dest.relative_to(CHECKS_DIR.parents[1])}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
