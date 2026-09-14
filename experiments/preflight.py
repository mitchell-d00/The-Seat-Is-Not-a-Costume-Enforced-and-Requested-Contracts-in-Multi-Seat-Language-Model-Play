"""Check a backend before committing to a multi-thousand-call run.

Verifies that the spec parses, the key is present, the SDK imports, the model
answers, and the prior gate passes for this model. Costs about a dozen calls.

    python experiments/preflight.py --backends openai:<model> gemini:<model>
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seatkit import available_backends, family_of, get_backend, make_facts, verify_prior


def check(spec: str, cache: str | None) -> dict:
    row = {"spec": spec, "family": None, "reachable": False,
           "prior_gate": None, "latency_s": None, "error": None}
    try:
        row["family"] = family_of(spec)
        backend = get_backend(spec, cache=cache or False)
        t0 = time.time()
        r = backend.generate("Reply with the single word: ready.", max_tokens=16)
        row["latency_s"] = round(time.time() - t0, 2)
        row["reachable"] = bool(r.text.strip())
        row["reported_prompt_tokens"] = r.prompt_tokens
        row["prompt_words"] = r.prompt_words

        # The gate is model-specific: a nonce pool with near-zero prior on one
        # model is not guaranteed to have near-zero prior on another, and a pool
        # that fails here would make every leak number from this model unreadable.
        passed, checks = verify_prior(make_facts(8, seed=7), backend)
        row["prior_gate"] = "pass" if passed else "FAIL"
        row["prior_recovered"] = sum(c.recovered for c in checks)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["mock"])
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    print("keys present:", {k: v for k, v in available_backends().items() if v})
    print()
    rows = [check(s, args.cache) for s in args.backends]
    for r in rows:
        status = "ok" if r["reachable"] and r["prior_gate"] == "pass" else "PROBLEM"
        print(f"{r['spec']:<28} {status:<8} family={r['family']} "
              f"gate={r['prior_gate']} latency={r['latency_s']}s")
        if r["error"]:
            print(f"    {r['error']}")

    families = {r["family"] for r in rows if r["reachable"]}
    print(f"\ndistinct reachable families: {len(families)} {sorted(families)}")
    if len(families) < 2:
        print("Paper 9.5 asks for at least two. Results from one family describe "
              "a deployment practice, not a general property.")
    return 0 if all(r["reachable"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
