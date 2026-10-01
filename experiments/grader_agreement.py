"""Inter-grader agreement for E1 leak labels (paper 9.1, Scoring).

E1 writes results/e1_audit.jsonl when run with --judge-backend: a deterministic
20% sample of scoring probes, each with the exact-match label and the model
grader's label, and an empty "human" field. Fill "human" with true or false,
then:

    python experiments/grader_agreement.py results/e1_audit.jsonl

Reports Cohen's kappa for model vs human and for exact-match vs human, and how
many leaks each automated grader missed or invented relative to the human.
Rows with "human": null are counted and excluded, never guessed.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from seatkit import cohens_kappa  # noqa: E402


def agreement(rows: list[dict]) -> dict:
    labelled = [r for r in rows if r.get("human") is not None]
    human = [bool(r["human"]) for r in labelled]
    out = {"n_rows": len(rows), "n_labelled": len(labelled),
           "n_unlabelled": len(rows) - len(labelled)}
    for key in ("model", "exact"):
        auto = [bool(r[key]) for r in labelled]
        k = cohens_kappa(auto, human)
        out[key] = {
            "kappa_vs_human": None if k != k else round(k, 4),
            "agreement": round(sum(a == h for a, h in zip(auto, human)) / len(human), 4)
            if human else None,
            "missed_leaks": sum(1 for a, h in zip(auto, human) if h and not a),
            "false_leaks": sum(1 for a, h in zip(auto, human) if a and not h),
        }
    out["human_leak_count"] = sum(human)
    if labelled and not any(human):
        out["note"] = ("No leaks in the labelled sample, so kappa is undefined. "
                       "Agreement on an all-negative sample shows nothing.")
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    res = agreement(rows)
    print(json.dumps(res, indent=2))
    dest = path.with_name(path.stem + "_agreement.json")
    dest.write_text(json.dumps(res, indent=2))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
