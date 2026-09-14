"""Collate per-family result files into one table (paper 9.5).

Deliberately does NOT pool across families. The question is whether ladder
effects replicate, and pooling would hide exactly the disagreement that matters.
If an effect shows up on one family and not another, the framework describes a
deployment practice rather than a general property - a weaker claim, and one
that has to be stated rather than averaged away.

    python experiments/compare_families.py results/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(root: Path) -> list[dict]:
    rows = []
    for e1 in sorted(root.glob("*/e1.json")):
        d = json.loads(e1.read_text())
        cells = d.get("cells", {})
        row = {
            "backend": d.get("backend", "?"),
            "family": d.get("family", "?"),
            "scenes": d.get("scenes"),
            "gate": d.get("prior_gate", "?"),
        }
        for lvl in ("L0", "L1", "L2", "L3", "L4"):
            row[lvl] = cells.get(lvl, {}).get("elicited_leak_rate")
            row[lvl + "_routing"] = cells.get(lvl, {}).get("routing_bleed")
        rows.append(row)
    return rows


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    rows = load(root)
    if not rows:
        print(f"no e1.json under {root}/*/ - run run_cross_family.sh first")
        return 1

    hdr = f"{'backend':<30}{'family':<12}{'gate':<6}" + "".join(
        f"{l:>8}" for l in ("L0", "L1", "L2", "L3", "L4"))
    print("\nE1 elicited leak rate by enforcement level\n")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        vals = "".join(
            f"{r[l]:>8.3f}" if isinstance(r[l], (int, float)) else f"{'-':>8}"
            for l in ("L0", "L1", "L2", "L3", "L4"))
        print(f"{r['backend']:<30}{r['family']:<12}{r['gate']:<6}{vals}")

    print("\nchecks")
    for r in rows:
        notes = []
        if r["gate"] != "pass":
            notes.append("prior gate did not pass - leak numbers unreadable")
        bad = [l for l in ("L3", "L4") if (r.get(l + "_routing") or 0) > 0]
        if bad:
            notes.append(f"routing bleed at {', '.join(bad)} - fix ladder.py, rerun")
        l1, l3 = r.get("L1"), r.get("L3")
        if isinstance(l1, float) and isinstance(l3, float) and l3 >= l1:
            notes.append("L3 leak >= L1: enforced kill condition may have fired")
        print(f"  {r['backend']:<30}{'; '.join(notes) or 'ok'}")

    fams = {r["family"] for r in rows}
    print(f"\ndistinct families: {len(fams)} {sorted(fams)}")
    if len(fams) < 2:
        print("Fewer than two families. Paper 9.5 is not satisfied by this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
