"""Freeze, check, and re-freeze the preregistration (paper 9.5).

The lock, docs/PREREGISTRATION.lock, records SHA-256 hashes of the
preregistration, protocol, configs (including the judge config, the judge gold
set, and the decisions file), and every source and experiment file, with the
UTC time and git commit. Every E1-E4 results file records whether it ran
against the locked tree.

Three ways to use it:

    python experiments/freeze_prereg.py --draft    # lock now; decisions may be unset
    python experiments/freeze_prereg.py --force    # final lock; refuses unset decisions
    python experiments/freeze_prereg.py --check    # verify the tree; exit 1 if not clean

A *draft* lock is a reusable baseline: it records the tree as it stands, and
`--check` lists exactly what has changed since. Results produced under a draft
lock are recorded as exploratory. A *final* lock requires every decision in
experiments/configs/prereg.json, the judge model in configs/judge.json, and every
"[to be set" placeholder in PREREGISTRATION.md to be filled. Re-freezing never
deletes history: the previous lock is kept beside it as a superseded file.

The lock is self-attested: it proves the tree did not change after it was
written, not when it was written. For an independent timestamp, deposit the
`digest` with OSF or AsPredicted, or push a signed git tag, and cite that.
"""
from __future__ import annotations

import argparse, datetime, hashlib, json, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (DECISIONS, JUDGE_CONFIG, PREREG_LOCK, ROOT,  # noqa: E402
                     prereg_hashes, prereg_status)


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, timeout=10)
        if r.returncode:
            return None
        return r.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


def unset_decisions() -> list[str]:
    """Everything a final lock requires to be filled in."""
    out = []
    dec = json.loads(DECISIONS.read_text()) if DECISIONS.exists() else {}
    out += [f"configs/prereg.json:{k}" for k, v in dec.items()
            if not k.startswith("_") and v is None]
    judge = json.loads(JUDGE_CONFIG.read_text())
    if not judge.get("model"):
        out.append("configs/judge.json:model")
    prereg = (ROOT / "docs" / "PREREGISTRATION.md").read_text()
    if "[to be set" in prereg:
        out.append("docs/PREREGISTRATION.md: '[to be set' placeholder")
    return out


def check() -> int:
    st = prereg_status()
    if st.get("locked"):
        changed = st["changed_since_freeze"]
        print(f"final lock from {st['frozen_at_utc']} (digest {st['lock_sha256']})")
        if changed:
            print("CHANGED since freeze - runs now are exploratory:")
            for f in changed:
                print(f"  {f}")
            return 1
        print("tree matches the lock: runs are under the frozen protocol")
        return 0
    if st.get("draft_lock"):
        print("draft lock only. Unset decisions at draft time:")
        for u in st["unset_decisions"]:
            print(f"  {u}")
        if st["changed_since_draft"]:
            print("changed since draft:")
            for f in st["changed_since_draft"]:
                print(f"  {f}")
        print("finalize with: python experiments/freeze_prereg.py --force")
        return 1
    print("no lock. Run freeze_prereg.py --draft or --force.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--draft", action="store_true", help="lock even with unset decisions")
    g.add_argument("--check", action="store_true", help="verify the tree against the lock")
    ap.add_argument("--force", action="store_true", help="replace an existing lock")
    args = ap.parse_args()

    if args.check:
        return check()

    unset = unset_decisions()
    if unset and not args.draft:
        print("cannot make a final lock; these are unset:")
        for u in unset:
            print(f"  {u}")
        print("fill them in, or pass --draft for a baseline lock.")
        return 1
    if PREREG_LOCK.exists() and not args.force:
        print(f"{PREREG_LOCK.relative_to(ROOT)} exists; pass --force to replace it "
              "(the old lock is kept as a superseded file).")
        return 1
    if PREREG_LOCK.exists():
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        PREREG_LOCK.rename(PREREG_LOCK.with_name(f"PREREGISTRATION.lock.superseded-{stamp}"))

    files = prereg_hashes()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    lock = {
        "status": "draft" if unset else "final",
        "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "digest": digest,
        "unset": unset,
        "files": files,
        "note": "Self-attested. Deposit `digest` with a third party for an "
                "independent timestamp.",
    }
    PREREG_LOCK.write_text(json.dumps(lock, indent=2))
    print(f"{lock['status']} lock: {len(files)} files, digest {digest}")
    if unset:
        print(f"{len(unset)} decision(s) unset; results under this lock are exploratory")
    if lock["git_commit"] is None or lock["git_commit"].endswith("-dirty"):
        print("[warning] no clean git commit recorded; commit first for a stronger record")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
