"""Plumbing shared by E1-E4. Nothing here changes what an experiment measures.

Kept in one place because four scripts each building scenes their own way is
how counterbalancing ends up in E1 and nowhere else.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seatkit import (Bench, Instrument, SeparatedBench, facing_pair,  # noqa: E402
                     get_backend, make_facts)

PREREG_LOCK = ROOT / "docs" / "PREREGISTRATION.lock"
CONFIGS = ROOT / "experiments" / "configs"
JUDGE_CONFIG = CONFIGS / "judge.json"
GOLD = CONFIGS / "judge_gold.json"
DECISIONS = CONFIGS / "prereg.json"
CHECKS_DIR = ROOT / "results" / "judge_checks"
PREREG_FILES = ("docs/PREREGISTRATION.md", "docs/protocol.md",
                "experiments/configs/default.yaml", "experiments/configs/judge.json",
                "experiments/configs/judge_gold.json", "experiments/configs/prereg.json")
PREREG_GLOBS = ("src/seatkit/*.py", "experiments/*.py", "experiments/*.sh")

MOCK_NOTICE = ("mock backend: these numbers check the harness and are not "
               "evidence for any prediction (paper 9.5)")

PERSONAS = {
    "geo": ("a geologist", "protect the rapid-event reading"),
    "bio": ("a paleobiologist", "protect the quiet-interval reading"),
}


# -- scenes -----------------------------------------------------------------

def build_pair(i: int, *, swap_contracts: bool = False,
               externalize: tuple[str, ...] = ()):
    """The standard two-seat scene for index `i`.

    Fact pools are counterbalanced across seats on odd scenes, in every
    experiment, so fact identity never confounds with seat identity.

    `swap_contracts` (E3, P8) exchanges the contracts between the two voices:
    each persona keeps its voice and takes the other seat's needs and facts.
    If disagreement half-life is a property of the contract pair, swapping
    should leave it unchanged; if it moves with the voices, it does not.
    """
    a = make_facts(3, seed=1000 + i, prefix="a")
    b = make_facts(3, seed=5000 + i, prefix="b")
    if i % 2:
        a, b = b, a
    (geo_persona, geo_need), (bio_persona, bio_need) = PERSONAS["geo"], PERSONAS["bio"]
    if swap_contracts:
        a, b = b, a
        geo_need, bio_need = bio_need, geo_need
    extra = {"externalize": externalize} if externalize else {}
    return facing_pair(
        "geo", "bio", a, b,
        a_extra={"persona": geo_persona, "needs": (geo_need,), **extra},
        b_extra={"persona": bio_persona, "needs": (bio_need,), **extra},
    )


def ladder_bench(seats, backend, instruments=None):
    instruments = instruments if instruments is not None else {
        s.seat_id: Instrument(host=s, backend=backend) for s in seats}
    return Bench(list(seats), backend, instruments)


def ceiling_factory(spec: str, cache: str | None):
    """One fresh backend per seat, each with its own cache directory.

    Separate caches matter: ceiling and L3 prompts are identical for stateless
    APIs, so a shared cache would replay L3's samples into the ceiling and make
    the two equal by construction rather than by measurement.
    """
    def make(seat_id: str):
        c = str(Path(cache) / "ceiling" / seat_id) if cache else False
        return get_backend(spec, cache=c)
    return make


def ceiling_bench(seats, spec: str, cache: str | None, instruments=None):
    return SeparatedBench(list(seats), ceiling_factory(spec, cache),
                          instruments=instruments)


# -- CLI --------------------------------------------------------------------

def add_common_args(ap, *, judges: bool = False):
    ap.add_argument("--backend", default="mock",
                    help="mock | openai:<model> | anthropic:<model> | "
                         "grok:<model> | gemini:<model>")
    ap.add_argument("--cache", default=None,
                    help="directory for the response cache; omit to disable")
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--turns", type=int, default=24)
    if judges:
        ap.add_argument("--judge-backend", default=None,
                        help="provider:model for judging, or 'rule' (offline "
                             "stand-in). Defaults to `model` in configs/judge.json; "
                             "with neither, the legacy stand-ins (mock only).")
        ap.add_argument("--judge-config", default=str(JUDGE_CONFIG))
        ap.add_argument("--allow-standin-judge", action="store_true",
                        help="run a live scene with stand-in judges anyway; "
                             "results are marked as stand-in")


def judge_cache(args):
    return str(Path(args.cache) / "judge") if args.cache else None


def judge_check_status(fingerprint: str) -> dict:
    """Did judge_check.py pass for exactly this judge?"""
    path = CHECKS_DIR / f"{fingerprint}.json"
    if not path.exists():
        return {"status": "missing",
                "note": "run experiments/judge_check.py with this judge first"}
    r = json.loads(path.read_text())
    return {"status": "passed" if r.get("passed") else "failed",
            "scores": r.get("scores")}


def resolve_judge(args):
    """Build the judge panel for E1-E3, or None for the legacy stand-ins.

    Resolution order: --judge-backend, then `model` in the judge config.
    Refuses a live scene backend with any stand-in judge (legacy or rule)
    unless --allow-standin-judge. Warns when the judge is a model under test,
    and when no passing judge check matches this judge's fingerprint.
    """
    from seatkit.judges import JudgeConfig, StandinJudgeRefused, build_panel
    cfg = JudgeConfig.load(args.judge_config)
    spec = args.judge_backend or cfg.model
    live = args.backend != "mock"
    standin = spec in (None, "", "mock", "standin", "rule")
    if live and standin and not args.allow_standin_judge:
        raise StandinJudgeRefused(
            "Live scene backend with a stand-in judge. Stand-ins bias E2 toward the "
            "hypothesis. Set `model` in experiments/configs/judge.json (a model not "
            "under test) or pass --judge-backend <provider:model>; or pass "
            "--allow-standin-judge to run anyway with results marked stand-in.")
    if spec and spec == args.backend:
        print(f"[warning] judge is the scene model ({spec}). Paper 9.2 specifies "
              "a separate model; results record this.")
    panel = build_panel(spec, cfg, judge_cache(args))
    if panel is not None:
        args._judge_check = judge_check_status(panel.fingerprint())
        if live and args._judge_check["status"] != "passed":
            print(f"[warning] judge check {args._judge_check['status']} for fingerprint "
                  f"{panel.fingerprint()}; run experiments/judge_check.py first")
    return panel


def judge_record(panel, args) -> dict:
    if panel is None:
        return {"panel": "legacy-standins", "standin": True}
    return {**panel.describe(), "check": getattr(args, "_judge_check", None),
            "usage": panel.usage()}


# -- preregistration --------------------------------------------------------

def _prereg_paths() -> list[Path]:
    paths = [ROOT / p for p in PREREG_FILES]
    for g in PREREG_GLOBS:
        paths += sorted(ROOT.glob(g))
    return [p for p in paths if p.is_file()]


def prereg_hashes() -> dict[str, str]:
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in _prereg_paths()}


def prereg_status() -> dict:
    """Compare the working tree with docs/PREREGISTRATION.lock.

    Recorded in every results file. A live run with `locked: false`, or with
    files listed under `changed_since_freeze`, was not run under the frozen
    protocol, and the results file says so.
    """
    if not PREREG_LOCK.exists():
        return {"locked": False,
                "note": "run experiments/freeze_prereg.py before data collection"}
    lock = json.loads(PREREG_LOCK.read_text())
    if lock.get("status") == "draft":
        now = prereg_hashes()
        frozen = lock.get("files", {})
        return {"locked": False, "draft_lock": True,
                "unset_decisions": lock.get("unset", []),
                "changed_since_draft": sorted(k for k in set(frozen) | set(now)
                                              if frozen.get(k) != now.get(k)),
                "note": "draft lock only; finalize with freeze_prereg.py --force"}
    now = prereg_hashes()
    frozen = lock.get("files", {})
    changed = sorted(k for k in set(frozen) | set(now) if frozen.get(k) != now.get(k))
    return {"locked": True, "frozen_at_utc": lock.get("frozen_at_utc"),
            "git_commit": lock.get("git_commit"),
            "lock_sha256": lock.get("digest"),
            "changed_since_freeze": changed}


def header(args, experiment: str) -> dict:
    from seatkit import __version__, family_of
    pre = prereg_status()
    if args.backend != "mock" and (not pre.get("locked") or pre.get("changed_since_freeze")):
        print("[warning] preregistration not frozen, or files changed since "
              "freeze; this run is recorded as exploratory")
    out = {"experiment": experiment, "seatkit_version": __version__,
           "backend": args.backend, "family": family_of(args.backend),
           "scenes": args.scenes, "turns": getattr(args, "turns", None),
           "preregistration": pre}
    if args.backend == "mock":
        out["notice"] = MOCK_NOTICE
        print(f"[{MOCK_NOTICE}]")
    return out


def decisions() -> dict:
    return json.loads(DECISIONS.read_text()) if DECISIONS.exists() else {}


def write(out: dict, path: str) -> None:
    from seatkit.backends import SUBSTITUTIONS
    out["parameter_substitutions"] = dict(SUBSTITUTIONS)
    if any(k.endswith(":temperature") for k in SUBSTITUTIONS):
        print("[warning] a provider rejected `temperature` and it was dropped for "
              "some calls; those cells did not run at the stated temperature. "
              "See parameter_substitutions in the results file.")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
