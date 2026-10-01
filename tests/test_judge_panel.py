"""The reusable judge (JudgePanel, judge.json, judge_check.py) and the lock
workflow (freeze_prereg.py --draft / --force / --check)."""
import argparse
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

from seatkit import Completion  # noqa: E402
from seatkit.judges import (JudgeConfig, JudgePanel, RuleJudgeBackend,  # noqa: E402
                            StandinJudgeRefused, build_panel)

common = importlib.import_module("_common")
jc = importlib.import_module("judge_check")
GOLD = json.loads(common.GOLD.read_text())


class Constant:
    name = "constant"

    def __init__(self, reply):
        self.reply = reply

    def generate(self, prompt, **kw):
        return Completion(self.reply, "constant", 0, 0)


def test_config_loads_and_merges_thresholds():
    cfg = JudgeConfig.load(common.JUDGE_CONFIG)
    assert cfg.temperature == 0.0
    assert set(cfg.thresholds) >= {"grader_accuracy", "max_unparseable_rate"}


def test_rule_judge_passes_the_gold_check():
    res = jc.run_check(build_panel("rule", JudgeConfig()), GOLD)
    assert res["passed"] and res["standin"]


@pytest.mark.parametrize("reply", ["YES", "AFFIRM", "A", "", "I am not sure."])
def test_degenerate_judges_fail_the_gold_check(reply):
    """A judge that always gives one answer must not qualify."""
    res = jc.run_check(JudgePanel(Constant(reply), JudgeConfig()), GOLD)
    assert not res["passed"]


def test_gold_set_has_both_classes():
    assert {x["leak"] for x in GOLD["grader"]} == {True, False}
    assert {x["stance"] for x in GOLD["stance"]} == {-1, 0, 1}


def test_fingerprint_tracks_what_changes_the_judge():
    base = JudgePanel(RuleJudgeBackend(), JudgeConfig()).fingerprint()
    assert base == JudgePanel(RuleJudgeBackend(), JudgeConfig()).fingerprint()
    assert base != JudgePanel(RuleJudgeBackend(), JudgeConfig(temperature=0.5)).fingerprint()
    assert base != JudgePanel(Constant("x"), JudgeConfig()).fingerprint()


def test_fingerprint_tracks_prompt_text(monkeypatch):
    import seatkit.judges as j
    base = JudgePanel(RuleJudgeBackend(), JudgeConfig()).fingerprint()
    monkeypatch.setitem(j.PROMPTS, "grader", j.GRADER_PROMPT + " ")
    assert JudgePanel(RuleJudgeBackend(), JudgeConfig()).fingerprint() != base


def test_usage_reports_unparseable_rate():
    panel = JudgePanel(Constant("??"), JudgeConfig())
    panel.stance.stance("x", "p")
    u = panel.usage()
    assert u["unparseable_rate"] == 1.0 and not u["within_limit"]


def args(**kw):
    base = dict(backend="mock", judge_backend=None, allow_standin_judge=False,
                judge_config=str(common.JUDGE_CONFIG), cache=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_resolve_judge_refuses_rule_for_live_scenes():
    with pytest.raises(StandinJudgeRefused):
        common.resolve_judge(args(backend="openai:x", judge_backend="rule"))


def test_resolve_judge_uses_config_model(tmp_path):
    cfg = json.loads(common.JUDGE_CONFIG.read_text())
    cfg["model"] = "rule"
    p = tmp_path / "judge.json"
    p.write_text(json.dumps(cfg))
    panel = common.resolve_judge(args(judge_config=str(p)))
    assert isinstance(panel.backend, RuleJudgeBackend)


def test_judge_check_status_reads_by_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "CHECKS_DIR", tmp_path)
    assert common.judge_check_status("abc")["status"] == "missing"
    (tmp_path / "abc.json").write_text(json.dumps({"passed": True, "scores": {}}))
    assert common.judge_check_status("abc")["status"] == "passed"


# -- lock workflow ----------------------------------------------------------

@pytest.fixture
def fz(tmp_path, monkeypatch):
    fz = importlib.import_module("freeze_prereg")
    lock = tmp_path / "PREREGISTRATION.lock"
    for mod in (fz, common):
        monkeypatch.setattr(mod, "PREREG_LOCK", lock)
    fz._lock = lock
    return fz


def run(fz, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["freeze_prereg.py", *argv])
    return fz.main()


def test_final_lock_refused_while_decisions_unset(fz, monkeypatch):
    monkeypatch.setattr(fz, "unset_decisions", lambda: ["configs/prereg.json:x"])
    assert run(fz, monkeypatch, "--force") == 1 and not fz._lock.exists()


def test_draft_lock_is_not_treated_as_frozen(fz, monkeypatch):
    monkeypatch.setattr(fz, "unset_decisions", lambda: ["configs/prereg.json:x"])
    assert run(fz, monkeypatch, "--draft") == 0
    st = common.prereg_status()
    assert st["locked"] is False and st["draft_lock"] is True
    assert run(fz, monkeypatch, "--check") == 1


def test_final_lock_check_passes_then_detects_change(fz, monkeypatch):
    monkeypatch.setattr(fz, "unset_decisions", lambda: [])
    assert run(fz, monkeypatch, "--force") == 0
    assert run(fz, monkeypatch, "--check") == 0
    real = common.prereg_hashes()
    monkeypatch.setattr(common, "prereg_hashes",
                        lambda: dict(real, **{"docs/protocol.md": "changed"}))
    assert run(fz, monkeypatch, "--check") == 1


def test_refreeze_keeps_the_superseded_lock(fz, monkeypatch):
    monkeypatch.setattr(fz, "unset_decisions", lambda: [])
    run(fz, monkeypatch, "--force")
    run(fz, monkeypatch, "--force")
    assert list(fz._lock.parent.glob("PREREGISTRATION.lock.superseded-*"))


def test_unset_decisions_reads_the_real_configs():
    fz = importlib.import_module("freeze_prereg")
    unset = fz.unset_decisions()
    assert "configs/judge.json:model" in unset
    assert "configs/prereg.json:p8_equivalence_margin_turns" in unset


def test_lock_covers_the_judge():
    files = common.prereg_hashes()
    for f in ("experiments/configs/judge.json", "experiments/configs/judge_gold.json",
              "experiments/configs/prereg.json", "src/seatkit/judges.py"):
        assert f in files
