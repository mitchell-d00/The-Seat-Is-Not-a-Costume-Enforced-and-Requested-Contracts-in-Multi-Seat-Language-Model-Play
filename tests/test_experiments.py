"""The experiment scripts, end to end on the mock, plus the pieces that decide
what a results file is allowed to claim.

These are harness tests. They check that every cell the paper specifies is
present, that verdicts are computed from data rather than printed as fixed
text, and that provenance (judges, preregistration) is recorded.
"""
import argparse
import importlib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(ROOT / "src"))


def run(script, tmp_path, *extra):
    out = tmp_path / f"{script}.json"
    subprocess.run([sys.executable, str(EXP / f"{script}.py"), "--scenes", "2",
                    "--turns", "4", "--out", str(out), *extra],
                   check=True, capture_output=True, text=True, cwd=ROOT)
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("res")
    return {s: run(s, tmp) for s in ("e1_leak_probe", "e2_attribution",
                                     "e3_instrument_symmetry", "e4_externalization")}


def test_every_experiment_has_a_ceiling_cell(results):
    assert "ceiling" in results["e1_leak_probe"]["cells"]
    assert "ceiling" in results["e2_attribution"]["cells"]
    assert "ceiling" in results["e3_instrument_symmetry"]["cells"]
    assert "ceiling" in results["e4_externalization"]["by_cell"]


def test_positions_are_reported(results):
    assert set(results["e1_leak_probe"]["positions_elicited"]) == {"L1", "L2", "L3", "L4"}
    assert "positions_content_accuracy" in results["e2_attribution"]
    assert "positions_half_life" in results["e3_instrument_symmetry"]
    assert "P11_floor_minus_ceiling" in results["e1_leak_probe"]


def test_provenance_is_recorded(results):
    for r in results.values():
        assert "preregistration" in r and "notice" in r   # mock notice
    assert results["e2_attribution"]["judges"]["standin"] is True
    assert results["e1_leak_probe"]["leak_scoring"] == "exact-match-only"


def test_e3_reports_swap_arm_and_construction_warning(results):
    e3 = results["e3_instrument_symmetry"]
    assert "L3_private_swapped" in e3["cells"] and "P8_contract_swap" in e3
    assert e3.get("P6_L1_identical_by_construction") is True


def test_e4_has_no_fixed_conclusion_text():
    src = (EXP / "e4_externalization.py").read_text()
    assert "A transcript tail degrades constraint survival at EVERY level" not in src


def test_e4_verdicts_follow_the_data():
    e4 = importlib.import_module("e4_externalization")

    def cell(tail_c, sheet_c, p10_supported):
        t = {"a_greater": 5, "b_greater": 0, "p": 0.01, "supported": True}
        return {"conditions": {"none": {"constraint_survival": 1.0},
                               "tail": {"constraint_survival": tail_c},
                               "sheet": {"constraint_survival": sheet_c}},
                "tests": {"P9_conventions_sheet_gt_tail": t,
                          "P9_conventions_tail_gt_none": t,
                          "P10_constraints_sheet_gt_tail":
                              dict(t, supported=p10_supported, p=0.01 if p10_supported else 0.6)}}
    a = e4.verdicts({"L1": cell(0.4, 0.9, True), "L3": cell(1.0, 1.0, False)}, False)
    b = e4.verdicts({"L1": cell(1.0, 1.0, False), "L3": cell(0.5, 1.0, True)}, False)
    assert a != b
    assert any("tail degrades constraint survival at: L1" == l for l in a)
    assert any("tail degrades constraint survival at: L3" == l for l in b)
    assert all("not evidence" in l for l in e4.verdicts({"L1": cell(.4, .9, True)}, True))


def test_live_backend_without_judge_is_refused(tmp_path):
    r = subprocess.run([sys.executable, str(EXP / "e2_attribution.py"),
                        "--backend", "openai:not-a-model", "--scenes", "1",
                        "--out", str(tmp_path / "x.json")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "stand-in judge" in (r.stdout + r.stderr)


def test_e2_runs_with_a_model_judge():
    """The live-judge path, with a fake model standing in for the provider."""
    from seatkit import Completion, MockBackend
    from seatkit.judges import ModelAttributionJudge, ModelParaphraser
    e2 = importlib.import_module("e2_attribution")

    class Fake:
        name = "fake"

        def generate(self, prompt, **kw):
            text = "A" if "Hidden utterance" in prompt else "plain restatement"
            return Completion(text, "fake", 0, 0)

    args = argparse.Namespace(scenes=2, turns=3, backend="mock", cache=None)
    judge, para = ModelAttributionJudge(Fake()), ModelParaphraser(Fake())
    per_scene = e2.run_cell(e2.Level.REQUESTED, args, MockBackend(), judge, para,
                            random.Random(0))
    assert len(per_scene) == 2 and all(s["content"] for s in per_scene)
    assert judge.unparseable == 0


def test_grader_agreement_counts_and_excludes_unlabelled():
    ga = importlib.import_module("grader_agreement")
    rows = [{"model": True, "exact": False, "human": True},
            {"model": False, "exact": False, "human": False},
            {"model": True, "exact": True, "human": True},
            {"model": False, "exact": False, "human": True},
            {"model": True, "exact": False, "human": None}]
    r = ga.agreement(rows)
    assert r["n_labelled"] == 4 and r["n_unlabelled"] == 1
    assert r["exact"]["missed_leaks"] == 2 and r["model"]["missed_leaks"] == 1


def test_prereg_status_detects_changes(tmp_path, monkeypatch):
    common = importlib.import_module("_common")
    lock = tmp_path / "PREREGISTRATION.lock"
    monkeypatch.setattr(common, "PREREG_LOCK", lock)
    assert common.prereg_status()["locked"] is False
    files = common.prereg_hashes()
    assert "docs/PREREGISTRATION.md" in files
    frozen = dict(files, **{"docs/PREREGISTRATION.md": "0" * 64})
    lock.write_text(json.dumps({"files": frozen, "frozen_at_utc": "t", "digest": "d"}))
    st = common.prereg_status()
    assert st["locked"] and st["changed_since_freeze"] == ["docs/PREREGISTRATION.md"]


def test_counterbalancing_applies_in_every_experiment():
    common = importlib.import_module("_common")
    g0, _ = common.build_pair(0)
    g1, _ = common.build_pair(1)
    assert g0.contract.holds[0].fact_id.startswith("a")
    assert g1.contract.holds[0].fact_id.startswith("b")


def test_contract_swap_keeps_voice_and_moves_needs():
    common = importlib.import_module("_common")
    geo, bio = common.build_pair(0)
    sgeo, sbio = common.build_pair(0, swap_contracts=True)
    assert sgeo.contract.persona == geo.contract.persona
    assert sgeo.contract.needs == bio.contract.needs
    assert sgeo.contract.holds == bio.contract.holds


def test_audit_sample_has_empty_human_column(tmp_path):
    from seatkit import ProbeResult, make_facts
    e1 = importlib.import_module("e1_leak_probe")
    f = make_facts(1, seed=9)[0]
    tagged = {"L1": [(0, ProbeResult(4, "geo", f.fact_id, False, "elicited",
                                     "resp", True, graded=True))] * 50}
    path = tmp_path / "audit.jsonl"
    e1.write_audit(tagged, {(0, f.fact_id): f}, str(path))
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert 0 < len(rows) < 50
    assert all(r["human"] is None and r["model"] is True for r in rows)


def test_e4_reports_probe_primary_and_context_mechanism(results):
    e4 = results["e4_externalization"]
    tail = e4["by_cell"]["L1"]["conditions"]["tail"]
    assert "constraint_survival" in tail and "constraint_survival_context" in tail
    assert e4["measures"]["not_implemented"]
