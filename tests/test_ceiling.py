"""The ceiling reference condition (paper 9; seatkit/ceiling.py).

For a stateless backend, an ideal L3 projection and two seats in separate
processes hand the model the same string. `SeparatedBench` never calls the
projection code, so asserting that equality is a differential test of
`ladder.py`: a routing defect shows up as an L3 prompt the ceiling never sent.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import seatkit.ladder as ladder
from seatkit import (CEILING, Bench, Instrument, Level, MockBackend, Origin,
                     SceneConfig, SeparatedBench, facing_pair, make_facts)


class Recording(MockBackend):
    def __init__(self):
        super().__init__()
        self.calls = []

    def generate(self, prompt, **kw):
        self.calls.append((prompt, kw.get("seed")))
        return super().generate(prompt, **kw)


def pair():
    a, b = make_facts(3, seed=31, prefix="a"), make_facts(3, seed=32, prefix="b")
    return facing_pair("geo", "bio", a, b,
                       a_extra={"persona": "a geologist", "needs": ("protect A",)},
                       b_extra={"persona": "a paleobiologist", "needs": ("protect B",)})


def run_l3(cfg):
    geo, bio = pair()
    be = Recording()
    Bench([geo, bio], be, {s.seat_id: Instrument(host=s, backend=be)
                           for s in (geo, bio)}).run(cfg)
    return be.calls


def run_ceiling(cfg):
    geo, bio = pair()
    made = {}

    def factory(sid):
        made[sid] = Recording()
        return made[sid]

    res = SeparatedBench([geo, bio], factory).run(cfg)
    return made, res


CFG = SceneConfig(turns=10, level=Level.PARTITIONED, seed=4, probe_turns=(4, 8))


def test_ceiling_prompts_equal_l3_prompts_seat_by_seat():
    l3_calls = run_l3(CFG)
    made, _ = run_ceiling(CFG)
    # Every call the ceiling made for a seat appears in L3, in the same order.
    for sid, be in made.items():
        persona = "geologist" if sid == "geo" else "paleobiologist"
        mine = [c for c in l3_calls if f"You are a {persona}" in c[0]
                or f"employed by {sid}" in c[0]]
        assert mine and be.calls == mine


def test_differential_test_catches_a_routing_bug(monkeypatch):
    """Break the projection so L3 shows every span to every seat. The ceiling,
    which never calls the projection, must now disagree with L3."""
    def leaky(transcript, seat_id):
        return [s for s in transcript if s.origin is not Origin.PROBE], 0
    monkeypatch.setattr(ladder, "_projection", leaky)
    l3_calls = {p for p, _ in run_l3(CFG)}
    made, res = run_ceiling(CFG)
    ceiling_calls = {p for be in made.values() for p, _ in be.calls}
    assert ceiling_calls - l3_calls, "ceiling should have sent prompts L3 no longer sends"
    assert res.routing_bleed_count == 0


def test_ceiling_result_is_labelled_and_clean():
    _, res = run_ceiling(CFG)
    assert res.condition == CEILING
    assert res.routing_bleed_count == 0
    assert any(p.kind == "elicited" for p in res.probes)


def test_sibling_facts_reach_a_seat_only_through_speech():
    geo, bio = pair()
    _, res = run_ceiling(CFG)
    for span in res.transcript:
        if span.origin in (Origin.SETUP, Origin.INSTRUMENT) and span.author:
            assert span.visible_to == frozenset({span.author})


def test_one_backend_per_seat_is_required():
    shared = MockBackend()
    with pytest.raises(ValueError, match="same object"):
        SeparatedBench(list(pair()), lambda sid: shared)


def test_shared_instrument_buffer_is_refused():
    geo, bio = pair()
    insts = {"geo": Instrument(host=geo, backend=MockBackend(), shares_buffer=True)}
    with pytest.raises(ValueError, match="hallway"):
        SeparatedBench([geo, bio], lambda sid: MockBackend(), instruments=insts)


def test_replay_log_instruments_are_used_verbatim():
    geo, bio = pair()
    logs = {s.seat_id: {t: f"note {s.seat_id} {t}" for t in range(12)} for s in (geo, bio)}
    insts = {s.seat_id: Instrument(host=s, replay_log=logs[s.seat_id]) for s in (geo, bio)}
    res = SeparatedBench([geo, bio], lambda sid: MockBackend(),
                         instruments=insts).run(SceneConfig(turns=3, seed=1))
    notes = [s.text for s in res.transcript if s.origin is Origin.INSTRUMENT]
    assert notes[:2] == ["note geo 1", "note bio 1"]
