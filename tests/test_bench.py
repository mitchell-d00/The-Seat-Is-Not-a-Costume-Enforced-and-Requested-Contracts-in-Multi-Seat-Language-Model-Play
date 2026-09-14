"""Scene orchestration: routing integrity, disclosure, and bench capture."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seatkit import (Bench, Instrument, Level, MockBackend, Origin, SceneConfig,
                     Span, facing_pair, make_facts)


def build(level, shared_buffer=False, backend=None):
    backend = backend or MockBackend()
    a, b = make_facts(2, seed=11, prefix="a"), make_facts(2, seed=22, prefix="b")
    geo, bio = facing_pair("geo", "bio", a, b,
                           a_extra={"persona": "a geologist"},
                           b_extra={"persona": "a paleobiologist"})
    insts = {s.seat_id: Instrument(host=s, backend=backend,
                                   shares_buffer=shared_buffer)
             for s in (geo, bio)}
    return Bench([geo, bio], backend, insts), (geo, bio)


@pytest.mark.parametrize("level", [Level.PARTITIONED, Level.ISOLATED])
def test_no_routing_bleed_at_enforced_levels(level):
    """The regression test for the whole framework."""
    bench, _ = build(level)
    res = bench.run(SceneConfig(turns=10, level=level, probe_turns=(4, 8)))
    assert res.routing_bleed == [], f"routing bleed: {res.routing_bleed}"


@pytest.mark.parametrize("level", [Level.PARTITIONED, Level.ISOLATED])
def test_private_instrument_output_never_reaches_sibling(level):
    bench, (geo, bio) = build(level, shared_buffer=False)
    res = bench.run(SceneConfig(turns=8, level=level, probe_turns=()))
    for span in res.transcript:
        if span.origin is Origin.INSTRUMENT:
            assert span.visible_to == frozenset({span.author})


def test_shared_buffer_marks_itself_as_a_hallway():
    bench, _ = build(Level.ISOLATED, shared_buffer=True)
    res = bench.run(SceneConfig(turns=6, level=Level.ISOLATED, probe_turns=()))
    inst = [s for s in res.transcript if s.origin is Origin.INSTRUMENT]
    assert inst and all("hallway" in s.tags for s in inst)


def test_disclosure_is_tracked_not_counted_as_leak():
    """A seat saying its own secret aloud is a transfer, not a wall failure."""
    bench, _ = build(Level.REQUESTED)
    res = bench.run(SceneConfig(turns=16, level=Level.REQUESTED,
                                probe_turns=(4, 8, 16)))
    scoring = res.scoring_probes()
    assert all(not p.disclosed for p in scoring)
    assert len(scoring) <= len(res.probes)


def test_bench_capture_is_detected():
    """A bench that contributes has become a hallway with a gavel."""
    bench, _ = build(Level.ISOLATED)
    res = bench.run(SceneConfig(turns=4, level=Level.ISOLATED, probe_turns=()))
    res.transcript.append(Span("Let me summarize where you both agree.",
                               Origin.BENCH, None,
                               frozenset({"geo", "bio"}), 5))
    with pytest.raises(AssertionError, match="bench capture"):
        bench.assert_transport_only(res)


def test_probes_do_not_contaminate_the_scene():
    """Probes fork the transcript; the live scene must not grow because of them."""
    backend = MockBackend()
    b1, _ = build(Level.REQUESTED, backend=backend)
    r1 = b1.run(SceneConfig(turns=8, level=Level.REQUESTED, probe_turns=()))
    b2, _ = build(Level.REQUESTED, backend=backend)
    r2 = b2.run(SceneConfig(turns=8, level=Level.REQUESTED, probe_turns=(4, 8)))
    spoken = lambda r: [s for s in r.transcript if s.origin is Origin.SEAT]
    assert [s.text for s in spoken(r1)] == [s.text for s in spoken(r2)]


def test_transcript_roundtrips_with_provenance():
    from seatkit import Transcript
    bench, _ = build(Level.PARTITIONED)
    res = bench.run(SceneConfig(turns=4, level=Level.PARTITIONED, probe_turns=()))
    back = Transcript.from_json(res.transcript.to_json())
    assert len(back) == len(res.transcript)
    assert [s.visible_to for s in back] == [s.visible_to for s in res.transcript]
