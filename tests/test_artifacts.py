"""Externalization: what a tail carries that a tagged sheet does not."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from seatkit import (Bench, Instrument, Level, MockBackend, SceneConfig,
                     SeatSheet, facing_pair, make_facts, reseed,
                     tail_carries_forbidden)


def scene(level=Level.REQUESTED, turns=12):
    backend = MockBackend()
    a, b = make_facts(2, seed=11, prefix="a"), make_facts(2, seed=22, prefix="b")
    geo, bio = facing_pair("geo", "bio", a, b,
                           a_extra={"persona": "a geologist",
                                    "needs": ("protect the rapid-event reading",)},
                           b_extra={"persona": "a paleobiologist"})
    insts = {s.seat_id: Instrument(host=s, backend=backend) for s in (geo, bio)}
    res = Bench([geo, bio], backend, insts).run(
        SceneConfig(turns=turns, level=level, probe_turns=()))
    return geo, bio, res


def test_sheet_roundtrips():
    geo, _, res = scene()
    sheet = SeatSheet.from_seat(geo)
    sheet.record("we will call it the lower member", ("geo", "bio"))
    back = SeatSheet.from_json(sheet.to_json())
    assert back.seat_id == "geo"
    assert back.ledger[0].visible_to == ("geo", "bio")


def test_sheet_filters_ledger_by_visibility():
    geo, _, _ = scene()
    sheet = SeatSheet.from_seat(geo)
    sheet.record("shared convention", ("geo", "bio"))
    sheet.record("bio-only convention", ("bio",))
    text = sheet.admissible_text("geo")
    assert "shared convention" in text
    assert "bio-only convention" not in text


def test_reseed_none_carries_nothing():
    geo, _, res = scene()
    ctx = reseed("none", geo, res.transcript)
    assert "Established" not in ctx


def test_reseed_sheet_requires_a_sheet():
    geo, _, res = scene()
    with pytest.raises(ValueError):
        reseed("sheet", geo, res.transcript, sheet=None)


def test_tail_ignores_visibility_tags_even_at_L3():
    """The E4 finding: enforcement during phase 1 does not survive a raw tail.

    A transcript tail is a flat dump. It discards the visibility tags the live
    system was maintaining, which is why reseeding from one re-opens walls that
    held throughout the scene.
    """
    geo, _, res = scene(level=Level.PARTITIONED, turns=16)
    tail = res.transcript.tail_tokens(600)
    private = [s for s in res.transcript
               if s.author == "bio" and len(s.visible_to) == 1]
    assert private, "no private spans produced; test is not exercising anything"
    assert any(s.text[:30] in tail for s in private)


def test_tail_carries_forbidden_needs_no_model_call():
    geo, _, res = scene(level=Level.REQUESTED, turns=16)
    carried = tail_carries_forbidden(res.transcript, geo)
    assert isinstance(carried, list)
