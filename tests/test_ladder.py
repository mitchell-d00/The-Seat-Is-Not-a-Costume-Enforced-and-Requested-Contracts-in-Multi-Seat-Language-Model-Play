"""The paper's central claim as a unit test.

At L0-L2 the forbidden fact is in the seat's view and the wall is a request.
At L3-L4 it is not in the view at all and there is nothing to request.

If `test_enforced_view_excludes_forbidden_fact` ever fails, the paper's
"unavoidable full availability (enforced)" kill condition has fired for this
implementation, and `ladder.py` is where to look.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seatkit import (Level, MockBackend, Origin, Span, Transcript,
                     assemble_view, facing_pair, make_facts, visibility_for)


@pytest.fixture
def pair():
    a, b = make_facts(2, seed=11, prefix="a"), make_facts(2, seed=22, prefix="b")
    return facing_pair("geo", "bio", a, b,
                       a_extra={"persona": "a geologist"},
                       b_extra={"persona": "a paleobiologist"})


def seeded(pair, level):
    geo, bio = pair
    t = Transcript()
    for seat in (geo, bio):
        for f in seat.contract.holds:
            vis = (frozenset({seat.seat_id}) if level.enforced
                   else frozenset({"geo", "bio"}))
            t.append(Span(f.statement(), Origin.SETUP, seat.seat_id, vis, 0,
                          frozenset({"forbidden_fact"})))
    return t


@pytest.mark.parametrize("level", [Level.COSTUME, Level.REQUESTED, Level.REMINDED])
def test_requested_view_contains_forbidden_fact(pair, level):
    """At L0-L2 alpha is the identity map. The fact is present and prohibited."""
    geo, _ = pair
    view = assemble_view(seeded(pair, level), geo, level, turn=1)
    for fact in geo.contract.forbidden:
        assert view.contains(fact.value), (
            f"{level.label}: forbidden value absent from a shared window; "
            "the identity map is not being applied"
        )


@pytest.mark.parametrize("level", [Level.PARTITIONED, Level.ISOLATED])
def test_enforced_view_excludes_forbidden_fact(pair, level):
    """At L3-L4 alpha is a projection. The fact is absent, not redacted.

    This is the assertion the paper rests on.
    """
    geo, _ = pair
    view = assemble_view(seeded(pair, level), geo, level, turn=1)
    for fact in geo.contract.forbidden:
        assert not view.contains(fact.value), f"{level.label}: routing bleed"
        assert not view.contains(fact.entity), f"{level.label}: entity leaked"
    assert view.leaked_spans == ()
    assert view.excluded > 0, "projection dropped nothing; check visibility tags"


@pytest.mark.parametrize("level", [Level.PARTITIONED, Level.ISOLATED])
def test_enforced_view_has_no_redaction_marker(pair, level):
    """A redaction marker is itself information about what was removed."""
    geo, _ = pair
    view = assemble_view(seeded(pair, level), geo, level, turn=1)
    for marker in ("[redacted]", "REDACTED", "withheld", "[removed]"):
        assert marker.lower() not in view.text.lower()


@pytest.mark.parametrize("level", [Level.PARTITIONED, Level.ISOLATED])
def test_enforced_view_omits_prohibition_paragraph(pair, level):
    """Naming a prohibition leaks the existence of the thing prohibited."""
    geo, _ = pair
    view = assemble_view(seeded(pair, level), geo, level, turn=1)
    assert "must not use it" not in view.text


def test_seat_keeps_its_own_facts(pair):
    """The projection must not be so aggressive that a seat forgets its job."""
    geo, _ = pair
    view = assemble_view(seeded(pair, Level.PARTITIONED), geo,
                         Level.PARTITIONED, turn=1)
    for fact in geo.contract.holds:
        assert view.contains(fact.value)


def test_private_instrument_output_is_invisible_to_sibling():
    vis = visibility_for("geo", Origin.INSTRUMENT, ("geo", "bio"),
                         Level.PARTITIONED, instrument_shares_buffer=False)
    assert vis == frozenset({"geo"})


def test_shared_buffer_is_a_hallway_by_construction():
    """Declaring a shared buffer makes the instrument a hallway. No paragraph
    about privacy changes that, which is the point of paper section 5.4."""
    vis = visibility_for("geo", Origin.INSTRUMENT, ("geo", "bio"),
                         Level.ISOLATED, instrument_shares_buffer=True)
    assert vis == frozenset({"geo", "bio"})


def test_shared_window_ignores_visibility_entirely():
    """At L0-L2 nothing is hidden, whatever the contract claims."""
    vis = visibility_for("geo", Origin.INSTRUMENT, ("geo", "bio"),
                         Level.REQUESTED, instrument_shares_buffer=False)
    assert vis == frozenset({"geo", "bio"})


def test_spoken_content_always_crosses():
    """Disclosure is legitimate. Walls govern what a seat knows, not what it says."""
    for level in Level:
        vis = visibility_for("geo", Origin.SEAT, ("geo", "bio"), level)
        assert vis == frozenset({"geo", "bio"})


def test_level_enforced_boundary():
    assert not Level.REMINDED.enforced
    assert Level.PARTITIONED.enforced
    assert Level.ISOLATED.enforced
