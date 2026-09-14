"""The enforcement ladder (paper section 5.2) and the admissibility operator.

This is the module the paper is about. Everything else is scaffolding.

The operator alpha(S_t, C_k) -> A_k,t is implemented as `assemble_view`. At L0-L2
alpha is the identity map and the contract is smuggled back in as text, so the
prohibition is *content* the generator may ignore. At L3-L4 alpha is a real
projection and the forbidden spans are absent, so there is nothing to ignore.

The difference is visible in the return value: at L1 the assembled view contains
the forbidden fact string; at L3 it does not. `tests/test_ladder.py` asserts
exactly that, which is the paper's central claim as a unit test.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .contracts import Seat
from .transcript import Origin, Span, Transcript


class Level(IntEnum):
    """Rungs of the enforcement ladder.

    The jump that matters is L2 -> L3. Below it, ignorance is a behaviour.
    At and above it, ignorance is a property of the input.
    """

    COSTUME = 0      # name and tone only; no admissibility claim
    REQUESTED = 1    # prohibition stated once, in a shared window
    REMINDED = 2     # prohibition re-injected at the context tail each turn
    PARTITIONED = 3  # per-call filtered view; forbidden spans never presented
    ISOLATED = 4     # L3 plus private instrument buffers and a declared bench

    @property
    def enforced(self) -> bool:
        """True when the wall lives in code rather than in a paragraph."""
        return self >= Level.PARTITIONED

    @property
    def label(self) -> str:
        return f"L{int(self)}"


@dataclass(frozen=True)
class View:
    """The assembled admissible state A_k,t for one seat at one turn.

    `text` is what actually goes to the model. `span_count` and `excluded` are
    bookkeeping used by the routing-bleed test suite: at L3/L4 a nonzero
    `leaked_spans` is a defect in this module, not a behaviour of the model.
    """

    seat_id: str
    level: Level
    text: str
    span_count: int
    excluded: int
    leaked_spans: tuple[str, ...] = ()

    def contains(self, needle: str) -> bool:
        return needle.lower() in self.text.lower()


def _shared_window(transcript: Transcript, level: Level) -> list[Span]:
    """L0-L2: everything is available. alpha is the identity map.

    Instrument output is included because in a single-window deployment there is
    nowhere else for it to go. That is the structural reason the paper says
    instruments become hallways unless barred: at these levels they cannot be.
    """
    return [s for s in transcript if s.origin is not Origin.PROBE]


def _projection(transcript: Transcript, seat_id: str) -> tuple[list[Span], int]:
    """L3-L4: alpha is a real projection onto spans visible to this seat.

    Forbidden spans are *absent*, not redacted. A redaction marker is itself
    information about what was removed, which is why there is no marker here.
    """
    kept, dropped = [], 0
    for span in transcript:
        if span.origin is Origin.PROBE:
            continue
        if span.visible(seat_id):
            kept.append(span)
        else:
            dropped += 1
    return kept, dropped


def _render(spans: list[Span], seat_id: str) -> str:
    lines = []
    for s in spans:
        if s.origin is Origin.SETUP:
            lines.append(s.text)
        elif s.origin is Origin.INSTRUMENT:
            who = "your instrument" if s.author == seat_id else f"{s.author}'s instrument"
            lines.append(f"[{who}] {s.text}")
        elif s.origin is Origin.TRANSFER:
            lines.append(f"[{s.author} said aloud] {s.text}")
        elif s.origin is Origin.BENCH:
            lines.append(f"[bench] {s.text}")
        else:
            who = "you" if s.author == seat_id else s.author
            lines.append(f"[{who}] {s.text}")
    return "\n".join(lines)


def assemble_view(
    transcript: Transcript,
    seat: Seat,
    level: Level,
    turn: int,
) -> View:
    """Construct A_k,t for `seat` at `turn` under `level`.

    This function *is* the admissibility operator. Note that the contract enters
    differently at each end of the ladder:

      - L1/L2 append `prohibition_text()`, so the forbidden facts are printed
        into the seat's own context along with an instruction not to use them.
      - L3/L4 never append it, because the facts are not there to prohibit, and
        naming a prohibition would leak the existence of the fact.
    """
    header = seat.contract.clause_text()
    leaked: list[str] = []

    if level.enforced:
        spans, dropped = _projection(transcript, seat.seat_id)
        body = _render(spans, seat.seat_id)
        text = f"{header}\n\n{body}".strip()
        # Routing-bleed self-check. At L3/L4 this must stay empty; if it does
        # not, the defect is here, is deterministic, and is fixed once.
        #
        # Only non-spoken channels count. A sibling saying its own secret out
        # loud is a *disclosure*: a legitimate transfer through the one channel
        # that is supposed to carry content across seats. Counting disclosure as
        # routing bleed would make the wall look broken every time the scene
        # worked as designed. What must never happen is a forbidden fact
        # arriving through setup or through someone else's instrument.
        covert = [sp for sp in spans
                  if sp.origin in (Origin.SETUP, Origin.INSTRUMENT)]
        covert_text = _render(covert, seat.seat_id).lower()
        for fact in seat.contract.forbidden:
            for form in fact.surface_forms():
                if form and form.lower() in covert_text:
                    leaked.append(fact.fact_id)
                    break
        return View(
            seat_id=seat.seat_id,
            level=level,
            text=text,
            span_count=len(spans),
            excluded=dropped,
            leaked_spans=tuple(sorted(set(leaked))),
        )

    # L0-L2: identity map. The whole window, for everyone.
    spans = _shared_window(transcript, level)
    body = _render(spans, seat.seat_id)

    if level is Level.COSTUME:
        text = f"{header}\n\n{body}".strip()
    elif level is Level.REQUESTED:
        # Prohibition stated once, up front, where it will be outnumbered.
        text = f"{header}\n\n{seat.contract.prohibition_text()}\n\n{body}".strip()
    else:  # REMINDED
        # Same paragraph, re-injected at the tail, where attention is better.
        text = (
            f"{header}\n\n{seat.contract.prohibition_text()}\n\n{body}\n\n"
            f"REMINDER, still binding at turn {turn}:\n"
            f"{seat.contract.prohibition_text()}"
        ).strip()

    # At L0-L2 the forbidden content is present by construction. Recording it
    # is not a bug report; it is the measurement the paper asks for.
    for fact in seat.contract.forbidden:
        if fact.value.lower() in body.lower():
            leaked.append(fact.fact_id)

    return View(
        seat_id=seat.seat_id,
        level=level,
        text=text,
        span_count=len(spans),
        excluded=0,
        leaked_spans=tuple(sorted(set(leaked))),
    )


def visibility_for(
    author: str,
    origin: Origin,
    all_seats: tuple[str, ...],
    level: Level,
    instrument_shares_buffer: bool = False,
) -> frozenset[str]:
    """Decide who may see a newly produced span.

    Encodes the E3 manipulation: `instrument_shares_buffer` routes instrument
    output to every seat while leaving its *content* identical. Same tokens
    enter the system either way; only the routing differs. That is what makes
    E3 interpretable where the original fetch-vs-critic comparison was not.
    """
    if not level.enforced:
        # Nothing is hidden in a shared window, whatever the paragraph claims.
        return frozenset(all_seats)

    if origin is Origin.SEAT or origin is Origin.TRANSFER:
        # Spoken content crosses. That is what "aloud" means.
        return frozenset(all_seats)

    if origin is Origin.INSTRUMENT:
        if instrument_shares_buffer:
            return frozenset(all_seats)   # a hallway, declared as such
        return frozenset({author})        # private buffer

    if origin is Origin.BENCH:
        return frozenset(all_seats)

    return frozenset({author}) if author else frozenset(all_seats)
