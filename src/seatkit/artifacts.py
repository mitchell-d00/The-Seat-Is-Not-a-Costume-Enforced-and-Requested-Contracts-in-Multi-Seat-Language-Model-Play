"""Externalized residue (clause five) and the three E4 reseeding conditions.

The interesting claim in E4 is that conditions (b) and (c) come apart: a
transcript tail preserves conventions and destroys the forbidden set in the same
move, because it carries both seats' content with no visibility tags. A
structured seat sheet preserves both, because it carries the tags.

`SeatSheet` is the artifact. `reseed` implements the three conditions so they
can be compared without one of them quietly doing the other's job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .contracts import Contract, Fact, Seat
from .transcript import Origin, Span, Transcript


@dataclass
class LedgerEntry:
    """An established fact plus who is allowed to carry it forward."""

    text: str
    visible_to: tuple[str, ...]
    kind: str = "convention"   # convention | claim | commitment


@dataclass
class SeatSheet:
    """Condition (c): the disciplined externalization.

    Carries the contract clauses and a visibility-tagged ledger. Re-admission
    filters the ledger by seat, which is why constraint survival holds.
    """

    seat_id: str
    needs: tuple[str, ...] = ()
    synergies: tuple[str, ...] = ()
    instrument_rights: dict = field(default_factory=dict)
    forbidden_ids: tuple[str, ...] = ()
    ledger: list[LedgerEntry] = field(default_factory=list)

    @classmethod
    def from_seat(cls, seat: Seat) -> "SeatSheet":
        r = seat.contract.instrument
        return cls(
            seat_id=seat.seat_id,
            needs=seat.contract.needs,
            synergies=seat.contract.synergies,
            instrument_rights={
                "can_fetch": r.can_fetch, "can_critique": r.can_critique,
                "can_remember": r.can_remember,
                "writes_to_shared": r.writes_to_shared,
                "may_referee": r.may_referee,
            },
            forbidden_ids=tuple(sorted(seat.contract.forbidden_ids())),
        )

    def record(self, text: str, visible_to: tuple[str, ...],
               kind: str = "convention") -> None:
        self.ledger.append(LedgerEntry(text, visible_to, kind))

    def harvest(self, transcript: Transcript, seat: Seat,
                markers: tuple[str, ...]) -> None:
        """Pull established conventions out of a transcript, with tags intact."""
        for span in transcript:
            if span.origin is not Origin.SEAT:
                continue
            for m in markers:
                if m.lower() in span.text.lower():
                    self.record(span.text, tuple(sorted(span.visible_to)))
                    break

    def admissible_text(self, seat_id: str) -> str:
        """Render only what this seat is allowed to carry forward."""
        lines = []
        if self.needs:
            lines.append("Assigned jobs: " + "; ".join(self.needs))
        if self.synergies:
            lines.append("Joint competence: " + "; ".join(self.synergies))
        entries = [e for e in self.ledger if seat_id in e.visible_to]
        if entries:
            lines.append("Established, and admissible to you:")
            lines += [f"  - {e.text}" for e in entries]
        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps({
            "seat_id": self.seat_id,
            "needs": list(self.needs),
            "synergies": list(self.synergies),
            "instrument_rights": self.instrument_rights,
            "forbidden_ids": list(self.forbidden_ids),
            "ledger": [{"text": e.text, "visible_to": list(e.visible_to),
                        "kind": e.kind} for e in self.ledger],
        }, indent=indent)

    @classmethod
    def from_json(cls, blob: str) -> "SeatSheet":
        r = json.loads(blob)
        s = cls(
            seat_id=r["seat_id"], needs=tuple(r["needs"]),
            synergies=tuple(r["synergies"]),
            instrument_rights=r["instrument_rights"],
            forbidden_ids=tuple(r["forbidden_ids"]),
        )
        s.ledger = [LedgerEntry(e["text"], tuple(e["visible_to"]), e["kind"])
                    for e in r["ledger"]]
        return s


def reseed(condition: str, seat: Seat, transcript: Transcript,
           sheet: SeatSheet | None = None, tail_tokens: int = 400) -> str:
    """Build the reseeding context for one of the three E4 conditions.

    Args:
        condition: "none" | "tail" | "sheet".

    Condition "tail" deliberately returns untagged text containing both seats'
    content. That is not an oversight in the implementation; it is the thing
    being measured.
    """
    header = seat.contract.clause_text()

    if condition == "none":
        return header

    if condition == "tail":
        return f"{header}\n\nEarlier in this session:\n{transcript.tail_tokens(tail_tokens)}"

    if condition == "sheet":
        if sheet is None:
            raise ValueError("condition 'sheet' requires a SeatSheet")
        return f"{header}\n\n{sheet.admissible_text(seat.seat_id)}"

    raise ValueError(f"unknown reseed condition: {condition!r}")


def tail_carries_forbidden(transcript: Transcript, seat: Seat,
                           tail_tokens: int = 400) -> list[str]:
    """Which of `seat`'s forbidden facts survive into an untagged tail.

    This is the mechanism behind E4 prediction 2, and it can be checked without
    running a model at all.
    """
    tail = transcript.tail_tokens(tail_tokens).lower()
    return [f.fact_id for f in seat.contract.forbidden
            if f.value.lower() in tail or f.entity.lower() in tail]
