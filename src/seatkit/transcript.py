"""Provenance-tagged transcript.

Section 5.3 of the paper: "Without provenance there is nothing to filter on."
Every span records who produced it and who is allowed to see it. A span with
no visibility set is a bug, not a default, so construction requires one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Iterable, Iterator, Sequence


class Origin(str, Enum):
    """Where a span came from. Used to distinguish host speech from tool output."""

    SETUP = "setup"          # contract text, scene framing
    SEAT = "seat"            # a seat speaking aloud
    INSTRUMENT = "instrument"  # a seat's private instrument
    TRANSFER = "transfer"    # content explicitly moved between seats by the bench
    BENCH = "bench"          # the arbiter, if it speaks at all
    PROBE = "probe"          # an experimental probe (never enters a live scene)


@dataclass(frozen=True)
class Span:
    """One unit of transcript content with full provenance.

    Attributes:
        text: the content itself.
        origin: what kind of component produced it.
        author: seat id of the producer, or None for setup/bench spans.
        visible_to: frozen set of seat ids permitted to see this span. The bench
            is assumed to see everything; that is what makes it the bench, and
            what makes bench capture (paper 5.4) the residual risk.
        turn: scene turn index.
        tags: free-form labels used by experiments (e.g. "forbidden_fact").
    """

    text: str
    origin: Origin
    author: str | None
    visible_to: frozenset[str]
    turn: int
    tags: frozenset[str] = field(default_factory=frozenset)

    def visible(self, seat_id: str) -> bool:
        return seat_id in self.visible_to

    def to_dict(self) -> dict:
        d = asdict(self)
        d["origin"] = self.origin.value
        d["visible_to"] = sorted(self.visible_to)
        d["tags"] = sorted(self.tags)
        return d


class Transcript:
    """An append-only sequence of spans.

    The transcript holds *everything*. It is never handed to a seat directly;
    seats receive views assembled by `seatkit.ladder`. Confusing the transcript
    with a view is the routing bug the paper calls L3/L4 failure.
    """

    def __init__(self, spans: Iterable[Span] | None = None) -> None:
        self._spans: list[Span] = list(spans or [])

    def append(self, span: Span) -> Span:
        self._spans.append(span)
        return span

    def extend(self, spans: Iterable[Span]) -> None:
        self._spans.extend(spans)

    def __iter__(self) -> Iterator[Span]:
        return iter(self._spans)

    def __len__(self) -> int:
        return len(self._spans)

    def __getitem__(self, idx):
        return self._spans[idx]

    @property
    def spans(self) -> Sequence[Span]:
        return tuple(self._spans)

    def by_author(self, seat_id: str) -> list[Span]:
        return [s for s in self._spans if s.author == seat_id]

    def by_turn(self, turn: int) -> list[Span]:
        return [s for s in self._spans if s.turn == turn]

    def tagged(self, tag: str) -> list[Span]:
        return [s for s in self._spans if tag in s.tags]

    DEFAULT_TAIL_ORIGINS = (Origin.SEAT, Origin.TRANSFER, Origin.INSTRUMENT)

    def tail_tokens(self, n_tokens: int,
                    origins: tuple[Origin, ...] | None = None) -> str:
        """Whitespace-token tail of the flat transcript.

        This is deliberately crude: it is the E4 condition (b) baseline, and
        condition (b) is *supposed* to be undisciplined. It flattens everything
        into one stream with no visibility tags, which is the point of the
        experiment.

        Instrument output is included by default because in a shared window
        (L0-L2) it is in the window, and it is precisely the untagged material
        that makes a transcript tail dangerous to reseed from. Excluding it
        would flatter the condition and hide the effect E4 is looking for.
        """
        keep = origins or self.DEFAULT_TAIL_ORIGINS
        flat = " ".join(
            f"[{s.author or s.origin.value}] {s.text}"
            for s in self._spans if s.origin in keep
        )
        toks = flat.split()
        return " ".join(toks[-n_tokens:])

    def to_json(self, indent: int = 2) -> str:
        return json.dumps([s.to_dict() for s in self._spans], indent=indent)

    @classmethod
    def from_json(cls, blob: str) -> "Transcript":
        raw = json.loads(blob)
        spans = [
            Span(
                text=r["text"],
                origin=Origin(r["origin"]),
                author=r["author"],
                visible_to=frozenset(r["visible_to"]),
                turn=r["turn"],
                tags=frozenset(r.get("tags", [])),
            )
            for r in raw
        ]
        return cls(spans)
