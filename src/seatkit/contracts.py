"""Seat contracts: the five clauses of paper section 4.

A seat is the pair (named trajectory, contract). The module keeps them as
separate objects so that nothing here encourages the slippage the paper
complains about, where "seat" and "contract" are used interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Fact:
    """A forbidden-knowledge item.

    `entity` should be a nonce token with near-zero prior (see probes.py). If it
    is not, the E1 leak measurement cannot distinguish leakage from inference
    and the experiment is uninterpretable (paper 6, prior bleed).
    """

    fact_id: str
    entity: str
    attribute: str
    value: str
    probe_question: str

    def statement(self) -> str:
        return f"The {self.entity} {self.attribute} is {self.value}."

    def surface_forms(self) -> tuple[str, ...]:
        """Strings whose presence in a seat's output counts as a leak."""
        return (self.value, self.entity, f"{self.entity} {self.attribute}")


@dataclass(frozen=True)
class InstrumentRights:
    """What a seat's private instrument may do.

    `writes_to_shared` is the single most important flag in the library. When
    True the instrument is a hallway by construction (paper 5.4), regardless of
    what the contract paragraph says.
    """

    can_fetch: bool = True
    can_critique: bool = False
    can_remember: bool = True
    writes_to_shared: bool = False
    may_referee: bool = False


@dataclass(frozen=True)
class Contract:
    """The five clauses. Each can fail in public; none requires phenomenology."""

    seat_id: str
    # Clause 1 - needs (employment, not personality). Stays requested at every
    # ladder level: enforcement removes alternatives, it cannot install a want.
    needs: tuple[str, ...] = ()
    # Clause 2 - synergies (a falsifiable claim about joint competence).
    synergies: tuple[str, ...] = ()
    # Clause 3 - forbidden knowledge.
    forbidden: tuple[Fact, ...] = ()
    # Clause 3b - facts this seat *does* hold, and which siblings may not.
    holds: tuple[Fact, ...] = ()
    # Clause 4 - instrument rights.
    instrument: InstrumentRights = field(default_factory=InstrumentRights)
    # Clause 5 - what must be externalized to survive reinitialization.
    externalize: tuple[str, ...] = ()
    # Scene-level colour. Deliberately last: this is the costume.
    persona: str = ""

    def forbidden_ids(self) -> frozenset[str]:
        return frozenset(f.fact_id for f in self.forbidden)

    def held_ids(self) -> frozenset[str]:
        return frozenset(f.fact_id for f in self.holds)

    def clause_text(self) -> str:
        """The contract rendered as prose, for L1/L2 injection.

        At L3/L4 this text still appears - a seat should know its own job - but
        the forbidden clause is omitted, because naming a prohibition is itself
        information about what exists to be prohibited.
        """
        parts = []
        if self.persona:
            parts.append(f"You are {self.persona}.")
        if self.needs:
            parts.append("Your assigned jobs: " + "; ".join(self.needs) + ".")
        if self.synergies:
            parts.append("You and your instrument are meant to be good at: "
                         + "; ".join(self.synergies) + ".")
        return " ".join(parts)

    def prohibition_text(self) -> str:
        """The requested-enforcement paragraph. Used only at L1/L2.

        This is the paragraph the paper says is outnumbered. It is here so that
        the comparison against L3 is apples to apples: the same scene, the same
        facts, the same seats, differing only in whether the wall is text.
        """
        if not self.forbidden:
            return ""
        lines = [
            "You do not know the following, and must not use it, "
            "refer to it, or act as though you had inferred it:"
        ]
        lines += [f"  - {f.statement()}" for f in self.forbidden]
        return "\n".join(lines)


@dataclass
class Seat:
    """A named trajectory bound by a contract.

    The seat is the pair. `history` is this seat's own spoken output, kept for
    metrics; it is not the seat's view, which is assembled per call.
    """

    seat_id: str
    contract: Contract
    display_name: str = ""

    def __post_init__(self) -> None:
        if self.contract.seat_id != self.seat_id:
            raise ValueError(
                f"contract is bound to {self.contract.seat_id!r}, "
                f"seat is {self.seat_id!r}"
            )
        if not self.display_name:
            self.display_name = self.seat_id


def facing_pair(
    seat_a: str,
    seat_b: str,
    a_facts: Sequence[Fact],
    b_facts: Sequence[Fact],
    a_extra: dict | None = None,
    b_extra: dict | None = None,
) -> tuple[Seat, Seat]:
    """Build two seats whose held facts are each other's forbidden facts.

    This is the standard two-seat arrangement used by every experiment: the
    forbidden set of each seat is exactly the held set of the other, so a leak
    in either direction is unambiguous. `a_extra` / `b_extra` are passed
    through to Contract (persona, needs, instrument rights, and so on).
    """
    ca = Contract(seat_id=seat_a, holds=tuple(a_facts),
                  forbidden=tuple(b_facts), **(a_extra or {}))
    cb = Contract(seat_id=seat_b, holds=tuple(b_facts),
                  forbidden=tuple(a_facts), **(b_extra or {}))
    return Seat(seat_a, ca), Seat(seat_b, cb)
