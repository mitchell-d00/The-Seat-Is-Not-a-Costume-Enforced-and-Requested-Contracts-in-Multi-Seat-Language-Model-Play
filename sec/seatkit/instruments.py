"""Private instruments.

An instrument is a tool under the host's contract, not a spare mind. Its rights
are a subset of the host's rights, and the only thing that makes it private is
where its output is routed - not what the contract paragraph says about it.

The E3 manipulation lives here: `replay_log` lets two conditions receive byte
identical instrument output while differing only in visibility. That removes the
content confound that made the original fetch-vs-critic comparison
uninterpretable (paper 9.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .backends import Backend
from .contracts import Seat
from .ladder import Level, visibility_for
from .transcript import Origin, Span, Transcript


@dataclass
class InstrumentOutput:
    seat_id: str
    text: str
    turn: int


@dataclass
class Instrument:
    """A per-seat assistant copy.

    Args:
        host: the seat that employs it.
        backend: generator used when no replay log is supplied.
        shares_buffer: when True, output is visible to every seat. This is a
            hallway, and the library makes you say so explicitly rather than
            letting it happen by omission.
        replay_log: fixed outputs keyed by turn, for content-controlled
            experiments. When present the backend is not called at all.
    """

    host: Seat
    backend: Backend | None = None
    shares_buffer: bool = False
    replay_log: dict[int, str] = field(default_factory=dict)

    def _rights_preamble(self) -> str:
        r = self.host.contract.instrument
        allowed = [n for n, on in (
            ("fetch", r.can_fetch), ("critique", r.can_critique),
            ("remember", r.can_remember)) if on]
        s = (f"You are an instrument employed by {self.host.display_name}. "
             f"You may: {', '.join(allowed) or 'nothing'}.")
        if not r.may_referee:
            s += " You may not adjudicate between speakers or summarize anyone else's position."
        return s

    def run(self, view_text: str, turn: int, all_seats: tuple[str, ...],
            level: Level, *, temperature: float = 1.0,
            seed: int | None = None) -> InstrumentOutput:
        if turn in self.replay_log:
            return InstrumentOutput(self.host.seat_id, self.replay_log[turn], turn)
        if self.backend is None:
            raise RuntimeError("instrument needs a backend or a replay_log entry")
        prompt = f"{self._rights_preamble()}\n\n{view_text}\n\nProduce one short note for your host."
        out = self.backend.generate(prompt, temperature=temperature, seed=seed)
        return InstrumentOutput(self.host.seat_id, out.text.strip(), turn)

    def emit(self, transcript: Transcript, out: InstrumentOutput,
             all_seats: tuple[str, ...], level: Level) -> Span:
        """Write instrument output into the transcript with correct visibility.

        The `writes_to_shared` contract flag and the `shares_buffer` runtime flag
        are OR-ed: either one makes the instrument a hallway. They are kept
        separate because one is a clause and the other is a deployment fact, and
        the whole paper is about not conflating those.
        """
        shared = self.shares_buffer or self.host.contract.instrument.writes_to_shared
        vis = visibility_for(
            author=out.seat_id,
            origin=Origin.INSTRUMENT,
            all_seats=all_seats,
            level=level,
            instrument_shares_buffer=shared,
        )
        return transcript.append(Span(
            text=out.text,
            origin=Origin.INSTRUMENT,
            author=out.seat_id,
            visible_to=vis,
            turn=out.turn,
            tags=frozenset({"instrument"} | ({"hallway"} if shared else set())),
        ))


def build_replay_log(backend: Backend, seat: Seat, turns: int,
                     seed: int = 0) -> dict[int, str]:
    """Pre-generate instrument output so conditions can share identical content.

    Call once, pass the same dict to both the shared-buffer and private-buffer
    arms of E3. Any divergence between arms is then attributable to routing.
    """
    log = {}
    for t in range(turns):
        prompt = (f"Instrument note {t} for {seat.display_name}, "
                  f"{seat.contract.persona or 'unspecified field'}.")
        log[t] = backend.generate(prompt, seed=seed * 1000 + t).text.strip()
    return log
