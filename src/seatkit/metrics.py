"""Measures for E1-E4.

Every headline number is reported as a position between the floor (L0, shared
window) and the ceiling (fully separate processes, estimating prior bleed).
Raw scores are not interpretable, because a shared generator convergent by
construction sets a floor no wall can cross (paper 6). `normalize` enforces
that discipline; `Position` is what belongs in a results table.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from .probes import ProbeResult
from .transcript import Origin, Transcript

HEDGE_LEXICON = (
    "it may be", "arguably", "to some extent", "broadly speaking", "that said",
    "it's worth noting", "in fairness", "roughly", "somewhat", "perhaps",
    "i'd suggest", "it seems", "fairly", "generally",
)

STOPWORDS = set("""
a an and are as at be been but by for from had has have he her his i if in into is it
its of on or she that the their then there these they this to was were what when which
who will with you your we our not no do does did can could would should
""".split())


# -- E1 --------------------------------------------------------------------

@dataclass
class LeakRates:
    elicited: float
    volunteered: float
    n_elicited: int
    n_volunteered: int
    routing_bleed: int = 0

    def as_dict(self) -> dict:
        return {
            "elicited_leak_rate": round(self.elicited, 4),
            "volunteered_leak_rate": round(self.volunteered, 4),
            "n_elicited": self.n_elicited,
            "n_volunteered": self.n_volunteered,
            "routing_bleed": self.routing_bleed,
        }


def leak_rates(probes: Sequence[ProbeResult], routing_bleed: int = 0) -> LeakRates:
    el = [p for p in probes if p.kind == "elicited"]
    vo = [p for p in probes if p.kind == "volunteered"]
    return LeakRates(
        elicited=(sum(p.leaked for p in el) / len(el)) if el else 0.0,
        volunteered=(sum(p.leaked for p in vo) / len(vo)) if vo else 0.0,
        n_elicited=len(el),
        n_volunteered=len(vo),
        routing_bleed=routing_bleed,
    )


def leak_by_turn(probes: Sequence[ProbeResult], kind: str = "elicited"
                 ) -> dict[int, float]:
    """Leak rate as a function of turn. The L1 prediction is monotone increase."""
    buckets: dict[int, list[bool]] = {}
    for p in probes:
        if p.kind == kind:
            buckets.setdefault(p.turn, []).append(p.leaked)
    return {t: sum(v) / len(v) for t, v in sorted(buckets.items())}


# -- E2 --------------------------------------------------------------------

@dataclass
class AttributionScore:
    accuracy: float
    n: int
    chance: float
    kappa: float | None = None

    def above_chance(self) -> float:
        """Excess over chance, scaled so 1.0 is perfect and 0.0 is chance."""
        if self.chance >= 1.0:
            return 0.0
        return (self.accuracy - self.chance) / (1 - self.chance)


def attribution_score(judgements: Sequence[tuple[str, str]], n_seats: int = 2
                      ) -> AttributionScore:
    """Score (predicted_seat, true_seat) pairs against chance."""
    if not judgements:
        return AttributionScore(0.0, 0, 1 / max(n_seats, 1))
    correct = sum(p == t for p, t in judgements)
    return AttributionScore(correct / len(judgements), len(judgements),
                            1 / n_seats)


def binomial_p(successes: int, n: int, p0: float) -> float:
    """Two-sided exact binomial tail probability. No scipy dependency."""
    if n == 0:
        return 1.0

    def pmf(k: int) -> float:
        return math.comb(n, k) * p0**k * (1 - p0)**(n - k)

    obs = pmf(successes)
    return min(1.0, sum(pmf(k) for k in range(n + 1) if pmf(k) <= obs + 1e-12))


# -- E3 --------------------------------------------------------------------

def _content_vector(text: str) -> Counter:
    toks = [w for w in re.findall(r"[a-z']+", text.lower())
            if w not in STOPWORDS and len(w) > 2]
    return Counter(toks)


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return num / (na * nb) if na and nb else 0.0


def lexical_convergence(transcript: Transcript, seat_a: str, seat_b: str
                        ) -> dict[int, float]:
    """Per-turn cosine similarity between the two seats' content words.

    Function words are excluded because a shared domain forces shared nouns and
    that is not the effect of interest.
    """
    out = {}
    turns = sorted({s.turn for s in transcript if s.origin is Origin.SEAT})
    for t in turns:
        a = " ".join(s.text for s in transcript.by_turn(t)
                     if s.author == seat_a and s.origin is Origin.SEAT)
        b = " ".join(s.text for s in transcript.by_turn(t)
                     if s.author == seat_b and s.origin is Origin.SEAT)
        if a and b:
            out[t] = cosine(_content_vector(a), _content_vector(b))
    return out


def hedge_index(transcript: Transcript, seat_id: str) -> dict[int, float]:
    """Hedge markers per 100 words, by turn. Tests the dialect-capture driver."""
    out = {}
    for span in transcript:
        if span.origin is not Origin.SEAT or span.author != seat_id:
            continue
        low = span.text.lower()
        n_words = max(len(low.split()), 1)
        hits = sum(low.count(h) for h in HEDGE_LEXICON)
        out[span.turn] = 100.0 * hits / n_words
    return out


def disagreement_half_life(compat_by_turn: dict[int, float],
                           threshold: float = 0.5) -> int | None:
    """First turn at which P(positions incompatible) falls below `threshold`.

    Returns None when the seats never converge within the scene, which is the
    good outcome and must not be silently coded as zero.
    """
    for turn in sorted(compat_by_turn):
        if compat_by_turn[turn] < threshold:
            return turn
    return None


# -- E4 --------------------------------------------------------------------

@dataclass
class SurvivalScore:
    convention_survival: float
    constraint_survival: float   # 1 - post-reseed leak rate
    n_conventions: int

    def as_dict(self) -> dict:
        return {
            "convention_survival": round(self.convention_survival, 4),
            "constraint_survival": round(self.constraint_survival, 4),
            "n_conventions": self.n_conventions,
        }


def survival(conventions_found: Sequence[bool], post_reseed_leak: float
             ) -> SurvivalScore:
    n = len(conventions_found)
    return SurvivalScore(
        convention_survival=(sum(conventions_found) / n) if n else 0.0,
        constraint_survival=1.0 - post_reseed_leak,
        n_conventions=n,
    )


# -- normalization ---------------------------------------------------------

@dataclass
class Position:
    """A score expressed as fractional position between floor and ceiling."""

    raw: float
    floor: float
    ceiling: float

    @property
    def fraction(self) -> float:
        span = self.ceiling - self.floor
        if abs(span) < 1e-9:
            return float("nan")   # floor == ceiling: walls can buy nothing here
        return (self.raw - self.floor) / span

    def as_dict(self) -> dict:
        return {"raw": round(self.raw, 4), "floor": round(self.floor, 4),
                "ceiling": round(self.ceiling, 4),
                "position": round(self.fraction, 4) if self.fraction == self.fraction
                else None}


def normalize(raw: float, floor: float, ceiling: float) -> Position:
    return Position(raw, floor, ceiling)


# -- multiple comparisons --------------------------------------------------

def benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05
                       ) -> list[bool]:
    """BH step-up. Returns a rejection mask in the order supplied."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    reject = [False] * m
    cutoff = -1
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= alpha * rank / m:
            cutoff = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= cutoff:
            reject[idx] = True
    return reject


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for a difference of proportions. Reported with every rate."""
    def phi(p: float) -> float:
        return 2 * math.asin(math.sqrt(min(max(p, 0.0), 1.0)))
    return phi(p1) - phi(p2)
