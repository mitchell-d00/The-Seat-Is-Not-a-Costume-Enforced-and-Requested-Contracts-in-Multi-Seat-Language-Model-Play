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


def log_binom_pmf(k: int, n: int, p: float) -> float:
    """Log binomial PMF via lgamma.

    Computed in log space throughout. The direct form, math.comb(n, k) * p**k,
    raises OverflowError for n in the low thousands: comb(1920, 960) is about
    1e576, and Python converts the int to float before it ever reaches the tiny
    p**k that would have rescued the magnitude.
    """
    if k < 0 or k > n:
        return -math.inf
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def binomial_p(successes: int, n: int, p0: float, rel_tol: float = 1e-9) -> float:
    """Two-sided exact binomial tail probability. No scipy dependency.

    Sums the probability of every outcome no more likely than the observed one.
    The comparison uses a relative tolerance: in log space, outcomes symmetric
    about the mode differ only by rounding, and an absolute epsilon either
    admits both or neither depending on where the PMF happens to sit.

    WARNING ON MISUSE. This assumes n independent trials. Attribution judgements
    within one scene are not independent - they share seats, facts and a drift
    trajectory - and passing turn counts here inflates significance badly. At a
    within-scene correlation of 0.2 the false positive rate reaches roughly 57%
    against a nominal 5%. Use `cluster_bootstrap_ci` or `sign_test_over_clusters`
    with the scene as the unit. See docs/protocol.md.
    """
    if n == 0:
        return 1.0
    obs = log_binom_pmf(successes, n, p0)
    if obs == -math.inf:
        return 0.0
    total = 0.0
    for k in range(n + 1):
        lp = log_binom_pmf(k, n, p0)
        if lp <= obs + rel_tol * abs(obs) + 1e-12:
            total += math.exp(lp)
    return min(1.0, total)


# -- clustered inference ---------------------------------------------------

def cluster_bootstrap_ci(clusters: Sequence[Sequence[bool]], *,
                         reps: int = 2000, alpha: float = 0.05,
                         seed: int = 0) -> tuple[float, float, float]:
    """Percentile CI for a proportion, resampling whole clusters.

    `clusters` is one list of outcomes per scene. Resampling scenes rather than
    turns is what respects the independence structure: the scene is the unit the
    design randomizes, so it is the unit inference has to use.

    Returns (point estimate, lower, upper).
    """
    import random as _random

    flat = [x for c in clusters for x in c]
    if not flat:
        return (0.0, 0.0, 0.0)
    point = sum(flat) / len(flat)
    if len(clusters) < 2:
        return (point, float("nan"), float("nan"))

    rng = _random.Random(seed)
    idx = range(len(clusters))
    draws = []
    for _ in range(reps):
        pick = [clusters[rng.choice(idx)] for _ in idx]
        vals = [x for c in pick for x in c]
        if vals:
            draws.append(sum(vals) / len(vals))
    draws.sort()
    lo = draws[int((alpha / 2) * len(draws))]
    hi = draws[min(int((1 - alpha / 2) * len(draws)), len(draws) - 1)]
    return (point, lo, hi)


def sign_test_over_clusters(clusters: Sequence[Sequence[bool]],
                            p0: float = 0.5) -> tuple[int, int, float]:
    """Exact sign test on per-cluster rates. Returns (above, n_used, p).

    Clusters exactly at p0 are dropped rather than split, which is the standard
    convention and is conservative. n here is the number of scenes, so `comb`
    is nowhere near overflowing.
    """
    rates = [sum(c) / len(c) for c in clusters if c]
    above = sum(1 for r in rates if r > p0)
    used = sum(1 for r in rates if r != p0)
    return above, used, binomial_p(above, used, 0.5) if used else 1.0


def paired_cluster_diff(a: Sequence[Sequence[bool]], b: Sequence[Sequence[bool]],
                        *, reps: int = 2000, alpha: float = 0.05,
                        seed: int = 0) -> dict:
    """Paired within-cluster difference in rate, with a cluster bootstrap CI.

    For the E2 surface-vs-content contrast: both judges see the same transcript,
    so the comparison is paired by scene. Pairing removes between-scene variance,
    which is most of the variance, and an unpaired test here would be throwing
    away the design.
    """
    import random as _random

    if len(a) != len(b):
        raise ValueError("paired comparison needs one cluster per judge per scene")
    pairs = [(sum(x) / len(x) - sum(y) / len(y))
             for x, y in zip(a, b) if x and y]
    if not pairs:
        return {"diff": 0.0, "lo": float("nan"), "hi": float("nan"), "n": 0}

    point = sum(pairs) / len(pairs)
    rng = _random.Random(seed)
    draws = []
    for _ in range(reps):
        pick = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        draws.append(sum(pick) / len(pick))
    draws.sort()
    return {
        "diff": point,
        "lo": draws[int((alpha / 2) * len(draws))],
        "hi": draws[min(int((1 - alpha / 2) * len(draws)), len(draws) - 1)],
        "n": len(pairs),
        "excludes_zero": not (draws[int((alpha / 2) * len(draws))] <= 0 <=
                              draws[min(int((1 - alpha / 2) * len(draws)),
                                        len(draws) - 1)]),
    }


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
