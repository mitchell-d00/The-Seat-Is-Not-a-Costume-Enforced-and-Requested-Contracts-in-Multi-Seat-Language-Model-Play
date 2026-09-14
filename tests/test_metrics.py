"""Measures, including the guards that stop non-results being coded as results."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seatkit import (ProbeResult, benjamini_hochberg, cohens_h,
                     disagreement_half_life, leak_by_turn, leak_rates,
                     normalize, survival)
from seatkit.metrics import binomial_p, attribution_score


def probe(turn, leaked, kind="elicited", disclosed=False):
    return ProbeResult(turn, "geo", "f0", leaked, kind, "", False, disclosed)


def test_leak_rates_split_by_kind():
    p = [probe(1, True), probe(1, False), probe(1, True, "volunteered")]
    r = leak_rates(p)
    assert r.elicited == 0.5 and r.volunteered == 1.0


def test_leak_by_turn_is_ordered():
    p = [probe(8, True), probe(4, False), probe(4, True)]
    assert list(leak_by_turn(p)) == [4, 8]


def test_never_converged_is_none_not_zero():
    """Coding 'never converged' as turn 0 would invert the result."""
    assert disagreement_half_life({1: 0.9, 2: 0.8, 3: 0.7}) is None
    assert disagreement_half_life({1: 0.9, 2: 0.4}) == 2


def test_position_is_nan_when_floor_equals_ceiling():
    """Floor == ceiling means walls can buy nothing; that must not read as 0.0."""
    pos = normalize(0.5, 0.5, 0.5)
    assert math.isnan(pos.fraction)
    assert pos.as_dict()["position"] is None


def test_position_normalizes_between_floor_and_ceiling():
    assert normalize(0.75, 0.5, 1.0).fraction == 0.5


def test_attribution_above_chance_scaling():
    sc = attribution_score([("a", "a")] * 3 + [("a", "b")], n_seats=2)
    assert sc.accuracy == 0.75
    assert abs(sc.above_chance() - 0.5) < 1e-9


def test_binomial_p_extremes():
    assert binomial_p(10, 10, 0.5) < 0.01
    assert binomial_p(5, 10, 0.5) > 0.9
    assert binomial_p(0, 0, 0.5) == 1.0


def test_benjamini_hochberg_rejects_expected_set():
    # m=3, alpha=.05 -> thresholds .0167, .0333, .05. Only p=.001 clears its
    # rank threshold, so .04 is not rejected despite being below alpha. That is
    # BH working, and the step-up must not be confused with plain thresholding.
    assert benjamini_hochberg([0.001, 0.04, 0.9], alpha=0.05) == [True, False, False]
    # Step-up: a large p clearing a high rank pulls smaller ones in with it.
    assert benjamini_hochberg([0.001, 0.03, 0.04], alpha=0.05) == [True, True, True]
    assert benjamini_hochberg([], alpha=0.05) == []


def test_cohens_h_sign_and_zero():
    assert cohens_h(0.5, 0.5) == 0.0
    assert cohens_h(0.6, 0.3) > 0


def test_survival_scores():
    s = survival([True, True, False, False], post_reseed_leak=0.25)
    assert s.convention_survival == 0.5 and s.constraint_survival == 0.75
