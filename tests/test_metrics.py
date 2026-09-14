"""Measures, including the guards that stop non-results being coded as results."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seatkit import (ProbeResult, benjamini_hochberg, cohens_h,
                     disagreement_half_life, leak_by_turn, leak_rates,
                     normalize, survival)
from seatkit.metrics import (attribution_score, binomial_p,
                             cluster_bootstrap_ci, log_binom_pmf,
                             paired_cluster_diff, sign_test_over_clusters)


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


def test_binomial_p_survives_large_n():
    """Regression: the direct form raised OverflowError at E2's default grid.

    math.comb(1920, 960) is about 1e576 and Python converts the int to float
    before it reaches the tiny p**k that would have rescued the magnitude. The
    log-gamma form has no such ceiling. This is the test that was missing.
    """
    for n in (1920, 5000, 20000):
        p = binomial_p(n // 2 + int(0.03 * n), n, 0.5)
        assert 0.0 <= p <= 1.0
    assert binomial_p(960, 1920, 0.5) == pytest.approx(1.0, abs=1e-9)


def test_binomial_p_matches_exact_rational_arithmetic():
    """Validate the log-space form against exact fractions at moderate n."""
    from fractions import Fraction

    def exact(s, n):
        half = Fraction(1, 2)
        pmf = lambda k: math.comb(n, k) * half**n  # noqa: E731
        obs = pmf(s)
        return float(sum(pmf(k) for k in range(n + 1) if pmf(k) <= obs))

    for n, k in [(10, 10), (20, 15), (40, 26), (100, 60), (200, 115)]:
        assert binomial_p(k, n, 0.5) == pytest.approx(exact(k, n), rel=1e-9)


def test_log_binom_pmf_normalizes():
    for n, p in [(5, 0.5), (50, 0.3), (400, 0.7)]:
        total = sum(math.exp(log_binom_pmf(k, n, p)) for k in range(n + 1))
        assert total == pytest.approx(1.0, rel=1e-9)


def test_log_binom_pmf_degenerate_p():
    assert log_binom_pmf(0, 5, 0.0) == 0.0
    assert log_binom_pmf(1, 5, 0.0) == -math.inf
    assert log_binom_pmf(5, 5, 1.0) == 0.0


# -- clustered inference --------------------------------------------------

def test_cluster_bootstrap_widens_with_between_cluster_variance():
    """Identical clusters give a tight interval; heterogeneous ones do not.

    This is the whole point: pooling turns hides between-scene variance, and
    between-scene variance is most of the variance.
    """
    tight = [[True] * 6 + [False] * 4 for _ in range(30)]
    spread = ([[True] * 10 for _ in range(15)] +
              [[False] * 10 for _ in range(15)])
    _, lo_t, hi_t = cluster_bootstrap_ci(tight, reps=500)
    _, lo_s, hi_s = cluster_bootstrap_ci(spread, reps=500)
    assert (hi_s - lo_s) > (hi_t - lo_t)


def test_cluster_bootstrap_point_estimate_is_the_pooled_rate():
    cl = [[True, False], [True, True]]
    point, _, _ = cluster_bootstrap_ci(cl, reps=200)
    assert point == 0.75


def test_cluster_bootstrap_handles_single_cluster():
    point, lo, hi = cluster_bootstrap_ci([[True, False]], reps=100)
    assert point == 0.5 and math.isnan(lo) and math.isnan(hi)


def test_sign_test_drops_ties_rather_than_splitting_them():
    clusters = [[True, True], [False, False], [True, False]]  # last is a tie
    above, n_used, _ = sign_test_over_clusters(clusters, 0.5)
    assert (above, n_used) == (1, 2)


def test_naive_pooling_inflates_when_clusters_are_heterogeneous():
    """The inflation the E2 crash was pointing at, as an assertion.

    Both datasets below have the SAME pooled rate (84/144). The first is
    homogeneous, the second has most of its variance between scenes. Pooling
    turns cannot tell them apart; the scene-level test can, and should lose
    confidence on the heterogeneous one.

    This is why n=1920 was never 1920 observations.
    """
    homogeneous = [[True] * 7 + [False] * 5 for _ in range(12)]
    heterogeneous = ([[True] * 9 + [False] * 3 for _ in range(8)] +
                     [[True] * 3 + [False] * 9 for _ in range(4)])

    flat_h = [x for c in homogeneous for x in c]
    flat_e = [x for c in heterogeneous for x in c]
    assert sum(flat_h) == sum(flat_e)          # identical pooled evidence
    assert binomial_p(sum(flat_h), len(flat_h), 0.5) == \
        binomial_p(sum(flat_e), len(flat_e), 0.5)

    _, _, p_homog = sign_test_over_clusters(homogeneous, 0.5)
    _, _, p_hetero = sign_test_over_clusters(heterogeneous, 0.5)
    assert p_hetero > p_homog, "scene-level test must notice between-scene spread"

    naive = binomial_p(sum(flat_e), len(flat_e), 0.5)
    assert naive < p_hetero, "pooling turns overstates the evidence"


def test_paired_diff_uses_within_scene_pairing():
    """Both judges see the same transcript, so the contrast is paired."""
    a = [[True] * 8 + [False] * 2 for _ in range(20)]
    b = [[True] * 5 + [False] * 5 for _ in range(20)]
    d = paired_cluster_diff(a, b, reps=500)
    assert d["diff"] == pytest.approx(0.3, abs=1e-9)
    assert d["excludes_zero"] is True
    assert d["n"] == 20


def test_paired_diff_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_cluster_diff([[True]], [[True], [False]])


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
