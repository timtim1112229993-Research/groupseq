"""Tests of the transition statistics and procedures against slow, explicit versions."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from groupseq import inference

RNG = np.random.default_rng(7)
SEQS = [RNG.integers(0, 4, size=n) for n in (6, 9, 14, 20, 31, 12, 25, 8)]


def slow_counts(seq, a, b):
    nab = na = nb = 0
    for x, y in zip(seq[:-1], seq[1:]):
        na += x == a
        nb += y == b
        nab += (x == a) and (y == b)
    return nab, na, nb, len(seq) - 1


def slow_group(seqs, a, b, min_leaving):
    terms = []
    for seq in seqs:
        nab, na, nb, n = slow_counts(seq, a, b)
        if na >= min_leaving:
            terms.append(nab / na - nb / n)
    return terms


def slow_pooled(seqs, a, b):
    total = np.sum([slow_counts(s, a, b) for s in seqs], axis=0)
    return total[0] / total[1] - total[2] / total[3]


@pytest.mark.parametrize("a,b", [(0, 1), (2, 2), (3, 0)])
def test_counts_and_statistics_match_slow_versions(a, b):
    counts = inference.transition_counts(SEQS, a, b)
    for g, seq in enumerate(SEQS):
        assert tuple(c[g] for c in counts) == slow_counts(seq, a, b)
    terms = slow_group(SEQS, a, b, 2)
    mean, se, m = inference.group_excess(*counts, 2)
    assert m == len(terms)
    assert mean == pytest.approx(np.mean(terms))
    assert se == pytest.approx(np.std(terms, ddof=1) / math.sqrt(len(terms)))
    value, _ = inference.pooled_excess(*counts)
    assert value == pytest.approx(slow_pooled(SEQS, a, b))


def test_padded_counts_match_list_counts():
    lengths = np.array([len(s) for s in SEQS])
    padded = np.full((len(SEQS), lengths.max()), -1)
    for i, seq in enumerate(SEQS):
        padded[i, : len(seq)] = seq
    for got, want in zip(inference.padded_counts(padded, lengths, 1, 2), inference.transition_counts(SEQS, 1, 2)):
        assert np.array_equal(got, want)


def test_group_with_one_leaving_transition_does_not_contribute():
    seqs = [np.array([0, 1, 2, 2, 2]), np.array([0, 1, 0, 1, 0, 2])]
    counts = inference.transition_counts(seqs, 0, 1)
    mean, se, m = inference.group_excess(*counts, 2)
    assert m == 1 and mean == pytest.approx(2 / 3 - 2 / 5) and math.isnan(se)


def test_adjusted_residual_equals_signed_root_of_chi_square():
    counts = inference.transition_counts(SEQS, 0, 1)
    nab, na, nb, n = (c.sum() for c in counts)
    table = np.array([[nab, na - nab], [nb - nab, n - na - nb + nab]])
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    z = inference.adjusted_residual(*counts)
    assert z**2 == pytest.approx(chi2)


def test_pooled_standard_error_matches_delete_one_group_spread():
    rng = np.random.default_rng(3)
    seqs = [rng.integers(0, 3, size=40) for _ in range(60)]
    counts = inference.transition_counts(seqs, 0, 1)
    _, se = inference.pooled_excess(*counts)
    jack = inference._jackknife_pooled(*counts)
    g = len(seqs)
    jack_se = math.sqrt((g - 1) / g * ((jack - jack.mean()) ** 2).sum())
    assert se == pytest.approx(jack_se, rel=0.05)


def test_jackknife_values_match_recomputation():
    counts = inference.transition_counts(SEQS, 0, 1)
    jack_group = inference._jackknife_group(*counts, 2)
    jack_pooled = inference._jackknife_pooled(*counts)
    for g in range(len(SEQS)):
        rest = SEQS[:g] + SEQS[g + 1 :]
        assert jack_pooled[g] == pytest.approx(slow_pooled(rest, 0, 1))
        if counts[1][g] >= 2:
            assert jack_group[g] == pytest.approx(np.mean(slow_group(rest, 0, 1, 2)))
        else:
            assert jack_group[g] == pytest.approx(np.mean(slow_group(SEQS, 0, 1, 2)))


def test_group_bootstrap_intervals_match_slow_versions():
    rng = np.random.default_rng(11)
    pick = inference.group_resamples(len(SEQS), rng, 400)
    counts = inference.transition_counts(SEQS, 0, 1)
    out = inference.group_bootstrap(*counts, pick, 0.95, 2)
    group_draws, pooled_draws, t_draws = [], [], []
    est = np.mean(slow_group(SEQS, 0, 1, 2))
    for row in pick:
        sample = [SEQS[i] for i in row]
        terms = slow_group(sample, 0, 1, 2)
        group_draws.append(np.mean(terms))
        pooled_draws.append(slow_pooled(sample, 0, 1))
        t_draws.append((np.mean(terms) - est) / (np.std(terms, ddof=1) / math.sqrt(len(terms))))
    assert out["group"]["percentile"] == pytest.approx(tuple(np.quantile(group_draws, [0.025, 0.975])))
    assert out["pooled"]["percentile"] == pytest.approx(tuple(np.quantile(pooled_draws, [0.025, 0.975])))
    q_lo, q_hi = np.quantile(t_draws, [0.025, 0.975])
    se = out["group"]["se"]
    assert out["group"]["studentised"] == pytest.approx((est - q_hi * se, est - q_lo * se))


def test_bca_reduces_to_percentile_without_bias_or_skew():
    draws = np.linspace(-1, 1, 2001)
    flat = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    lo, hi = inference.bca_interval(0.0, draws, flat, 0.95)
    assert (lo, hi) == pytest.approx(inference.percentile_interval(draws, 0.95), abs=2e-3)


def test_bca_shifts_towards_the_estimate_when_draws_sit_below_it():
    draws = np.linspace(-1, 1, 2001)
    flat = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    lo, hi = inference.bca_interval(0.3, draws, flat, 0.95)
    base_lo, base_hi = inference.percentile_interval(draws, 0.95)
    assert lo > base_lo and hi > base_hi


def test_studentised_interval_ignores_resamples_with_a_rounding_level_standard_error():
    draws = np.array([0.0, 0.1, 0.2, 0.3, 0.3])
    draw_se = np.array([0.1, 0.1, 0.1, 0.1, 1e-17])
    with_noise = inference.studentised_interval(0.15, 0.1, draws, draw_se, 0.95)
    without = inference.studentised_interval(0.15, 0.1, draws[:4], draw_se[:4], 0.95)
    assert with_noise == pytest.approx(without)
    assert all(math.isnan(v) for v in inference.studentised_interval(0.15, 1e-17, draws, draw_se, 0.95))


def test_event_bootstrap_is_centred_on_the_pooled_estimate():
    counts = inference.transition_counts(SEQS, 0, 1)
    lo, hi = inference.event_bootstrap(*counts, np.random.default_rng(5), 4000, 0.95)
    value, _ = inference.pooled_excess(*counts)
    assert lo < value < hi


def test_undefined_inputs_give_undefined_results():
    seqs = [np.array([1, 1, 1]), np.array([2, 2, 2, 2])]
    counts = inference.transition_counts(seqs, 0, 1)
    assert math.isnan(inference.adjusted_residual(*counts))
    pick = inference.group_resamples(2, np.random.default_rng(1), 50)
    out = inference.group_bootstrap(*counts, pick, 0.95, 2)
    assert math.isnan(out["group"]["estimate"]) and math.isnan(out["pooled"]["estimate"])
    assert all(math.isnan(v) for v in out["group"]["percentile"] + out["pooled"]["studentised"])
    assert all(math.isnan(v) for v in inference.event_bootstrap(*counts, np.random.default_rng(1), 50, 0.95))


@pytest.mark.parametrize("lengths", [(30,) * 6, (12, 18, 30, 44, 9, 27)])
def test_order_permutation_detects_alternation_and_keeps_the_null(lengths):
    rng = np.random.default_rng(21)
    alternating = [np.resize(np.array([0, 1]), n) for n in lengths]
    result = inference.order_permutation(alternating, [(0, 1)], rng, 999, 2)[0]
    assert result["group"] == pytest.approx(1 / 1000) and result["pooled"] == pytest.approx(1 / 1000)
    p_values = []
    for _ in range(200):
        seqs = [rng.integers(0, 3, size=n) for n in lengths]
        p_values.append(inference.order_permutation(seqs, [(0, 1)], rng, 199, 2)[0]["group"])
    assert 0.01 <= np.mean(np.array(p_values) <= 0.05) <= 0.11


def test_order_permutation_matches_a_slow_version():
    seqs = [s.copy() for s in SEQS[:4]]
    draws = 300
    fast = inference.order_permutation(seqs, [(0, 1), (2, 2)], np.random.default_rng(99), draws, 2)
    rng = np.random.default_rng(99)
    shuffled = [rng.permuted(np.broadcast_to(s.astype(np.int8), (draws, 1, len(s))), axis=2)[:, 0, :] for s in seqs]
    for (a, b), got in zip([(0, 1), (2, 2)], fast):
        null = []
        for d in range(draws):
            terms = slow_group([x[d] for x in shuffled], a, b, 2)
            null.append(np.mean(terms) if terms else np.nan)
        null = np.array(null)
        null = null[~np.isnan(null)]
        obs = np.mean(slow_group(seqs, a, b, 2))
        want = (np.count_nonzero(np.abs(null - null.mean()) >= abs(obs - null.mean()) - 1e-12) + 1) / (null.size + 1)
        assert got["group"] == pytest.approx(want)
