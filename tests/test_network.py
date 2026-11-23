"""Tests of the co-occurrence vectors and the permutation tests of a two-arm contrast."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from groupseq import network

SPEC = {"nodes": 8, "baseline": [0.70, 0.15], "segments": [3, 7], "shift": 1.0, "shift_nodes": 3}


def test_segment_vectors_score_a_pair_when_both_nodes_occur():
    presence = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]], dtype=bool)
    vectors = network.segment_vectors(presence)
    assert vectors.shape == (3, 6)
    assert vectors[0].tolist() == [1, 0, 0, 0, 0, 0]
    assert vectors[1].tolist() == [1, 1, 0, 1, 0, 0]
    assert vectors[2].sum() == 0
    assert network.segment_vectors(np.ones((1, 8), dtype=bool)).shape == (1, 28)


def test_group_vectors_are_means_scaled_to_unit_length_and_zero_stays_zero():
    vectors = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    owner = np.array([0, 0, 1, 1])
    raw = network.group_vectors(vectors, owner, 2, unit=False)
    assert raw.tolist() == [[1.0, 0.5], [0.0, 0.0]]
    unit = network.group_vectors(vectors, owner, 2, unit=True)
    assert np.linalg.norm(unit[0]) == pytest.approx(1.0)
    assert unit[1].tolist() == [0.0, 0.0]


def test_arms_split():
    assert network.arms(23).sum() == 12 and (network.arms(23) == 0).sum() == 11
    assert network.arms(8).sum() == 4


def test_statistics_match_slow_versions():
    rng = np.random.default_rng(1)
    labels = network.arms(13)
    vectors = rng.random((13, 5))
    vectors[:, 4] = 0.0
    permuted = network.label_permutations(labels, rng, 300)
    assert np.all(permuted.sum(axis=1) == labels.sum())
    first, second = vectors[labels == 1], vectors[labels == 0]
    squared = network.squared_difference_test(labels, vectors, permuted)
    assert squared["statistic"] == pytest.approx(((first.mean(axis=0) - second.mean(axis=0)) ** 2).sum())
    null = [((vectors[row == 1].mean(axis=0) - vectors[row == 0].mean(axis=0)) ** 2).sum() for row in permuted]
    want = (np.count_nonzero(np.array(null) >= squared["statistic"] - 1e-12) + 1) / 301
    assert squared["p"] == pytest.approx(want)
    student = network.studentised_test(labels, vectors, permuted)
    welch = stats.ttest_ind(first[:, :4], second[:, :4], equal_var=False).statistic
    assert np.allclose(student["pair_t"][:4], np.abs(welch))
    assert student["pair_t"][4] == 0.0
    assert student["p"] == pytest.approx(student["pair_p"].min())
    assert np.all(student["pair_p"] >= student["p"] - 1e-12)


def test_simulated_presence_follows_the_stated_rates():
    rng = np.random.default_rng(2)
    labels = network.arms(4000)
    presence, owner = network.simulate_presence(rng, 4000, labels, SPEC, 0.0, False)
    counts = np.bincount(owner)
    assert counts.min() == 3 and counts.max() == 7
    assert np.allclose(presence.mean(axis=0), np.linspace(0.70, 0.15, 8), atol=0.012)
    shifted, owner = network.simulate_presence(rng, 4000, labels, SPEC, 0.0, True)
    second = shifted[labels[owner] == 1].mean(axis=0)
    first = shifted[labels[owner] == 0].mean(axis=0)
    base = np.linspace(0.70, 0.15, 8)
    lifted = 1 / (1 + np.exp(-(np.log(base / (1 - base)) + 1.0)))
    assert np.allclose(second[:3], lifted[:3], atol=0.02)
    assert np.allclose(second[3:], base[3:], atol=0.02) and np.allclose(first, base, atol=0.02)


def test_segment_level_test_is_liberal_when_groups_differ_and_group_level_is_not():
    rng = np.random.default_rng(3)
    labels = network.arms(12)
    hits = {"segment": 0, "group": 0, "student": 0}
    runs = 300
    for _ in range(runs):
        presence, owner = network.simulate_presence(rng, 12, labels, SPEC, 1.0, False)
        segments = network.segment_vectors(presence)
        seg_labels = labels[owner]
        p1 = network.squared_difference_test(seg_labels, segments, network.label_permutations(seg_labels, rng, 199))["p"]
        vectors = network.group_vectors(segments, owner, 12)
        permuted = network.label_permutations(labels, rng, 199)
        hits["segment"] += p1 <= 0.05
        hits["group"] += network.squared_difference_test(labels, vectors, permuted)["p"] <= 0.05
        hits["student"] += network.studentised_test(labels, vectors, permuted)["p"] <= 0.05
    assert hits["segment"] / runs > 0.12
    assert hits["group"] / runs < 0.10 and hits["student"] / runs < 0.10
