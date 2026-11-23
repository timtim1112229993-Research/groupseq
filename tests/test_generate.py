"""Tests of the synthetic data generators and their true values."""
from __future__ import annotations

import numpy as np
import pytest

from groupseq import generate, inference

SPEC = {"median_length": 24, "log_sd": 0.5, "min_length": 6, "extreme_length": 144}


def test_common_rates_fall_by_a_fixed_ratio():
    rates = generate.common_rates(6, 0.7)
    assert rates.sum() == pytest.approx(1.0)
    assert np.allclose(rates[1:] / rates[:-1], 0.7)


def test_closed_form_stationary_matches_the_general_solver():
    rng = np.random.default_rng(1)
    base = generate.common_rates(13, 0.7)
    rates = generate.draw_rates(rng, base, 8, 50)
    weight = generate.mixing_weight(base, 2, 0.2)
    matrices = generate.controlled_matrices(rates, 1, 2, weight)
    assert np.allclose(matrices.sum(axis=2), 1.0)
    closed = generate.controlled_stationary(rates, 1, 2, weight)
    assert np.allclose(closed, generate.stationary(matrices), atol=1e-9)
    assert np.allclose(np.einsum("gi,gij->gj", closed, matrices), closed, atol=1e-12)


def test_null_matrices_have_identical_rows_and_zero_excess():
    base = generate.common_rates(6, 0.7)
    rates = generate.draw_rates(np.random.default_rng(2), base, 8, 20)
    matrices = generate.controlled_matrices(rates, 1, 2, 0.0)
    assert np.allclose(matrices, rates[:, None, :])
    truth = generate.controlled_truth(np.random.default_rng(2), base, 8, 1, 2, 0.0, 20000)
    assert truth["group"] == 0.0
    assert truth["pooled"] != 0.0


def test_truth_without_heterogeneity_has_a_closed_form():
    base = generate.common_rates(6, 0.7)
    truth = generate.controlled_truth(np.random.default_rng(3), base, None, 1, 2, 0.10, 1000)
    weight = 0.10 / (1 - base[2])
    share = base[1] / (1 + weight * base[1])
    want = (1 - share) * weight * (1 - base[2])
    assert truth["group"] == pytest.approx(want)
    assert truth["pooled"] == pytest.approx(want)
    assert truth["group"] < 0.10


def test_length_profiles():
    rng = np.random.default_rng(4)
    assert np.all(generate.draw_lengths(rng, "equal", 5, 8, SPEC) == 24)
    skewed = generate.draw_lengths(rng, "skewed", 4000, 8, SPEC)
    assert skewed.min() >= 6 and np.median(skewed) == pytest.approx(24, abs=1)
    extreme = generate.draw_lengths(rng, "extreme", 10, 8, SPEC)
    assert np.all(extreme[:, -1] == 144) and extreme[:, :-1].max() < 144
    with pytest.raises(ValueError):
        generate.draw_lengths(rng, "other", 1, 1, SPEC)


def test_simulated_chains_follow_their_matrix_and_padding():
    rng = np.random.default_rng(5)
    matrix = np.array([[0.1, 0.6, 0.3], [0.5, 0.25, 0.25], [0.2, 0.2, 0.6]])
    size = 400
    matrices = np.broadcast_to(matrix, (size, 3, 3)).copy()
    lengths = rng.integers(50, 300, size=size)
    chains = generate.simulate_chains(rng, matrices, generate.stationary(matrices), lengths)
    assert chains.shape == (size, lengths.max())
    for i in (0, 1, 2):
        assert np.all(chains[i, : lengths[i]] >= 0) and np.all(chains[i, lengths[i] :] == generate.PAD)
    sequences = generate.unpad(chains, lengths)
    counts = np.zeros((3, 3))
    for seq in sequences:
        np.add.at(counts, (seq[:-1], seq[1:]), 1)
    assert np.allclose(counts / counts.sum(axis=1, keepdims=True), matrix, atol=0.01)
    first = np.bincount([s[0] for s in sequences], minlength=3) / size
    assert np.allclose(first, generate.stationary(matrix[None])[0], atol=0.08)


def test_long_sequences_recover_the_true_excess():
    rng = np.random.default_rng(6)
    base = generate.common_rates(6, 0.7)
    weight = generate.mixing_weight(base, 2, 0.2)
    rates = generate.draw_rates(rng, base, 8, 600)
    lengths = np.full(600, 1500)
    chains = generate.simulate_chains(
        rng, generate.controlled_matrices(rates, 1, 2, weight), generate.controlled_stationary(rates, 1, 2, weight), lengths
    )
    counts = inference.padded_counts(chains, lengths, 1, 2)
    mean, se, _ = inference.group_excess(*counts, 2)
    truth = generate.controlled_truth(np.random.default_rng(7), base, 8, 1, 2, 0.2, 400000)
    assert abs(mean - truth["group"]) < 4 * se + 0.003
    pooled, pooled_se = inference.pooled_excess(*counts)
    assert abs(pooled - truth["pooled"]) < 4 * pooled_se + 0.003


def test_rowwise_matrices_and_calibrated_truth():
    rng = np.random.default_rng(8)
    common = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.3, 0.4]])
    same = generate.rowwise_matrices(rng, common, None, 4)
    assert np.allclose(same, common)
    drawn = generate.rowwise_matrices(rng, common, 30, 20000)
    assert np.allclose(drawn.sum(axis=2), 1.0) and np.allclose(drawn.mean(axis=0), common, atol=0.01)
    targets = [(0, 0), (0, 1)]
    exact = generate.calibrated_truth(rng, common, None, targets, 10)
    stat = generate.stationary(common[None])[0]
    assert exact[0]["group"] == pytest.approx(0.5 - stat[0]) and exact[0]["pooled"] == pytest.approx(0.5 - stat[0])
    spread = generate.calibrated_truth(rng, common, 30, targets, 60000, chunk=25000)
    assert spread[1]["group"] == pytest.approx(exact[1]["group"], abs=0.01)
