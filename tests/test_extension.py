"""Tests of the corrected estimates, the opposing-dependence generator and the extension runner."""
from __future__ import annotations

import copy
import itertools

import numpy as np
import pytest

from groupseq import extension, generate, inference, study


def pad(sequences: list) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.array([len(s) for s in sequences])
    out = np.full((len(sequences), lengths.max()), generate.PAD, dtype=np.int8)
    for i, seq in enumerate(sequences):
        out[i, : len(seq)] = seq
    return out, lengths


def slow_loo(seq, a, b):
    seq = list(seq)
    leaving = [(x, y) for x, y in zip(seq[:-1], seq[1:]) if x == a]
    if not leaving:
        return None, 0
    share = sum(y == b for _, y in leaving) / len(leaving)
    return share - (seq.count(b) - (a == b)) / (len(seq) - 1), len(leaving)


@pytest.mark.parametrize("a,b", [(0, 1), (0, 0), (2, 1)])
def test_leave_one_out_term_has_zero_mean_over_every_ordering(a, b):
    """Exhaustive check: the term averages to exactly zero over all orderings in which it is defined."""
    for multiset in ([0, 0, 1, 1, 2, 2, 2], [0, 0, 0, 1, 2], [0, 1, 1, 1, 2, 2]):
        orderings = sorted(set(itertools.permutations(multiset)))
        padded, lengths = pad([np.array(o) for o in orderings])
        terms, na = extension.loo_terms(padded, lengths, a, b)
        for leaving in np.unique(na[na > 0]):
            assert terms[na == leaving].mean() == pytest.approx(0.0, abs=1e-12)
        raw = inference.group_terms(*inference.padded_counts(padded, lengths, a, b), 1)[0][na > 0]
        assert abs(raw.mean()) > 1e-3


def test_terms_match_slow_versions_and_contribution_rules():
    rng = np.random.default_rng(1)
    sequences = [rng.integers(0, 3, size=n) for n in (6, 7, 12, 19, 30, 9, 15, 24)]
    padded, lengths = pad(sequences)
    out = extension.corrected_terms(padded, lengths, 0, 1, 2)
    loo, keep = out["group_loo"]
    split, both = out["group_split"]
    for g, seq in enumerate(sequences):
        full, na = slow_loo(seq, 0, 1)
        assert keep[g] == (na >= 2)
        if keep[g]:
            assert loo[g] == pytest.approx(full)
        half = len(seq) // 2
        first, na1 = slow_loo(seq[:half], 0, 1)
        second, na2 = slow_loo(seq[half:], 0, 1)
        assert both[g] == (na >= 2 and na1 >= 1 and na2 >= 1)
        if both[g]:
            assert split[g] == pytest.approx(2 * full - 0.5 * (first + second))
        else:
            assert split[g] == 0.0


def test_second_half_returns_the_tail_of_every_sequence():
    padded, lengths = pad([np.array([0, 1, 2, 0, 1]), np.array([2, 2, 1])])
    rest, rest_lengths = extension.second_half(padded, lengths, lengths // 2)
    assert rest_lengths.tolist() == [3, 2]
    assert rest[0, :3].tolist() == [2, 0, 1] and rest[1, :2].tolist() == [2, 1]


def test_term_bootstrap_reproduces_the_group_bootstrap_on_uncorrected_terms():
    rng = np.random.default_rng(2)
    sequences = [rng.integers(0, 4, size=n) for n in (8, 14, 20, 31, 12, 25, 9, 17, 22)]
    counts = inference.transition_counts(sequences, 0, 1)
    pick = inference.group_resamples(len(sequences), rng, 300)
    want = inference.group_bootstrap(*counts, pick, 0.95, 2)["group"]
    terms, mask = inference.group_terms(*counts, 2)
    got = extension.term_bootstrap(terms, mask, pick, 0.95)
    assert got["estimate"] == pytest.approx(want["estimate"]) and got["contributing"] == want["contributing"]
    for key in ("percentile", "bca", "studentised"):
        assert got[key] == pytest.approx(want[key])


def test_opposing_effects_give_each_group_its_stated_excess_and_average_to_zero():
    rng = np.random.default_rng(3)
    base = generate.common_rates(6, 0.7)
    rates = generate.draw_rates(rng, base, 8, 200000)
    weight, excess = extension.opposing_effects(rng, rates, 1, 2, 0.10)
    matrices = extension.weighted_matrices(rates[:500], 1, 2, weight[:500])
    assert np.allclose(matrices.sum(axis=2), 1.0) and matrices.min() >= 0.0 and matrices.max() <= 1.0 + 1e-12
    stat = extension.weighted_stationary(rates, 1, 2, weight)
    assert np.allclose(stat[:500], generate.stationary(matrices), atol=1e-9)
    true_excess = (1 - weight) * rates[:, 2] + weight - stat[:, 2]
    assert np.allclose(true_excess, excess, atol=1e-12)
    assert np.abs(excess).max() <= 0.10 + 1e-12
    assert np.mean(np.abs(excess) > 0.0999) > 0.4
    same = generate.draw_rates(rng, base, None, 1000)
    assert np.allclose(np.abs(extension.opposing_effects(rng, same, 1, 2, 0.10)[1]), 0.10)
    assert abs(excess.mean()) < 4 * excess.std() / np.sqrt(len(excess))
    assert abs(np.mean(excess > 0) - 0.5) < 0.01
    truth = extension.opposing_truth(np.random.default_rng(4), base, 8, 1, 2, 0.10, 50000)
    assert truth["group"] == 0.0 and 0.05 < truth["mean_size"] <= 0.10


def bias_under_persistence(length: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    matrix = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
    size = 20000
    matrices = np.broadcast_to(matrix, (size, 3, 3)).copy()
    lengths = np.full(size, length)
    chains = generate.simulate_chains(rng, matrices, generate.stationary(matrices), lengths)
    truth = 0.6 - 1 / 3
    raw_terms, raw_mask = inference.group_terms(*inference.padded_counts(chains, lengths, 0, 0), 2)
    out = extension.corrected_terms(chains, lengths, 0, 0, 2)
    loo = out["group_loo"][0][out["group_loo"][1]].mean() - truth
    split = out["group_split"][0][out["group_split"][1]].mean() - truth
    return raw_terms[raw_mask].mean() - truth, loo, split


def test_leave_one_out_base_rate_roughly_halves_the_bias_under_persistence():
    raw, loo, _ = bias_under_persistence(22, 5)
    assert raw < -0.05
    assert raw < loo < 0 and abs(loo) < 0.6 * abs(raw)


def test_split_sequence_estimate_works_for_long_sequences_only():
    """Recorded behaviour: the jackknife removes the bias at 80 events and does not at 22."""
    _, loo_long, split_long = bias_under_persistence(80, 6)
    assert abs(split_long) < 0.004 < abs(loo_long)
    _, loo_short, split_short = bias_under_persistence(22, 5)
    assert abs(split_short) > 0.5 * abs(loo_short)


def test_grid_and_streams_of_the_extension(pipeline_config):
    cells = extension.all_cells(pipeline_config["simulation"])
    by_arm = {arm: sum(1 for c in cells if c["arm"] == arm) for arm in extension.ARMS}
    assert by_arm == {"opposing": 90, "controlled_new": 270, "calibrated_new": 3}
    ext = pipeline_config["extension"]
    assert ext["seed"] != pipeline_config["seed"]
    assert (ext["replications"], ext["resamples"], ext["permutations"]) == (2000, 1999, 4999)
    new = [g.random() for g in extension.cell_streams(ext["seed"], cells[0], 4)]
    old = [g.random() for g in study.cell_streams(pipeline_config["seed"], {"arm": "controlled", "index": 0}, 4)]
    assert not set(new) & set(old)


def test_cells_are_reproducible_and_carry_every_statistic(pipeline_config):
    ext = copy.deepcopy(pipeline_config["extension"])
    ext.update({"replications": 5, "resamples": 99, "permutations": 99, "batch": 3, "truth_groups": 2000, "truth_datasets": 100})
    calibration = {
        "families": pipeline_config["families"]["order"],
        "matrix": (np.full((6, 6), 0.1) + np.eye(6) * 0.4).tolist(),
        "lengths": [12, 20, 30, 16, 25, 18, 22, 40],
    }
    cells = extension.all_cells(pipeline_config["simulation"])
    picked = [cells[7], cells[90 + 40], cells[-2]]
    tasks = [
        {"cell": c, "extension": ext, "simulation": pipeline_config["simulation"], "analysis": pipeline_config["analysis"],
         "calibration": calibration}
        for c in picked
    ]
    first = [extension.run_cell(t) for t in tasks]
    second = [extension.run_cell(t) for t in reversed(tasks)][::-1]
    for x, y in zip(first, second):
        x.pop("seconds"), y.pop("seconds")
        assert repr(x) == repr(y)
    frames = extension.assemble(first)
    procedures = frames["extension_procedures"]
    opposing = procedures[procedures["arm"] == "opposing"]
    assert len(opposing) == 4 * 3 + 1 + 1 + 2
    assert set(opposing["statistic"]) == {"group", "pooled", "group_loo", "group_split"}
    other = procedures[procedures["arm"] == "controlled_new"]
    assert len(other) == 12 and set(other["procedure"]) == {"T3_percentile", "T4_bca", "T5_studentised"}
    assert len(procedures[procedures["arm"] == "calibrated_new"]) == 8 * 12
    assert len(frames["extension_estimates"]) == 4 + 4 + 8 * 4
