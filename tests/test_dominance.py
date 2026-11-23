"""Tests of the dominant-group generator, the t interval and the runner of the second extension."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from groupseq import application, dominance, dominance_summary, extension, generate, inference, study

SPEC = {"median_length": 24, "log_sd": 0.5, "min_length": 6, "extreme_length": 144}


@pytest.mark.parametrize("df", [1, 2, 3, 7, 11, 22, 39, 79, 400])
def test_t_quantile_matches_the_reference_library(df):
    for prob in (0.95, 0.975, 0.995):
        assert dominance.t_quantile(df, prob) == pytest.approx(stats.t.ppf(prob, df), rel=1e-9)


def test_t_interval_is_the_one_sample_interval_and_is_undefined_without_spread():
    values = np.array([0.10, -0.02, 0.07, 0.31, 0.05, 0.00, 0.12])
    se = values.std(ddof=1) / np.sqrt(len(values))
    low, high = dominance.t_interval(values.mean(), se, len(values), 0.95)
    want = stats.t.interval(0.95, len(values) - 1, loc=values.mean(), scale=se)
    assert (low, high) == pytest.approx(want)
    assert np.isnan(dominance.t_interval(0.1, 0.0, 9, 0.95)[0])
    assert np.isnan(dominance.t_interval(0.1, 0.02, 1, 0.95)[0])
    assert np.isnan(dominance.t_interval(float("nan"), 0.02, 9, 0.95)[1])


def test_tilted_rates_raise_the_two_target_codes_and_keep_the_others_in_proportion():
    base = generate.common_rates(6, 0.7)
    tilted = dominance.tilted_rates(base, 1, 2, 0.35)
    assert tilted.sum() == pytest.approx(1.0) and tilted[1] == tilted[2] == 0.35
    others = [0, 3, 4, 5]
    assert np.allclose(tilted[others] / tilted[others].sum(), base[others] / base[others].sum())


def test_weight_for_excess_gives_the_stated_excess_or_the_largest_possible():
    rng = np.random.default_rng(1)
    tilted = dominance.tilted_rates(generate.common_rates(6, 0.7), 1, 2, 0.35)
    rates = generate.draw_rates(rng, tilted, 8, 50000)
    weight, reached = dominance.weight_for_excess(rates, 1, 2, 0.30)
    assert weight.min() >= 0.0 and weight.max() <= 1.0
    matrices = extension.weighted_matrices(rates[:400], 1, 2, weight[:400])
    assert np.allclose(matrices.sum(axis=2), 1.0)
    stat = extension.weighted_stationary(rates, 1, 2, weight)
    assert np.allclose(stat[:400], generate.stationary(matrices), atol=1e-9)
    true_excess = (1 - weight) * rates[:, 2] + weight - stat[:, 2]
    assert np.allclose(true_excess, reached, atol=1e-10)
    assert reached.max() <= 0.30 + 1e-12 and np.mean(reached > 0.2999) > 0.85
    same = generate.draw_rates(rng, tilted, None, 3)
    assert np.allclose(dominance.weight_for_excess(same, 1, 2, 0.30)[1], 0.30)


def cell(level: str, excess: float, heterogeneity: str = "none", groups: int = 8, codes: int = 6) -> dict:
    return {"arm": "controlled_dom", "index": 0, "codes": codes, "groups": groups, "heterogeneity": heterogeneity,
            "length_profile": "extreme", "excess": excess, "dominant": level}


def test_truth_without_a_dominant_group_equals_that_of_the_first_study(pipeline_config):
    spec, dom = pipeline_config["simulation"]["controlled"], pipeline_config["dominance"]
    base = generate.common_rates(6, spec["rate_ratio"])
    for excess in (0.0, 0.1):
        got = dominance.controlled_truth(np.random.default_rng(2), cell("none", excess), spec, dom, 1000)
        want = generate.controlled_truth(np.random.default_rng(2), base, None, 1, 2, excess, 1000)
        assert got["group"] == pytest.approx(want["group"]) and got["pooled"] == pytest.approx(want["pooled"])


def test_truth_with_a_dominant_group(pipeline_config):
    spec, dom = pipeline_config["simulation"]["controlled"], pipeline_config["dominance"]
    rng = np.random.default_rng(3)
    rates_only = dominance.controlled_truth(rng, cell("rates", 0.0), spec, dom, 20000)
    assert rates_only["group"] == 0.0 and rates_only["dominant_excess"] == pytest.approx(0.0, abs=1e-12)
    assert rates_only["pooled"] > 0.01
    both = dominance.controlled_truth(rng, cell("rates_dependence", 0.0), spec, dom, 20000)
    assert both["dominant_excess"] == pytest.approx(0.30) and both["group"] == pytest.approx(0.30 / 8)
    ordinary = generate.controlled_truth(rng, generate.common_rates(6, spec["rate_ratio"]), None, 1, 2, 0.1, 10)["group"]
    mixed = dominance.controlled_truth(rng, cell("rates_dependence", 0.1, groups=23), spec, dom, 20000)
    assert mixed["group"] == pytest.approx((22 * ordinary + 0.30) / 23)
    assert mixed["pooled"] > mixed["group"]


def test_pooled_truth_agrees_with_counts_summed_over_many_simulated_data_sets(pipeline_config):
    spec, dom = pipeline_config["simulation"]["controlled"], pipeline_config["dominance"]
    target = cell("rates_dependence", 0.1, heterogeneity="high")
    truth = dominance.controlled_truth(np.random.default_rng(4), target, spec, dom, 200000)
    rng = np.random.default_rng(5)
    base = generate.common_rates(6, spec["rate_ratio"])
    sets, groups = 6000, 8
    rates = generate.draw_rates(rng, base, 8, sets * groups)
    lengths = generate.draw_lengths(rng, "extreme", sets, groups, spec)
    weight = np.full(len(rates), generate.mixing_weight(base, 2, 0.1))
    long_rows = np.arange(sets) * groups + groups - 1
    rates[long_rows] = generate.draw_rates(rng, dominance.tilted_rates(base, 1, 2, dom["target_rate"]), 8, sets)
    weight[long_rows], reached = dominance.weight_for_excess(rates[long_rows], 1, 2, dom["dominant_excess"])
    chains = generate.simulate_chains(
        rng, extension.weighted_matrices(rates, 1, 2, weight), extension.weighted_stationary(rates, 1, 2, weight),
        lengths.reshape(-1),
    )
    nab, na, nb, n = inference.padded_counts(chains, lengths.reshape(-1), 1, 2)
    pooled = nab.sum() / na.sum() - nb.sum() / n.sum()
    assert pooled == pytest.approx(truth["pooled"], abs=0.004)
    assert truth["dominant_excess"] == pytest.approx(reached.mean(), abs=0.003)


def synthetic_events() -> pd.DataFrame:
    rows = []
    for case, codes in (("Z01", "PQRQPQRQ"), ("Z02", "PPPPPPPPPPPP"), ("Z03", "QRQRQP")):
        rows += [{"case": case, "order_in_case": i + 1, "family": code} for i, code in enumerate(codes)]
    return pd.DataFrame(rows)


def test_calibration_separates_the_longest_group_from_the_others():
    out = dominance.calibration_from_events(synthetic_events(), ["P", "Q", "R"], 0.5)
    assert out["lengths"] == [8, 12, 6] and out["dominant_index"] == 1
    ordinary, own = np.array(out["matrix_ordinary"]), np.array(out["matrix_dominant"])
    assert np.allclose(ordinary.sum(axis=1), 1.0) and np.allclose(own.sum(axis=1), 1.0)
    assert own[0, 0] == pytest.approx(11.5 / 12.5)
    assert ordinary[0, 0] == pytest.approx(0.5 / 3.5) and ordinary[0, 1] == pytest.approx(2.5 / 3.5)


def test_calibrated_truth_weights_the_dominant_group_by_its_transitions():
    calibration = {
        "families": ["P", "Q", "R"],
        "matrix_ordinary": [[0.5, 0.3, 0.2], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]],
        "matrix_dominant": [[0.9, 0.05, 0.05], [0.4, 0.3, 0.3], [0.4, 0.3, 0.3]],
        "lengths": [11, 21, 101, 16],
        "dominant_index": 2,
    }
    truth = dominance.calibrated_truth(np.random.default_rng(6), calibration, None, [(0, 0), (0, 1)], 10)[0]
    common, own = np.array(calibration["matrix_ordinary"]), np.array(calibration["matrix_dominant"])
    s0, s1 = generate.stationary(common[None])[0], generate.stationary(own[None])[0]
    e0, e1 = common[0, 0] - s0[0], own[0, 0] - s1[0]
    assert truth["group"] == pytest.approx((3 * e0 + e1) / 4) and truth["dominant_excess"] == pytest.approx(e1)
    steps0, steps1 = 10 + 20 + 15, 100
    moved = steps0 * s0[0] * common[0, 0] + steps1 * s1[0] * own[0, 0]
    leaving = steps0 * s0[0] + steps1 * s1[0]
    arriving = steps0 * s0[0] + steps1 * s1[0]
    assert truth["pooled"] == pytest.approx(moved / leaving - arriving / (steps0 + steps1))
    drawn = dominance.calibrated_truth(np.random.default_rng(6), calibration, 30, [(0, 0)], 20000, chunk=7000)[0]
    assert drawn["group"] == pytest.approx(truth["group"], abs=0.01)


def test_grid_and_streams_of_the_second_extension(pipeline_config):
    dom = pipeline_config["dominance"]
    cells = dominance.all_cells(dom, pipeline_config["simulation"])
    by_arm = {arm: sum(1 for c in cells if c["arm"] == arm) for arm in dominance.ARMS}
    assert by_arm == {"controlled_dom": 144, "calibrated_dom": 3}
    controlled = [c for c in cells if c["arm"] == "controlled_dom"]
    assert {c["dominant"] for c in controlled} == set(dominance.LEVELS)
    assert {c["length_profile"] for c in controlled} == {"extreme"} and {c["groups"] for c in controlled} == {8, 12, 23, 40}
    assert len({dom["seed"], pipeline_config["seed"], pipeline_config["extension"]["seed"]}) == 3
    assert (dom["replications"], dom["resamples"], dom["permutations"]) == (2000, 1999, 4999)
    new = [g.random() for g in dominance.cell_streams(dom["seed"], cells[0], 4)]
    old = [g.random() for g in study.cell_streams(pipeline_config["seed"], {"arm": "controlled", "index": 0}, 4)]
    other = [g.random() for g in extension.cell_streams(pipeline_config["extension"]["seed"], {"arm": "opposing", "index": 0}, 4)]
    assert len(set(new) | set(old) | set(other)) == 12


def test_cells_are_reproducible_and_carry_every_procedure(pipeline_config):
    dom = copy.deepcopy(pipeline_config["dominance"])
    dom.update({"replications": 5, "resamples": 99, "permutations": 99, "batch": 3, "truth_groups": 2000})
    calibration = {
        "families": pipeline_config["families"]["order"],
        "matrix_ordinary": (np.full((6, 6), 0.1) + np.eye(6) * 0.4).tolist(),
        "matrix_dominant": (np.full((6, 6), 0.05) + np.eye(6) * 0.7).tolist(),
        "lengths": [12, 20, 30, 16, 25, 90, 22, 40],
        "dominant_index": 5,
    }
    cells = dominance.all_cells(dom, pipeline_config["simulation"])
    picked = [cells[1], cells[2], cells[100], cells[-1]]
    tasks = [
        {"cell": c, "dominance": dom, "simulation": pipeline_config["simulation"], "analysis": pipeline_config["analysis"],
         "calibration": calibration}
        for c in picked
    ]
    first = [dominance.run_cell(t) for t in tasks]
    second = [dominance.run_cell(t) for t in reversed(tasks)][::-1]
    for x, y in zip(first, second):
        x.pop("seconds"), y.pop("seconds")
        assert repr(x) == repr(y)
    frames = dominance.assemble(first)
    procedures = frames["dominance_procedures"]
    one = procedures[(procedures["arm"] == "controlled_dom") & (procedures["index"] == 1)]
    assert len(one) == 4 * 3 + 1 + 1 + 2
    assert set(one["statistic"]) == {"group", "pooled", "group_loo"}
    assert set(one["procedure"]) == {
        "T1_adjusted_residual", "T2_event_bootstrap", "T3_percentile", "T4_bca", "T5_studentised",
        "T6_order_permutation", dominance.T_INTERVAL,
    }
    intervals = one[one["coverage"].notna()]
    assert intervals["out_of_range"].between(0, 1).all() and one[one["coverage"].isna()]["out_of_range"].isna().all()
    assert len(procedures[procedures["arm"] == "calibrated_dom"]) == 8 * 16
    assert len(frames["dominance_estimates"]) == 3 * 3 + 8 * 3
    assert set(procedures["dominant"]) == {"rates", "rates_dependence", "observed"}


def test_null_cells_of_the_summary_are_those_without_dependence_in_any_group():
    frame = pd.DataFrame(
        {"excess": [0.0, 0.0, 0.0, 0.1, 0.1], "dominant": ["none", "rates", "rates_dependence", "none", "rates"]}
    )
    assert dominance_summary.null_cells(frame).tolist() == [True, True, False, False, False]


def test_added_intervals_of_the_application_follow_the_estimates(pipeline_config):
    rng = np.random.default_rng(7)
    sequences = [rng.integers(0, 3, size=int(n)).astype(np.int8) for n in rng.integers(10, 40, size=11)]
    targets = [("P>P", 0, 0), ("P>Q", 0, 1)]
    estimates, _ = application.focal_analysis(
        sequences, targets, pipeline_config["analysis"], {"resamples": 99, "permutations": 99}, rng, rng
    )
    estimates = [{"variant": "main", **row} for row in estimates]
    rows = application.added_intervals(estimates, 0.95)
    assert len(rows) == 2 * 3 and {r["status"] for r in rows} == {"added"}
    assert {r["statistic"] for r in rows} == {"group", "pooled", "group_loo"}
    for row in rows:
        reach = stats.t.ppf(0.975, row["contributing"] - 1) * row["se"]
        assert row["low"] == pytest.approx(row["estimate"] - reach) and row["high"] == pytest.approx(row["estimate"] + reach)
