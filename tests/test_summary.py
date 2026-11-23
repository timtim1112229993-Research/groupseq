"""Tests of the decision rules applied to simulation tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from groupseq import summary

RULES = {"coverage": [0.925, 0.975], "false_positive": [0.025, 0.075]}


def rows(specs: list[tuple]) -> pd.DataFrame:
    """Each spec is (procedure, statistic, excess, coverage, reject, width_median)."""
    frame = pd.DataFrame(specs, columns=["procedure", "statistic", "excess", "coverage", "reject", "width_median"])
    frame["coverage_se"] = 0.005
    frame["reject_se"] = 0.005
    frame["groups"] = 8
    return frame


def test_decision_bands_follow_the_plan(pipeline_config):
    assert pipeline_config["decision"] == RULES


def test_interval_is_judged_on_coverage_everywhere_and_on_false_positives_at_the_null():
    frame = rows(
        [
            ("T3", "group", 0.0, 0.95, 0.05, 0.2),
            ("T3", "group", 0.1, 0.95, 0.60, 0.2),
            ("T3", "group", 0.1, 0.90, 0.60, 0.2),
            ("T3", "pooled", 0.0, 0.95, 0.20, 0.2),
            ("T3", "pooled", 0.0, 0.99, 0.05, 0.2),
        ]
    )
    judged = summary.judge(frame, RULES, frame["excess"] == 0)
    assert list(judged["valid"]) == [True, True, False, False, False]
    assert list(judged["judged"]) == [True] * 5
    assert np.isnan(judged["false_positive_ok"][1]) and judged["false_positive_ok"][3] == 0.0


def test_test_without_interval_is_judged_in_null_cells_only():
    frame = rows([("T6", "group", 0.0, np.nan, 0.05, np.nan), ("T6", "group", 0.0, np.nan, 0.09, np.nan), ("T6", "group", 0.2, np.nan, 0.9, np.nan)])
    judged = summary.judge(frame, RULES, frame["excess"] == 0)
    assert list(judged["judged"]) == [True, True, False]
    assert list(judged["valid"]) == [True, False, False]
    counts = summary.validity_counts(judged, ["excess"])
    overall = counts[counts["factor"] == "all"].iloc[0]
    assert overall["cells_judged"] == 2 and overall["cells_valid"] == 1
    assert overall["power_mean"] != overall["power_mean"]


def test_band_edges_are_inside_and_borderline_cells_are_flagged():
    frame = rows([("T3", "group", 0.1, 0.925, 0.5, 0.2), ("T3", "group", 0.1, 0.9249, 0.5, 0.2), ("T3", "group", 0.1, 0.95, 0.5, 0.2)])
    judged = summary.judge(frame, RULES, frame["excess"] == 0)
    assert list(judged["valid"]) == [True, False, True]
    assert list(judged["borderline"]) == [True, True, False]


def test_ranking_prefers_more_valid_cells_then_narrower_intervals_and_keeps_tests_apart():
    frame = rows(
        [
            ("A", "group", 0.1, 0.95, 0.5, 0.30),
            ("A", "group", 0.2, 0.95, 0.5, 0.30),
            ("B", "group", 0.1, 0.95, 0.5, 0.20),
            ("B", "group", 0.2, 0.95, 0.5, 0.20),
            ("C", "group", 0.1, 0.95, 0.5, 0.10),
            ("C", "group", 0.2, 0.80, 0.5, 0.10),
            ("Z", "group", 0.0, np.nan, 0.05, np.nan),
        ]
    )
    judged = summary.judge(frame, RULES, frame["excess"] == 0)
    order = summary.ranking(summary.validity_counts(judged, ["excess"]))
    intervals = order[order["kind"] == "interval"]
    assert list(intervals["procedure"]) == ["B", "A", "C"] and list(intervals["rank"]) == [1, 2, 3]
    tests = order[order["kind"] == "test"]
    assert list(tests["procedure"]) == ["Z"] and list(tests["rank"]) == [1]


def test_network_counts_split_null_and_shifted_cells():
    frame = pd.DataFrame(
        {
            "index": [0, 1, 2, 3],
            "groups": [8, 8, 12, 12],
            "heterogeneity": ["none", "none", "high", "high"],
            "shift": [False, True, False, True],
            "procedure": ["N1"] * 4,
            "replications": 2000,
            "reject": [0.05, 0.60, 0.30, 0.90],
            "reject_se": 0.005,
        }
    )
    out = summary.network_tables(frame, RULES)
    overall = out["network_counts"].query("factor == 'all'").iloc[0]
    assert overall["null_cells"] == 2 and overall["null_cells_valid"] == 1
    assert overall["false_positive_max"] == pytest.approx(0.30) and overall["power_mean"] == pytest.approx(0.75)


def test_incomplete_run_is_refused(pipeline_config):
    class Stub:
        pipeline = pipeline_config

    procedures = pd.DataFrame({"arm": ["controlled", "calibrated"], "index": [0, 0], "replications": [2000, 2000]})
    networks = pd.DataFrame({"index": [0], "replications": [2000]})
    with pytest.raises(summary.SummaryError):
        summary.check_complete(Stub, procedures, networks)
