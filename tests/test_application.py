"""Tests of the application stage on synthetic events only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from groupseq import application, inference

FAMILIES = ["P", "Q", "R"]


def events(rows: list[tuple]) -> pd.DataFrame:
    """Each row is (class, case, second, family, node, stage)."""
    frame = pd.DataFrame(rows, columns=["class", "case", "second", "family", "node", "stage"])
    frame["t_start"] = [f"2030-01-01 09:{s // 60:02d}:{s % 60:02d}" for s in frame["second"]]
    frame["order_in_case"] = frame.groupby("case").cumcount() + 1
    frame["gap"] = frame.groupby("case")["second"].diff()
    counts = frame.groupby(["case", "t_start"])["family"].transform("size")
    frame["tied"] = np.where(counts > 1, "yes", "no")
    return frame


def random_events(seed: int, groups: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(groups):
        second = 0
        for _ in range(int(rng.integers(8, 30))):
            second += int(rng.integers(0, 40))
            family = FAMILIES[int(rng.integers(0, 3))]
            rows.append(("A" if g < groups // 2 else "B", f"Z{g:02d}", second, family, family, str(int(rng.integers(0, 4)))))
    return events(rows)


def test_rapid_repeats_of_one_family_count_once():
    frame = events(
        [("A", "Z01", 0, "P", "P", "0"), ("A", "Z01", 3, "P", "P", "0"), ("A", "Z01", 7, "P", "P", "0"),
         ("A", "Z01", 20, "P", "P", "0"), ("A", "Z01", 22, "Q", "Q", "0"), ("A", "Z02", 23, "Q", "Q", "0")]
    )
    merged = application.merge_rapid(frame, 5)
    assert list(merged["second"]) == [0, 20, 22, 23]


def test_tie_shuffle_moves_only_events_that_share_a_start_time():
    frame = events(
        [("A", "Z01", 0, "P", "P", "0"), ("A", "Z01", 5, "Q", "Q", "0"), ("A", "Z01", 5, "R", "R", "0"),
         ("A", "Z01", 5, "P", "P", "0"), ("A", "Z01", 9, "Q", "Q", "0"), ("A", "Z02", 5, "R", "R", "0")]
    )
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(60):
        out = application.shuffle_ties(frame, rng)
        assert list(out["second"]) == list(frame["second"]) and list(out["case"]) == list(frame["case"])
        assert out["family"].iloc[0] == "P" and out["family"].iloc[4] == "Q" and out["family"].iloc[5] == "R"
        seen.add(tuple(out["family"].iloc[1:4]))
    assert len(seen) == 6


def test_holm_adjustment_matches_a_worked_example():
    got = application.holm([0.01, 0.04, 0.03, np.nan, 0.5])
    assert got[:3] == pytest.approx([0.04, 0.09, 0.09])
    assert np.isnan(got[3]) and got[4] == pytest.approx(0.5)
    assert application.holm([0.6, 0.7]).tolist() == [1.0, 1.0]


def test_longest_group_and_sequences():
    frame = random_events(2)
    cases, sequences = application.to_sequences(frame, FAMILIES)
    assert cases == sorted(frame["case"].unique())
    longest = application.longest_case(frame)
    assert len(sequences[cases.index(longest)]) == max(len(s) for s in sequences)
    padded, lengths = application.pad(sequences)
    assert padded.shape == (len(cases), lengths.max()) and (padded[0, lengths[0]:] == -1).all()


def test_permutation_draws_reproduce_the_order_permutation_test():
    _, sequences = application.to_sequences(random_events(3), FAMILIES)
    targets = [(0, 1), (1, 1), (2, 0)]
    want = inference.order_permutation(sequences, targets, np.random.default_rng(8), 400, 2)
    nulls = application.permutation_nulls(sequences, targets, np.random.default_rng(8), 400, 2)
    for k in range(3):
        assert inference._two_sided_p(nulls["obs_group"][k], nulls["null_group"][:, k]) == pytest.approx(want[k]["group"])
        assert inference._two_sided_p(nulls["obs_pooled"][k], nulls["null_pooled"][:, k]) == pytest.approx(want[k]["pooled"])


def test_maximum_statistic_adjustment_is_never_below_the_single_test_value():
    rng = np.random.default_rng(4)
    null = rng.normal(size=(999, 6)) * np.array([1, 2, 3, 1, 1, 0.5])
    null[:, 5] = 0.25
    observed = np.array([2.5, 0.1, -12.0, np.nan, 0.0, 0.25])
    out = application.max_statistic(observed, null)
    assert np.isnan(out["z"][3]) and np.isnan(out["p_adjusted"][5])
    ok = [0, 1, 2, 4]
    assert np.all(out["p_adjusted"][ok] >= out["p"][ok] - 0.02)
    assert out["p_adjusted"][2] < out["p_adjusted"][0] < out["p_adjusted"][1]
    alone = application.max_statistic(observed[:1], null[:, :1])
    assert alone["p_adjusted"][0] == pytest.approx(alone["p"][0], abs=0.01)


def test_focal_analysis_rows_and_holm_groups(pipeline_config):
    _, sequences = application.to_sequences(random_events(5, groups=12), FAMILIES)
    targets = [("P>P", 0, 0), ("P>Q", 0, 1), ("R>P", 2, 0)]
    rng = np.random.default_rng(6)
    estimates, tests = application.focal_analysis(
        sequences, targets, pipeline_config["analysis"], {"resamples": 199, "permutations": 199}, rng, rng
    )
    assert len(estimates) == 3 * 4 and len(tests) == 3 * 3
    frame = pd.DataFrame(estimates)
    assert set(frame["status"]) == {"registered", "added"}
    group = frame[(frame["transition"] == "P>Q") & (frame["statistic"] == "group")].iloc[0]
    counts = inference.transition_counts(sequences, 0, 1)
    assert group["estimate"] == pytest.approx(float(inference.group_excess(*counts, 2)[0]))
    assert group["n_ab"] == counts[0].sum() and group["percentile_low"] <= group["estimate"] <= group["percentile_high"]
    assert np.isnan(group["event_low"])
    pooled = frame[(frame["transition"] == "P>Q") & (frame["statistic"] == "pooled")].iloc[0]
    assert pooled["event_low"] < pooled["estimate"] < pooled["event_high"]
    tested = pd.DataFrame(tests)
    for _, block in tested.groupby(["test", "statistic"]):
        assert np.allclose(block["p_holm"], application.holm(block["p"].to_numpy()))
        assert (block["p_holm"] >= block["p"] - 1e-12).all()


def test_network_data_builds_one_segment_per_group_and_stage():
    frame = events(
        [("A", "Z01", 0, "P", "P", "0"), ("A", "Z01", 5, "Q", "", "0"), ("A", "Z01", 9, "R", "R", "1"),
         ("B", "Z02", 1, "P", "P", "2"), ("B", "Z02", 2, "Q", "Q", "2"), ("B", "Z02", 3, "Q", "", "3")]
    )
    data = application.network_data(frame, ["P", "Q", "R"], ["A", "B"])
    assert data["cases"] == ["Z01", "Z02"] and data["arm"].tolist() == [0, 1]
    assert data["owner"].tolist() == [0, 0, 1, 1]
    assert data["presence"].tolist() == [[True, False, False], [False, False, True], [True, True, False], [False, False, False]]


def test_network_analysis_returns_every_test_and_pair():
    frame = random_events(7, groups=12)
    data = application.network_data(frame, FAMILIES, ["A", "B"])
    tests, edges = application.network_analysis(data, FAMILIES, 199, np.random.default_rng(9))
    assert [(t["test"], t["vectors"]) for t in tests] == [
        ("N1_segment_label", "segment"), ("N2_group_label", "unit_length"), ("N3_group_studentised", "unit_length"),
        ("N2_group_label", "unscaled"), ("N3_group_studentised", "unscaled"),
    ]
    assert len(edges) == 2 * 3 and all(0 < e["p_adjusted"] <= 1 for e in edges)
    assert all(0 < t["p"] <= 1 for t in tests)


def test_tie_sensitivity_is_constant_without_ties(pipeline_config):
    frame = random_events(10, groups=9)
    frame["t_start"] = [f"{i:06d}" for i in range(len(frame))]
    rows = application.tie_sensitivity(frame, FAMILIES, [("P>Q", 0, 1)], pipeline_config["analysis"], 5, np.random.default_rng(1))
    assert len(rows) == 3
    for row in rows:
        assert row["min"] == pytest.approx(row["max"])
