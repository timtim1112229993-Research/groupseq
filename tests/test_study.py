"""Tests of the simulation grids, random streams, scoring and reproducibility."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from groupseq import study


@pytest.fixture()
def small(pipeline_config) -> dict:
    sim = copy.deepcopy(pipeline_config["simulation"])
    sim.update({"replications": 6, "resamples": 99, "permutations": 99, "batch": 4, "truth_groups": 2000, "truth_datasets": 200})
    return sim


def test_grid_sizes_follow_the_plan(pipeline_config):
    sim = pipeline_config["simulation"]
    assert len(study.controlled_cells(sim["controlled"])) == 270
    assert len(study.calibrated_cells(sim["calibrated"])) == 3
    assert len(study.network_cells(sim["network"])) == 30
    assert (sim["replications"], sim["resamples"], sim["permutations"]) == (2000, 1999, 4999)
    assert pipeline_config["seed"] == 20261123
    cells = study.all_cells(sim)
    assert len({(c["arm"], c["index"]) for c in cells}) == 303


def test_streams_differ_between_cells_and_repeat_for_one_cell(pipeline_config):
    cells = study.controlled_cells(pipeline_config["simulation"]["controlled"])
    one = [g.random() for g in study.cell_streams(1, cells[0], 3)]
    again = [g.random() for g in study.cell_streams(1, cells[0], 3)]
    other = [g.random() for g in study.cell_streams(1, cells[1], 3)]
    network_cell = study.network_cells(pipeline_config["simulation"]["network"])[0]
    assert one == again and len(set(one)) == 3
    assert not set(one) & set(other)
    assert not set(one) & {g.random() for g in study.cell_streams(1, network_cell, 3)}


def test_scoring_of_a_hand_built_tally():
    tally = study.Tally(4)
    for statistic in ("group", "pooled"):
        tally.estimate[statistic][:] = [0.1, 0.2, 0.3, np.nan]
    tally.contributing[:] = [8, 9, 7, 8]
    key = ("T3_percentile", "group")
    tally.low[key][:] = [0.05, 0.15, -0.1, np.nan]
    tally.high[key][:] = [0.30, 0.40, 0.15, np.nan]
    tally.residual[:] = [2.5, -2.5, 0.1, np.nan]
    tally.order_p["group"][:] = [0.01, 0.05, 0.2, 0.9]
    procedures, estimates = study.score(tally, {"group": 0.2, "pooled": 0.2}, 8)
    rows = {(r["procedure"], r["statistic"]): r for r in procedures}
    t3 = rows["T3_percentile", "group"]
    assert t3["defined"] == 3 and t3["coverage"] == pytest.approx(2 / 3)
    assert t3["miss_low"] == pytest.approx(1 / 3) and t3["miss_high"] == 0.0
    assert t3["reject"] == pytest.approx(2 / 3) and t3["width"] == pytest.approx(0.25)
    assert rows["T1_adjusted_residual", "pooled"]["reject"] == pytest.approx(2 / 3)
    assert rows["T6_order_permutation", "group"]["reject"] == pytest.approx(0.5)
    assert rows["T4_bca", "group"]["defined"] == 0
    group = [e for e in estimates if e["statistic"] == "group"][0]
    assert group["bias"] == pytest.approx(0.0) and group["defined"] == 3
    assert group["estimable_rate"] == pytest.approx(0.75)


def test_calibration_counts_transitions_within_groups_only():
    events = pd.DataFrame(
        {
            "case": ["Z01"] * 3 + ["Z02"] * 2,
            "order_in_case": [1, 2, 3, 1, 2],
            "family": ["X", "Y", "X", "Y", "Y"],
        }
    )
    out = study.calibration_from_events(events, ["X", "Y"], 0.5)
    assert out["lengths"] == [3, 2]
    assert np.allclose(out["matrix"], [[0.5 / 2, 1.5 / 2], [1.5 / 3, 1.5 / 3]])


def test_cells_are_reproducible_and_tables_assemble(small, pipeline_config):
    analysis = pipeline_config["analysis"]
    calibration = {
        "families": pipeline_config["families"]["order"],
        "matrix": (np.full((6, 6), 0.1) + np.eye(6) * 0.4).tolist(),
        "lengths": [12, 20, 30, 16, 25, 18, 22, 40],
    }
    cells = [
        study.controlled_cells(small["controlled"])[5],
        study.controlled_cells(small["controlled"])[40],
        study.calibrated_cells(small["calibrated"])[1],
        study.network_cells(small["network"])[3],
    ]
    tasks = [
        {"cell": c, "simulation": small, "analysis": analysis, "seed": 5, "calibration": calibration, "nodes": 8}
        for c in cells
    ]
    first = [study.run_cell(t) for t in tasks]
    second = [study.run_cell(t) for t in reversed(tasks)][::-1]
    for a, b in zip(first, second):
        a.pop("seconds"), b.pop("seconds")
        assert repr(a) == repr(b)
    frames = study.assemble(first)
    procedures = frames["transition_procedures"]
    assert len(procedures) == 2 * 10 + 8 * 10
    assert set(procedures["procedure"]) == {
        "T1_adjusted_residual", "T2_event_bootstrap", "T3_percentile", "T4_bca", "T5_studentised", "T6_order_permutation"
    }
    assert len(frames["transition_estimates"]) == 2 * 2 + 8 * 2
    assert list(frames["network_procedures"]["procedure"]) == ["N1_segment_label", "N2_group_label", "N3_group_studentised"]
    assert procedures["replications"].eq(6).all()
