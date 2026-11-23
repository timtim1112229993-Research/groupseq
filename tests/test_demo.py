"""Tests of the synthetic demonstration inputs.

The inputs must be reproducible from their seed, must pass through the data
stages and the application without any restricted file, and must stay inside
the ignored working folder.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from groupseq import demo, tables
from groupseq.config import locate_inputs

PIPELINE_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory):
    """One run of the demonstration in a temporary folder, with small resampling sizes."""
    settings = demo.demo_settings(root=tmp_path_factory.mktemp("demo"))
    pipeline = json.loads(json.dumps(settings.pipeline))
    pipeline["application"] = {"resamples": 99, "permutations": 99}
    pipeline["analysis"]["tie_orderings"] = 20
    settings = dataclasses.replace(settings, pipeline=pipeline)
    written = demo.write_inputs(settings)
    return settings, written, demo.run_stages(settings)


def test_inputs_are_fixed_by_the_seed(pipeline_config):
    first = demo.build(pipeline_config, seed=5)
    assert first == demo.build(pipeline_config, seed=5)
    other = demo.build(pipeline_config, seed=6)
    assert first.keys() == other.keys()
    assert all(first[label]["agent_all"] != other[label]["agent_all"] for label in first)


def test_every_export_is_found_exactly_once_through_the_demonstration_paths(demo_run):
    settings, written, _ = demo_run
    found = locate_inputs(settings)
    kinds = set(demo.FILE_NAMES)
    assert set(found) == {f"{label}/{kind}" for label in settings.pipeline["classes"] for kind in kinds}
    assert sorted(found.values()) == sorted(written)


def test_headers_and_labels_come_from_the_column_map(pipeline_config):
    files = demo.build(pipeline_config)
    for label in pipeline_config["classes"]:
        for kind in ("agent_all", "agent_logs", "groups", "quiz_wrong", "process"):
            header = files[label][kind].decode("utf-8").split("\n", 1)[0]
            assert header.split(",") == [item["raw"] for item in pipeline_config["column_map"][kind]]


def test_data_stages_and_application_run_on_the_inputs(demo_run):
    settings, _, reports = demo_run
    cfg = settings.pipeline
    events = reports["events"]
    assert events["records_in_scope"] == events["records_read"] - sum(events["records_excluded"].values())
    assert set(events["records_excluded"]) == {cfg["scope"]["excluded_group_reason"], cfg["scope"]["window_reason"]}
    assert events["pairs_with_stage_mismatch"] == 0
    for trigger, rule in (("IDLE", cfg["pairing"]["idle"]), ("HELP", cfg["pairing"]["help"])):
        offsets = events["pair_offset_seconds"][trigger]
        assert rule["min_offset_s"] <= offsets["min"] and offsets["max"] <= rule["max_offset_s"]
    coded = reports["coded"]
    assert coded["cases"] == sum(demo.ORDINARY_GROUPS) and coded["events"] == events["exchanges"]
    assert set(coded["families"]) == set(cfg["families"]["order"])
    assert max(max(block["sorted"]) for block in coded["sequence_length"].values()) == demo.LONG_LENGTH
    assert coded["tied_events"] > 0
    applied = reports["application"]
    assert applied["groups"] == coded["cases"] and applied["transitions"] == coded["events"] - coded["cases"]
    assert sum(applied["arms"].values()) == applied["groups"]


def test_every_planned_exchange_is_recovered(demo_run):
    """Each paired trigger finds its decision, so the rebuilt sequences are the generated ones."""
    settings, _, reports = demo_run
    unpaired = reports["events"]["unpaired_by_trigger_and_action"]
    assert not any(key.startswith(("IDLE/", "HELP/")) for key in unpaired)
    estimates = tables.read_csv(settings.work_dir / "05_application" / "focal_estimates.csv")
    assert set(estimates["variant"]) == {"main", "without_longest_group", "rapid_repeats_merged"}


def test_a_demonstration_run_writes_nothing_outside_its_own_folders(demo_run):
    settings, _, _ = demo_run
    root = settings.pipeline_dir
    top = {path.relative_to(root).parts[0] for path in root.rglob("*") if path.is_file()}
    assert top == {"work"}
    assert not (settings.provenance_dir / "manifests").exists()


def test_demonstration_paths_stay_inside_the_ignored_working_folder():
    paths = tables.read_json(PIPELINE_DIR / "config" / "paths.demo.json")
    for key in ("input_root", "work_dir", "provenance_dir"):
        assert paths[key] == "work/demo" or paths[key].startswith("work/demo/"), key
    assert "inherited_event_table" not in paths
    example = tables.read_json(PIPELINE_DIR / "config" / "paths.example.json")
    assert paths["file_patterns"] == example["file_patterns"]
