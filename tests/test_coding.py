"""Tests of the decision table, pooling rule and configuration consistency."""
from __future__ import annotations

import copy

import pandas as pd
import pytest

from groupseq import coding


def exchanges(rows: list[tuple]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["trigger", "agent_action", "off_topic"])
    frame["class"] = "Z"
    frame["case"] = "Z01"
    return frame


def test_rules_are_unique_and_codes_are_unique(pipeline_config):
    rules = pipeline_config["coding"]["rules"]
    lookup = coding.rule_lookup(rules)
    assert len(lookup) == len(rules)
    assert len(set(lookup.values())) == len(rules)


def test_duplicate_rule_is_rejected(pipeline_config):
    rules = copy.deepcopy(pipeline_config["coding"]["rules"])
    rules.append(dict(rules[0]))
    with pytest.raises(coding.CodingError):
        coding.rule_lookup(rules)


def test_every_decision_combination_has_a_rule(pipeline_config):
    lookup = coding.rule_lookup(pipeline_config["coding"]["rules"])
    actions = [a for a in pipeline_config["allowed"]["action"] if a] + ["none"]
    for trigger in pipeline_config["allowed"]["record_trigger"]["DECISION"]:
        for action in actions:
            assert (trigger, action, "") in lookup


def test_assignment_and_pooling(pipeline_config):
    cfg = copy.deepcopy(pipeline_config["coding"])
    cfg["pool_min_count"] = 3
    frame = exchanges([("IDLE", "scaffold", "")] * 3 + [("ASK", "reply", "yes")] * 2 + [("UPTAKE", "none", "")])
    coded = coding.assign_codes(frame, cfg)
    assert list(coded["fine_code"]) == ["IDLE_SCAFFOLD"] * 3 + ["ASK_OFF"] * 2 + ["UPTAKE"]
    assert list(coded["code"]) == ["IDLE_SCAFFOLD"] * 3 + ["OTHER"] * 3


def test_uncoded_exchange_stops_the_run(pipeline_config):
    with pytest.raises(coding.CodingError):
        coding.assign_codes(exchanges([("ASK", "reply", "")]), pipeline_config["coding"])


def test_pooled_name_must_not_shadow_a_fine_code(pipeline_config):
    cfg = copy.deepcopy(pipeline_config["coding"])
    cfg["pooled_code"] = "UPTAKE"
    cfg["pool_min_count"] = 1
    with pytest.raises(coding.CodingError):
        coding.assign_codes(exchanges([("UPTAKE", "none", "")]), cfg)


def test_every_fine_code_has_a_family_and_a_node_entry(pipeline_config):
    fine = {rule["code"] for rule in pipeline_config["coding"]["rules"]}
    families, net = pipeline_config["families"], pipeline_config["network"]
    assert set(families["of_fine_code"]) == fine
    assert set(net["node_of_fine_code"]) == fine
    assert set(families["of_fine_code"].values()) == set(families["order"])
    assert set(net["node_of_fine_code"].values()) - {""} == set(net["node_order"])
    assert len(families["order"]) == 6 and len(net["node_order"]) == 8
    focal = pipeline_config["analysis"]["focal_transitions"]
    assert len(focal) == 8 and all(x in families["order"] and y in families["order"] for x, y in focal)


def test_levels_are_assigned_and_reminder_is_not_a_node(pipeline_config):
    frame = exchanges([("IDLE", "redirect", ""), ("HELP", "probe", ""), ("REMINDER", "none", ""), ("ASK", "reply", "no")])
    coded = coding.assign_codes(frame, pipeline_config["coding"])
    levelled = coding.assign_levels(coded, pipeline_config["families"], pipeline_config["network"])
    assert list(levelled["family"]) == ["IDLE", "HELP", "REMINDER", "ASK"]
    assert list(levelled["node"]) == ["", "HELP", "", "ASK_ON"]


def test_fine_code_without_family_stops_the_run(pipeline_config):
    families = copy.deepcopy(pipeline_config["families"])
    del families["of_fine_code"]["UPTAKE"]
    coded = coding.assign_codes(exchanges([("UPTAKE", "none", "")]), pipeline_config["coding"])
    with pytest.raises(coding.CodingError):
        coding.assign_levels(coded, families, pipeline_config["network"])


def test_configuration_is_internally_consistent(pipeline_config):
    cfg = pipeline_config
    exchange_triggers = {rule["trigger"] for rule in cfg["coding"]["rules"]}
    assert exchange_triggers == set(cfg["initiator"])
    mapped = set(cfg["value_map"]["trigger"].values())
    declared = {t for triggers in cfg["allowed"]["record_trigger"].values() for t in triggers}
    assert mapped == declared
    for rule in cfg["pairing"].values():
        assert rule["min_offset_s"] <= rule["max_offset_s"]
        assert rule["decision_trigger"] in cfg["allowed"]["record_trigger"]["DECISION"]
        assert rule["operation_trigger"] in cfg["allowed"]["record_trigger"]["OPERATION"]
        assert rule["exchange_trigger"] in exchange_triggers
