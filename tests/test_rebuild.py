"""Tests of scope rules, pairing and exchange construction on synthetic records."""
from __future__ import annotations

import pandas as pd
import pytest
from conftest import BASE, make_records

from groupseq import rebuild


def at(seconds: int) -> pd.Timestamp:
    return BASE + pd.Timedelta(seconds=seconds)


def test_pair_inside_window():
    pairs, lone_replies, lone_triggers = rebuild.pair_ordered([at(4)], [at(0)], 0, 30)
    assert pairs == [(0, 0)] and lone_replies == [] and lone_triggers == []


def test_trigger_without_reply_is_left_unpaired():
    pairs, lone_replies, lone_triggers = rebuild.pair_ordered([at(104)], [at(0), at(100)], 0, 30)
    assert pairs == [(0, 1)] and lone_replies == [] and lone_triggers == [0]


def test_reply_logged_before_trigger_pairs_under_negative_window():
    pairs, lone_replies, lone_triggers = rebuild.pair_ordered([at(0)], [at(2)], -30, 5)
    assert pairs == [(0, 0)] and lone_replies == [] and lone_triggers == []


def test_burst_pairs_in_rank_order():
    pairs, lone_replies, lone_triggers = rebuild.pair_ordered([at(0), at(1), at(2)], [at(3), at(3), at(3)], -30, 5)
    assert pairs == [(0, 0), (1, 1), (2, 2)] and lone_replies == [] and lone_triggers == []


def test_reply_far_ahead_of_trigger_stays_unpaired():
    pairs, lone_replies, lone_triggers = rebuild.pair_ordered([at(0)], [at(100)], -30, 5)
    assert pairs == [] and lone_replies == [0] and lone_triggers == [0]


def test_trailing_items_are_reported_unpaired():
    pairs, lone_replies, lone_triggers = rebuild.pair_ordered([at(1), at(50), at(90)], [at(0)], 0, 30)
    assert pairs == [(0, 0)] and lone_replies == [1, 2] and lone_triggers == []


@pytest.fixture()
def synthetic():
    return make_records(
        [
            (0, 1, "OPERATION", "DWELL", "", "", ""),
            (4, 1, "DECISION", "IDLE", "scaffold", "llm", ""),
            (60, 1, "DECISION", "NEXT", "probe", "fallback", ""),
            (100, 1, "DECISION", "HELP", "scaffold", "local", ""),
            (101, 1, "DECISION", "HELP", "scaffold", "local", ""),
            (102, 1, "OPERATION", "HELP_OP", "", "", ""),
            (102, 1, "OPERATION", "HELP_OP", "", "", ""),
            (102, 1, "OPERATION", "SCAFFOLD_OP", "", "", ""),
            (200, 1, "QUESTION", "ASK", "", "llm", "yes"),
            (300, 1, "OPERATION", "DWELL", "", "", ""),
            (400, 1, "OPERATION", "REMINDER", "", "", ""),
            (10, 2, "OPERATION", "DWELL", "", "", ""),
            (13, 2, "DECISION", "IDLE", "probe", "llm", ""),
        ]
    )


def build(records, pipeline_config, quiz_keys=frozenset()):
    return rebuild.build_exchanges(records, pipeline_config["pairing"], pipeline_config["initiator"], quiz_keys)


def test_exchange_counts_and_triggers(synthetic, pipeline_config):
    exchanges, _ = build(synthetic, pipeline_config)
    first = exchanges[exchanges["case"] == "Z01"]
    assert list(first["trigger"]) == ["IDLE", "NEXT", "HELP", "HELP", "UPTAKE", "ASK", "IDLE", "REMINDER"]
    assert list(first["agent_action"]) == ["scaffold", "probe", "scaffold", "scaffold", "none", "reply", "none", "none"]
    assert list(first["n_records"]) == [2, 1, 2, 2, 1, 1, 1, 1]
    assert len(exchanges[exchanges["case"] == "Z02"]) == 1


def test_every_record_belongs_to_exactly_one_exchange(synthetic, pipeline_config):
    exchanges, links = build(synthetic, pipeline_config)
    assert sorted(links["record_id"]) == sorted(synthetic["record_id"])
    assert links["record_id"].is_unique
    assert set(links["exchange_id"]) == set(exchanges["exchange_id"])


def test_roles_follow_who_opened_the_exchange(synthetic, pipeline_config):
    _, links = build(synthetic, pipeline_config)
    role = dict(zip(links["record_id"], links["role"]))
    assert role["Z-R0001"] == "trigger" and role["Z-R0002"] == "reply"
    assert role["Z-R0004"] == "reply" and role["Z-R0006"] == "trigger"
    assert role["Z-R0003"] == "single" and role["Z-R0010"] == "single"


def test_order_identifiers_and_gaps(synthetic, pipeline_config):
    exchanges, _ = build(synthetic, pipeline_config)
    first = exchanges[exchanges["case"] == "Z01"]
    assert list(first["order_in_case"]) == list(range(1, 9))
    assert first["exchange_id"].iloc[0] == "Z01-X001"
    assert first["exchange_id"].is_unique
    assert list(first["gap_prev_s"]) == ["", 60, 40, 1, 1, 98, 100, 100]
    assert list(first["pair_offset_s"]) == [4, "", -2, -1, "", "", "", ""]


def test_initiator_and_timestamps(synthetic, pipeline_config):
    exchanges, _ = build(synthetic, pipeline_config)
    first = exchanges[exchanges["case"] == "Z01"]
    assert list(first["initiator"]) == ["timer", "student", "student", "student", "student", "student", "timer", "system"]
    assert first["t_start"].iloc[2] == "2030-01-01 09:01:40"
    assert first["t_end"].iloc[2] == "2030-01-01 09:01:42"


def test_tied_start_times_are_flagged(pipeline_config):
    records = make_records(
        [
            (5, 1, "QUESTION", "ASK", "", "llm", "no"),
            (5, 1, "OPERATION", "SCAFFOLD_OP", "", "", ""),
            (9, 1, "DECISION", "NEXT", "release", "fallback", ""),
        ]
    )
    exchanges, _ = build(records, pipeline_config)
    assert list(exchanges["tied"]) == ["yes", "yes", "no"]
    assert list(exchanges["trigger"]) == ["ASK", "UPTAKE", "NEXT"]


def test_reminder_is_linked_to_a_quiz_error(pipeline_config):
    records = make_records([(7, 3, "OPERATION", "REMINDER", "", "", ""), (9, 3, "OPERATION", "REMINDER", "", "", "")])
    exchanges, _ = build(records, pipeline_config, frozenset({("Z", 3, at(7))}))
    assert list(exchanges["quiz_linked"]) == ["yes", "no"]


def test_scope_excludes_listed_groups_and_records_outside_the_window():
    records = make_records(
        [
            (0, 1, "DECISION", "NEXT", "probe", "llm", ""),
            (0, 9, "DECISION", "NEXT", "probe", "llm", ""),
            (7200, 1, "DECISION", "NEXT", "probe", "llm", ""),
        ]
    )
    sessions = {"Z": {"window_start_local": "2029-12-31 23:00:00", "window_end_local": "2030-01-01 10:00:00"}}
    scope = {"excluded_groups": {"Z": [9]}, "excluded_group_reason": "listed", "window_reason": "window"}
    assert list(rebuild.apply_scope(records, sessions, scope)) == ["", "listed", "window"]


def test_decision_log_must_align(pipeline_config):
    records = make_records([(0, 1, "DECISION", "NEXT", "probe", "llm", "")])
    log = pd.DataFrame(
        [{"class": "Z", "source_row": "1", "time_local": records["time_local"].iloc[0], "group": "1", "trigger": "NEXT",
          "action": "scaffold", "source": "llm", "help_count": "2", "items_done": "1", "items_total": "4"}]
    )
    with pytest.raises(ValueError):
        rebuild.attach_decision_log(records, log)
    log["action"] = "probe"
    joined = rebuild.attach_decision_log(records, log)
    assert joined["help_count"].iloc[0] == "2" and joined["items_total"].iloc[0] == "4"
