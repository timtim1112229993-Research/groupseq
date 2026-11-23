"""Stage 2: rebuild the event table with one record per exchange.

The platform writes a trigger and the agent decision bound to it as two log
records. An exchange joins the two, so that a structurally bound pair is not
counted as a sequence of two independent events.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import tables
from .config import Settings, locate_inherited, stage_dir

EXCHANGE_COLUMNS = [
    "exchange_id", "class", "group", "case", "order_in_case", "t_start", "t_end", "stage",
    "initiator", "trigger", "agent_action", "reply_source", "off_topic", "help_count",
    "items_done", "items_total", "n_records", "pair_offset_s", "stage_mismatch", "quiz_linked",
    "tied", "gap_prev_s", "first_row",
]


def load_records(settings: Settings) -> pd.DataFrame:
    """Load the normalised agent records with typed keys and without free text."""
    frame = tables.read_csv(stage_dir(settings, "ingest") / "agent_records.csv")
    frame = frame.drop(columns=["question", "reply", "message", "note"])
    for name in ("source_row", "group", "stage"):
        frame[name] = frame[name].astype(int)
    frame["t"] = pd.to_datetime(frame["time_local"], format=tables.ISO_SECONDS)
    return frame


def attach_decision_log(records: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    """Add the decision-log counters to decision records by position within each class.

    The two exports list the same decisions in the same order. The alignment is
    asserted on five fields, so a divergence stops the run.
    """
    out = records.copy()
    for name in ("help_count", "items_done", "items_total"):
        out[name] = ""
    for label, block in records.groupby("class"):
        decisions = block[block["record_type"] == "DECISION"].sort_values("source_row")
        rows = log[log["class"] == label].copy()
        rows["source_row"] = rows["source_row"].astype(int)
        rows = rows.sort_values("source_row")
        if len(decisions) != len(rows):
            raise ValueError(f"class {label}: {len(decisions)} decisions against {len(rows)} decision-log rows")
        for name in ("time_local", "trigger", "action", "source"):
            if list(decisions[name]) != list(rows[name]):
                raise ValueError(f"class {label}: decision log disagrees with agent records on {name}")
        if list(decisions["group"]) != list(rows["group"].astype(int)):
            raise ValueError(f"class {label}: decision log disagrees with agent records on group")
        for name in ("help_count", "items_done", "items_total"):
            out.loc[decisions.index, name] = list(rows[name])
    return out


def apply_scope(records: pd.DataFrame, sessions: dict, scope: dict) -> pd.Series:
    """Return the exclusion reason of every record; empty text means the record is in scope."""
    reason = pd.Series("", index=records.index)
    for label, groups in scope["excluded_groups"].items():
        mask = (records["class"] == label) & records["group"].isin(groups)
        reason[mask] = scope["excluded_group_reason"]
    for label, session in sessions.items():
        start = pd.Timestamp(session["window_start_local"])
        end = pd.Timestamp(session["window_end_local"])
        outside = (records["t"] < start) | (records["t"] > end)
        mask = (records["class"] == label) & outside & (reason == "")
        reason[mask] = scope["window_reason"]
    return reason


def pair_ordered(reply_times: list, trigger_times: list, min_offset: float, max_offset: float):
    """Pair replies with triggers in time order.

    Both lists are sorted. A reply and a trigger are paired when the reply time
    minus the trigger time lies inside the window; otherwise the earlier of the
    two is left unpaired and the scan moves on.
    """
    pairs, lone_replies, lone_triggers = [], [], []
    i = j = 0
    while i < len(reply_times) and j < len(trigger_times):
        offset = (reply_times[i] - trigger_times[j]).total_seconds()
        if min_offset <= offset <= max_offset:
            pairs.append((i, j))
            i += 1
            j += 1
        elif offset < min_offset:
            lone_replies.append(i)
            i += 1
        else:
            lone_triggers.append(j)
            j += 1
    lone_replies.extend(range(i, len(reply_times)))
    lone_triggers.extend(range(j, len(trigger_times)))
    return pairs, lone_replies, lone_triggers


def _exchange_row(label, group, trigger, reply, opener, initiator, quiz_keys):
    members = [m for m in (opener, reply) if m is not None]
    times = [m["t"] for m in members]
    start = min(times)
    if reply is None:
        action, source = "none", "none"
    elif reply["record_type"] == "QUESTION":
        action, source = "reply", reply["source"]
    else:
        action, source = reply["action"], reply["source"]
    is_decision = reply is not None and reply["record_type"] == "DECISION"
    both = reply is not None and opener is not None
    row = {
        "class": label,
        "group": group,
        "case": f"{label}{group:02d}",
        "t_start": start,
        "t_end": max(times),
        "stage": (reply if reply is not None else opener)["stage"],
        "initiator": initiator[trigger],
        "trigger": trigger,
        "agent_action": action,
        "reply_source": source,
        "off_topic": reply["off_topic"] if reply is not None else "",
        "help_count": reply["help_count"] if is_decision else "",
        "items_done": reply["items_done"] if is_decision else "",
        "items_total": reply["items_total"] if is_decision else "",
        "n_records": len(members),
        "pair_offset_s": int((reply["t"] - opener["t"]).total_seconds()) if both else "",
        "stage_mismatch": "yes" if both and reply["stage"] != opener["stage"] else "no",
        "quiz_linked": "",
        "first_row": min(m["source_row"] for m in members),
    }
    if trigger == "REMINDER":
        row["quiz_linked"] = "yes" if (label, group, start) in quiz_keys else "no"
    return row


def build_exchanges(records: pd.DataFrame, pairing: dict, initiator: dict, quiz_keys=frozenset()):
    """Join bound records into exchanges and order the exchanges within each group.

    Returns the exchange table and a table that links every record to its exchange.
    """
    rows, link_rows = [], []

    def add(label, group, trigger, reply, opener):
        key = len(rows)
        rows.append(_exchange_row(label, group, trigger, reply, opener, initiator, quiz_keys))
        rows[-1]["_key"] = key
        if reply is not None and opener is not None:
            link_rows.append({"record_id": opener["record_id"], "_key": key, "role": "trigger"})
            link_rows.append({"record_id": reply["record_id"], "_key": key, "role": "reply"})
        else:
            member = reply if reply is not None else opener
            link_rows.append({"record_id": member["record_id"], "_key": key, "role": "single"})

    for (label, group), block in records.groupby(["class", "group"], sort=True):
        block = block.sort_values(["t", "source_row"], kind="stable")
        used: set[str] = set()
        for rule in pairing.values():
            replies = block[(block["record_type"] == "DECISION") & (block["trigger"] == rule["decision_trigger"])]
            openers = block[(block["record_type"] == "OPERATION") & (block["trigger"] == rule["operation_trigger"])]
            reply_list = replies.to_dict("records")
            opener_list = openers.to_dict("records")
            pairs, lone_replies, lone_openers = pair_ordered(
                [r["t"] for r in reply_list], [o["t"] for o in opener_list],
                rule["min_offset_s"], rule["max_offset_s"],
            )
            for i, j in pairs:
                add(label, group, rule["exchange_trigger"], reply_list[i], opener_list[j])
            for i in lone_replies:
                add(label, group, rule["exchange_trigger"], reply_list[i], None)
            for j in lone_openers:
                add(label, group, rule["exchange_trigger"], None, opener_list[j])
            used.update(replies["record_id"])
            used.update(openers["record_id"])
        for record in block[~block["record_id"].isin(used)].to_dict("records"):
            if record["record_type"] == "DECISION":
                add(label, group, record["trigger"], record, None)
            elif record["record_type"] == "QUESTION":
                add(label, group, "ASK", record, None)
            elif record["trigger"] == "SCAFFOLD_OP":
                add(label, group, "UPTAKE", None, record)
            elif record["trigger"] == "REMINDER":
                add(label, group, "REMINDER", None, record)
            else:
                raise ValueError(f"no exchange rule for record {record['record_id']}")

    exchanges = pd.DataFrame(rows)
    exchanges = exchanges.sort_values(["class", "group", "t_start", "first_row"], kind="stable").reset_index(drop=True)
    exchanges["order_in_case"] = exchanges.groupby("case").cumcount() + 1
    exchanges["exchange_id"] = [f"{c}-X{n:03d}" for c, n in zip(exchanges["case"], exchanges["order_in_case"])]
    exchanges["tied"] = exchanges.duplicated(["case", "t_start"], keep=False).map({True: "yes", False: "no"})
    gap = exchanges.groupby("case")["t_start"].diff().dt.total_seconds()
    exchanges["gap_prev_s"] = gap.map(lambda v: "" if pd.isna(v) else int(v))
    links = pd.DataFrame(link_rows).merge(exchanges[["_key", "exchange_id"]], on="_key", how="left")
    links = links[["record_id", "exchange_id", "role"]].sort_values("record_id", kind="stable").reset_index(drop=True)
    exchanges["t_start"] = exchanges["t_start"].dt.strftime(tables.ISO_SECONDS)
    exchanges["t_end"] = exchanges["t_end"].dt.strftime(tables.ISO_SECONDS)
    return exchanges[EXCHANGE_COLUMNS], links


def reconcile(inherited: pd.DataFrame, records: pd.DataFrame, links: pd.DataFrame, value_map: dict):
    """Match every inherited event to a raw record and report every difference."""
    inh = pd.DataFrame()
    inh["inherited_event_id"] = inherited["event_id"]
    inh["class"] = inherited["class_label"].str[0]
    inh["group"] = tables.to_int(inherited["group"], "inherited.group")
    stamps = pd.to_datetime(inherited["time"], format=tables.ISO_SECONDS)
    inh["time_local"] = stamps.dt.strftime(tables.ISO_SECONDS)
    inh["trigger"] = tables.map_values(inherited["trigger"], value_map["trigger"], "inherited.trigger")
    inh["inherited_initiator"] = inherited["initiator"]
    inh["inherited_code"] = tables.map_values(inherited["action_code"], value_map["inherited_code"], "inherited.action_code")
    inh = inh.sort_values("inherited_event_id", kind="stable")
    key = ["class", "group", "time_local", "trigger"]
    inh["_n"] = inh.groupby(key).cumcount()
    raw = records.sort_values(["class", "source_row"], kind="stable")[
        ["record_id", "class", "group", "time_local", "trigger", "record_type", "scope"]
    ].copy()
    raw["_n"] = raw.groupby(key).cumcount()
    merged = inh.merge(raw, how="outer", on=key + ["_n"], indicator=True)
    merged = merged.merge(links, how="left", on="record_id")
    labels = {"both": "matched", "left_only": "inherited_only", "right_only": "raw_only"}
    merged["status"] = merged["_merge"].astype(str).map(labels)
    merged = merged.drop(columns=["_merge", "_n"]).fillna("")
    merged = merged.sort_values(["class", "group", "time_local", "record_id"], kind="stable").reset_index(drop=True)
    columns = [
        "status", "inherited_event_id", "record_id", "class", "group", "time_local", "trigger", "record_type",
        "inherited_initiator", "inherited_code", "scope", "exchange_id", "role",
    ]
    merged = merged[columns]
    matched = merged[merged["status"] == "matched"]
    raw_only = merged[merged["status"] == "raw_only"]
    summary = {
        "inherited_rows": int(len(inh)),
        "matched_to_a_raw_record": int(len(matched)),
        "inherited_without_raw_record": int((merged["status"] == "inherited_only").sum()),
        "raw_records_absent_from_inherited": {
            reason or "in_scope": int(n) for reason, n in raw_only.groupby("scope").size().items()
        },
        "matched_rows_now_out_of_scope": {
            reason: int(n) for reason, n in matched[matched["scope"] != ""].groupby("scope").size().items()
        },
        "matched_rows_in_scope": int((matched["scope"] == "").sum()),
        "matched_rows_joined_into_a_pair": int(matched["role"].isin(["trigger", "reply"]).sum()),
        "inherited_student_initiated_rows_that_are_agent_decisions": int(
            ((matched["inherited_initiator"] == "S") & (matched["record_type"] == "DECISION")).sum()
        ),
        "inherited_student_initiated_rows_that_are_timer_triggers": int(
            ((matched["inherited_initiator"] == "S") & (matched["trigger"] == "DWELL")).sum()
        ),
        "exchanges_covering_matched_rows": int(matched.loc[matched["exchange_id"] != "", "exchange_id"].nunique()),
    }
    return merged, summary


def run(settings: Settings) -> dict:
    """Run stage 2 and return the written paths and a report of counts."""
    cfg = settings.pipeline
    in_dir = stage_dir(settings, "ingest")
    out_dir = stage_dir(settings, "events")
    sessions = tables.read_json(in_dir / "sessions.json")
    records = attach_decision_log(load_records(settings), tables.read_csv(in_dir / "decision_log.csv"))
    records["scope"] = apply_scope(records, sessions, cfg["scope"])
    in_scope = records[records["scope"] == ""]

    quiz = tables.read_csv(in_dir / "quiz_errors.csv")
    quiz_keys = frozenset(
        zip(quiz["class"], quiz["group"].astype(int), pd.to_datetime(quiz["time_local"], format=tables.ISO_SECONDS))
    )
    exchanges, links = build_exchanges(in_scope, cfg["pairing"], cfg["initiator"], quiz_keys)

    exclusions = records.loc[
        records["scope"] != "", ["record_id", "class", "group", "time_local", "stage", "record_type", "trigger", "scope"]
    ].rename(columns={"scope": "reason"})

    outputs: list[Path] = []
    for name, frame in (("exchanges", exchanges), ("record_links", links), ("exclusions", exclusions)):
        path = out_dir / f"{name}.csv"
        tables.write_csv(frame, path)
        outputs.append(path)

    paired = exchanges[exchanges["pair_offset_s"] != ""]
    offsets = paired["pair_offset_s"].astype(int)
    report = {
        "records_read": int(len(records)),
        "records_excluded": {reason: int(n) for reason, n in exclusions.groupby("reason").size().items()},
        "records_in_scope": int(len(in_scope)),
        "exchanges": int(len(exchanges)),
        "exchanges_by_class": {k: int(v) for k, v in exchanges.groupby("class").size().items()},
        "exchanges_by_trigger": {k: int(v) for k, v in exchanges.groupby("trigger").size().items()},
        "exchanges_by_initiator": {k: int(v) for k, v in exchanges.groupby("initiator").size().items()},
        "paired_exchanges_by_trigger": {k: int(v) for k, v in paired.groupby("trigger").size().items()},
        "unpaired_by_trigger_and_action": {
            f"{t}/{a}": int(n)
            for (t, a), n in exchanges[exchanges["n_records"] == 1].groupby(["trigger", "agent_action"]).size().items()
        },
        "pair_offset_seconds": {
            trigger: {"min": int(block.min()), "median": float(block.median()), "max": int(block.max())}
            for trigger, block in offsets.groupby(paired["trigger"])
        },
        "pairs_with_stage_mismatch": int((exchanges["stage_mismatch"] == "yes").sum()),
        "reminders_linked_to_a_quiz_error": {
            k: int(v) for k, v in exchanges[exchanges["trigger"] == "REMINDER"].groupby("quiz_linked").size().items()
        },
        "cases": int(exchanges["case"].nunique()),
        "tied_exchanges": int((exchanges["tied"] == "yes").sum()),
        "records_linked": int(len(links)),
    }
    if report["records_linked"] != report["records_in_scope"]:
        raise ValueError("every record in scope must belong to exactly one exchange")

    inherited_path = locate_inherited(settings)
    if inherited_path is None:
        report["reconciliation"] = "inherited table not found"
    else:
        inherited = tables.read_mapped_csv(inherited_path, cfg["column_map"]["inherited_events"])
        rows, summary = reconcile(inherited, records, links, cfg["value_map"])
        path = out_dir / "reconciliation.csv"
        tables.write_csv(rows, path)
        outputs.append(path)
        report["reconciliation"] = summary

    report_path = out_dir / "rebuild_report.json"
    tables.write_json(report, report_path)
    outputs.append(report_path)
    return {"outputs": outputs, "report": report}
