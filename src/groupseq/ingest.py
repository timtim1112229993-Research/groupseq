"""Stage 1: read the raw exports through the column map and normalise them."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import tables
from .config import Settings, locate_inputs, stage_dir


def parse_raw_time(series: pd.Series, fmt: str, field: str, allow_blank: bool = False) -> pd.Series:
    """Parse local clock times written by the platform export."""
    blank = series == ""
    if blank.any() and not allow_blank:
        raise ValueError(f"blank timestamps in {field}")
    parsed = pd.to_datetime(series.where(~blank), format=fmt, errors="coerce")
    if parsed[~blank].isna().any():
        raise ValueError(f"unparseable timestamps in {field}")
    return parsed


def parse_stage_label(series: pd.Series, field: str) -> pd.Series:
    """Extract the stage number from a label that begins with 'S' and a digit."""
    number = series.str.extract(r"^S(\d+)", expand=False)
    if number.isna().any():
        raise ValueError(f"stage labels without a stage number in {field}")
    return number.astype(int)


def utc_to_local(text: str | None, offset_hours: int) -> datetime | None:
    """Convert an ISO timestamp with an offset to a naive local clock time."""
    if not text:
        return None
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError("timestamp without an offset cannot be placed on the local clock")
    return moment.astimezone(timezone(timedelta(hours=offset_hours))).replace(tzinfo=None)


def _local_text(text: str | None, offset_hours: int) -> str:
    moment = utc_to_local(text, offset_hours)
    return moment.strftime(tables.ISO_SECONDS) if moment else ""


def _agent_records(raw: pd.DataFrame, label: str, cfg: dict) -> pd.DataFrame:
    vm, allowed = cfg["value_map"], cfg["allowed"]
    out = pd.DataFrame()
    out["source_row"] = range(1, len(raw) + 1)
    out["record_id"] = [f"{label}-R{n:04d}" for n in out["source_row"]]
    out["class"] = label
    out["time_local"] = tables.iso(parse_raw_time(raw["time"], cfg["raw_time_format"], "agent_all.time"))
    out["group"] = tables.to_int(raw["group"], "agent_all.group")
    out["stage"] = parse_stage_label(raw["stage"], "agent_all.stage")
    out["record_type"] = tables.map_values(raw["record_type"], vm["record_type"], "agent_all.record_type")
    out["trigger"] = tables.map_values(raw["trigger"], vm["trigger"], "agent_all.trigger")
    tables.require_allowed(raw["action"], allowed["action"], "agent_all.action")
    tables.require_allowed(raw["source"], allowed["source"], "agent_all.source")
    out["action"] = raw["action"]
    out["source"] = raw["source"]
    out["off_topic"] = tables.map_values(raw["off_topic"], vm["yes_no"], "agent_all.off_topic", allow_blank=True)
    for name in ("question", "reply", "message", "note"):
        out[name] = raw[name]
    for record_type, triggers in allowed["record_trigger"].items():
        block = out[out["record_type"] == record_type]
        tables.require_allowed(block["trigger"], triggers, f"trigger of {record_type}")
    decisions = out["record_type"] == "DECISION"
    if (out.loc[decisions, "action"] == "").any() or (out.loc[~decisions, "action"] != "").any():
        raise ValueError("an action must be present on every decision and absent elsewhere")
    questions = out["record_type"] == "QUESTION"
    if (out.loc[questions, "off_topic"] == "").any() or (out.loc[~questions, "off_topic"] != "").any():
        raise ValueError("the off-topic flag must be present on every question and absent elsewhere")
    columns = ["record_id", "class", "source_row"] + [c for c in out.columns if c not in ("record_id", "class", "source_row")]
    return out[columns]


def _decision_log(raw: pd.DataFrame, label: str, cfg: dict) -> pd.DataFrame:
    vm, allowed = cfg["value_map"], cfg["allowed"]
    out = pd.DataFrame()
    out["class"] = [label] * len(raw)
    out["source_row"] = range(1, len(raw) + 1)
    out["time_local"] = tables.iso(parse_raw_time(raw["time"], cfg["raw_time_format"], "agent_logs.time"))
    out["group"] = tables.to_int(raw["group"], "agent_logs.group")
    out["stage"] = parse_stage_label(raw["stage"], "agent_logs.stage")
    out["trigger"] = tables.map_values(raw["trigger"], vm["trigger"], "agent_logs.trigger")
    tables.require_allowed(out["trigger"], allowed["record_trigger"]["DECISION"], "agent_logs.trigger")
    tables.require_allowed(raw["action"], [a for a in allowed["action"] if a], "agent_logs.action")
    tables.require_allowed(raw["source"], [s for s in allowed["source"] if s], "agent_logs.source")
    out["action"] = raw["action"]
    out["source"] = raw["source"]
    for name in ("help_count", "items_done", "items_total"):
        out[name] = tables.to_int(raw[name], f"agent_logs.{name}")
    out["message"] = raw["message"]
    out["diagnosis"] = raw["diagnosis"]
    return out


def _stage_fields(raw: pd.DataFrame, label: str, cfg: dict) -> pd.DataFrame:
    vm = cfg["value_map"]
    out = pd.DataFrame()
    out["class"] = [label] * len(raw)
    out["group"] = tables.to_int(raw["group"], "process.group")
    out["stage"] = tables.to_int(raw["stage"], "process.stage")
    out["stage_complete"] = tables.map_values(raw["stage_complete"], vm["yes_no"], "process.stage_complete")
    out["completion_pct"] = tables.to_int(raw["completion_pct"].str.rstrip("%"), "process.completion_pct")
    out["field"] = raw["field"]
    out["content"] = raw["content"]
    out["updated_local"] = tables.iso(
        parse_raw_time(raw["updated"], cfg["raw_time_format"], "process.updated", allow_blank=True)
    )
    return out


def _group_summary(raw: pd.DataFrame, label: str, cfg: dict) -> pd.DataFrame:
    vm = cfg["value_map"]
    out = pd.DataFrame()
    out["class"] = [label] * len(raw)
    out["group"] = tables.to_int(raw["group"], "groups.group")
    out["current_stage"] = parse_stage_label(raw["current_stage"], "groups.current_stage")
    for name in ("stages_completed", "mean_completion_pct", "help_clicks", "quiz_errors", "evidence_completeness"):
        out[name] = tables.to_int(raw[name], f"groups.{name}")
    out["risk_flag"] = tables.map_values(raw["risk_flag"], vm["risk_flag"], "groups.risk_flag", allow_blank=True)
    return out


def _quiz_errors(raw: pd.DataFrame, label: str, cfg: dict) -> pd.DataFrame:
    out = pd.DataFrame()
    out["class"] = [label] * len(raw)
    out["time_local"] = tables.iso(parse_raw_time(raw["time"], cfg["raw_time_format"], "quiz_wrong.time"))
    out["group"] = tables.to_int(raw["group"], "quiz_wrong.group")
    out["stage"] = tables.to_int(raw["stage"], "quiz_wrong.stage")
    for name in ("block", "crop", "item", "field", "answer", "correct"):
        out[name] = raw[name]
    return out


def _session(stats: dict, cfg: dict) -> dict:
    offset = cfg["timezone_offset_hours"]
    session = stats["session"]
    return {
        "session_id": session["id"],
        "group_count": session["group_count"],
        "started_utc": session["started_at"],
        "ended_utc": session["ended_at"],
        "window_start_local": utc_to_local(session["started_at"], offset).isoformat(sep=" "),
        "window_end_local": utc_to_local(session["ended_at"], offset).isoformat(sep=" "),
    }


def _stage_records(stats: dict, label: str, cfg: dict) -> pd.DataFrame:
    offset = cfg["timezone_offset_hours"]
    number = {g["group_id"]: g["group_number"] for g in stats["groups"]}
    rows = []
    for item in stats["stage_submissions"]:
        rows.append(
            {
                "class": label,
                "group": number[item["group_id"]],
                "stage": item["stage"],
                "stage_key": item["stage_key"],
                "is_completed": "yes" if item["is_completed"] else "no",
                "completion_rate": item["completion_rate"],
                "completed_local": _local_text(item.get("completed_at"), offset),
                "updated_local": _local_text(item.get("updated_at"), offset),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["group", "stage"], kind="stable").reset_index(drop=True)


def _device_summary(device: dict, sessions: dict, cfg: dict) -> dict:
    """Count device-export events that belong to a class session and fall inside its window."""
    offset = cfg["timezone_offset_hours"]
    runs = device.get("runs", {})
    events = device.get("events", [])
    run_session = {run_id: run.get("sessionCode") for run_id, run in runs.items()}
    summary = {"runs": len(runs), "events": len(events), "by_class_session": {}}
    for label, session in sessions.items():
        start = datetime.fromisoformat(session["window_start_local"])
        end = datetime.fromisoformat(session["window_end_local"])
        mine = [e for e in events if run_session.get(e["runId"]) == session["session_id"]]
        inside = [e for e in mine if start <= utc_to_local(e["ts"], offset) <= end]
        summary["by_class_session"][label] = {
            "runs": sum(1 for s in run_session.values() if s == session["session_id"]),
            "events": len(mine),
            "events_inside_window": len(inside),
        }
    summary["events_other_sessions"] = len(events) - sum(v["events"] for v in summary["by_class_session"].values())
    summary["used"] = any(v["events_inside_window"] > 0 for v in summary["by_class_session"].values())
    return summary


def run(settings: Settings) -> dict:
    """Run stage 1 and return the written paths and a report of counts."""
    cfg = settings.pipeline
    inputs = locate_inputs(settings)
    out_dir = stage_dir(settings, "ingest")
    builders = {
        "agent_all": ("agent_records", _agent_records),
        "agent_logs": ("decision_log", _decision_log),
        "process": ("stage_fields", _stage_fields),
        "groups": ("group_summary", _group_summary),
        "quiz_wrong": ("quiz_errors", _quiz_errors),
    }
    frames: dict[str, list[pd.DataFrame]] = {name: [] for name, _ in builders.values()}
    frames["stage_records"] = []
    sessions: dict[str, dict] = {}
    report: dict = {"inputs": {}, "tables": {}}
    for label in cfg["classes"]:
        for kind, (name, builder) in builders.items():
            raw = tables.read_mapped_csv(inputs[f"{label}/{kind}"], cfg["column_map"][kind])
            frame = builder(raw, label, cfg)
            frames[name].append(frame)
            report["inputs"][f"{label}/{kind}"] = {"rows": int(len(raw)), "columns": int(raw.shape[1])}
        stats = tables.read_json(inputs[f"{label}/stats"])
        sessions[label] = _session(stats, cfg)
        frames["stage_records"].append(_stage_records(stats, label, cfg))
        report["inputs"][f"{label}/stats"] = {"stage_submissions": len(stats["stage_submissions"])}
    outputs: list[Path] = []
    for name, parts in frames.items():
        frame = pd.concat(parts, ignore_index=True)
        path = out_dir / f"{name}.csv"
        tables.write_csv(frame, path)
        outputs.append(path)
        report["tables"][name] = {"rows": int(len(frame))}
    records = pd.concat(frames["agent_records"], ignore_index=True)
    report["agent_records"] = {
        "by_class_type_trigger": {
            f"{c}/{r}/{t}": int(n)
            for (c, r, t), n in records.groupby(["class", "record_type", "trigger"]).size().items()
        },
        "file_order_is_time_order": {
            label: bool(block["time_local"].is_monotonic_increasing) for label, block in records.groupby("class")
        },
    }
    report["device_export"] = {}
    for label in cfg["classes"]:
        key = f"{label}/device"
        if key in inputs:
            report["device_export"][key] = _device_summary(tables.read_json(inputs[key]), sessions, cfg)
    sessions_path = out_dir / "sessions.json"
    tables.write_json(sessions, sessions_path)
    report_path = out_dir / "ingest_report.json"
    tables.write_json(report, report_path)
    outputs += [sessions_path, report_path]
    return {"outputs": outputs, "report": report, "inputs": inputs}
