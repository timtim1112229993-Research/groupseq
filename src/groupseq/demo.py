"""Synthetic demonstration inputs with the structure of the platform exports.

Every value written here is drawn from a seeded random generator or is a fixed
placeholder. No record, count, rate or estimate of any real data set enters the
construction: the transition matrix, the action shares and the sequence lengths
below are invented round numbers. The files exist so that the data stages and
the application can be run end to end by a reader who does not hold the
restricted records; the numbers they yield mean nothing.

Raw headers and raw labels are taken from the column map and the value map of
the analysis configuration, so this module holds no label of its own.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import application, coding, ingest, rebuild, tables
from .config import PIPELINE_DIR, Settings

DEFAULT_SEED = 20300107
FAMILIES = ["IDLE", "NEXT", "HELP", "ASK", "UPTAKE", "REMINDER"]
# Invented values. Rows: the family just seen; columns: the family that follows.
MATRIX = [
    [0.45, 0.15, 0.15, 0.12, 0.03, 0.10],
    [0.35, 0.25, 0.12, 0.15, 0.05, 0.08],
    [0.25, 0.10, 0.35, 0.10, 0.15, 0.05],
    [0.25, 0.12, 0.10, 0.40, 0.05, 0.08],
    [0.30, 0.25, 0.20, 0.10, 0.05, 0.10],
    [0.40, 0.20, 0.10, 0.15, 0.05, 0.10],
]
FIRST = [0.55, 0.15, 0.10, 0.12, 0.03, 0.05]
GROUP_CONCENTRATION = 40.0
ACTIONS = {
    "IDLE": (["scaffold", "probe", "release", "redirect"], [0.45, 0.35, 0.18, 0.02]),
    "NEXT": (["release", "probe", "scaffold"], [0.60, 0.25, 0.15]),
    "HELP": (["scaffold", "probe", "release"], [0.60, 0.30, 0.10]),
}
SOURCES = (["llm", "fallback", "local"], [0.80, 0.15, 0.05])
PAIRED = {"IDLE": ("DWELL", [0, 1, 2, 3]), "HELP": ("HELP_OP", [-1, 0, 1, 2])}
ORDINARY_GROUPS = (11, 12)
LONG_GROUP = (1, 3)  # class position and group number of the one long sequence
LONG_LENGTH = 90
MEDIAN_LENGTH = 20
LOG_SD = 0.5
MIN_LENGTH = 8
MAX_LENGTH = 60
TEST_GROUP_LENGTH = 6
SESSION_SECONDS = 2700
MARGIN_SECONDS = 60
TIE_SHARE = 0.04
OFF_TOPIC_SHARE = 0.20
QUIZ_LINK_SHARE = 0.60
ITEMS_TOTAL = 6
FILE_NAMES = {
    "agent_all": "export-agent-all-{code}.csv",
    "agent_logs": "export-agent-logs-{code}.csv",
    "groups": "export-groups-{code}.csv",
    "quiz_wrong": "export-quiz-wrong-{code}.csv",
    "stats": "export-{code}-stats.json",
    "process": "export-{code}-process.csv",
}


def _raw_label(mapping: dict, value: str) -> str:
    """The first raw label that the value map translates to the given value."""
    for raw, mapped in mapping.items():
        if mapped == value:
            return raw
    raise KeyError(value)


def _csv_bytes(header: list[str], rows: list[list]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _session_start(position: int) -> datetime:
    return datetime(2030, 1, 7 + position, 9 + position, 0, 0)


def _utc_text(local: datetime, offset_hours: int) -> str:
    moment = local.replace(tzinfo=timezone(timedelta(hours=offset_hours))).astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _length(rng: np.random.Generator) -> int:
    drawn = int(round(float(rng.lognormal(np.log(MEDIAN_LENGTH), LOG_SD))))
    return min(MAX_LENGTH, max(MIN_LENGTH, drawn))


def _families(rng: np.random.Generator, length: int) -> list[str]:
    """One first-order sequence of families from a matrix drawn around the common one."""
    own = np.array([rng.dirichlet(GROUP_CONCENTRATION * np.array(row)) for row in MATRIX])
    state = int(rng.choice(len(FAMILIES), p=FIRST))
    out = [FAMILIES[state]]
    for _ in range(length - 1):
        state = int(rng.choice(len(FAMILIES), p=own[state]))
        out.append(FAMILIES[state])
    return out


def _start_seconds(rng: np.random.Generator, families: list[str]) -> list[int]:
    """Distinct start seconds inside the session, with a few deliberate ties between unpaired families."""
    span = np.arange(MARGIN_SECONDS, SESSION_SECONDS - MARGIN_SECONDS)
    seconds = sorted(int(v) for v in rng.choice(span, size=len(families), replace=False))
    for i in range(1, len(families)):
        free = families[i] not in PAIRED and families[i - 1] not in PAIRED and families[i] != families[i - 1]
        if free and rng.random() < TIE_SHARE:
            seconds[i] = seconds[i - 1]
    return seconds


def _exchange_records(rng: np.random.Generator, family: str, counter: list[int]) -> list[dict]:
    """The log records of one exchange, as offsets in seconds from its start."""
    blank = {"action": "", "source": "", "off_topic": "", "question": "", "reply": "", "message": ""}
    if family in ACTIONS:
        names, shares = ACTIONS[family]
        decision = {
            **blank, "record_type": "DECISION", "trigger": family, "offset": 0,
            "action": str(rng.choice(names, p=shares)), "source": str(rng.choice(SOURCES[0], p=SOURCES[1])),
            "message": "synthetic message",
        }
        if family not in PAIRED:
            return [decision]
        operation, offsets = PAIRED[family]
        decision["offset"] = int(rng.choice(offsets))
        return [{**blank, "record_type": "OPERATION", "trigger": operation, "offset": 0}, decision]
    if family == "ASK":
        counter[0] += 1
        return [{
            **blank, "record_type": "QUESTION", "trigger": "ASK", "offset": 0, "source": "llm",
            "off_topic": "yes" if rng.random() < OFF_TOPIC_SHARE else "no",
            "question": f"synthetic question {counter[0]}", "reply": "synthetic reply",
        }]
    trigger = "SCAFFOLD_OP" if family == "UPTAKE" else "REMINDER"
    return [{**blank, "record_type": "OPERATION", "trigger": trigger, "offset": 0}]


def _group_records(rng: np.random.Generator, group: int, length: int, start: datetime, counter: list[int]):
    """Records, quiz rows and a summary of one synthetic group."""
    families = _families(rng, length)
    seconds = _start_seconds(rng, families)
    stages = int(rng.integers(3, 6))
    cuts = sorted(int(v) for v in rng.choice(np.arange(300, SESSION_SECONDS - 300), size=stages - 1, replace=False))
    records, quiz = [], []
    for family, second in zip(families, seconds):
        stage = 1 + sum(second >= cut for cut in cuts)
        for record in _exchange_records(rng, family, counter):
            moment = start + timedelta(seconds=second + record.pop("offset"))
            records.append({**record, "t": moment, "group": group, "stage": stage, "progress": second / SESSION_SECONDS})
        if family == "REMINDER" and rng.random() < QUIZ_LINK_SHARE:
            quiz.append({"t": start + timedelta(seconds=second), "group": group, "stage": stage})
    summary = {
        "group": group, "stages": stages, "help": families.count("HELP"), "quiz": len(quiz),
        "completion": int(rng.integers(60, 101)), "evidence": int(rng.integers(40, 101)),
        "extra_help": int(rng.integers(0, 3)), "cuts": cuts,
    }
    return records, quiz, summary


def _outside_window(group: int, start: datetime) -> list[dict]:
    """Two records before the session opens and one after it closes; the scope rule must drop them."""
    blank = {"action": "", "source": "", "off_topic": "", "question": "", "reply": "", "message": ""}
    early = start - timedelta(minutes=20)
    late = start + timedelta(seconds=SESSION_SECONDS + 300)
    return [
        {**blank, "record_type": "OPERATION", "trigger": "DWELL", "t": early, "group": group, "stage": 1, "progress": 0.0},
        {**blank, "record_type": "DECISION", "trigger": "IDLE", "action": "probe", "source": "llm",
         "message": "synthetic message", "t": early + timedelta(seconds=1), "group": group, "stage": 1, "progress": 0.0},
        {**blank, "record_type": "DECISION", "trigger": "NEXT", "action": "release", "source": "llm",
         "message": "synthetic message", "t": late, "group": group, "stage": 1, "progress": 1.0},
    ]


def _class_files(pipeline: dict, label: str, position: int, seed: int) -> dict[str, bytes]:
    """The six export files of one synthetic class, keyed by kind."""
    vm, cmap, fmt = pipeline["value_map"], pipeline["column_map"], pipeline["raw_time_format"]
    start = _session_start(position)
    ordinary = list(range(1, ORDINARY_GROUPS[position % len(ORDINARY_GROUPS)] + 1))
    excluded = [g for g in pipeline["scope"]["excluded_groups"].get(label, []) if g not in ordinary]
    records, quiz, summaries = [], [], []
    counter = [0]
    for group in ordinary + excluded:
        rng = np.random.default_rng(np.random.SeedSequence([seed, position, group]))
        if group in excluded:
            length = TEST_GROUP_LENGTH
        elif (position, group) == LONG_GROUP:
            length = LONG_LENGTH
        else:
            length = _length(rng)
        own, own_quiz, summary = _group_records(rng, group, length, start, counter)
        records += own
        quiz += own_quiz
        summaries.append(summary)
    records += _outside_window(ordinary[0], start)
    order = sorted(range(len(records)), key=lambda i: (records[i]["t"], i))
    records = [records[i] for i in order]
    quiz.sort(key=lambda row: (row["t"], row["group"]))

    def name(group: int) -> str:
        return f"Group {group}"

    def clock(moment: datetime) -> str:
        return moment.strftime(fmt)

    def raw(kind: str, value: str) -> str:
        return _raw_label(vm[kind], value) if value else ""

    all_rows, log_rows = [], []
    helped: dict[int, int] = {}
    for record in records:
        fields = {
            "record_type": raw("record_type", record["record_type"]), "time": clock(record["t"]),
            "group": record["group"], "group_name": name(record["group"]), "stage": f"S{record['stage']}",
            "trigger": raw("trigger", record["trigger"]), "action": record["action"], "question": record["question"],
            "reply": record["reply"], "off_topic": raw("yes_no", record["off_topic"]), "source": record["source"],
            "message": record["message"], "note": "",
        }
        all_rows.append([fields[item["name"]] for item in cmap["agent_all"]])
        if record["record_type"] != "DECISION":
            continue
        if record["trigger"] == "HELP":
            helped[record["group"]] = helped.get(record["group"], 0) + 1
        fields.update({
            "help_count": helped.get(record["group"], 0),
            "items_done": min(ITEMS_TOTAL, int(record["progress"] * (ITEMS_TOTAL + 1))),
            "items_total": ITEMS_TOTAL, "diagnosis": "synthetic diagnosis",
        })
        log_rows.append([fields[item["name"]] for item in cmap["agent_logs"]])

    group_rows, process_rows, submissions = [], [], []
    for summary in summaries:
        group, stages = summary["group"], summary["stages"]
        help_clicks = summary["help"] + summary["extra_help"]
        fields = {
            "group": group, "group_name": name(group), "current_stage": f"S{stages}", "stages_completed": stages - 1,
            "mean_completion_pct": summary["completion"], "help_clicks": help_clicks, "quiz_errors": summary["quiz"],
            "evidence_completeness": summary["evidence"],
            "risk_flag": raw("risk_flag", "frequent_help") if help_clicks >= 8 else "",
        }
        group_rows.append([fields[item["name"]] for item in cmap["groups"]])
        for stage in range(1, stages + 1):
            done = stage < stages
            closed = start + timedelta(seconds=summary["cuts"][stage - 1]) if done else None
            for number in (1, 2):
                fields = {
                    "group": group, "group_name": name(group), "stage": stage, "stage_name": f"Stage {stage}",
                    "stage_complete": raw("yes_no", "yes" if done else "no"),
                    "completion_pct": "100%" if done else f"{summary['completion']}%",
                    "field": f"field_{number}", "content": "synthetic content", "updated": clock(closed) if closed else "",
                }
                process_rows.append([fields[item["name"]] for item in cmap["process"]])
            stamp = _utc_text(closed, pipeline["timezone_offset_hours"]) if closed else None
            submissions.append({
                "group_id": f"demo-{label}-{group:02d}", "stage": stage, "stage_key": f"stage_{stage}",
                "is_completed": done, "completion_rate": 100 if done else summary["completion"],
                "completed_at": stamp, "updated_at": stamp,
            })

    quiz_rows = []
    for number, row in enumerate(quiz, start=1):
        fields = {
            "time": clock(row["t"]), "group": row["group"], "group_name": name(row["group"]), "stage": row["stage"],
            "block": "block_1", "crop": "item_set_1", "item": f"item_{number}", "field": "field_1",
            "answer": "synthetic answer", "correct": "synthetic key",
        }
        quiz_rows.append([fields[item["name"]] for item in cmap["quiz_wrong"]])

    stats = {
        "session": {
            "id": f"demo-session-{label}", "group_count": len(summaries),
            "started_at": _utc_text(start, pipeline["timezone_offset_hours"]),
            "ended_at": _utc_text(start + timedelta(seconds=SESSION_SECONDS), pipeline["timezone_offset_hours"]),
        },
        "groups": [{"group_id": f"demo-{label}-{s['group']:02d}", "group_number": s["group"]} for s in summaries],
        "stage_submissions": submissions,
    }

    def header(kind: str) -> list[str]:
        return [item["raw"] for item in cmap[kind]]

    return {
        "agent_all": _csv_bytes(header("agent_all"), all_rows),
        "agent_logs": _csv_bytes(header("agent_logs"), log_rows),
        "groups": _csv_bytes(header("groups"), group_rows),
        "quiz_wrong": _csv_bytes(header("quiz_wrong"), quiz_rows),
        "process": _csv_bytes(header("process"), process_rows),
        "stats": (json.dumps(stats, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii"),
    }


def build(pipeline: dict, seed: int = DEFAULT_SEED) -> dict[str, dict[str, bytes]]:
    """Return the content of every synthetic export, by class label and kind of file."""
    return {label: _class_files(pipeline, label, position, seed) for position, label in enumerate(pipeline["classes"])}


def write_inputs(settings: Settings, seed: int = DEFAULT_SEED) -> list[Path]:
    """Write the synthetic exports under the input root of the given settings and return their paths."""
    written = []
    for label, files in build(settings.pipeline, seed).items():
        code = settings.paths["class_folders"][label]
        folder = settings.input_root / f"{code}_synthetic"
        folder.mkdir(parents=True, exist_ok=True)
        for kind, content in files.items():
            path = folder / FILE_NAMES[kind].format(code=code)
            path.write_bytes(content)
            written.append(path)
    return written


def demo_settings(root: Path | None = None, pipeline_dir: Path | None = None) -> Settings:
    """Settings that point every folder at the demonstration area.

    The demonstration paths file keeps inputs, stage outputs and run records
    inside the ignored working folder. With a root, the same layout is placed
    under that folder instead, which the tests use.
    """
    base = Path(pipeline_dir) if pipeline_dir else PIPELINE_DIR
    paths = tables.read_json(base / "config" / "paths.demo.json")
    home = Path(root) if root else base
    return Settings(
        pipeline=tables.read_json(base / "config" / "pipeline.json"),
        paths=paths,
        pipeline_dir=home,
        input_root=(home / paths["input_root"]).resolve(),
        work_dir=home / paths["work_dir"],
        provenance_dir=home / paths["provenance_dir"],
    )


def run_stages(settings: Settings) -> dict[str, dict]:
    """Run the three data stages and the application on the demonstration inputs; return the stage reports."""
    reports = {}
    for name, run in (("ingest", ingest.run), ("events", rebuild.run), ("coded", coding.run), ("application", application.run)):
        reports[name] = run(settings)["report"]
    return reports
