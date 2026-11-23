"""Stage 3: assign action codes to exchanges from a written decision table.

Every exchange matches exactly one rule, so no priority order is needed.
Codes that occur fewer times than the pooling threshold are merged into one
pooled code for analysis; the fine code is kept alongside.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import tables
from .config import Settings, stage_dir

CODED_COLUMNS = [
    "exchange_id", "class", "group", "case", "order_in_case", "t_start", "stage", "initiator",
    "fine_code", "code", "family", "node", "tied", "gap_prev_s",
]


class CodingError(ValueError):
    """Raised when the decision table is ambiguous or leaves an exchange uncoded."""


def rule_lookup(rules: list[dict]) -> dict:
    """Index the decision table and fail on duplicate conditions."""
    lookup: dict = {}
    for rule in rules:
        key = (rule["trigger"], rule["agent_action"], rule["off_topic"])
        if key in lookup:
            raise CodingError(f"duplicate rule for {key}")
        lookup[key] = rule["code"]
    return lookup


def assign_codes(exchanges: pd.DataFrame, coding: dict) -> pd.DataFrame:
    """Return the exchanges with a fine code and an analysis code."""
    lookup = rule_lookup(coding["rules"])
    keys = list(zip(exchanges["trigger"], exchanges["agent_action"], exchanges["off_topic"]))
    missing = sorted({k for k in keys if k not in lookup})
    if missing:
        raise CodingError(f"no rule for {missing}")
    out = exchanges.copy()
    out["fine_code"] = [lookup[k] for k in keys]
    counts = out["fine_code"].value_counts()
    kept = set(counts[counts >= coding["pool_min_count"]].index)
    if coding["pooled_code"] in kept:
        raise CodingError("the pooled code must not be the name of a fine code")
    out["code"] = [c if c in kept else coding["pooled_code"] for c in out["fine_code"]]
    return out


def assign_levels(coded: pd.DataFrame, families: dict, network: dict) -> pd.DataFrame:
    """Add the family of each event and its network node; an empty node means the event is not a node."""
    family_of, node_of = families["of_fine_code"], network["node_of_fine_code"]
    missing = sorted(set(coded["fine_code"]) - set(family_of) | set(coded["fine_code"]) - set(node_of))
    if missing:
        raise CodingError(f"fine codes without a family or node entry: {missing}")
    out = coded.copy()
    out["family"] = out["fine_code"].map(family_of)
    out["node"] = out["fine_code"].map(node_of)
    if not set(out["family"]) <= set(families["order"]):
        raise CodingError("a family is missing from the declared family order")
    if not set(out["node"]) - {""} <= set(network["node_order"]):
        raise CodingError("a node is missing from the declared node order")
    return out


def frequency_table(coded: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Count each code at every level, overall, by class and by number of groups."""
    rows = []
    for level in ("fine_code", "code", "family", "node"):
        for code, block in coded.groupby(level):
            row = {"level": level, "code": code, "n": int(len(block))}
            for label in classes:
                row[f"n_class_{label}"] = int((block["class"] == label).sum())
            row["groups_with_code"] = int(block["case"].nunique())
            row["largest_share_from_one_group"] = round(float(block["case"].value_counts().iloc[0] / len(block)), 4)
            rows.append(row)
    table = pd.DataFrame(rows)
    return table.sort_values(["level", "n", "code"], ascending=[True, False, True], kind="stable").reset_index(drop=True)


def case_table(coded: pd.DataFrame) -> pd.DataFrame:
    """Summarise the length and span of each group's sequence."""
    rows = []
    for case, block in coded.groupby("case", sort=True):
        stages = block["stage"].astype(int)
        rows.append(
            {
                "case": case,
                "class": block["class"].iloc[0],
                "group": block["group"].iloc[0],
                "n_events": int(len(block)),
                "n_codes": int(block["code"].nunique()),
                "n_fine_codes": int(block["fine_code"].nunique()),
                "n_tied_events": int((block["tied"] == "yes").sum()),
                "first_stage": int(stages.min()),
                "last_stage": int(stages.max()),
                "n_stages": int(stages.nunique()),
                "t_first": block["t_start"].iloc[0],
                "t_last": block["t_start"].iloc[-1],
            }
        )
    return pd.DataFrame(rows)


def code_crosswalk(reconciliation: pd.DataFrame, coded: pd.DataFrame) -> pd.DataFrame:
    """Cross the inherited action code with the rebuilt code for every matched record in scope."""
    matched = reconciliation[(reconciliation["status"] == "matched") & (reconciliation["exchange_id"] != "")]
    joined = matched.merge(coded[["exchange_id", "fine_code"]], on="exchange_id", how="left")
    table = joined.groupby(["inherited_code", "inherited_initiator", "role", "fine_code"]).size().reset_index(name="records")
    return table.sort_values(["inherited_code", "records"], ascending=[True, False], kind="stable").reset_index(drop=True)


def run(settings: Settings) -> dict:
    """Run stage 3 and return the written paths and a report of counts."""
    cfg = settings.pipeline
    in_dir = stage_dir(settings, "events")
    out_dir = stage_dir(settings, "coded")
    exchanges = tables.read_csv(in_dir / "exchanges.csv")
    coded = assign_levels(assign_codes(exchanges, cfg["coding"]), cfg["families"], cfg["network"])
    events = coded[CODED_COLUMNS]
    frequencies = frequency_table(coded, cfg["classes"])
    cases = case_table(coded)

    outputs: list[Path] = []
    for name, frame in (("coded_events", events), ("code_frequencies", frequencies), ("case_summary", cases)):
        path = out_dir / f"{name}.csv"
        tables.write_csv(frame, path)
        outputs.append(path)

    recon_path = in_dir / "reconciliation.csv"
    if recon_path.is_file():
        path = out_dir / "code_crosswalk.csv"
        tables.write_csv(code_crosswalk(tables.read_csv(recon_path), coded), path)
        outputs.append(path)

    analysis = frequencies[frequencies["level"] == "code"]
    fine = frequencies[frequencies["level"] == "fine_code"]
    pooled = sorted(set(coded.loc[coded["code"] == cfg["coding"]["pooled_code"], "fine_code"]))
    codebook = {
        "rules": cfg["coding"]["rules"],
        "pool_min_count": cfg["coding"]["pool_min_count"],
        "pooled_code": cfg["coding"]["pooled_code"],
        "analysis_codes": list(analysis["code"]),
        "fine_codes_pooled": pooled,
    }
    codebook_path = out_dir / "codebook.json"
    tables.write_json(codebook, codebook_path)
    lengths = cases["n_events"]
    report = {
        "events": int(len(coded)),
        "cases": int(len(cases)),
        "cases_by_class": {k: int(v) for k, v in cases.groupby("class").size().items()},
        "fine_codes_observed": int(len(fine)),
        "analysis_codes": {row["code"]: int(row["n"]) for _, row in analysis.iterrows()},
        "families": {
            row["code"]: int(row["n"]) for _, row in frequencies[frequencies["level"] == "family"].iterrows()
        },
        "network_nodes": {
            (row["code"] or "not_a_node"): int(row["n"])
            for _, row in frequencies[frequencies["level"] == "node"].iterrows()
        },
        "fine_codes_pooled": {row["code"]: int(row["n"]) for _, row in fine[fine["code"].isin(pooled)].iterrows()},
        "sequence_length": {
            label: {
                "min": int(block.min()),
                "median": float(block.median()),
                "max": int(block.max()),
                "sorted": sorted(int(v) for v in block),
            }
            for label, block in lengths.groupby(cases["class"])
        },
        "longest_case_share_of_events": round(float(lengths.max() / lengths.sum()), 4),
        "tied_events": int((coded["tied"] == "yes").sum()),
    }
    report_path = out_dir / "coding_report.json"
    tables.write_json(report, report_path)
    outputs += [codebook_path, report_path]
    return {"outputs": outputs, "report": report}
