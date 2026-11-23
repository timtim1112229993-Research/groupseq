"""Summary of the simulation study under the decision rules fixed in advance.

A procedure is valid in a cell when every criterion that applies there holds:
interval coverage inside the coverage band, and, in a null cell, the rate of
excluding zero (or of rejecting) inside the false-positive band. A test without
an interval is judged in null cells only. Nothing here changes a simulated
value; the module only reads the simulation tables and counts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import study, tables
from .config import Settings, stage_dir

FACTORS = ["codes", "groups", "heterogeneity", "length_profile", "excess"]
KEYS = ["procedure", "statistic"]
BORDERLINE_SE = 2.0


class SummaryError(ValueError):
    """Raised when the simulation tables are incomplete or inconsistent with the configuration."""


def _inside(values: pd.Series, band: list[float]) -> pd.Series:
    return values.between(band[0], band[1])


def _near_edge(values: pd.Series, errors: pd.Series, band: list[float]) -> pd.Series:
    """True when a rate lies within a few Monte Carlo standard errors of an edge of its band."""
    distance = np.minimum((values - band[0]).abs(), (values - band[1]).abs())
    return distance <= BORDERLINE_SE * errors


def judge(frame: pd.DataFrame, rules: dict, null: pd.Series) -> pd.DataFrame:
    """Flag each row against the criteria that apply to it."""
    out = frame.copy()
    null = pd.Series(np.asarray(null, dtype=bool), index=out.index)
    has_interval = out["coverage"].notna()
    coverage_ok = _inside(out["coverage"], rules["coverage"])
    false_positive_ok = _inside(out["reject"], rules["false_positive"])
    out["null_cell"] = null
    out["has_interval"] = has_interval
    out["coverage_ok"] = np.where(has_interval, coverage_ok.astype(float), np.nan)
    out["false_positive_ok"] = np.where(null, false_positive_ok.astype(float), np.nan)
    out["judged"] = has_interval | null
    out["valid"] = out["judged"] & (coverage_ok | ~has_interval) & (false_positive_ok | ~null)
    out["borderline"] = (has_interval & _near_edge(out["coverage"], out["coverage_se"], rules["coverage"])) | (
        null & _near_edge(out["reject"], out["reject_se"], rules["false_positive"])
    )
    return out


def _count_row(block: pd.DataFrame) -> dict:
    null, other = block[block["null_cell"]], block[~block["null_cell"]]
    return {
        "cells_judged": len(block),
        "cells_valid": int(block["valid"].sum()),
        "share_valid": round(float(block["valid"].mean()), 4),
        "cells_borderline": int(block["borderline"].sum()),
        "coverage_mean": block["coverage"].mean(),
        "coverage_min": block["coverage"].min(),
        "coverage_max": block["coverage"].max(),
        "coverage_cells_ok": int(np.nansum(block["coverage_ok"])),
        "null_cells": len(null),
        "false_positive_cells_ok": int(np.nansum(null["false_positive_ok"])),
        "false_positive_mean": null["reject"].mean(),
        "false_positive_max": null["reject"].max(),
        "power_mean": other["reject"].mean(),
        "width_median_mean": block["width_median"].mean(),
    }


def validity_counts(judged: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """Count valid cells for each procedure, overall and within each level of each factor."""
    used = judged[judged["judged"]]
    rows = []
    slices = [("all", "all", used)]
    for factor in factors:
        slices += [(factor, level, part) for level, part in used.groupby(factor, sort=True)]
    for factor, level, part in slices:
        for (procedure, statistic), block in part.groupby(KEYS, sort=True):
            rows.append({"factor": factor, "level": level, "procedure": procedure, "statistic": statistic, **_count_row(block)})
    return pd.DataFrame(rows)


def ranking(counts: pd.DataFrame) -> pd.DataFrame:
    """Order procedures by the number of valid cells, then by narrower intervals.

    Interval procedures and tests are ranked apart, because a test is judged in
    null cells only and the two counts are not comparable.
    """
    overall = counts[counts["factor"] == "all"].drop(columns=["factor", "level"]).copy()
    overall["kind"] = np.where(overall["coverage_mean"].notna(), "interval", "test")
    ordered = overall.sort_values(
        ["kind", "cells_valid", "width_median_mean", "procedure", "statistic"],
        ascending=[True, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ordered.insert(0, "rank", ordered.groupby("kind").cumcount() + 1)
    return ordered[["kind", "rank"] + [c for c in ordered.columns if c not in ("kind", "rank")]]


def check_complete(settings: Settings, procedures: pd.DataFrame, networks: pd.DataFrame) -> None:
    """Refuse to summarise a partial or reduced run."""
    sim = settings.pipeline["simulation"]
    expected = {arm: 0 for arm in study.ARMS}
    for cell in study.all_cells(sim):
        expected[cell["arm"]] += 1
    found = procedures.groupby("arm")["index"].nunique().to_dict()
    found["network"] = networks["index"].nunique()
    if found != expected:
        raise SummaryError(f"cells found {found} differ from the grid {expected}")
    sizes = set(procedures["replications"]) | set(networks["replications"])
    if sizes != {sim["replications"]}:
        raise SummaryError(f"replications {sorted(sizes)} differ from the configured {sim['replications']}")


def calibrated_tables(procedures: pd.DataFrame, estimates: pd.DataFrame, rules: dict) -> dict[str, pd.DataFrame]:
    """Coverage of every focal transition in the calibrated scenario; no true value is zero there."""
    drop = [c for c in FACTORS if c != "heterogeneity"]
    rows = procedures[procedures["arm"] == "calibrated"].drop(columns=drop, errors="ignore").reset_index(drop=True)
    judged = judge(rows, rules, pd.Series(False, index=rows.index))
    counts = validity_counts(judged, ["heterogeneity", "target"])
    kept = estimates[estimates["arm"] == "calibrated"].drop(columns=drop, errors="ignore").reset_index(drop=True)
    return {"calibrated_by_target": judged, "calibrated_counts": counts, "calibrated_estimates": kept}


def network_tables(networks: pd.DataFrame, rules: dict) -> dict[str, pd.DataFrame]:
    """False-positive rate in null cells and power in shifted cells for the network tests."""
    rows = networks.copy()
    null = ~rows["shift"].astype(bool)
    rows["null_cell"] = null
    rows["false_positive_ok"] = np.where(null, _inside(rows["reject"], rules["false_positive"]).astype(float), np.nan)
    rows["borderline"] = null & _near_edge(rows["reject"], rows["reject_se"], rules["false_positive"])
    out = []
    slices = [("all", "all", rows)]
    for factor in ("groups", "heterogeneity"):
        slices += [(factor, level, part) for level, part in rows.groupby(factor, sort=True)]
    for factor, level, part in slices:
        for procedure, block in part.groupby("procedure", sort=True):
            at_null, shifted = block[block["null_cell"]], block[~block["null_cell"]]
            out.append(
                {
                    "factor": factor,
                    "level": level,
                    "procedure": procedure,
                    "null_cells": len(at_null),
                    "null_cells_valid": int(np.nansum(at_null["false_positive_ok"])),
                    "null_cells_borderline": int(at_null["borderline"].sum()),
                    "false_positive_mean": at_null["reject"].mean(),
                    "false_positive_min": at_null["reject"].min(),
                    "false_positive_max": at_null["reject"].max(),
                    "power_mean": shifted["reject"].mean(),
                    "power_min": shifted["reject"].min(),
                    "power_max": shifted["reject"].max(),
                }
            )
    return {"network_by_cell": rows, "network_counts": pd.DataFrame(out)}


def estimate_table(estimates: pd.DataFrame) -> pd.DataFrame:
    """Bias and spread of both statistics in the controlled grid, by design factor."""
    rows = estimates[estimates["arm"] == "controlled"]
    return (
        rows.groupby(["statistic", "codes", "heterogeneity", "length_profile", "excess"], sort=True)
        .agg(
            cells=("index", "size"),
            truth_mean=("truth", "mean"),
            bias_mean=("bias", "mean"),
            bias_min=("bias", "min"),
            bias_max=("bias", "max"),
            bias_se_max=("bias_se", "max"),
            sd_mean=("sd", "mean"),
            estimable_rate_min=("estimable_rate", "min"),
            defined_min=("defined", "min"),
        )
        .reset_index()
    )


def run(settings: Settings) -> dict:
    """Apply the decision rules to the simulation tables and write the summary tables."""
    rules = settings.pipeline["decision"]
    folder = stage_dir(settings, "simulation")
    procedures = pd.read_csv(folder / "transition_procedures.csv")
    estimates = pd.read_csv(folder / "transition_estimates.csv")
    networks = pd.read_csv(folder / "network_procedures.csv")
    check_complete(settings, procedures, networks)

    controlled = procedures[procedures["arm"] == "controlled"].reset_index(drop=True)
    judged = judge(controlled, rules, controlled["excess"] == 0)
    counts = validity_counts(judged, FACTORS)
    written = {
        "validity_by_cell": judged,
        "validity_counts": counts,
        "procedure_ranking": ranking(counts),
        "controlled_estimates": estimate_table(estimates),
        **calibrated_tables(procedures, estimates, rules),
        **network_tables(networks, rules),
    }
    outputs = []
    for name, frame in written.items():
        path = stage_dir(settings, "summary") / f"{name}.csv"
        tables.write_csv(frame, path)
        outputs.append(path)
    undefined = procedures["replications"] - procedures["defined"]
    report = {
        "cells": {
            "controlled": int(controlled["index"].nunique()),
            "calibrated": int(written["calibrated_by_target"]["index"].nunique()),
            "network": int(networks["index"].nunique()),
        },
        "replications": int(procedures["replications"].max()),
        "undefined_replications_max": int(undefined.max()),
        "rows_with_undefined_replications": int((undefined > 0).sum()),
        "rules": rules,
        "borderline_se": BORDERLINE_SE,
    }
    path = stage_dir(settings, "summary") / "summary_report.json"
    tables.write_json(report, path)
    outputs.append(path)
    return {"outputs": outputs, "report": report}
