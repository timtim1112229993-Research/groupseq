"""Summary of the extension under the decision rules of the first study.

Opposing dependence: every cell is a null cell for the mean excess. New
controlled grid: as in the first study. New calibrated cells: coverage only.
The comparison table sets each corrected estimate beside the uncorrected one
under the same interval procedure, which is the criterion fixed in advance.
"""
from __future__ import annotations

import pandas as pd

from . import extension, summary, tables
from .config import Settings, stage_dir

GROUP_STATISTICS = ["group", "group_loo", "group_split"]


def check_complete(settings: Settings, procedures: pd.DataFrame) -> None:
    """Refuse to summarise a partial or reduced run."""
    cfg = settings.pipeline
    expected = {arm: 0 for arm in extension.ARMS}
    for cell in extension.all_cells(cfg["simulation"]):
        expected[cell["arm"]] += 1
    found = procedures.groupby("arm")["index"].nunique().to_dict()
    if found != expected:
        raise summary.SummaryError(f"cells found {found} differ from the grid {expected}")
    sizes = set(procedures["replications"])
    if sizes != {cfg["extension"]["replications"]}:
        raise summary.SummaryError(f"replications {sorted(sizes)} differ from the configured size")


def comparison(counts_by_arm: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Valid cells of every group-level estimate under each interval procedure, arm by arm."""
    rows = []
    for arm, counts in counts_by_arm.items():
        overall = counts[(counts["factor"] == "all") & counts["statistic"].isin(GROUP_STATISTICS)]
        overall = overall[overall["coverage_mean"].notna()]
        for _, row in overall.iterrows():
            rows.append(
                {
                    "arm": arm,
                    "procedure": row["procedure"],
                    "statistic": row["statistic"],
                    "cells_judged": row["cells_judged"],
                    "cells_valid": row["cells_valid"],
                    "cells_borderline": row["cells_borderline"],
                    "coverage_mean": row["coverage_mean"],
                    "coverage_min": row["coverage_min"],
                    "false_positive_mean": row["false_positive_mean"],
                    "false_positive_max": row["false_positive_max"],
                    "power_mean": row["power_mean"],
                    "width_median_mean": row["width_median_mean"],
                }
            )
    out = pd.DataFrame(rows)
    order = {name: i for i, name in enumerate(GROUP_STATISTICS)}
    out["_order"] = out["statistic"].map(order)
    return out.sort_values(["arm", "procedure", "_order"], kind="mergesort").drop(columns="_order").reset_index(drop=True)


def estimate_table(estimates: pd.DataFrame) -> pd.DataFrame:
    """Bias, spread and root mean squared error of every estimate, by arm and design factor."""
    frame = estimates.copy()
    for column in ("heterogeneity", "excess", "target"):
        if column not in frame.columns:
            frame[column] = ""
    frame["excess"] = frame["excess"].fillna(-1.0)
    grouped = frame.groupby(["arm", "statistic", "heterogeneity", "excess", "target"], sort=True)
    out = grouped.agg(
        cells=("index", "size"),
        truth_mean=("truth", "mean"),
        bias_mean=("bias", "mean"),
        bias_min=("bias", "min"),
        bias_max=("bias", "max"),
        bias_se_max=("bias_se", "max"),
        sd_mean=("sd", "mean"),
        rmse_mean=("rmse", "mean"),
        contributing_mean=("contributing_mean", "mean"),
        defined_min=("defined", "min"),
    ).reset_index()
    out["excess"] = out["excess"].where(out["excess"] >= 0)
    return out


def run(settings: Settings) -> dict:
    """Apply the decision rules to the extension tables and write the summary tables."""
    rules = settings.pipeline["decision"]
    folder = stage_dir(settings, "extension")
    procedures = pd.read_csv(folder / "extension_procedures.csv")
    estimates = pd.read_csv(folder / "extension_estimates.csv")
    check_complete(settings, procedures)

    def part(arm: str) -> pd.DataFrame:
        rows = procedures[procedures["arm"] == arm].dropna(axis=1, how="all").reset_index(drop=True)
        for column in ("coverage", "coverage_se", "width_median"):
            if column not in rows.columns:
                rows[column] = float("nan")
        return rows

    opposing = part("opposing")
    judged_opposing = summary.judge(opposing, rules, pd.Series(True, index=opposing.index))
    controlled = part("controlled_new")
    judged_controlled = summary.judge(controlled, rules, controlled["excess"] == 0)
    calibrated = part("calibrated_new")
    judged_calibrated = summary.judge(calibrated, rules, pd.Series(False, index=calibrated.index))
    counts = {
        "opposing": summary.validity_counts(judged_opposing, ["codes", "groups", "heterogeneity", "length_profile"]),
        "controlled_new": summary.validity_counts(judged_controlled, summary.FACTORS),
        "calibrated_new": summary.validity_counts(judged_calibrated, ["heterogeneity", "target"]),
    }
    written = {
        "opposing_by_cell": judged_opposing,
        "opposing_counts": counts["opposing"],
        "controlled_new_by_cell": judged_controlled,
        "controlled_new_counts": counts["controlled_new"],
        "controlled_new_ranking": summary.ranking(counts["controlled_new"]),
        "calibrated_new_by_target": judged_calibrated,
        "calibrated_new_counts": counts["calibrated_new"],
        "correction_comparison": comparison(counts),
        "estimates_by_factor": estimate_table(estimates),
    }
    outputs = []
    for name, frame in written.items():
        path = stage_dir(settings, "extension_summary") / f"{name}.csv"
        tables.write_csv(frame, path)
        outputs.append(path)
    undefined = procedures["replications"] - procedures["defined"]
    report = {
        "cells": procedures.groupby("arm")["index"].nunique().to_dict(),
        "replications": int(procedures["replications"].max()),
        "undefined_replications_max": int(undefined.max()),
        "rows_with_undefined_replications": int((undefined > 0).sum()),
        "rules": rules,
        "seed": settings.pipeline["extension"]["seed"],
    }
    path = stage_dir(settings, "extension_summary") / "summary_report.json"
    tables.write_json(report, path)
    outputs.append(path)
    return {"outputs": outputs, "report": report}
