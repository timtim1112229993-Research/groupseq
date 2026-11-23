"""Summary of the second extension under the decision rules of the first study.

A null cell is one in which no group has any sequential dependence: the excess
of ordinary groups is zero and the long group, if it differs at all, differs in
its code rates only. The procedure recommended for data with a dominant group
is ranked over the cells that hold one; the cells without one are a reference.
"""
from __future__ import annotations

import pandas as pd

from . import dominance, summary, tables
from .config import Settings, stage_dir

FACTORS = ["dominant", "codes", "groups", "heterogeneity", "excess"]
NULL_LEVELS = ("none", "rates")


def check_complete(settings: Settings, procedures: pd.DataFrame) -> None:
    """Refuse to summarise a partial or reduced run."""
    cfg = settings.pipeline
    expected = {arm: 0 for arm in dominance.ARMS}
    for cell in dominance.all_cells(cfg["dominance"], cfg["simulation"]):
        expected[cell["arm"]] += 1
    found = procedures.groupby("arm")["index"].nunique().to_dict()
    if found != expected:
        raise summary.SummaryError(f"cells found {found} differ from the grid {expected}")
    sizes = set(procedures["replications"])
    if sizes != {cfg["dominance"]["replications"]}:
        raise summary.SummaryError(f"replications {sorted(sizes)} differ from the configured size")


def null_cells(controlled: pd.DataFrame) -> pd.Series:
    """Cells in which no group has any sequential dependence."""
    return (controlled["excess"] == 0) & controlled["dominant"].isin(NULL_LEVELS)


def range_table(procedures: pd.DataFrame) -> pd.DataFrame:
    """Share of intervals with a limit outside the admissible range, and interval width, by dominance level."""
    rows = procedures[procedures["coverage"].notna()]
    return (
        rows.groupby(["arm", "dominant", "procedure", "statistic"], sort=True)
        .agg(
            rows=("index", "size"),
            out_of_range_mean=("out_of_range", "mean"),
            out_of_range_max=("out_of_range", "max"),
            width_mean=("width", "mean"),
            width_max=("width", "max"),
            width_median_mean=("width_median", "mean"),
        )
        .reset_index()
    )


def estimate_table(estimates: pd.DataFrame) -> pd.DataFrame:
    """Bias, spread and root mean squared error of every estimate, by arm and design factor."""
    frame = estimates.copy()
    frame["excess"] = frame["excess"].fillna(-1.0)
    grouped = frame.groupby(["arm", "statistic", "dominant", "heterogeneity", "excess", "target"], sort=True)
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
    """Apply the decision rules to the tables of the second extension and write the summary tables."""
    rules = settings.pipeline["decision"]
    folder = stage_dir(settings, "dominance")
    procedures = pd.read_csv(folder / "dominance_procedures.csv")
    estimates = pd.read_csv(folder / "dominance_estimates.csv")
    check_complete(settings, procedures)

    def part(arm: str) -> pd.DataFrame:
        return procedures[procedures["arm"] == arm].dropna(axis=1, how="all").reset_index(drop=True)

    controlled = part("controlled_dom")
    judged = summary.judge(controlled, rules, null_cells(controlled))
    with_dominant = judged[judged["dominant"] != "none"]
    reference = judged[judged["dominant"] == "none"]
    calibrated = part("calibrated_dom")
    judged_calibrated = summary.judge(calibrated, rules, pd.Series(False, index=calibrated.index))
    dominant_counts = summary.validity_counts(with_dominant, ["dominant", "groups", "heterogeneity", "excess"])
    reference_counts = summary.validity_counts(reference, ["groups", "heterogeneity", "excess"])
    written = {
        "controlled_by_cell": judged,
        "controlled_counts": summary.validity_counts(judged, FACTORS),
        "dominant_counts": dominant_counts,
        "dominant_ranking": summary.ranking(dominant_counts),
        "reference_counts": reference_counts,
        "reference_ranking": summary.ranking(reference_counts),
        "calibrated_by_target": judged_calibrated,
        "calibrated_counts": summary.validity_counts(judged_calibrated, ["heterogeneity", "target"]),
        "range_and_width": range_table(procedures),
        "estimates_by_factor": estimate_table(estimates),
    }
    outputs = []
    for name, frame in written.items():
        path = stage_dir(settings, "dominance_summary") / f"{name}.csv"
        tables.write_csv(frame, path)
        outputs.append(path)
    undefined = procedures["replications"] - procedures["defined"]
    report = {
        "cells": procedures.groupby("arm")["index"].nunique().to_dict(),
        "cells_with_dominant_group": int(with_dominant["index"].nunique()),
        "reference_cells": int(reference["index"].nunique()),
        "null_cells": int(judged.loc[judged["null_cell"], "index"].nunique()),
        "replications": int(procedures["replications"].max()),
        "undefined_replications_max": int(undefined.max()),
        "rows_with_undefined_replications": int((undefined > 0).sum()),
        "rules": rules,
        "seed": settings.pipeline["dominance"]["seed"],
    }
    path = stage_dir(settings, "dominance_summary") / "summary_report.json"
    tables.write_json(report, path)
    outputs.append(path)
    return {"outputs": outputs, "report": report}
