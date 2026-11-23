"""Stage 6: one frozen record of every summary-level result.

The snapshot copies every summary table and every application table into one
document, lists each source file with its digest and the manifest that
describes it, and renders the headline tables in a readable form. It computes
nothing: a value in the snapshot is a value of a stage output. Cell-level
tables stay in their stage folders and are listed in the index only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import provenance, tables
from .config import STAGE_FOLDERS, Settings, stage_dir  # noqa: F401  (the folder map is replaced in tests)

# Stage folder, label in the snapshot, and whether its tables are copied or only listed.
SOURCES = (
    ("simulation", "stage4_cells", False),
    ("summary", "stage4", True),
    ("extension", "extension_cells", False),
    ("extension_summary", "extension", True),
    ("dominance", "dominance_cells", False),
    ("dominance_summary", "dominance", True),
    ("application", "application", True),
)
DETAIL_SUFFIX = "_by_cell"
RANKING = [
    "kind", "rank", "procedure", "statistic", "cells_judged", "cells_valid", "cells_borderline", "coverage_mean",
    "false_positive_mean", "power_mean", "width_median_mean",
]
OVERALL = ["procedure", "statistic", "cells_judged", "cells_valid", "coverage_mean", "coverage_min", "coverage_max"]
# Title, table, columns shown, and an optional (column, value) filter.
HEADLINES = (
    ("Analysed sequences", "application.variant_sizes", ["variant", "groups", "events", "transitions"], None),
    ("Stage 4, controlled grid: ranking of procedures", "stage4.procedure_ranking", RANKING, None),
    ("Stage 4, calibrated scenario", "stage4.calibrated_counts", OVERALL, ("factor", "all")),
    ("Stage 4, network tests", "stage4.network_counts",
     ["procedure", "null_cells", "null_cells_valid", "false_positive_mean", "false_positive_max", "power_mean"],
     ("factor", "all")),
    ("First extension, controlled grid on a new seed: ranking", "extension.controlled_new_ranking", RANKING, None),
    ("First extension, calibrated scenario", "extension.calibrated_new_counts", OVERALL, ("factor", "all")),
    ("First extension, opposing dependence", "extension.opposing_counts",
     ["procedure", "statistic", "cells_judged", "cells_valid", "coverage_mean", "false_positive_mean",
      "false_positive_max"], ("factor", "all")),
    ("First extension, corrected estimates", "extension.correction_comparison",
     ["arm", "procedure", "statistic", "cells_judged", "cells_valid", "coverage_mean", "coverage_min",
      "false_positive_mean", "width_median_mean"], None),
    ("Second extension, cells with a dominant group: ranking", "dominance.dominant_ranking", RANKING, None),
    ("Second extension, reference cells: ranking", "dominance.reference_ranking", RANKING, None),
    ("Second extension, calibrated scenario with a dominant group", "dominance.calibrated_counts", OVERALL,
     ("factor", "all")),
    ("Second extension, limits outside the admissible range and interval width", "dominance.range_and_width",
     ["arm", "dominant", "procedure", "statistic", "out_of_range_mean", "out_of_range_max", "width_mean",
      "width_median_mean"], None),
    ("Application, focal transitions: estimates and intervals", "application.focal_estimates",
     ["transition", "statistic", "status", "estimate", "se", "contributing", "estimable", "percentile_low",
      "percentile_high", "bca_low", "bca_high", "studentised_low", "studentised_high", "event_low", "event_high"],
     ("variant", "main")),
    ("Application, focal transitions: added t intervals", "application.focal_added_intervals",
     ["transition", "statistic", "estimate", "se", "contributing", "low", "high"], ("variant", "main")),
    ("Application, focal transitions: tests", "application.focal_tests",
     ["transition", "test", "statistic", "value", "p", "p_holm"], ("variant", "main")),
    ("Application, network contrast", "application.network_tests", ["test", "vectors", "units", "statistic", "p"], None),
)


class SnapshotError(ValueError):
    """Raised when a stage output is missing or is not described by any manifest."""


def as_table(frame: pd.DataFrame) -> dict:
    """A table as column names and rows, with missing values written as null."""


    def plain(value):
        if pd.isna(value):
            return None
        return value.item() if isinstance(value, np.generic) else value

    rows = [[plain(value) for value in record] for record in frame.itertuples(index=False, name=None)]
    return {"columns": list(frame.columns), "rows": rows}


def manifest_digests(settings: Settings) -> list[tuple[str, dict]]:
    """Name and output digests of every manifest, oldest first."""
    folder = settings.provenance_dir / "manifests"
    return [(p.name, tables.read_json(p)["output_digests"]) for p in sorted(folder.glob("manifest_*.json"))]


def describe_source(settings: Settings, path, recorded: list[tuple[str, dict]]) -> dict:
    """Digest of one stage output and the latest manifest that states the same digest."""
    name = path.relative_to(settings.pipeline_dir).as_posix()
    digest = provenance.sha256_file(path)
    matching = [manifest for manifest, digests in recorded if digests.get(name) == digest]
    if not matching:
        raise SnapshotError(f"{name} is not described by any manifest")
    return {"file": name, "sha256": digest, "manifest": matching[-1]}


def cell_text(value) -> str:
    """A value as it is shown in the readable snapshot."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render(title: str, table: dict, columns: list[str], where) -> list[str]:
    """One headline table as lines of text."""
    missing = [c for c in columns + ([where[0]] if where else []) if c not in table["columns"]]
    if missing:
        raise SnapshotError(f"{title}: columns {missing} are absent")
    position = {name: i for i, name in enumerate(table["columns"])}
    rows = table["rows"]
    if where:
        rows = [row for row in rows if row[position[where[0]]] == where[1]]
    lines = [f"## {title}", "", "| " + " | ".join(columns) + " |", "|" + "|".join(" --- " for _ in columns) + "|"]
    lines += ["| " + " | ".join(cell_text(row[position[c]]) for c in columns) + " |" for row in rows]
    return lines + [""]


def run(settings: Settings) -> dict:
    """Write the snapshot of all results and its readable form."""
    recorded = manifest_digests(settings)
    sources, reports, copied = [], {}, {}
    for stage, label, keep in SOURCES:
        folder = stage_dir(settings, stage)
        files = sorted(folder.glob("*.csv")) + sorted(folder.glob("*.json"))
        if not files:
            raise SnapshotError(f"stage {stage} has no output")
        for path in files:
            sources.append(describe_source(settings, path, recorded))
            if path.suffix == ".json":
                reports[f"{label}.{path.stem}"] = tables.read_json(path)
            elif keep and not path.stem.endswith(DETAIL_SUFFIX):
                copied[f"{label}.{path.stem}"] = as_table(pd.read_csv(path, float_precision="round_trip"))
    for key in [k for k in reports if k.endswith(".calibration")]:
        del reports[key]
    snapshot = {"sources": sources, "reports": reports, "tables": copied}
    folder = stage_dir(settings, "snapshot")
    document = folder / "results_snapshot.json"
    tables.write_json(snapshot, document)

    lines = [
        "# Results snapshot",
        "",
        "Internal record, not for release. Every value is copied from a stage output; the index at the end of the",
        "full snapshot names the digest and the manifest of each source file. Floats are shown to four decimals.",
        "",
    ]
    for title, key, columns, where in HEADLINES:
        if key not in copied:
            raise SnapshotError(f"table {key} is absent")
        lines += render(title, copied[key], columns, where)
    readable = folder / "results_snapshot.md"
    readable.write_bytes(("\n".join(lines) + "\n").encode("ascii"))

    report = {
        "source_files": len(sources),
        "tables_copied": len(copied),
        "reports_copied": len(reports),
        "headline_tables": len(HEADLINES),
        "manifests_cited": len({s["manifest"] for s in sources}),
    }
    path = folder / "snapshot_report.json"
    tables.write_json(report, path)
    return {"outputs": [document, readable, path], "report": report}
