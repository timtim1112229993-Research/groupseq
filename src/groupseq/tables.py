"""Deterministic reading and writing of tables and JSON documents."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ISO_SECONDS = "%Y-%m-%d %H:%M:%S"


class ColumnMapError(ValueError):
    """Raised when a file header does not match the written column map."""


class ValueMapError(ValueError):
    """Raised when a field holds a label that the value map does not cover."""


def read_mapped_csv(path: Path, spec: list[dict]) -> pd.DataFrame:
    """Read a raw export through its column map and return English column names.

    The header must equal the mapped raw header exactly, in the same order.
    Every cell is kept as text so that no value is silently reinterpreted.
    """
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    expected = [item["raw"] for item in spec]
    found = list(frame.columns)
    if found != expected:
        raise ColumnMapError(
            f"header mismatch: expected {[ascii(x) for x in expected]}, found {[ascii(x) for x in found]}"
        )
    frame.columns = [item["name"] for item in spec]
    return frame


def map_values(series: pd.Series, mapping: dict, field: str, allow_blank: bool = False) -> pd.Series:
    """Translate raw labels to analysis labels and fail on any unmapped label."""
    lookup = dict(mapping)
    if allow_blank:
        lookup[""] = ""
    unknown = sorted(set(series.unique()) - set(lookup))
    if unknown:
        raise ValueMapError(f"unmapped labels in {field}: {[ascii(x) for x in unknown]}")
    return series.map(lookup)


def require_allowed(series: pd.Series, allowed: list[str], field: str) -> None:
    """Fail when a field holds a value outside its declared set."""
    unknown = sorted(set(series.unique()) - set(allowed))
    if unknown:
        raise ValueMapError(f"unexpected values in {field}: {[ascii(x) for x in unknown]}")


def to_int(series: pd.Series, field: str) -> pd.Series:
    """Convert a text column of whole numbers and fail on anything else."""
    bad = series[~series.str.fullmatch(r"-?\d+")]
    if len(bad):
        raise ValueError(f"non-integer values in {field}: {[ascii(x) for x in bad.unique()[:5]]}")
    return series.astype(int)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a table as UTF-8 with line feeds and no index, so that bytes are reproducible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = frame.to_csv(index=False, lineterminator="\n")
    path.write_bytes(text.encode("utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    """Read a pipeline table back with every cell as text."""
    return pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)


def write_json(obj, path: Path) -> None:
    """Write a JSON document in ASCII with sorted keys, so that bytes are reproducible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    path.write_bytes(text.encode("ascii"))


def read_json(path: Path):
    """Read a JSON document, tolerating a byte order mark."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def iso(series: pd.Series) -> pd.Series:
    """Format timestamps to whole seconds; missing values become empty text."""
    return series.dt.strftime(ISO_SECONDS).fillna("")
