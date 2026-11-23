"""Tests of table input and output, value maps and source hygiene."""
from __future__ import annotations

import pandas as pd
import pytest
from conftest import PIPELINE_DIR

from groupseq import ingest, tables

SPEC = [{"raw": "first", "name": "a"}, {"raw": "second", "name": "b"}]


def test_mapped_read_renames_and_keeps_text(tmp_path):
    path = tmp_path / "in.csv"
    path.write_bytes(b"\xef\xbb\xbffirst,second\n007,\n")
    frame = tables.read_mapped_csv(path, SPEC)
    assert list(frame.columns) == ["a", "b"]
    assert frame["a"].iloc[0] == "007" and frame["b"].iloc[0] == ""


def test_header_mismatch_is_rejected(tmp_path):
    path = tmp_path / "in.csv"
    path.write_bytes(b"second,first\n1,2\n")
    with pytest.raises(tables.ColumnMapError):
        tables.read_mapped_csv(path, SPEC)


def test_written_tables_are_byte_stable(tmp_path):
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", ""]})
    tables.write_csv(frame, tmp_path / "one.csv")
    tables.write_csv(frame, tmp_path / "two.csv")
    data = (tmp_path / "one.csv").read_bytes()
    assert data == (tmp_path / "two.csv").read_bytes()
    assert b"\r" not in data and not data.startswith(b"\xef\xbb\xbf")


def test_written_json_is_ascii_and_sorted(tmp_path):
    tables.write_json({"b": 1, "a": "\u00e9"}, tmp_path / "doc.json")
    data = (tmp_path / "doc.json").read_bytes()
    assert data.isascii() and data.index(b'"a"') < data.index(b'"b"')


def test_unmapped_label_is_rejected():
    with pytest.raises(tables.ValueMapError):
        tables.map_values(pd.Series(["x", "y"]), {"x": "X"}, "field")
    assert list(tables.map_values(pd.Series(["x", ""]), {"x": "X"}, "field", allow_blank=True)) == ["X", ""]


def test_non_integer_text_is_rejected():
    with pytest.raises(ValueError):
        tables.to_int(pd.Series(["1", "1.5"]), "field")


def test_stage_label_and_clock_conversion():
    assert list(ingest.parse_stage_label(pd.Series(["S0 x", "S6 y"]), "f")) == [0, 6]
    with pytest.raises(ValueError):
        ingest.parse_stage_label(pd.Series(["stage 1"]), "f")
    local = ingest.utc_to_local("2030-01-01T02:30:00.250000+00:00", 8)
    assert local.isoformat(sep=" ") == "2030-01-01 10:30:00.250000"
    assert ingest.utc_to_local("2030-01-01T02:30:00Z", 8).hour == 10
    assert ingest.utc_to_local(None, 8) is None


def test_source_and_configuration_files_are_ascii():
    files = list((PIPELINE_DIR / "src").rglob("*.py")) + list((PIPELINE_DIR / "tests").rglob("*.py"))
    files += list((PIPELINE_DIR / "config").glob("*.json")) + [PIPELINE_DIR / "run_stage.py"]
    assert files
    dirty = [p.name for p in files if not p.read_bytes().isascii()]
    assert dirty == []
