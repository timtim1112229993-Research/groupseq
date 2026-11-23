"""Tests of the results snapshot on synthetic tables only."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from groupseq import provenance, snapshot, tables
from groupseq.config import load_settings


def test_tables_keep_column_order_and_write_missing_values_as_null():
    frame = pd.DataFrame({"z": ["u", "v"], "a": [1, 2], "m": [0.25, np.nan], "flag": [True, False]})
    out = snapshot.as_table(frame)
    assert out["columns"] == ["z", "a", "m", "flag"]
    assert out["rows"] == [["u", 1, 0.25, True], ["v", 2, None, False]]


def test_a_copied_table_keeps_every_digit_of_its_source(tmp_path, monkeypatch):
    settings = dataclasses.replace(
        load_settings(), pipeline_dir=tmp_path, work_dir=tmp_path / "work", provenance_dir=tmp_path / "provenance"
    )
    values = [0.12357748905872215, 0.023155115329458784, 1 / 3, 2 / 3, 0.1 + 0.2, 1e-17, 123456.78901234567]
    frame = pd.DataFrame({"name": [f"r{i}" for i in range(len(values))], "value": values})
    path = settings.work_dir / "stage" / "numbers.csv"
    tables.write_csv(frame, path)
    name = path.relative_to(tmp_path).as_posix()
    manifest = settings.provenance_dir / "manifests" / "manifest_1.json"
    tables.write_json({"output_digests": {name: provenance.sha256_file(path)}}, manifest)
    monkeypatch.setattr(snapshot, "SOURCES", (("stage", "label", True),))
    monkeypatch.setattr(snapshot, "HEADLINES", (("Numbers", "label.numbers", ["name", "value"], None),))
    monkeypatch.setitem(snapshot.STAGE_FOLDERS, "stage", "stage")
    monkeypatch.setitem(snapshot.STAGE_FOLDERS, "snapshot", "snap")
    result = snapshot.run(settings)
    stored = tables.read_json(result["outputs"][0])
    assert [row[1] for row in stored["tables"]["label.numbers"]["rows"]] == values
    assert stored["sources"] == [{"file": name, "sha256": provenance.sha256_file(path), "manifest": "manifest_1.json"}]
    assert all(p.read_bytes().isascii() for p in result["outputs"])


def test_headline_tables_are_filtered_and_formatted():
    table = {"columns": ["factor", "name", "rate", "count", "ok"],
             "rows": [["all", "x", 0.93456, 12, True], ["groups", "y", 0.5, 3, False], ["all", "z", None, 7, True]]}
    lines = snapshot.render("Title", table, ["name", "rate", "count", "ok"], ("factor", "all"))
    assert lines[0] == "## Title" and lines[2] == "| name | rate | count | ok |"
    assert lines[4:6] == ["| x | 0.9346 | 12 | yes |", "| z |  | 7 | yes |"] and lines[-1] == ""
    assert len(snapshot.render("Title", table, ["name"], None)) == 4 + 3 + 1
    with pytest.raises(snapshot.SnapshotError):
        snapshot.render("Title", table, ["name", "absent"], None)


def test_a_source_must_be_described_by_a_manifest(tmp_path):
    settings = dataclasses.replace(load_settings(), pipeline_dir=tmp_path, provenance_dir=tmp_path / "provenance")
    path = tmp_path / "work" / "stage" / "table.csv"
    tables.write_csv(pd.DataFrame({"a": [1]}), path)
    digest = provenance.sha256_file(path)
    name = "work/stage/table.csv"
    for stamp, value in (("1", "0" * 64), ("2", digest), ("3", digest), ("4", "f" * 64)):
        tables.write_json({"output_digests": {name: value}}, settings.provenance_dir / "manifests" / f"manifest_{stamp}.json")
    recorded = snapshot.manifest_digests(settings)
    assert [m for m, _ in recorded] == [f"manifest_{k}.json" for k in "1234"]
    assert snapshot.describe_source(settings, path, recorded) == {"file": name, "sha256": digest, "manifest": "manifest_3.json"}
    with pytest.raises(snapshot.SnapshotError):
        snapshot.describe_source(settings, path, recorded[:1])


def test_every_headline_names_a_summary_or_application_table():
    labels = {label for _, label, keep in snapshot.SOURCES if keep}
    for title, key, columns, where in snapshot.HEADLINES:
        assert key.split(".")[0] in labels and columns and title
        assert where is None or len(where) == 2
