"""Guard tests on the release boundary and on reproducibility.

Release boundary: inputs, intermediate tables and outputs are never tracked;
run provenance holds digests and settings only; no tracked file names a
manuscript, a working document or an automated tool. Reproducibility: the
modules that produced each simulation are unchanged since their run, stored
outputs match their manifests, and stored work can be produced again.

Tests that need the local working folder or the raw inputs are skipped where
those are absent, as in a public copy of the code.
"""
from __future__ import annotations

import dataclasses
import fnmatch
import json
import re
import subprocess
from pathlib import Path

import pytest

from groupseq import application, coding, dominance, extension, ingest, provenance, rebuild, study, tables
from groupseq.config import InputError, load_settings, locate_inputs

PIPELINE_DIR = Path(__file__).resolve().parents[1]
MANIFEST_DIR = PIPELINE_DIR / "provenance" / "manifests"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_IGNORES = {
    "work/", "config/paths.local.json", "*.csv", "*.tsv", "*.xlsx", "*.xls", "*.parquet", "*.feather", "*.pkl",
    "*.pickle", "*.npy", "*.npz", "*.png", "*.jpg", "*.jpeg", "*.svg", "*.pdf", "*.docx", "*.log", "__pycache__/",
}
TRACKABLE_SUFFIXES = {".py", ".json", ".ini", ".txt", ".md", ".cff", ".toml", ".cfg", ".yml", ".yaml"}
TRACKABLE_NAMES = {".gitignore", ".gitattributes", "LICENSE"}
MANIFEST_KEYS = {
    "code_state", "command", "environment", "input_digests", "output_digests", "parameters", "run_started_utc", "seed",
    "stages",
}
PARAMETER_KEYS = {
    "timezone_offset_hours", "scope", "pairing", "pool_min_count", "pooled_code", "analysis", "simulation", "extension",
    "dominance", "decision", "application", "cells_run",
}
# Fragments are joined at run time so that this file does not contain the names it looks for.
FORBIDDEN_TEXT = [
    "cur" + "sor", "co" + "pilot", "clau" + "de", "anth" + "ropic", "open" + "ai", "chat" + "gpt", "pap" + "er1",
    "pap" + "er 1", "pap" + "er 4", "pap" + "er 5", "analysis" + "_plan", "writing" + "_rules", "cluster" + "-aware",
    "gg5" + "iag", "gn6" + "jzj",
]
FROZEN = {
    "simulation": ["generate.py", "inference.py", "network.py", "study.py", "tables.py"],
    "extension": ["generate.py", "inference.py", "study.py", "tables.py", "extension.py"],
    "dominance": ["generate.py", "inference.py", "study.py", "tables.py", "extension.py", "dominance.py"],
}
RUNNERS = {"simulation": "run_simulation.py", "extension": "run_extension.py", "dominance": "run_dominance.py"}


# ---------------------------------------------------------------- helpers


def ignore_patterns() -> list[str]:
    lines = (PIPELINE_DIR / ".gitignore").read_text(encoding="ascii").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def is_ignored(relative: str, patterns: list[str]) -> bool:
    """A small reading of the ignore rules: folder names, anchored paths and file-name patterns."""
    parts = relative.split("/")
    for pattern in patterns:
        if pattern.endswith("/"):
            if pattern[:-1] in parts[:-1]:
                return True
        elif "/" in pattern:
            if fnmatch.fnmatch(relative, pattern):
                return True
        elif fnmatch.fnmatch(parts[-1], pattern):
            return True
    return False


def would_be_tracked() -> list[str]:
    patterns = ignore_patterns()
    files = []
    for path in sorted(PIPELINE_DIR.rglob("*")):
        relative = path.relative_to(PIPELINE_DIR).as_posix()
        if path.is_file() and not relative.startswith(".git/") and not is_ignored(relative, patterns):
            files.append(relative)
    return files


def boundary_faults(relative: str) -> list[str]:
    """Reasons why a file must not be tracked; an empty list means it may be."""
    path = Path(relative)
    faults = []
    if path.parts[0] == "work":
        faults.append("stage output")
    if relative == "config/paths.local.json":
        faults.append("local paths")
    if path.suffix not in TRACKABLE_SUFFIXES and path.name not in TRACKABLE_NAMES:
        faults.append("file type")
    if path.suffix == ".json" and path.parts[0] not in ("config", "provenance"):
        faults.append("data file outside configuration and provenance")
    return faults


def manifests() -> list[tuple[str, dict]]:
    return [(p.name, json.loads(p.read_text(encoding="ascii"))) for p in sorted(MANIFEST_DIR.glob("manifest_*.json"))]


def latest_manifest(stage: str) -> dict:
    found = [m for _, m in manifests() if m["stages"] == [stage]]
    if not found:
        pytest.skip(f"no manifest of stage {stage}")
    return found[-1]


def recorded_outputs() -> dict[str, str]:
    """The digest of every output, as stated by the latest manifest that lists it."""
    out = {}
    for _, manifest in manifests():
        out.update(manifest["output_digests"])
    return out


def plain(result: dict) -> str:
    kept = {k: v for k, v in result.items() if k != "seconds"}
    return json.dumps(json.loads(json.dumps(kept)), sort_keys=True)


# ------------------------------------------------------- release boundary


def test_ignore_rules_cover_every_data_and_output_location():
    assert REQUIRED_IGNORES <= set(ignore_patterns())


def test_files_are_stored_exactly_as_written():
    """Recorded digests of source files can be checked in a copy only if no line ending is converted."""
    lines = (PIPELINE_DIR / ".gitattributes").read_text(encoding="ascii").splitlines()
    rules = [line.split() for line in lines if line.strip() and not line.startswith("#")]
    assert rules == [["*", "-text"]]


def test_ignore_reader_follows_the_rules_it_is_given():
    patterns = ["work/", "config/paths.local.json", "*.csv", "__pycache__/"]
    assert is_ignored("work/05_application/focal_estimates.csv", patterns)
    assert is_ignored("work/04_simulation/cells/controlled_000.json", patterns)
    assert is_ignored("config/paths.local.json", patterns) and not is_ignored("config/pipeline.json", patterns)
    assert is_ignored("tests/data/table.csv", patterns) and is_ignored("src/groupseq/__pycache__/x.pyc", patterns)
    assert not is_ignored("src/groupseq/study.py", patterns) and not is_ignored("workbench.py", patterns)


def test_boundary_check_names_each_kind_of_forbidden_file():
    assert boundary_faults("work/03_coded/coded_events.csv")
    assert boundary_faults("config/paths.local.json") and boundary_faults("figures/plot.png")
    assert boundary_faults("results/estimates.json") and boundary_faults("notes/table.docx")
    for fine in ("src/groupseq/study.py", "config/pipeline.json", "provenance/input_digests.json", ".gitignore", "pytest.ini"):
        assert boundary_faults(fine) == []


def test_no_file_that_would_be_tracked_crosses_the_boundary():
    tracked = would_be_tracked()
    assert tracked, "the pipeline folder should hold code"
    faults = {name: boundary_faults(name) for name in tracked if boundary_faults(name)}
    assert faults == {}


def test_files_tracked_by_a_repository_respect_the_boundary():
    probe = subprocess.run(["git", "ls-files"], cwd=PIPELINE_DIR, capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("the pipeline folder is not under version control")
    tracked = [line for line in probe.stdout.splitlines() if line]
    faults = {name: boundary_faults(name) for name in tracked if boundary_faults(name)}
    assert faults == {}


def test_commit_history_names_no_manuscript_document_or_tool():
    """Authors, committers and messages of every commit are held to the same rule as tracked text."""
    probe = subprocess.run(
        ["git", "log", "--all", "--format=%an%n%ae%n%cn%n%ce%n%B"], cwd=PIPELINE_DIR, capture_output=True
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        pytest.skip("the pipeline folder has no commit history")
    assert probe.stdout.isascii(), "the commit history holds a byte outside ASCII"
    text = probe.stdout.decode("ascii").lower()
    found = [word for word in FORBIDDEN_TEXT if word in text]
    assert found == [], f"the commit history names {found}"


def test_tracked_text_is_plain_and_names_no_manuscript_document_or_tool():
    for relative in would_be_tracked():
        raw = (PIPELINE_DIR / relative).read_bytes()
        assert raw.isascii(), f"{relative} holds a byte outside ASCII"
        text = raw.decode("ascii").lower()
        found = [word for word in FORBIDDEN_TEXT if word in text]
        assert found == [], f"{relative} names {found}"


def test_provenance_holds_digests_and_settings_only():
    folder = PIPELINE_DIR / "provenance"
    files = [p for p in folder.rglob("*") if p.is_file()]
    assert files and all(p.suffix == ".json" for p in files)
    names = {p.relative_to(folder).as_posix() for p in files if p.parent == folder}
    assert names <= {"column_map.json", "input_digests.json"}
    digests = json.loads((folder / "input_digests.json").read_text(encoding="ascii"))
    assert all(HEX64.match(v) for v in digests.values())
    assert set(json.loads((folder / "column_map.json").read_text(encoding="ascii"))) == {"column_map", "value_map"}
    for name, manifest in manifests():
        assert set(manifest) == MANIFEST_KEYS, name
        assert set(manifest["parameters"]) <= PARAMETER_KEYS, name
        assert set(manifest["code_state"]) == {"digest", "files"}, name
        for block in (manifest["input_digests"], manifest["output_digests"], manifest["code_state"]["files"]):
            assert all(isinstance(v, str) and HEX64.match(v) for v in block.values()), name
        assert (MANIFEST_DIR / name).stat().st_size < 100_000, name


# -------------------------------------------------------- reproducibility


@pytest.mark.parametrize("stage", ["simulation", "extension", "dominance"])
def test_modules_and_settings_of_each_simulation_are_unchanged_since_its_run(stage, pipeline_config):
    manifest = latest_manifest(stage)
    recorded = manifest["code_state"]["files"]
    for name in FROZEN[stage]:
        key = f"src/groupseq/{name}"
        assert provenance.sha256_file(PIPELINE_DIR / key) == recorded[key], f"{key} changed after the {stage} run"
    assert provenance.sha256_file(PIPELINE_DIR / RUNNERS[stage]) == recorded[RUNNERS[stage]]
    blocks = {"simulation": ["analysis", "simulation"], "extension": ["analysis", "simulation", "extension"],
              "dominance": ["analysis", "simulation", "dominance"]}[stage]
    for block in blocks:
        assert manifest["parameters"][block] == pipeline_config[block], f"{block} settings changed after the {stage} run"


def test_stored_outputs_match_the_manifests_that_describe_them():
    recorded = recorded_outputs()
    present = {name: digest for name, digest in recorded.items() if (PIPELINE_DIR / name).is_file()}
    if not present:
        pytest.skip("no stage output is stored here")
    changed = [name for name, digest in present.items() if provenance.sha256_file(PIPELINE_DIR / name) != digest]
    assert changed == []
    missing = sorted(set(recorded) - set(present))
    assert missing == [], "outputs named by a manifest are absent"


@pytest.mark.slow
def test_data_stages_and_application_reproduce_from_the_raw_inputs(tmp_path):
    settings = load_settings()
    try:
        locate_inputs(settings)
    except (InputError, FileNotFoundError):
        pytest.skip("the raw inputs are not available here")
    recorded = recorded_outputs()
    scratch = dataclasses.replace(settings, work_dir=tmp_path / "work")
    checked = 0
    for run in (ingest.run, rebuild.run, coding.run, application.run):
        for path in run(scratch)["outputs"]:
            name = "work/" + path.relative_to(scratch.work_dir).as_posix()
            assert provenance.sha256_file(path) == recorded[name], f"{name} is not reproduced"
            checked += 1
    assert checked >= 20


@pytest.mark.slow
def test_stored_simulation_cells_rerun_identically(pipeline_config):
    work = load_settings().work_dir
    cfg = pipeline_config

    def stored(folder: str, arm: str, index: int) -> dict:
        path = work / folder / "cells" / f"{arm}_{index:03d}.json"
        if not path.is_file():
            pytest.skip("no simulation cell is stored here")
        return json.loads(path.read_text(encoding="ascii"))["result"]

    def calibration(folder: str) -> dict:
        return tables.read_json(work / folder / "calibration.json")

    first = {c["arm"] + str(c["index"]): c for c in study.all_cells(cfg["simulation"])}
    base = {"simulation": cfg["simulation"], "analysis": cfg["analysis"], "seed": cfg["seed"],
            "nodes": len(cfg["network"]["node_order"])}
    for arm, index in (("network", 0), ("controlled", 0)):
        if arm == "controlled":
            base["calibration"] = calibration("04_simulation")
        fresh = study.run_cell({**base, "cell": first[arm + str(index)]})
        assert plain(fresh) == plain(stored("04_simulation", arm, index))

    second = {c["arm"] + str(c["index"]): c for c in extension.all_cells(cfg["simulation"])}
    task = {"cell": second["controlled_new0"], "extension": cfg["extension"], "simulation": cfg["simulation"],
            "analysis": cfg["analysis"], "calibration": calibration("04b_extension")}
    assert plain(extension.run_cell(task)) == plain(stored("04b_extension", "controlled_new", 0))

    third = {c["arm"] + str(c["index"]): c for c in dominance.all_cells(cfg["dominance"], cfg["simulation"])}
    task = {"cell": third["controlled_dom1"], "dominance": cfg["dominance"], "simulation": cfg["simulation"],
            "analysis": cfg["analysis"], "calibration": calibration("04c_dominance")}
    assert plain(dominance.run_cell(task)) == plain(stored("04c_dominance", "controlled_dom", 1))
