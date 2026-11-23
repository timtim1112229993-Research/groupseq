"""Settings, folder layout and discovery of input files."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import tables

PIPELINE_DIR = Path(__file__).resolve().parents[2]

STAGE_FOLDERS = {
    "ingest": "01_ingest",
    "events": "02_events",
    "coded": "03_coded",
    "simulation": "04_simulation",
    "simulation_trial": "04_simulation_trial",
    "summary": "04_simulation/summary",
    "extension": "04b_extension",
    "extension_trial": "04b_extension_trial",
    "extension_summary": "04b_extension/summary",
    "dominance": "04c_dominance",
    "dominance_trial": "04c_dominance_trial",
    "dominance_summary": "04c_dominance/summary",
    "application": "05_application",
    "snapshot": "06_snapshot",
}


class InputError(FileNotFoundError):
    """Raised when an expected input cannot be located unambiguously."""


@dataclass(frozen=True)
class Settings:
    pipeline: dict
    paths: dict
    pipeline_dir: Path
    input_root: Path
    work_dir: Path
    provenance_dir: Path


def load_settings(pipeline_dir: Path | None = None) -> Settings:
    """Load the analysis configuration and the local path configuration."""
    base = Path(pipeline_dir) if pipeline_dir else PIPELINE_DIR
    pipeline = tables.read_json(base / "config" / "pipeline.json")
    paths = tables.read_json(base / "config" / "paths.local.json")
    return Settings(
        pipeline=pipeline,
        paths=paths,
        pipeline_dir=base,
        input_root=(base / paths["input_root"]).resolve(),
        work_dir=base / paths["work_dir"],
        provenance_dir=base / paths["provenance_dir"],
    )


def stage_dir(settings: Settings, stage: str) -> Path:
    """Return the working folder of a stage."""
    return settings.work_dir / STAGE_FOLDERS[stage]


def _single(matches: list[Path], what: str) -> Path:
    if len(matches) != 1:
        raise InputError(f"expected exactly one {what}, found {len(matches)}")
    return matches[0]


def locate_inputs(settings: Settings) -> dict[str, Path]:
    """Find every raw input and return it under a neutral label such as 'A/agent_all'."""
    found: dict[str, Path] = {}
    optional = set(settings.paths.get("optional_kinds", []))
    for label, code in settings.paths["class_folders"].items():
        folders = [p for p in settings.input_root.iterdir() if p.is_dir() and p.name.startswith(code)]
        folder = _single(folders, f"folder for class {label}")
        for kind, template in settings.paths["file_patterns"].items():
            pattern = re.compile(template.replace("{code}", re.escape(code)))
            matches = sorted(p for p in folder.iterdir() if p.is_file() and pattern.match(p.name))
            if not matches and kind in optional:
                continue
            found[f"{label}/{kind}"] = _single(matches, f"{kind} file for class {label}")
    return found


def locate_inherited(settings: Settings) -> Path | None:
    """Return the inherited coded event table used for reconciliation, if present."""
    parts = settings.paths.get("inherited_event_table")
    if not parts:
        return None
    path = settings.input_root.joinpath(*parts)
    return path if path.is_file() else None
