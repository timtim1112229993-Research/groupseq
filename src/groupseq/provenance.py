"""Run provenance: input digests, code state and per-run manifests.

Provenance holds no estimate, no table, no figure and no part of a record.
"""
from __future__ import annotations

import hashlib
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import tables
from .config import Settings


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_files(pipeline_dir: Path) -> list[Path]:
    """List the files whose content defines the behaviour of a run."""
    files = sorted((pipeline_dir / "src").rglob("*.py"))
    files += sorted(pipeline_dir.glob("run_*.py"))
    files.append(pipeline_dir / "config" / "pipeline.json")
    return [p for p in files if p.is_file()]


def code_state(pipeline_dir: Path) -> dict:
    """Digest every behaviour-defining file and combine the digests into one state value."""
    per_file = {p.relative_to(pipeline_dir).as_posix(): sha256_file(p) for p in code_files(pipeline_dir)}
    combined = hashlib.sha256()
    for name in sorted(per_file):
        combined.update(name.encode("ascii"))
        combined.update(per_file[name].encode("ascii"))
    return {"digest": combined.hexdigest(), "files": per_file}


def environment() -> dict:
    """Describe the interpreter and the numerical libraries, without host or user names."""
    import numpy
    import pandas

    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": {"numpy": numpy.__version__, "pandas": pandas.__version__},
    }


def digest_inputs(inputs: dict[str, Path]) -> dict[str, str]:
    """Digest each input under its neutral label."""
    return {label: sha256_file(path) for label, path in sorted(inputs.items())}


def digest_outputs(settings: Settings, outputs: list[Path]) -> dict[str, str]:
    """Digest each output under its path relative to the pipeline folder."""
    return {p.relative_to(settings.pipeline_dir).as_posix(): sha256_file(p) for p in sorted(outputs)}


def write_manifest(
    settings: Settings,
    stages: list[str],
    input_digests: dict,
    output_digests: dict,
    parameters: dict | None = None,
    state: dict | None = None,
) -> Path:
    """Write the manifest of one run and return its path.

    Extra parameters are settings only; an estimate never belongs here. A long
    run passes the code state it recorded when it started.
    """
    stamp = datetime.now(timezone.utc)
    cfg = settings.pipeline
    stated = {
        "timezone_offset_hours": cfg["timezone_offset_hours"],
        "scope": cfg["scope"],
        "pairing": cfg["pairing"],
        "pool_min_count": cfg["coding"]["pool_min_count"],
        "pooled_code": cfg["coding"]["pooled_code"],
    }
    stated.update(parameters or {})
    manifest = {
        "run_started_utc": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stages": stages,
        "code_state": state or code_state(settings.pipeline_dir),
        "environment": environment(),
        "seed": cfg["seed"],
        "parameters": stated,
        "input_digests": input_digests,
        "output_digests": output_digests,
        "command": [Path(sys.argv[0]).name] + sys.argv[1:],
    }
    path = settings.provenance_dir / "manifests" / f"manifest_{stamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    tables.write_json(manifest, path)
    return path


def write_column_map(settings: Settings) -> Path:
    """Write the column map that records were read through."""
    path = settings.provenance_dir / "column_map.json"
    tables.write_json(
        {"column_map": settings.pipeline["column_map"], "value_map": settings.pipeline["value_map"]}, path
    )
    return path
