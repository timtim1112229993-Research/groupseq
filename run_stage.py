"""Command-line entry point: run one stage or all stages and write the run manifest.

Usage: python run_stage.py {ingest,events,coded,all,summary,extension_summary,dominance_summary,application,snapshot}

"all" runs the three data stages. Each summary stage applies the decision rules
to a finished simulation run, "application" analyses the coded sequences, and
"snapshot" collects the results of every finished stage; each is run on its own.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from groupseq import (  # noqa: E402
    application,
    coding,
    dominance_summary,
    extension_summary,
    ingest,
    provenance,
    rebuild,
    snapshot,
    summary,
    tables,
)
from groupseq.config import load_settings, locate_inputs, stage_dir  # noqa: E402

STAGES = {"ingest": ingest.run, "events": rebuild.run, "coded": coding.run}
LATER_STAGES = {
    "summary": summary.run,
    "extension_summary": extension_summary.run,
    "dominance_summary": dominance_summary.run,
    "application": application.run,
    "snapshot": snapshot.run,
}
ASCII_ONLY_STAGES = (
    "events", "coded", "summary", "extension_summary", "dominance_summary", "application", "snapshot",
)


def verify_ascii(paths: list[Path]) -> list[str]:
    """Return the files that hold a byte outside ASCII; an empty list means all are clean."""
    return [p.name for p in paths if not p.read_bytes().isascii()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pipeline stages.")
    parser.add_argument("stage", choices=list(STAGES) + ["all"] + list(LATER_STAGES))
    parser.add_argument("--quiet", action="store_true", help="do not print stage reports")
    args = parser.parse_args()
    settings = load_settings()
    names = list(STAGES) if args.stage == "all" else [args.stage]
    runners = {**STAGES, **LATER_STAGES}
    parameters = None
    if names[0] == "application":
        parameters = {"analysis": settings.pipeline["analysis"], "application": settings.pipeline["application"]}
    elif names[0] == "snapshot":
        parameters = None
    elif names[0] in LATER_STAGES:
        parameters = {"decision": settings.pipeline["decision"]}

    outputs: list[Path] = []
    for name in names:
        result = runners[name](settings)
        outputs += result["outputs"]
        if name in ASCII_ONLY_STAGES:
            dirty = verify_ascii(result["outputs"])
            if dirty:
                raise SystemExit(f"non-ASCII content in stage {name}: {dirty}")
        if not args.quiet:
            print(f"== stage {name}")
            print(json.dumps(result["report"], ensure_ascii=True, indent=1, sort_keys=True))

    inputs = locate_inputs(settings)
    input_digests = provenance.digest_inputs(inputs)
    tables.write_json(input_digests, settings.provenance_dir / "input_digests.json")
    column_map = provenance.write_column_map(settings)
    manifest = provenance.write_manifest(
        settings, names, input_digests, provenance.digest_outputs(settings, outputs), parameters
    )
    dirty = verify_ascii([settings.provenance_dir / "input_digests.json", column_map, manifest])
    if dirty:
        raise SystemExit(f"non-ASCII content in provenance: {dirty}")
    print(f"manifest written: {manifest.relative_to(settings.pipeline_dir).as_posix()}")
    print(f"outputs written: {len(outputs)} under {stage_dir(settings, names[0]).parent.name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
