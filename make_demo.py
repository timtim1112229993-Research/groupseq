"""Write synthetic demonstration inputs and, on request, run the analysis on them.

Usage: python make_demo.py [--seed N] [--run] [--resamples N] [--permutations N]

The inputs, the stage outputs and nothing else are written under the folders
named in config/paths.demo.json, which lie inside the ignored working folder.
No run record is written and the stored provenance is not touched. The numbers
produced from synthetic inputs carry no meaning; the run shows that the code
works from raw exports to result tables.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from groupseq import demo  # noqa: E402

REPORT_KEYS = {
    "ingest": ["tables"],
    "events": ["records_read", "records_excluded", "records_in_scope", "exchanges", "cases"],
    "coded": ["events", "cases", "cases_by_class", "families"],
    "application": ["groups", "events", "transitions", "segments", "arms", "sizes"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic demonstration inputs.")
    parser.add_argument("--seed", type=int, default=demo.DEFAULT_SEED)
    parser.add_argument("--run", action="store_true", help="run the data stages and the application on the inputs")
    parser.add_argument("--resamples", type=int, default=0, help="bootstrap resamples of the application, if not the configured number")
    parser.add_argument("--permutations", type=int, default=0, help="permutations of the application, if not the configured number")
    args = parser.parse_args()
    settings = demo.demo_settings()
    written = demo.write_inputs(settings, args.seed)
    print(f"inputs written: {len(written)} files under {settings.paths['input_root']}/")
    if not args.run:
        return 0
    pipeline = json.loads(json.dumps(settings.pipeline))
    if args.resamples:
        pipeline["application"]["resamples"] = args.resamples
    if args.permutations:
        pipeline["application"]["permutations"] = args.permutations
    reports = demo.run_stages(dataclasses.replace(settings, pipeline=pipeline))
    for stage, keys in REPORT_KEYS.items():
        print(f"== stage {stage}")
        print(json.dumps({key: reports[stage][key] for key in keys}, ensure_ascii=True, indent=1, sort_keys=True))
    print(f"outputs written under {settings.paths['work_dir']}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
