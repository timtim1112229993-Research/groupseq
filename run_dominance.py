"""Command-line entry point of the second extension of the simulation study.

Usage: python run_dominance.py [--workers N] [--arms ...] [--resume]

Without size options the run uses the size fixed in the configuration. Any
size option, arm subset or cell limit marks the run as a trial, which is
written to a separate folder. The code state is recorded when the run starts.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from groupseq import dominance, provenance, tables  # noqa: E402
from groupseq.config import load_settings, locate_inputs, stage_dir  # noqa: E402

SIZE_KEYS = ("replications", "resamples", "permutations")


def cell_name(cell: dict) -> str:
    return f"{cell['arm']}_{cell['index']:03d}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the second extension of the simulation study.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 10))
    parser.add_argument("--arms", nargs="+", choices=dominance.ARMS, default=list(dominance.ARMS))
    parser.add_argument("--limit", type=int, default=0, help="run only every k-th cell of each arm")
    parser.add_argument("--resume", action="store_true", help="reuse finished cells of the same code state and size")
    for key in SIZE_KEYS:
        parser.add_argument(f"--{key}", type=int, default=0)
    args = parser.parse_args()

    settings = load_settings()
    cfg = settings.pipeline
    state = provenance.code_state(settings.pipeline_dir)
    dom = copy.deepcopy(cfg["dominance"])
    trial = args.limit > 0 or set(args.arms) != set(dominance.ARMS)
    for key in SIZE_KEYS:
        if getattr(args, key):
            dom[key] = getattr(args, key)
            trial = True
    out_dir = stage_dir(settings, "dominance_trial" if trial else "dominance")
    cell_dir = out_dir / "cells"
    cell_dir.mkdir(parents=True, exist_ok=True)

    events = tables.read_csv(stage_dir(settings, "coded") / "coded_events.csv")
    events["order_in_case"] = events["order_in_case"].astype(int)
    calibration = dominance.calibration_from_events(
        events, cfg["families"]["order"], cfg["simulation"]["calibrated"]["smoothing"]
    )
    tables.write_json(calibration, out_dir / "calibration.json")

    signature = {"code_state": state["digest"], "seed": dom["seed"], **{key: dom[key] for key in SIZE_KEYS}}
    cells = [c for c in dominance.all_cells(dom, cfg["simulation"]) if c["arm"] in args.arms]
    if args.limit:
        cells = [c for c in cells if c["index"] % args.limit == 0]

    results, tasks = [], []
    for cell in sorted(cells, key=dominance.cell_weight, reverse=True):
        path = cell_dir / cell_name(cell)
        if args.resume and path.is_file():
            stored = json.loads(path.read_text(encoding="ascii"))
            if stored["signature"] == signature:
                results.append(stored["result"])
                continue
        tasks.append(
            {
                "cell": cell,
                "dominance": dom,
                "simulation": cfg["simulation"],
                "analysis": cfg["analysis"],
                "calibration": calibration,
            }
        )
    print(f"cells: {len(cells)} in all, {len(results)} reused, {len(tasks)} to run on {args.workers} processes", flush=True)

    started = time.perf_counter()
    if tasks:
        with Pool(processes=args.workers) as pool:
            for done, result in enumerate(pool.imap_unordered(dominance.run_cell, tasks, chunksize=1), start=1):
                cell = result["cell"]
                text = json.dumps({"signature": signature, "result": result}, ensure_ascii=True, sort_keys=True)
                (cell_dir / cell_name(cell)).write_bytes(text.encode("ascii"))
                results.append(result)
                if done % 10 == 0 or done == len(tasks):
                    print(f"done {done}/{len(tasks)}; elapsed {(time.perf_counter() - started) / 60:.1f} min", flush=True)

    outputs = [out_dir / "calibration.json"]
    for name, frame in dominance.assemble(results).items():
        if frame.empty:
            continue
        path = out_dir / f"{name}.csv"
        tables.write_csv(frame, path)
        outputs.append(path)
    report = {
        "trial": trial,
        "cells": len(cells),
        "cells_by_arm": {arm: sum(1 for c in cells if c["arm"] == arm) for arm in dominance.ARMS},
        "size": {key: dom[key] for key in SIZE_KEYS},
        "seed": dom["seed"],
    }
    tables.write_json(report, out_dir / "dominance_report.json")
    outputs.append(out_dir / "dominance_report.json")

    manifest = provenance.write_manifest(
        settings,
        ["dominance_trial" if trial else "dominance"],
        provenance.digest_inputs(locate_inputs(settings)),
        provenance.digest_outputs(settings, outputs),
        parameters={"analysis": cfg["analysis"], "simulation": cfg["simulation"], "dominance": dom, "cells_run": len(cells)},
        state=state,
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    print(f"manifest written: {manifest.relative_to(settings.pipeline_dir).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
