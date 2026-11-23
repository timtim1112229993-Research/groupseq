# groupseq

Inference on transitions between coded events when the events are nested in a
small number of groups whose sequences differ in length. The repository holds
the procedures, the simulation code that compares them, and the code that
applies them to exported interaction logs.

It holds source code, configuration, an environment specification, a generator
of synthetic demonstration inputs and the provenance of the recorded runs. It
holds no data and no result. Inputs, intermediate tables and outputs of every
kind stay in restricted custody and are never tracked; the ignore rules and
the guard tests enforce this.

## Requirements

Python 3.13 with the packages pinned in `requirements.txt`: NumPy and pandas
for the analysis, SciPy and pytest for the tests only.

    python -m pip install -r requirements.txt

## Layout

| Path | Content |
| --- | --- |
| `src/groupseq/inference.py` | Group-averaged and pooled transition excess; adjusted residual; event bootstrap; group bootstrap with percentile, BCa and studentised intervals; order permutation test |
| `src/groupseq/network.py` | Co-occurrence vectors of segments and groups; permutation tests of a two-arm contrast with labels reassigned among segments or among groups |
| `src/groupseq/generate.py` | Generating model: group rates, transition matrices, sequence lengths, simulated chains and true values |
| `src/groupseq/study.py` | Cells of the simulation study and their scoring |
| `src/groupseq/extension.py` | First addition: groups with dependence of opposing sign, and two corrected estimates |
| `src/groupseq/dominance.py` | Second addition: one long group that behaves differently, and the t interval |
| `src/groupseq/summary.py`, `extension_summary.py`, `dominance_summary.py` | Decision rules applied to a finished run |
| `src/groupseq/ingest.py`, `rebuild.py`, `coding.py` | Data stages: read exports through the column map, join bound log records into exchanges, assign codes from a written decision table |
| `src/groupseq/application.py` | The procedures applied to coded sequences, with sensitivity analyses |
| `src/groupseq/demo.py` | Generator of synthetic demonstration inputs |
| `src/groupseq/provenance.py`, `snapshot.py`, `tables.py`, `config.py` | Run records, a collected copy of result tables, deterministic reading and writing, settings |
| `config/pipeline.json` | Every analysis setting: column map, coding rules, grids, sizes, seeds, decision rules |
| `config/paths.example.json`, `config/paths.demo.json` | Path settings for restricted exports and for the demonstration |
| `provenance/` | Column map, input digests and one manifest per recorded run |
| `tests/` | Unit tests, release-boundary tests and reproducibility tests |

## Tests

    python -m pytest

Tests that need the restricted records or stored outputs are skipped where
those are absent, as in a fresh copy.

## Demonstration on synthetic inputs

    python make_demo.py --run

This writes synthetic exports with the structure of the platform exports and
runs the three data stages and the application on them. Inputs and outputs go
to `work/demo/`, which is ignored. The run takes well under a minute.

The synthetic inputs are drawn from a seeded generator with invented
parameters. They use no part of any real record, so the tables they yield
carry no meaning. Their purpose is to show that the code runs from raw exports
to result tables.

To use the command-line stages on the same inputs, copy
`config/paths.demo.json` to `config/paths.local.json` and run, for example:

    python run_stage.py all
    python run_stage.py application

Run records of these commands are written under `work/demo/provenance/`, not
to the tracked `provenance/` folder.

## Running on other exports

Copy `config/paths.example.json` to `config/paths.local.json` and set the
input root and the folder prefix of each class. The header of every export
must equal the column map in `config/pipeline.json`; a mismatch stops the run
and no value is reinterpreted silently. `config/paths.local.json` is ignored
and must not be tracked.

## Simulation study

    python run_simulation.py
    python run_extension.py
    python run_dominance.py

followed by

    python run_stage.py summary
    python run_stage.py extension_summary
    python run_stage.py dominance_summary

Grids, sizes and seeds are fixed in `config/pipeline.json`. At the configured
sizes the runs are long and are meant for a machine with many cores; set the
number of processes with `--workers`. The options `--replications`,
`--resamples`, `--permutations`, `--arms` and `--limit` give a reduced trial,
which is written to a separate folder and is never mixed with a full run.
`--resume` reuses finished cells of the same code state and size.

Each cell draws from its own random stream, derived from the seed and the
cell. Cells of the controlled grids and of the network grid depend on the
configuration and the seed alone, so they can be reproduced without any data.
The calibrated cells take a transition matrix and the sequence lengths from
the coded records. Run on the synthetic inputs they execute, but they do not
reproduce the recorded runs.

## Provenance

`provenance/` records how the reported runs were made and nothing about what
they found:

- `column_map.json`: the column map and value map through which records were
  read;
- `input_digests.json`: SHA-256 digests that fix the analysed set of records;
- `manifests/`: one manifest per run, stating the code state (a digest of
  every source file), the interpreter and library versions, the seed, the
  parameter values, and digests of inputs and outputs.

A manifest never holds an estimate, a table, a figure or any part of a
record, and a test fails if one does. The tests also check that the modules
and settings of each simulation run are unchanged since that run.

## Licence and citation

The code is released under the MIT licence; see `LICENSE`. Citation metadata
are in `CITATION.cff`.
