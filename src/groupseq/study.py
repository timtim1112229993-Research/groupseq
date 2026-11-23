"""Simulation study: cell grids, per-cell runs and scoring.

Each cell owns a random stream derived from the master seed, the arm and the
cell's position in its grid, so a cell's result does not depend on the order
in which cells are run or on the number of worker processes.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from . import generate, inference, network

ARMS = ("controlled", "calibrated", "network")
INTERVAL_PROCEDURES = ("T3_percentile", "T4_bca", "T5_studentised")
BOOT_KEYS = {"T3_percentile": "percentile", "T4_bca": "bca", "T5_studentised": "studentised"}
ALPHA = 0.05
Z_CRITICAL = 1.959963984540054


def controlled_cells(spec: dict) -> list[dict]:
    """Every cell of the controlled grid, in a fixed order."""
    cells = []
    grid = itertools.product(
        spec["codes"], spec["groups"], spec["concentration"], spec["length_profiles"], spec["excess"]
    )
    for index, (codes, groups, level, profile, excess) in enumerate(grid):
        cells.append(
            {"arm": "controlled", "index": index, "codes": codes, "groups": groups, "heterogeneity": level,
             "length_profile": profile, "excess": excess}
        )
    return cells


def calibrated_cells(spec: dict) -> list[dict]:
    """Every cell of the calibrated scenario."""
    return [{"arm": "calibrated", "index": i, "heterogeneity": level} for i, level in enumerate(spec["concentration"])]


def network_cells(spec: dict) -> list[dict]:
    """Every cell of the network grid."""
    grid = itertools.product(spec["groups"], spec["group_sd"], (False, True))
    return [
        {"arm": "network", "index": i, "groups": groups, "heterogeneity": level, "shift": shift}
        for i, (groups, level, shift) in enumerate(grid)
    ]


def all_cells(sim: dict) -> list[dict]:
    """Every cell of the study."""
    return controlled_cells(sim["controlled"]) + calibrated_cells(sim["calibrated"]) + network_cells(sim["network"])


def cell_streams(seed: int, cell: dict, count: int) -> list[np.random.Generator]:
    """Independent random streams of one cell."""
    root = np.random.SeedSequence(seed, spawn_key=(ARMS.index(cell["arm"]), cell["index"]))
    return [np.random.default_rng(child) for child in root.spawn(count)]


def cell_weight(cell: dict) -> float:
    """Rough relative cost of a cell, used only to order the work."""
    if cell["arm"] == "network":
        return 0.02 * cell["groups"]
    if cell["arm"] == "calibrated":
        return 60.0
    return cell["groups"] * (1.6 if cell["length_profile"] == "extreme" else 1.0)


class Tally:
    """Collects the per-replication results of one target transition."""

    def __init__(self, replications: int):
        nan = lambda: np.full(replications, np.nan)  # noqa: E731
        self.estimate = {s: nan() for s in inference.STATISTICS}
        self.contributing = nan()
        self.low = {}
        self.high = {}
        for procedure in INTERVAL_PROCEDURES:
            for statistic in inference.STATISTICS:
                self.low[procedure, statistic], self.high[procedure, statistic] = nan(), nan()
        self.low["T2_event_bootstrap", "pooled"], self.high["T2_event_bootstrap", "pooled"] = nan(), nan()
        self.residual = nan()
        self.order_p = {s: nan() for s in inference.STATISTICS}

    def record(self, r: int, boot: dict, event: tuple, residual: float, order_p: dict) -> None:
        for statistic in inference.STATISTICS:
            self.estimate[statistic][r] = boot[statistic]["estimate"]
            self.order_p[statistic][r] = order_p[statistic]
            for procedure in INTERVAL_PROCEDURES:
                low, high = boot[statistic][BOOT_KEYS[procedure]]
                self.low[procedure, statistic][r], self.high[procedure, statistic][r] = low, high
        self.contributing[r] = boot["group"]["contributing"]
        self.low["T2_event_bootstrap", "pooled"][r], self.high["T2_event_bootstrap", "pooled"][r] = event
        self.residual[r] = residual


def _rate(flags: np.ndarray) -> tuple[float, float]:
    """A proportion and its Monte Carlo standard error."""
    if flags.size == 0:
        return float("nan"), float("nan")
    rate = float(flags.mean())
    return rate, float(np.sqrt(rate * (1 - rate) / flags.size))


def score(tally: Tally, truth: dict, min_groups: int) -> tuple[list[dict], list[dict]]:
    """Turn the collected replications into procedure rows and estimate rows."""
    procedures, estimates = [], []
    replications = tally.contributing.size
    for statistic in inference.STATISTICS:
        est = tally.estimate[statistic]
        defined = est[~np.isnan(est)]
        spread = float(defined.std(ddof=1)) if defined.size > 1 else float("nan")
        estimates.append(
            {
                "statistic": statistic,
                "truth": truth[statistic],
                "replications": replications,
                "defined": int(defined.size),
                "mean": float(defined.mean()) if defined.size else float("nan"),
                "bias": float(defined.mean() - truth[statistic]) if defined.size else float("nan"),
                "bias_se": spread / np.sqrt(defined.size) if defined.size > 1 else float("nan"),
                "sd": spread,
                "contributing_mean": float(np.nanmean(tally.contributing)),
                "estimable_rate": float(np.mean(tally.contributing >= min_groups)),
            }
        )
    for (procedure, statistic), low in tally.low.items():
        high = tally.high[procedure, statistic]
        ok = ~np.isnan(low) & ~np.isnan(high)
        low_ok, high_ok = low[ok], high[ok]
        value = truth[statistic]
        coverage, coverage_se = _rate((low_ok <= value) & (value <= high_ok))
        reject, reject_se = _rate((low_ok > 0) | (high_ok < 0))
        procedures.append(
            {
                "procedure": procedure,
                "statistic": statistic,
                "truth": value,
                "replications": replications,
                "defined": int(ok.sum()),
                "coverage": coverage,
                "coverage_se": coverage_se,
                "miss_low": float(np.mean(high_ok < value)) if ok.any() else float("nan"),
                "miss_high": float(np.mean(low_ok > value)) if ok.any() else float("nan"),
                "reject": reject,
                "reject_se": reject_se,
                "width": float(np.mean(high_ok - low_ok)) if ok.any() else float("nan"),
                "width_median": float(np.median(high_ok - low_ok)) if ok.any() else float("nan"),
            }
        )
    tests = [("T1_adjusted_residual", "pooled", np.abs(tally.residual), Z_CRITICAL, True)]
    tests += [("T6_order_permutation", s, tally.order_p[s], ALPHA, False) for s in inference.STATISTICS]
    for procedure, statistic, values, cut, above in tests:
        ok = ~np.isnan(values)
        reject, reject_se = _rate(values[ok] > cut if above else values[ok] <= cut)
        procedures.append(
            {
                "procedure": procedure,
                "statistic": statistic,
                "truth": truth[statistic],
                "replications": replications,
                "defined": int(ok.sum()),
                "coverage": float("nan"),
                "coverage_se": float("nan"),
                "miss_low": float("nan"),
                "miss_high": float("nan"),
                "reject": reject,
                "reject_se": reject_se,
                "width": float("nan"),
                "width_median": float("nan"),
            }
        )
    return procedures, estimates


def analyse_replication(sequences, targets, tallies, r, rng_boot, rng_perm, size: dict, analysis: dict) -> None:
    """Apply every procedure to one simulated data set and record the results for each target."""
    level, min_leaving = analysis["level"], analysis["min_leaving"]
    pick = inference.group_resamples(len(sequences), rng_boot, size["resamples"])
    order = inference.order_permutation(sequences, targets, rng_perm, size["permutations"], min_leaving)
    for (a, b), tally, order_p in zip(targets, tallies, order):
        counts = inference.transition_counts(sequences, a, b)
        boot = inference.group_bootstrap(*counts, pick, level, min_leaving)
        event = inference.event_bootstrap(*counts, rng_boot, size["resamples"], level)
        tally.record(r, boot, event, inference.adjusted_residual(*counts), order_p)


def run_controlled(cell: dict, sim: dict, analysis: dict, seed: int) -> dict:
    """Run one cell of the controlled grid."""
    spec, size = sim["controlled"], sim
    a, b = spec["target"]
    base = generate.common_rates(cell["codes"], spec["rate_ratio"])
    concentration = spec["concentration"][cell["heterogeneity"]]
    weight = generate.mixing_weight(base, b, cell["excess"])
    rng_truth, rng_data, rng_boot, rng_perm = cell_streams(seed, cell, 4)
    truth = generate.controlled_truth(rng_truth, base, concentration, a, b, cell["excess"], sim["truth_groups"])
    replications, groups = size["replications"], cell["groups"]
    tally = Tally(replications)
    done = 0
    while done < replications:
        batch = min(size["batch"], replications - done)
        rates = generate.draw_rates(rng_data, base, concentration, batch * groups)
        lengths = generate.draw_lengths(rng_data, cell["length_profile"], batch, groups, spec)
        chains = generate.simulate_chains(
            rng_data,
            generate.controlled_matrices(rates, a, b, weight),
            generate.controlled_stationary(rates, a, b, weight),
            lengths.reshape(-1),
        )
        for i in range(batch):
            rows = slice(i * groups, (i + 1) * groups)
            sequences = generate.unpad(chains[rows], lengths[i])
            analyse_replication(sequences, [(a, b)], [tally], done + i, rng_boot, rng_perm, size, analysis)
        done += batch
    procedures, estimates = score(tally, truth, analysis["min_groups"])
    return {"cell": cell, "target": "target", "procedures": procedures, "estimates": estimates}


def run_calibrated(cell: dict, sim: dict, analysis: dict, seed: int, calibration: dict) -> dict:
    """Run one cell of the calibrated scenario; every focal transition is scored."""
    spec, size = sim["calibrated"], sim
    families = calibration["families"]
    common = np.array(calibration["matrix"])
    lengths = np.array(calibration["lengths"])
    targets = [(families.index(x), families.index(y)) for x, y in analysis["focal_transitions"]]
    concentration = spec["concentration"][cell["heterogeneity"]]
    rng_truth, rng_data, rng_boot, rng_perm = cell_streams(seed, cell, 4)
    groups = len(lengths)
    truths = generate.calibrated_truth(rng_truth, common, concentration, targets, sim["truth_datasets"] * groups)
    replications = size["replications"]
    tallies = [Tally(replications) for _ in targets]
    done = 0
    while done < replications:
        batch = min(size["batch"], replications - done)
        matrices = generate.rowwise_matrices(rng_data, common, concentration, batch * groups)
        tiled = np.tile(lengths, batch)
        chains = generate.simulate_chains(rng_data, matrices, generate.stationary(matrices), tiled)
        for i in range(batch):
            rows = slice(i * groups, (i + 1) * groups)
            sequences = generate.unpad(chains[rows], lengths)
            analyse_replication(sequences, targets, tallies, done + i, rng_boot, rng_perm, size, analysis)
        done += batch
    out = []
    for (x, y), tally, truth in zip(analysis["focal_transitions"], tallies, truths):
        procedures, estimates = score(tally, truth, analysis["min_groups"])
        out.append({"target": f"{x}>{y}", "procedures": procedures, "estimates": estimates})
    return {"cell": cell, "targets": out}


def run_network(cell: dict, sim: dict, nodes: int, seed: int) -> dict:
    """Run one cell of the network grid."""
    spec = dict(sim["network"], nodes=nodes)
    group_sd = spec["group_sd"][cell["heterogeneity"]]
    rng_data, rng_perm = cell_streams(seed, cell, 2)
    groups, replications, draws = cell["groups"], sim["replications"], sim["permutations"]
    arm_of_group = network.arms(groups)
    p_values = {name: np.full(replications, np.nan) for name in ("N1_segment_label", "N2_group_label", "N3_group_studentised")}
    flagged = np.zeros(replications)
    for r in range(replications):
        presence, group_of_segment = network.simulate_presence(rng_data, groups, arm_of_group, spec, group_sd, cell["shift"])
        segments = network.segment_vectors(presence)
        segment_labels = arm_of_group[group_of_segment]
        permuted = network.label_permutations(segment_labels, rng_perm, draws)
        p_values["N1_segment_label"][r] = network.squared_difference_test(segment_labels, segments, permuted)["p"]
        vectors = network.group_vectors(segments, group_of_segment, groups, unit=True)
        permuted = network.label_permutations(arm_of_group, rng_perm, draws)
        p_values["N2_group_label"][r] = network.squared_difference_test(arm_of_group, vectors, permuted)["p"]
        result = network.studentised_test(arm_of_group, vectors, permuted)
        p_values["N3_group_studentised"][r] = result["p"]
        flagged[r] = np.count_nonzero(result["pair_p"] <= ALPHA)
    rows = []
    for name, values in p_values.items():
        reject, reject_se = _rate(values <= ALPHA)
        rows.append(
            {
                "procedure": name,
                "replications": replications,
                "reject": reject,
                "reject_se": reject_se,
                "pairs_flagged_mean": float(flagged.mean()) if name == "N3_group_studentised" else float("nan"),
            }
        )
    return {"cell": cell, "procedures": rows}


def run_cell(task: dict) -> dict:
    """Run any cell; the entry point used by worker processes."""
    started = time.perf_counter()
    cell, sim, analysis, seed = task["cell"], task["simulation"], task["analysis"], task["seed"]
    if cell["arm"] == "controlled":
        result = run_controlled(cell, sim, analysis, seed)
    elif cell["arm"] == "calibrated":
        result = run_calibrated(cell, sim, analysis, seed, task["calibration"])
    else:
        result = run_network(cell, sim, task["nodes"], seed)
    result["seconds"] = round(time.perf_counter() - started, 1)
    return result


def calibration_from_events(events: pd.DataFrame, families: list[str], smoothing: float) -> dict:
    """Common transition matrix and sequence lengths of the calibrated scenario."""
    index = {name: i for i, name in enumerate(families)}
    counts = np.zeros((len(families), len(families)))
    lengths = []
    for _, case in events.sort_values(["case", "order_in_case"]).groupby("case", sort=True):
        codes = case["family"].map(index).to_numpy()
        np.add.at(counts, (codes[:-1], codes[1:]), 1)
        lengths.append(len(codes))
    smoothed = counts + smoothing
    matrix = smoothed / smoothed.sum(axis=1, keepdims=True)
    return {"families": families, "matrix": matrix.tolist(), "lengths": lengths}


CELL_COLUMNS = {
    "controlled": ["arm", "index", "codes", "groups", "heterogeneity", "length_profile", "excess"],
    "calibrated": ["arm", "index", "heterogeneity"],
    "network": ["arm", "index", "groups", "heterogeneity", "shift"],
}


def assemble(results: list[dict]) -> dict[str, pd.DataFrame]:
    """Combine cell results into the output tables."""
    procedures, estimates, networks = [], [], []
    for result in sorted(results, key=lambda r: (ARMS.index(r["cell"]["arm"]), r["cell"]["index"])):
        cell = result["cell"]
        if cell["arm"] == "network":
            networks += [{**cell, **row} for row in result["procedures"]]
            continue
        blocks = result["targets"] if cell["arm"] == "calibrated" else [result]
        for block in blocks:
            head = {**cell, "target": block["target"]}
            procedures += [{**head, **row} for row in block["procedures"]]
            estimates += [{**head, **row} for row in block["estimates"]]
    lead = ["arm", "index", "codes", "groups", "heterogeneity", "length_profile", "excess", "target"]

    def table(rows: list[dict], first: list[str]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        ordered = [c for c in first if c in frame.columns]
        return frame[ordered + [c for c in frame.columns if c not in ordered]]

    return {
        "transition_procedures": table(procedures, lead),
        "transition_estimates": table(estimates, lead),
        "network_procedures": table(networks, ["arm", "index", "groups", "heterogeneity", "shift"]),
    }
