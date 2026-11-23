"""Second extension of the simulation study: one long group that behaves differently.

Added after the application showed such a group in the analysed sequences. In
every data set one sequence is long and may have its own code rates and its own
dependence on the target transition. A t interval over group values is run
beside the registered procedures as an added comparator.

The modules of the first study and of the first extension are used as they
stand and are not edited.
"""
from __future__ import annotations

import itertools
import math
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from . import extension, generate, inference, study

ARMS = ("controlled_dom", "calibrated_dom")
LEVELS = ("none", "rates", "rates_dependence")
T_INTERVAL = "T7_t_interval"
QUADRATURE = np.polynomial.legendre.leggauss(64)


# ---------------------------------------------------------------- t interval


def _t_mass(angle: float, df: int) -> float:
    """Probability that a t variable lies between zero and sqrt(df) * tan(angle)."""
    nodes, weights = QUADRATURE
    points = 0.5 * angle * (nodes + 1.0)
    integral = 0.5 * angle * float(np.sum(weights * np.cos(points) ** (df - 1)))
    constant = math.exp(math.lgamma((df + 1) / 2.0) - math.lgamma(df / 2.0)) / math.sqrt(math.pi)
    return constant * integral


@lru_cache(maxsize=None)
def t_quantile(df: int, prob: float) -> float:
    """Quantile of the t distribution for a probability above one half, by bisection on an angle."""
    if df < 1 or not 0.5 < prob < 1.0:
        raise ValueError("t_quantile needs at least one degree of freedom and a probability between 0.5 and 1")
    target = prob - 0.5
    low, high = 0.0, math.pi / 2.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if _t_mass(middle, df) < target:
            low = middle
        else:
            high = middle
    return math.sqrt(df) * math.tan(0.5 * (low + high))


def t_interval(estimate: float, se: float, contributing: int, level: float) -> tuple[float, float]:
    """Estimate plus or minus the standard error times the t quantile; undefined without a usable standard error."""
    if contributing < 2 or np.isnan(estimate) or np.isnan(se) or se <= inference.SE_FLOOR:
        return float("nan"), float("nan")
    reach = t_quantile(int(contributing) - 1, 1.0 - (1.0 - level) / 2.0) * se
    return float(estimate - reach), float(estimate + reach)


# ----------------------------------------------------------------- generator


def tilted_rates(base: np.ndarray, a: int, b: int, rate: float) -> np.ndarray:
    """Base rates with the two target codes each raised to one rate and the other codes scaled in proportion."""
    out = np.array(base, dtype=float)
    others = out.sum() - out[a] - out[b]
    out *= (1.0 - 2.0 * rate) / others
    out[a] = out[b] = rate
    return out


def weight_for_excess(rates: np.ndarray, a: int, b: int, excess: float) -> tuple[np.ndarray, np.ndarray]:
    """Mixing weight of the target row that gives each group a stated positive excess.

    The excess is capped at the largest value the group's rates allow. Returns
    the weight and the excess reached.
    """
    pa, pb = rates[:, a], rates[:, b]
    reached = np.minimum(excess, (1.0 - pb) / (1.0 + pa))
    lead = (1.0 - pb) * pa
    middle = (1.0 - pb) * (1.0 - pa) - reached * pa
    root = np.sqrt(np.clip(middle**2 + 4.0 * lead * reached, 0.0, None))
    weight = np.clip(2.0 * reached / (middle + root), 0.0, 1.0)
    return weight, reached


def type_parts(rates: np.ndarray, a: int, b: int, weight: np.ndarray) -> dict:
    """Means, over groups of one type, of the quantities the true values are built from."""
    stat = extension.weighted_stationary(rates, a, b, weight)
    p_ab = (1.0 - weight) * rates[:, b] + weight
    return {
        "excess": float(np.mean(p_ab - stat[:, b])),
        "from_to": float(np.mean(stat[:, a] * p_ab)),
        "from": float(np.mean(stat[:, a])),
        "to": float(np.mean(stat[:, b])),
    }


def combine(ordinary: dict, dominant: dict | None, groups: int, ordinary_steps: float, dominant_steps: float) -> dict:
    """True group-averaged and pooled excess of a data set with at most one dominant group.

    ordinary_steps is the expected number of transitions of all ordinary groups
    together, dominant_steps that of the dominant group.
    """
    if dominant is None:
        return {"group": ordinary["excess"], "pooled": ordinary["from_to"] / ordinary["from"] - ordinary["to"]}
    leaving = ordinary_steps * ordinary["from"] + dominant_steps * dominant["from"]
    moved = ordinary_steps * ordinary["from_to"] + dominant_steps * dominant["from_to"]
    arriving = ordinary_steps * ordinary["to"] + dominant_steps * dominant["to"]
    return {
        "group": ((groups - 1) * ordinary["excess"] + dominant["excess"]) / groups,
        "pooled": moved / leaving - arriving / (ordinary_steps + dominant_steps),
    }


def controlled_truth(rng: np.random.Generator, cell: dict, spec: dict, dom: dict, size: int) -> dict:
    """True values of one controlled cell with a dominant group."""
    a, b = spec["target"]
    base = generate.common_rates(cell["codes"], spec["rate_ratio"])
    concentration = spec["concentration"][cell["heterogeneity"]]
    fixed = generate.mixing_weight(base, b, cell["excess"])
    draws = 1 if concentration is None else size
    rates = generate.draw_rates(rng, base, concentration, draws)
    ordinary = type_parts(rates, a, b, np.full(len(rates), fixed))
    level = cell["dominant"]
    if level == "none":
        truth = combine(ordinary, None, cell["groups"], 0.0, 0.0)
    else:
        own = generate.draw_rates(rng, tilted_rates(base, a, b, dom["target_rate"]), concentration, draws)
        weight = np.full(len(own), fixed)
        if level == "rates_dependence":
            weight, _ = weight_for_excess(own, a, b, dom["dominant_excess"])
        dominant = type_parts(own, a, b, weight)
        typical = generate.draw_lengths(rng, "skewed", 1, size, spec).mean() - 1.0
        truth = combine(
            ordinary, dominant, cell["groups"], (cell["groups"] - 1) * typical, spec["extreme_length"] - 1.0
        )
        truth["dominant_excess"] = dominant["excess"]
    if cell["excess"] == 0.0 and level != "rates_dependence":
        truth["group"] = 0.0
    return truth


def matrix_parts(matrices: np.ndarray, targets: list[tuple[int, int]]) -> np.ndarray:
    """Sums, over the given matrices, of the four quantities of type_parts for every target."""
    stat = generate.stationary(matrices)
    out = np.zeros((len(targets), 4))
    for i, (a, b) in enumerate(targets):
        p_ab = matrices[:, a, b]
        out[i] = [np.sum(p_ab - stat[:, b]), np.sum(stat[:, a] * p_ab), np.sum(stat[:, a]), np.sum(stat[:, b])]
    return out


def calibrated_truth(
    rng: np.random.Generator, calibration: dict, concentration, targets: list[tuple[int, int]], size: int,
    chunk: int = 100000,
) -> list[dict]:
    """True values of every target in a calibrated cell with a dominant group."""
    common = np.array(calibration["matrix_ordinary"])
    own = np.array(calibration["matrix_dominant"])
    lengths = np.array(calibration["lengths"])
    position = calibration["dominant_index"]
    if concentration is None:
        sums, count = matrix_parts(common[None, :, :], targets), 1
    else:
        sums, count = np.zeros((len(targets), 4)), 0
        while count < size:
            step = min(chunk, size - count)
            sums += matrix_parts(generate.rowwise_matrices(rng, common, concentration, step), targets)
            count += step
    special = matrix_parts(own[None, :, :], targets)
    keys = ("excess", "from_to", "from", "to")
    dominant_steps = float(lengths[position] - 1)
    ordinary_steps = float((lengths - 1).sum() - dominant_steps)
    out = []
    for i in range(len(targets)):
        ordinary = dict(zip(keys, sums[i] / count))
        dominant = dict(zip(keys, special[i]))
        truth = combine(ordinary, dominant, len(lengths), ordinary_steps, dominant_steps)
        truth["dominant_excess"] = dominant["excess"]
        out.append(truth)
    return out


def calibration_from_events(events: pd.DataFrame, families: list[str], smoothing: float) -> dict:
    """Transition matrices of the longest group and of all other groups together, and the sequence lengths."""
    index = {name: i for i, name in enumerate(families)}
    size = len(families)
    counts, lengths = [], []
    for _, case in events.sort_values(["case", "order_in_case"]).groupby("case", sort=True):
        codes = case["family"].map(index).to_numpy()
        own = np.zeros((size, size))
        np.add.at(own, (codes[:-1], codes[1:]), 1)
        counts.append(own)
        lengths.append(len(codes))
    position = int(np.argmax(lengths))
    special = counts[position] + smoothing
    rest = sum(c for i, c in enumerate(counts) if i != position) + smoothing
    return {
        "families": families,
        "matrix_ordinary": (rest / rest.sum(axis=1, keepdims=True)).tolist(),
        "matrix_dominant": (special / special.sum(axis=1, keepdims=True)).tolist(),
        "lengths": lengths,
        "dominant_index": position,
    }


# --------------------------------------------------------------------- cells


def controlled_cells(dom: dict, spec: dict) -> list[dict]:
    """Every cell of the controlled grid with a dominant group, in a fixed order."""
    grid = itertools.product(spec["codes"], dom["groups"], spec["concentration"], dom["excess"], dom["levels"])
    return [
        {"arm": "controlled_dom", "index": i, "codes": codes, "groups": groups, "heterogeneity": level,
         "length_profile": "extreme", "excess": excess, "dominant": dominance}
        for i, (codes, groups, level, excess, dominance) in enumerate(grid)
    ]


def calibrated_cells(sim: dict) -> list[dict]:
    """Every cell of the calibrated scenario with a dominant group."""
    return [
        {"arm": "calibrated_dom", "index": i, "heterogeneity": level, "dominant": "observed"}
        for i, level in enumerate(sim["calibrated"]["concentration"])
    ]


def all_cells(dom: dict, sim: dict) -> list[dict]:
    """Every cell of the second extension."""
    return controlled_cells(dom, sim["controlled"]) + calibrated_cells(sim)


def cell_streams(seed: int, cell: dict, count: int) -> list[np.random.Generator]:
    """Independent random streams of one cell."""
    root = np.random.SeedSequence(seed, spawn_key=(ARMS.index(cell["arm"]), cell["index"]))
    return [np.random.default_rng(child) for child in root.spawn(count)]


def cell_weight(cell: dict) -> float:
    """Rough relative cost of a cell, used only to order the work."""
    return 60.0 if cell["arm"] == "calibrated_dom" else cell["groups"] + 5.0


# ------------------------------------------------------------------- scoring


def score(tally: extension.Tally, truth: dict, min_groups: int) -> tuple[list[dict], list[dict]]:
    """Procedure rows and estimate rows, with the share of intervals that leave the admissible range."""
    procedures, estimates = extension.score(tally, truth, min_groups)
    for row in procedures:
        key = (row["procedure"], row["statistic"])
        row["out_of_range"] = float("nan")
        if key in tally.low:
            low, high = tally.low[key], tally.high[key]
            ok = ~np.isnan(low) & ~np.isnan(high)
            if ok.any():
                row["out_of_range"] = float(np.mean((low[ok] < -1.0) | (high[ok] > 1.0)))
    return procedures, estimates


# ------------------------------------------------------------------- runners


def analyse(padded, lengths, targets, tallies, r, rng_boot, rng_perm, size: dict, analysis: dict) -> None:
    """Apply every procedure to one simulated data set and record the results for each target."""
    level, min_leaving = analysis["level"], analysis["min_leaving"]
    sequences = generate.unpad(padded, lengths)
    pick = inference.group_resamples(len(sequences), rng_boot, size["resamples"])
    order = inference.order_permutation(sequences, targets, rng_perm, size["permutations"], min_leaving)
    for (a, b), tally, order_p in zip(targets, tallies, order):
        counts = inference.transition_counts(sequences, a, b)
        boot = inference.group_bootstrap(*counts, pick, level, min_leaving)
        terms, leaving = extension.loo_terms(padded, lengths, a, b)
        keep = leaving >= min_leaving
        boot["group_loo"] = extension.term_bootstrap(np.where(keep, terms, 0.0), keep, pick, level)
        for statistic, block in boot.items():
            tally.point(r, statistic, block["estimate"], block["contributing"])
            for procedure, key in study.BOOT_KEYS.items():
                tally.interval(r, procedure, statistic, block[key])
            tally.interval(
                r, T_INTERVAL, statistic, t_interval(block["estimate"], block["se"], block["contributing"], level)
            )
        event = inference.event_bootstrap(*counts, rng_boot, size["resamples"], level)
        tally.interval(r, "T2_event_bootstrap", "pooled", event)
        tally.tested(r, "T1_adjusted_residual", "pooled", inference.adjusted_residual(*counts))
        for statistic in inference.STATISTICS:
            tally.tested(r, "T6_order_permutation", statistic, order_p[statistic])


def run_controlled(cell: dict, dom: dict, sim: dict, analysis: dict) -> dict:
    """Run one controlled cell; the last group of every data set is the long one."""
    spec = sim["controlled"]
    a, b = spec["target"]
    base = generate.common_rates(cell["codes"], spec["rate_ratio"])
    concentration = spec["concentration"][cell["heterogeneity"]]
    fixed = generate.mixing_weight(base, b, cell["excess"])
    tilted = tilted_rates(base, a, b, dom["target_rate"])
    level = cell["dominant"]
    rng_truth, rng_data, rng_boot, rng_perm = cell_streams(dom["seed"], cell, 4)
    truth = controlled_truth(rng_truth, cell, spec, dom, dom["truth_groups"])
    replications, groups = dom["replications"], cell["groups"]
    tally = extension.Tally(replications)
    done = 0
    while done < replications:
        batch = min(dom["batch"], replications - done)
        rates = generate.draw_rates(rng_data, base, concentration, batch * groups)
        lengths = generate.draw_lengths(rng_data, "extreme", batch, groups, spec)
        weight = np.full(len(rates), fixed)
        if level != "none":
            long_rows = np.arange(batch) * groups + groups - 1
            rates[long_rows] = generate.draw_rates(rng_data, tilted, concentration, batch)
            if level == "rates_dependence":
                weight[long_rows], _ = weight_for_excess(rates[long_rows], a, b, dom["dominant_excess"])
        chains = generate.simulate_chains(
            rng_data,
            extension.weighted_matrices(rates, a, b, weight),
            extension.weighted_stationary(rates, a, b, weight),
            lengths.reshape(-1),
        )
        for i in range(batch):
            rows = slice(i * groups, (i + 1) * groups)
            analyse(chains[rows], lengths[i], [(a, b)], [tally], done + i, rng_boot, rng_perm, dom, analysis)
        done += batch
    procedures, estimates = score(tally, truth, analysis["min_groups"])
    extra = {"dominant_excess": truth.get("dominant_excess", float("nan"))}
    return {"cell": cell, "targets": [{"target": "target", "procedures": procedures, "estimates": estimates, **extra}]}


def run_calibrated(cell: dict, dom: dict, sim: dict, analysis: dict, calibration: dict) -> dict:
    """Run one calibrated cell; every focal transition is scored."""
    families = calibration["families"]
    common = np.array(calibration["matrix_ordinary"])
    own = np.array(calibration["matrix_dominant"])
    lengths = np.array(calibration["lengths"])
    position = calibration["dominant_index"]
    targets = [(families.index(x), families.index(y)) for x, y in analysis["focal_transitions"]]
    concentration = sim["calibrated"]["concentration"][cell["heterogeneity"]]
    rng_truth, rng_data, rng_boot, rng_perm = cell_streams(dom["seed"], cell, 4)
    groups = len(lengths)
    truths = calibrated_truth(rng_truth, calibration, concentration, targets, dom["truth_groups"])
    replications = dom["replications"]
    tallies = [extension.Tally(replications) for _ in targets]
    done = 0
    while done < replications:
        batch = min(dom["batch"], replications - done)
        matrices = generate.rowwise_matrices(rng_data, common, concentration, batch * groups)
        matrices[np.arange(batch) * groups + position] = own
        chains = generate.simulate_chains(rng_data, matrices, generate.stationary(matrices), np.tile(lengths, batch))
        for i in range(batch):
            rows = slice(i * groups, (i + 1) * groups)
            analyse(chains[rows], lengths, targets, tallies, done + i, rng_boot, rng_perm, dom, analysis)
        done += batch
    out = []
    for (x, y), tally, truth in zip(analysis["focal_transitions"], tallies, truths):
        procedures, estimates = score(tally, truth, analysis["min_groups"])
        out.append(
            {"target": f"{x}>{y}", "procedures": procedures, "estimates": estimates,
             "dominant_excess": truth["dominant_excess"]}
        )
    return {"cell": cell, "targets": out}


def run_cell(task: dict) -> dict:
    """Run any cell of the second extension; the entry point used by worker processes."""
    started = time.perf_counter()
    cell, dom, sim, analysis = task["cell"], task["dominance"], task["simulation"], task["analysis"]
    if cell["arm"] == "calibrated_dom":
        result = run_calibrated(cell, dom, sim, analysis, task["calibration"])
    else:
        result = run_controlled(cell, dom, sim, analysis)
    result["seconds"] = round(time.perf_counter() - started, 1)
    return result


def assemble(results: list[dict]) -> dict[str, pd.DataFrame]:
    """Combine cell results into the output tables."""
    procedures, estimates = [], []
    for result in sorted(results, key=lambda r: (ARMS.index(r["cell"]["arm"]), r["cell"]["index"])):
        for block in result["targets"]:
            head = {**result["cell"], "target": block["target"], "dominant_excess": block["dominant_excess"]}
            procedures += [{**head, **row} for row in block["procedures"]]
            estimates += [{**head, **row} for row in block["estimates"]]
    lead = [
        "arm", "index", "codes", "groups", "heterogeneity", "length_profile", "excess", "dominant", "target",
        "dominant_excess",
    ]

    def table(rows: list[dict]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        ordered = [c for c in lead if c in frame.columns]
        return frame[ordered + [c for c in frame.columns if c not in ordered]]

    return {"dominance_procedures": table(procedures), "dominance_estimates": table(estimates)}
