"""Extension of the simulation study, added after its first results were known.

Two additions. A null in which groups have real but opposing dependence, so
that the mean excess is zero while almost no group is free of dependence. And
two corrected estimates of the group-averaged excess: one with a leave-one-out
base rate, which is exactly unbiased under any reordering of a group's events,
and a split-sequence jackknife of it, which removes bias of order one over the
sequence length when dependence is present.

The modules of the first study are used as they stand and are not edited.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from . import generate, inference, study

ARMS = ("opposing", "controlled_new", "calibrated_new")
CORRECTED = ("group_loo", "group_split")


# ----------------------------------------------------------------- estimates


def code_count(padded: np.ndarray, lengths: np.ndarray, code: int) -> np.ndarray:
    """Number of events of one code in every sequence of a padded matrix."""
    valid = np.arange(padded.shape[1])[None, :] < lengths[:, None]
    return ((padded == code) & valid).sum(axis=1).astype(float)


def loo_terms(padded: np.ndarray, lengths: np.ndarray, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    """Excess of a to b against the share of b among the group's other events.

    Returns the term of every group (missing when no transition leaves a) and
    the number of transitions leaving a.
    """
    nab, na, _, n = inference.padded_counts(padded, lengths, a, b)
    others = code_count(padded, lengths, b) - (1.0 if a == b else 0.0)
    ok = (na > 0) & (n > 0)
    term = np.where(ok, nab / np.where(ok, na, 1.0) - others / np.where(ok, n, 1.0), np.nan)
    return term, na


def second_half(padded: np.ndarray, lengths: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every sequence from its own start position onwards, as a padded matrix with new lengths."""
    width = padded.shape[1]
    index = np.minimum(start[:, None] + np.arange(width)[None, :], width - 1)
    return np.take_along_axis(padded, index, axis=1), lengths - start


def corrected_terms(padded: np.ndarray, lengths: np.ndarray, a: int, b: int, min_leaving: int) -> dict:
    """Group terms and contribution flags of the two corrected estimates."""
    full, na = loo_terms(padded, lengths, a, b)
    keep = na >= min_leaving
    half = lengths // 2
    first, na_first = loo_terms(padded, half, a, b)
    rest, rest_lengths = second_half(padded, lengths, half)
    second, na_second = loo_terms(rest, rest_lengths, a, b)
    both = keep & (na_first >= 1) & (na_second >= 1)
    split = 2.0 * full - 0.5 * (first + second)
    return {
        "group_loo": (np.where(keep, full, 0.0), keep),
        "group_split": (np.where(both, split, 0.0), both),
    }


def masked_mean(terms: np.ndarray, mask: np.ndarray):
    """Mean over contributing groups, its standard error and the number contributing."""
    m = mask.sum(axis=-1)
    kept = np.where(mask, terms, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(m > 0, kept.sum(axis=-1) / np.maximum(m, 1), np.nan)
        centred = np.where(mask, kept - np.where(m > 0, mean, 0.0)[..., None], 0.0)
        var = np.where(m > 1, (centred**2).sum(axis=-1) / np.maximum(m - 1, 1), np.nan)
        se = np.sqrt(var / np.maximum(m, 1))
    return mean, se, m


def term_bootstrap(terms: np.ndarray, mask: np.ndarray, pick: np.ndarray, level: float) -> dict:
    """Percentile, BCa and studentised intervals for a mean of group terms, from given group resamples."""
    est, se, m = masked_mean(terms, mask)
    d_est, d_se, _ = masked_mean(terms[pick], mask[pick])
    est, se = float(est), float(se)
    count = int(mask.sum())
    total = float(np.where(mask, terms, 0.0).sum())
    if count > 1:
        jackknife = np.where(mask, (total - terms) / (count - 1), total / count)
    else:
        jackknife = np.full(len(terms), total / count if count else np.nan)
    return {
        "estimate": est,
        "se": se,
        "contributing": count,
        "percentile": inference.percentile_interval(d_est, level),
        "bca": inference.bca_interval(est, d_est, jackknife, level),
        "studentised": inference.studentised_interval(est, se, d_est, d_se, level),
    }


# ----------------------------------------------------------------- generator


def opposing_effects(rng: np.random.Generator, rates: np.ndarray, a: int, b: int, magnitude: float):
    """Give every group a true excess of one size and a random sign.

    The size is the requested magnitude, or the largest that both signs allow
    given the group's rates. Returns the mixing weight of the target row and the
    true excess of every group.
    """
    pa, pb = rates[:, a], rates[:, b]
    # With weight w on the target code, the excess is w(1-pb)(1-pa+w*pa)/(1+w*pa),
    # which rises with w. The weight runs from -pb/(1-pb) (a never moves to b) to 1.
    with np.errstate(invalid="ignore", divide="ignore"):
        cap_up = (1.0 - pb) / (1.0 + pa)
        cap_down = pb * (1.0 - pa - pb) / ((1.0 - pb) - pa * pb)
    cap = np.clip(np.nan_to_num(np.minimum(cap_up, cap_down), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    size = np.minimum(magnitude, cap)
    sign = np.where(rng.random(len(rates)) < 0.5, 1.0, -1.0)
    excess = sign * size
    # Solve (1-pb)*pa*w^2 + ((1-pb)(1-pa) - excess*pa)*w - excess = 0 for the root that vanishes with the excess.
    lead = (1.0 - pb) * pa
    middle = (1.0 - pb) * (1.0 - pa) - excess * pa
    with np.errstate(invalid="ignore", divide="ignore"):
        root = np.sqrt(np.clip(middle**2 + 4.0 * lead * excess, 0.0, None))
        weight = np.nan_to_num(2.0 * excess / (middle + root), nan=0.0, posinf=0.0, neginf=0.0)
    return weight, excess


def weighted_matrices(rates: np.ndarray, a: int, b: int, weight: np.ndarray) -> np.ndarray:
    """Transition matrices whose rows equal the group's rates, except the target row, with one weight per group."""
    codes = rates.shape[1]
    matrices = np.repeat(rates[:, None, :], codes, axis=1)
    matrices[:, a, :] *= (1.0 - weight)[:, None]
    matrices[:, a, b] += weight
    return np.clip(matrices, 0.0, None)


def weighted_stationary(rates: np.ndarray, a: int, b: int, weight: np.ndarray) -> np.ndarray:
    """Stationary distribution of each of those chains, in closed form."""
    share = weight * rates[:, a] / (1.0 + weight * rates[:, a])
    out = (1.0 - share)[:, None] * rates
    out[:, b] += share
    return out


def opposing_truth(rng, base: np.ndarray, concentration, a: int, b: int, magnitude: float, size: int) -> dict:
    """True values under opposing dependence; the group-averaged excess is zero by construction."""
    rates = generate.draw_rates(rng, base, concentration, size)
    weight, excess = opposing_effects(rng, rates, a, b, magnitude)
    stat = weighted_stationary(rates, a, b, weight)
    truth = generate.excess_truth((1.0 - weight) * rates[:, b] + weight, stat[:, a], stat[:, b])
    truth["group"] = 0.0
    truth["mean_size"] = float(np.mean(np.abs(excess)))
    return truth


# --------------------------------------------------------------------- cells


def opposing_cells(spec: dict) -> list[dict]:
    """Every cell of the opposing-dependence grid, in a fixed order."""
    grid = itertools.product(spec["codes"], spec["groups"], spec["concentration"], spec["length_profiles"])
    return [
        {"arm": "opposing", "index": i, "codes": codes, "groups": groups, "heterogeneity": level, "length_profile": profile}
        for i, (codes, groups, level, profile) in enumerate(grid)
    ]


def all_cells(sim: dict) -> list[dict]:
    """Every cell of the extension."""
    cells = opposing_cells(sim["controlled"])
    cells += [dict(c, arm="controlled_new") for c in study.controlled_cells(sim["controlled"])]
    cells += [dict(c, arm="calibrated_new") for c in study.calibrated_cells(sim["calibrated"])]
    return cells


def cell_streams(seed: int, cell: dict, count: int) -> list[np.random.Generator]:
    """Independent random streams of one cell of the extension."""
    root = np.random.SeedSequence(seed, spawn_key=(ARMS.index(cell["arm"]), cell["index"]))
    return [np.random.default_rng(child) for child in root.spawn(count)]


def cell_weight(cell: dict) -> float:
    """Rough relative cost of a cell, used only to order the work."""
    if cell["arm"] == "calibrated_new":
        return 30.0
    groups = cell["groups"] * (1.6 if cell["length_profile"] == "extreme" else 1.0)
    return groups * (8.0 if cell["arm"] == "opposing" else 1.0)


# ------------------------------------------------------------------- scoring


class Tally:
    """Collects per-replication results under free keys."""

    def __init__(self, replications: int):
        self.size = replications
        self.low, self.high, self.estimate, self.contributing, self.tests = {}, {}, {}, {}, {}

    def _slot(self, store: dict, key):
        if key not in store:
            store[key] = np.full(self.size, np.nan)
        return store[key]

    def point(self, r: int, statistic: str, estimate: float, contributing: int) -> None:
        self._slot(self.estimate, statistic)[r] = estimate
        self._slot(self.contributing, statistic)[r] = contributing

    def interval(self, r: int, procedure: str, statistic: str, pair: tuple) -> None:
        self._slot(self.low, (procedure, statistic))[r] = pair[0]
        self._slot(self.high, (procedure, statistic))[r] = pair[1]

    def tested(self, r: int, procedure: str, statistic: str, value: float) -> None:
        self._slot(self.tests, (procedure, statistic))[r] = value


def truth_of(truth: dict, statistic: str) -> float:
    """The corrected estimates share the estimand of the group-averaged excess."""
    return truth["pooled"] if statistic == "pooled" else truth["group"]


def score(tally: Tally, truth: dict, min_groups: int) -> tuple[list[dict], list[dict]]:
    """Turn the collected replications into procedure rows and estimate rows."""
    procedures, estimates = [], []
    nan = float("nan")
    for statistic, est in tally.estimate.items():
        value = truth_of(truth, statistic)
        defined = est[~np.isnan(est)]
        spread = float(defined.std(ddof=1)) if defined.size > 1 else nan
        contributing = tally.contributing[statistic]
        estimates.append(
            {
                "statistic": statistic,
                "truth": value,
                "replications": tally.size,
                "defined": int(defined.size),
                "mean": float(defined.mean()) if defined.size else nan,
                "bias": float(defined.mean() - value) if defined.size else nan,
                "bias_se": spread / np.sqrt(defined.size) if defined.size > 1 else nan,
                "sd": spread,
                "rmse": float(np.sqrt(np.mean((defined - value) ** 2))) if defined.size else nan,
                "contributing_mean": float(np.nanmean(contributing)),
                "estimable_rate": float(np.mean(contributing >= min_groups)),
            }
        )
    for (procedure, statistic), low in tally.low.items():
        high = tally.high[procedure, statistic]
        value = truth_of(truth, statistic)
        ok = ~np.isnan(low) & ~np.isnan(high)
        low_ok, high_ok = low[ok], high[ok]
        coverage, coverage_se = study._rate((low_ok <= value) & (value <= high_ok))
        reject, reject_se = study._rate((low_ok > 0) | (high_ok < 0))
        procedures.append(
            {
                "procedure": procedure,
                "statistic": statistic,
                "truth": value,
                "replications": tally.size,
                "defined": int(ok.sum()),
                "coverage": coverage,
                "coverage_se": coverage_se,
                "miss_low": float(np.mean(high_ok < value)) if ok.any() else nan,
                "miss_high": float(np.mean(low_ok > value)) if ok.any() else nan,
                "reject": reject,
                "reject_se": reject_se,
                "width": float(np.mean(high_ok - low_ok)) if ok.any() else nan,
                "width_median": float(np.median(high_ok - low_ok)) if ok.any() else nan,
            }
        )
    for (procedure, statistic), values in tally.tests.items():
        ok = ~np.isnan(values)
        if procedure == "T1_adjusted_residual":
            flags = np.abs(values[ok]) > study.Z_CRITICAL
        else:
            flags = values[ok] <= study.ALPHA
        reject, reject_se = study._rate(flags)
        procedures.append(
            {
                "procedure": procedure,
                "statistic": statistic,
                "truth": truth_of(truth, statistic),
                "replications": tally.size,
                "defined": int(ok.sum()),
                "coverage": nan,
                "coverage_se": nan,
                "miss_low": nan,
                "miss_high": nan,
                "reject": reject,
                "reject_se": reject_se,
                "width": nan,
                "width_median": nan,
            }
        )
    return procedures, estimates


# ------------------------------------------------------------------- runners


def analyse(padded, lengths, targets, tallies, r, rng_boot, rng_perm, size: dict, analysis: dict, tests: bool) -> None:
    """Apply the procedures to one simulated data set and record the results for each target."""
    level, min_leaving = analysis["level"], analysis["min_leaving"]
    sequences = generate.unpad(padded, lengths)
    pick = inference.group_resamples(len(sequences), rng_boot, size["resamples"])
    order = (
        inference.order_permutation(sequences, targets, rng_perm, size["permutations"], min_leaving)
        if tests
        else [None] * len(targets)
    )
    for (a, b), tally, order_p in zip(targets, tallies, order):
        counts = inference.transition_counts(sequences, a, b)
        boot = inference.group_bootstrap(*counts, pick, level, min_leaving)
        for name, (terms, mask) in corrected_terms(padded, lengths, a, b, min_leaving).items():
            boot[name] = term_bootstrap(terms, mask, pick, level)
        for statistic, block in boot.items():
            tally.point(r, statistic, block["estimate"], block["contributing"])
            for procedure, key in study.BOOT_KEYS.items():
                tally.interval(r, procedure, statistic, block[key])
        if tests:
            event = inference.event_bootstrap(*counts, rng_boot, size["resamples"], level)
            tally.interval(r, "T2_event_bootstrap", "pooled", event)
            tally.tested(r, "T1_adjusted_residual", "pooled", inference.adjusted_residual(*counts))
            for statistic in inference.STATISTICS:
                tally.tested(r, "T6_order_permutation", statistic, order_p[statistic])


def run_controlled(cell: dict, ext: dict, sim: dict, analysis: dict) -> dict:
    """Run one cell with a single target: opposing dependence, or the controlled grid on the new seed."""
    spec = sim["controlled"]
    a, b = spec["target"]
    opposing = cell["arm"] == "opposing"
    base = generate.common_rates(cell["codes"], spec["rate_ratio"])
    concentration = spec["concentration"][cell["heterogeneity"]]
    rng_truth, rng_data, rng_boot, rng_perm = cell_streams(ext["seed"], cell, 4)
    if opposing:
        magnitude = ext["opposing"]["magnitude"]
        truth = opposing_truth(rng_truth, base, concentration, a, b, magnitude, ext["truth_groups"])
    else:
        fixed = generate.mixing_weight(base, b, cell["excess"])
        truth = generate.controlled_truth(rng_truth, base, concentration, a, b, cell["excess"], ext["truth_groups"])
    replications, groups = ext["replications"], cell["groups"]
    tally = Tally(replications)
    done = 0
    while done < replications:
        batch = min(ext["batch"], replications - done)
        rates = generate.draw_rates(rng_data, base, concentration, batch * groups)
        lengths = generate.draw_lengths(rng_data, cell["length_profile"], batch, groups, spec)
        if opposing:
            weight, _ = opposing_effects(rng_data, rates, a, b, magnitude)
        else:
            weight = np.full(len(rates), fixed)
        chains = generate.simulate_chains(
            rng_data, weighted_matrices(rates, a, b, weight), weighted_stationary(rates, a, b, weight), lengths.reshape(-1)
        )
        for i in range(batch):
            rows = slice(i * groups, (i + 1) * groups)
            analyse(chains[rows], lengths[i], [(a, b)], [tally], done + i, rng_boot, rng_perm, ext, analysis, opposing)
        done += batch
    procedures, estimates = score(tally, truth, analysis["min_groups"])
    extra = {"mean_size": truth["mean_size"]} if opposing else {}
    return {"cell": cell, "targets": [{"target": "target", "procedures": procedures, "estimates": estimates, **extra}]}


def run_calibrated(cell: dict, ext: dict, sim: dict, analysis: dict, calibration: dict) -> dict:
    """Run one calibrated cell on the new seed; every focal transition is scored."""
    families = calibration["families"]
    common = np.array(calibration["matrix"])
    lengths = np.array(calibration["lengths"])
    targets = [(families.index(x), families.index(y)) for x, y in analysis["focal_transitions"]]
    concentration = sim["calibrated"]["concentration"][cell["heterogeneity"]]
    rng_truth, rng_data, rng_boot, rng_perm = cell_streams(ext["seed"], cell, 4)
    groups = len(lengths)
    truths = generate.calibrated_truth(rng_truth, common, concentration, targets, ext["truth_datasets"] * groups)
    replications = ext["replications"]
    tallies = [Tally(replications) for _ in targets]
    done = 0
    while done < replications:
        batch = min(ext["batch"], replications - done)
        matrices = generate.rowwise_matrices(rng_data, common, concentration, batch * groups)
        chains = generate.simulate_chains(rng_data, matrices, generate.stationary(matrices), np.tile(lengths, batch))
        for i in range(batch):
            rows = slice(i * groups, (i + 1) * groups)
            analyse(chains[rows], lengths, targets, tallies, done + i, rng_boot, rng_perm, ext, analysis, False)
        done += batch
    out = []
    for (x, y), tally, truth in zip(analysis["focal_transitions"], tallies, truths):
        procedures, estimates = score(tally, truth, analysis["min_groups"])
        out.append({"target": f"{x}>{y}", "procedures": procedures, "estimates": estimates})
    return {"cell": cell, "targets": out}


def run_cell(task: dict) -> dict:
    """Run any cell of the extension; the entry point used by worker processes."""
    started = time.perf_counter()
    cell, ext, sim, analysis = task["cell"], task["extension"], task["simulation"], task["analysis"]
    if cell["arm"] == "calibrated_new":
        result = run_calibrated(cell, ext, sim, analysis, task["calibration"])
    else:
        result = run_controlled(cell, ext, sim, analysis)
    result["seconds"] = round(time.perf_counter() - started, 1)
    return result


def assemble(results: list[dict]) -> dict[str, pd.DataFrame]:
    """Combine cell results into the output tables."""
    procedures, estimates = [], []
    for result in sorted(results, key=lambda r: (ARMS.index(r["cell"]["arm"]), r["cell"]["index"])):
        for block in result["targets"]:
            head = {**result["cell"], "target": block["target"]}
            if "mean_size" in block:
                head["mean_size"] = block["mean_size"]
            procedures += [{**head, **row} for row in block["procedures"]]
            estimates += [{**head, **row} for row in block["estimates"]]
    lead = ["arm", "index", "codes", "groups", "heterogeneity", "length_profile", "excess", "mean_size", "target"]

    def table(rows: list[dict]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        ordered = [c for c in lead if c in frame.columns]
        return frame[ordered + [c for c in frame.columns if c not in ordered]]

    return {"extension_procedures": table(procedures), "extension_estimates": table(estimates)}
