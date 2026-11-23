"""Stage 5: application of every procedure to the classroom event sequences.

Focal transitions carry estimates, intervals and tests, with Holm adjustment
across the focal set. The full matrix of transitions is exploratory and uses a
maximum-statistic adjustment. Three sensitivity analyses and a two-class
network contrast follow. The estimates fixed before the simulation are marked
"registered"; the corrected ones are marked "added", as is the table of t
intervals, which is derived from the estimates and their standard errors.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from . import dominance, extension, generate, inference, network, tables
from .config import Settings, stage_dir

NORMAL = NormalDist()
STATUS = {"group": "registered", "pooled": "registered", "group_loo": "added", "group_split": "added"}
INTERVAL_KEYS = ("percentile", "bca", "studentised")


# ------------------------------------------------------------------ helpers


def load_events(settings: Settings) -> pd.DataFrame:
    """Read the coded events in sequence order, with numeric order and gap columns."""
    events = tables.read_csv(stage_dir(settings, "coded") / "coded_events.csv")
    events["order_in_case"] = events["order_in_case"].astype(int)
    events["gap"] = pd.to_numeric(events["gap_prev_s"], errors="coerce")
    return events.sort_values(["case", "order_in_case"], kind="mergesort").reset_index(drop=True)


def to_sequences(events: pd.DataFrame, families: list[str]) -> tuple[list[str], list[np.ndarray]]:
    """One integer sequence per group, in the order of the group identifiers."""
    index = {name: i for i, name in enumerate(families)}
    cases, sequences = [], []
    for case, block in events.groupby("case", sort=True):
        cases.append(case)
        sequences.append(block["family"].map(index).to_numpy(dtype=np.int8))
    return cases, sequences


def pad(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Sequences as rows of a padded matrix, with their lengths."""
    lengths = np.array([len(s) for s in sequences])
    padded = np.full((len(sequences), int(lengths.max())), generate.PAD, dtype=np.int8)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq
    return padded, lengths


def merge_rapid(events: pd.DataFrame, seconds: float) -> pd.DataFrame:
    """Count once any run of events of one family that follow each other within the given gap."""
    previous = events.groupby("case")["family"].shift()
    repeat = (events["family"] == previous) & (events["gap"] <= seconds)
    return events[~repeat].reset_index(drop=True)


def shuffle_ties(events: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Put events that share a start time within a group in a random order; all other order is kept."""
    keyed = events.assign(_key=rng.random(len(events)))
    return keyed.sort_values(["case", "t_start", "_key"], kind="mergesort").drop(columns="_key").reset_index(drop=True)


def longest_case(events: pd.DataFrame) -> str:
    """The group with the most events; ties go to the first identifier."""
    sizes = events.groupby("case", sort=True).size()
    return str(sizes.idxmax())


def holm(p_values) -> np.ndarray:
    """Holm-adjusted P values; missing values stay missing and do not count as tests."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    if ok.size == 0:
        return out
    order = ok[np.argsort(p[ok], kind="mergesort")]
    scaled = p[order] * (ok.size - np.arange(ok.size))
    out[order] = np.minimum(np.maximum.accumulate(scaled), 1.0)
    return out


def normal_p(z: float) -> float:
    """Two-sided P value of a standard normal statistic."""
    return float("nan") if np.isnan(z) else float(2.0 * (1.0 - NORMAL.cdf(abs(z))))


def streams(seed: int, count: int) -> list[np.random.Generator]:
    """Random streams of the application, apart from every simulation stream."""
    root = np.random.SeedSequence(seed, spawn_key=(9,))
    return [np.random.default_rng(child) for child in root.spawn(count)]


# ------------------------------------------------------------- transitions


def focal_analysis(sequences, targets: list[tuple], analysis: dict, sizes: dict, rng_boot, rng_perm) -> tuple[list, list]:
    """Estimates, intervals and tests of every focal transition.

    targets holds (label, a, b). Returns estimate rows and test rows.
    """
    level, min_leaving, min_groups = analysis["level"], analysis["min_leaving"], analysis["min_groups"]
    pick = inference.group_resamples(len(sequences), rng_boot, sizes["resamples"])
    pairs = [(a, b) for _, a, b in targets]
    order = inference.order_permutation(sequences, pairs, rng_perm, sizes["permutations"], min_leaving)
    padded, lengths = pad(sequences)
    estimates, tests = [], []
    for (label, a, b), order_p in zip(targets, order):
        counts = inference.transition_counts(sequences, a, b)
        boot = inference.group_bootstrap(*counts, pick, level, min_leaving)
        for name, (terms, mask) in extension.corrected_terms(padded, lengths, a, b, min_leaving).items():
            boot[name] = extension.term_bootstrap(terms, mask, pick, level)
        event = inference.event_bootstrap(*counts, rng_boot, sizes["resamples"], level)
        totals = [int(c.sum()) for c in counts]
        for statistic, block in boot.items():
            row = {
                "transition": label,
                "statistic": statistic,
                "status": STATUS[statistic],
                "estimate": block["estimate"],
                "se": block["se"],
                "groups": len(sequences),
                "contributing": block["contributing"],
                "estimable": bool(statistic == "pooled" or block["contributing"] >= min_groups),
                "n_ab": totals[0],
                "n_a": totals[1],
                "n_b": totals[2],
                "n": totals[3],
            }
            for key in INTERVAL_KEYS:
                row[f"{key}_low"], row[f"{key}_high"] = block[key]
            row["event_low"], row["event_high"] = event if statistic == "pooled" else (float("nan"), float("nan"))
            estimates.append(row)
        z = inference.adjusted_residual(*counts)
        tests.append({"transition": label, "test": "T1_adjusted_residual", "statistic": "pooled", "value": z, "p": normal_p(z)})
        for statistic in inference.STATISTICS:
            tests.append(
                {"transition": label, "test": "T6_order_permutation", "statistic": statistic,
                 "value": boot[statistic]["estimate"], "p": order_p[statistic]}
            )
    frame = pd.DataFrame(tests)
    frame["p_holm"] = np.nan
    for _, index in frame.groupby(["test", "statistic"]).groups.items():
        frame.loc[index, "p_holm"] = holm(frame.loc[index, "p"].to_numpy())
    return estimates, frame.to_dict("records")


def added_intervals(estimates: list[dict], level: float) -> list[dict]:
    """The t interval of every reported estimate, from values already computed; it draws no random number."""
    rows = []
    for row in estimates:
        if row["statistic"] == "group_split":
            continue
        low, high = dominance.t_interval(row["estimate"], row["se"], row["contributing"], level)
        rows.append(
            {
                "variant": row["variant"],
                "transition": row["transition"],
                "statistic": row["statistic"],
                "procedure": dominance.T_INTERVAL,
                "status": "added",
                "estimate": row["estimate"],
                "se": row["se"],
                "contributing": row["contributing"],
                "estimable": row["estimable"],
                "low": low,
                "high": high,
            }
        )
    return rows


def permutation_nulls(sequences, targets: list[tuple[int, int]], rng, draws: int, min_leaving: int) -> dict:
    """Observed values and order-permutation draws of both statistics for many transitions.

    The shuffles are drawn exactly as in the order permutation test, so the P
    values derived from these draws equal those of that test for the same stream.
    """
    groups = len(sequences)
    lengths = np.array([len(s) for s in sequences])
    if groups and np.all(lengths == lengths[0]):
        stacked = np.broadcast_to(np.stack(sequences).astype(np.int8), (draws, groups, int(lengths[0])))
        shuffled, blocks = [rng.permuted(stacked, axis=2)], [slice(0, groups)]
    else:
        shuffled = [rng.permuted(np.broadcast_to(s.astype(np.int8), (draws, 1, len(s))), axis=2) for s in sequences]
        blocks = [slice(g, g + 1) for g in range(groups)]
    n = (lengths - 1).astype(float)
    out = {key: [] for key in ("obs_group", "obs_pooled", "null_group", "null_pooled", "contributing")}
    for a, b in targets:
        p_ab, p_a, p_b = (np.empty((draws, groups)) for _ in range(3))
        for block, x in zip(blocks, shuffled):
            left, right = x[:, :, :-1] == a, x[:, :, 1:] == b
            p_ab[:, block], p_a[:, block], p_b[:, block] = (left & right).sum(axis=2), left.sum(axis=2), right.sum(axis=2)
        nab, na, nb, _ = inference.transition_counts(sequences, a, b)
        obs_group, _, m = inference.group_excess(nab, na, nb, n, min_leaving)
        obs_pooled, _ = inference.pooled_excess(nab, na, nb, n)
        null_group, _, _ = inference.group_excess(p_ab, p_a, p_b, n, min_leaving)
        null_pooled, _ = inference.pooled_excess(p_ab, p_a, p_b, np.broadcast_to(n, (draws, groups)))
        out["obs_group"].append(float(obs_group))
        out["obs_pooled"].append(float(obs_pooled))
        out["null_group"].append(null_group)
        out["null_pooled"].append(null_pooled)
        out["contributing"].append(int(m))
    return {
        "obs_group": np.array(out["obs_group"]),
        "obs_pooled": np.array(out["obs_pooled"]),
        "null_group": np.column_stack(out["null_group"]),
        "null_pooled": np.column_stack(out["null_pooled"]),
        "contributing": np.array(out["contributing"]),
    }


def max_statistic(observed: np.ndarray, null: np.ndarray) -> dict:
    """Standardised values, single-test P values and maximum-statistic adjusted P values.

    Each transition is standardised by the mean and spread of its own permutation
    draws. The adjusted P value compares it with the largest absolute standardised
    value across all transitions within each draw.
    """
    with np.errstate(invalid="ignore"):
        centre = np.nanmean(null, axis=0)
        spread = np.nanstd(null, axis=0, ddof=1)
    ok = ~np.isnan(observed) & (spread > 0)
    z = np.full(observed.shape, np.nan)
    z[ok] = (observed[ok] - centre[ok]) / spread[ok]
    scaled = np.abs(np.nan_to_num((null[:, ok] - centre[ok]) / spread[ok], nan=0.0))
    top = scaled.max(axis=1) if ok.any() else np.zeros(len(null))
    single, adjusted = np.full(observed.shape, np.nan), np.full(observed.shape, np.nan)
    for k in np.flatnonzero(ok):
        single[k] = inference._two_sided_p(float(observed[k]), null[:, k])
        adjusted[k] = (np.count_nonzero(top >= abs(z[k]) - 1e-12) + 1) / (len(top) + 1)
    return {"z": z, "p": single, "p_adjusted": adjusted}


def matrix_analysis(sequences, families: list[str], analysis: dict, draws: int, rng) -> list[dict]:
    """Exploratory analysis of every transition between families."""
    k = len(families)
    targets = [(a, b) for a in range(k) for b in range(k)]
    nulls = permutation_nulls(sequences, targets, rng, draws, analysis["min_leaving"])
    by_group = max_statistic(nulls["obs_group"], nulls["null_group"])
    by_pool = max_statistic(nulls["obs_pooled"], nulls["null_pooled"])
    rows = []
    for i, (a, b) in enumerate(targets):
        nab, na, _, _ = inference.transition_counts(sequences, a, b)
        rows.append(
            {
                "from": families[a],
                "to": families[b],
                "n_ab": int(nab.sum()),
                "n_a": int(na.sum()),
                "contributing": int(nulls["contributing"][i]),
                "estimable": bool(nulls["contributing"][i] >= analysis["min_groups"]),
                "group_excess": nulls["obs_group"][i],
                "group_z": by_group["z"][i],
                "group_p": by_group["p"][i],
                "group_p_adjusted": by_group["p_adjusted"][i],
                "pooled_excess": nulls["obs_pooled"][i],
                "pooled_z": by_pool["z"][i],
                "pooled_p": by_pool["p"][i],
                "pooled_p_adjusted": by_pool["p_adjusted"][i],
            }
        )
    return rows


def tie_sensitivity(events, families, targets, analysis: dict, orderings: int, rng) -> list[dict]:
    """Point estimates of the focal transitions under random orderings of events that share a start time."""
    names = ("group", "pooled", "group_loo")
    draws = {(label, name): [] for label, _, _ in targets for name in names}
    for _ in range(orderings):
        _, sequences = to_sequences(shuffle_ties(events, rng), families)
        padded, lengths = pad(sequences)
        for label, a, b in targets:
            counts = inference.transition_counts(sequences, a, b)
            draws[label, "group"].append(float(inference.group_excess(*counts, analysis["min_leaving"])[0]))
            draws[label, "pooled"].append(float(inference.pooled_excess(*counts)[0]))
            terms, mask = extension.corrected_terms(padded, lengths, a, b, analysis["min_leaving"])["group_loo"]
            draws[label, "group_loo"].append(float(extension.masked_mean(terms, mask)[0]))
    rows = []
    for (label, name), values in draws.items():
        values = np.array(values)
        low, mid, high = np.nanquantile(values, [0.025, 0.5, 0.975])
        rows.append(
            {
                "transition": label,
                "statistic": name,
                "orderings": orderings,
                "min": float(np.nanmin(values)),
                "q025": float(low),
                "median": float(mid),
                "q975": float(high),
                "max": float(np.nanmax(values)),
                "share_positive": float(np.mean(values > 0)),
            }
        )
    return rows


# ----------------------------------------------------------------- network


def network_data(events: pd.DataFrame, nodes: list[str], classes: list[str]) -> dict:
    """Node presence in every segment, the group of each segment and the arm of each group.

    A segment is one group in one stage with at least one event. The second
    class listed is arm 1.
    """
    index = {name: i for i, name in enumerate(nodes)}
    cases = sorted(events["case"].unique())
    case_index = {case: i for i, case in enumerate(cases)}
    presence, owner = [], []
    for (case, _), block in events.groupby(["case", "stage"], sort=True):
        row = np.zeros(len(nodes), dtype=bool)
        for node in block["node"]:
            if node:
                row[index[node]] = True
        presence.append(row)
        owner.append(case_index[case])
    class_of = events.drop_duplicates("case").set_index("case")["class"]
    arm = np.array([classes.index(class_of[case]) for case in cases])
    return {"cases": cases, "presence": np.array(presence), "owner": np.array(owner), "arm": arm}


def network_analysis(data: dict, nodes: list[str], draws: int, rng) -> tuple[list[dict], list[dict]]:
    """The three permutation tests of the two-arm contrast, and the pair-level table."""
    segments = network.segment_vectors(data["presence"])
    owner, arm = data["owner"], data["arm"]
    groups = len(arm)
    tests, edges = [], []
    segment_arm = arm[owner]
    permuted = network.label_permutations(segment_arm, rng, draws)
    result = network.squared_difference_test(segment_arm, segments, permuted)
    tests.append({"test": "N1_segment_label", "vectors": "segment", "units": len(owner), **result})
    first, second = network.pair_index(len(nodes))
    permuted = network.label_permutations(arm, rng, draws)
    for scaling, unit in (("unit_length", True), ("unscaled", False)):
        vectors = network.group_vectors(segments, owner, groups, unit=unit)
        squared = network.squared_difference_test(arm, vectors, permuted)
        tests.append({"test": "N2_group_label", "vectors": scaling, "units": groups, **squared})
        student = network.studentised_test(arm, vectors, permuted)
        tests.append(
            {"test": "N3_group_studentised", "vectors": scaling, "units": groups, "statistic": student["statistic"], "p": student["p"]}
        )
        labels = arm.astype(float)
        signed = network._welch(labels, vectors)
        for k in range(len(first)):
            edges.append(
                {
                    "vectors": scaling,
                    "node_1": nodes[first[k]],
                    "node_2": nodes[second[k]],
                    "mean_arm_0": float(vectors[arm == 0, k].mean()),
                    "mean_arm_1": float(vectors[arm == 1, k].mean()),
                    "difference": float(vectors[arm == 1, k].mean() - vectors[arm == 0, k].mean()),
                    "t": float(signed[k]),
                    "p_adjusted": float(student["pair_p"][k]),
                }
            )
    return tests, edges


# --------------------------------------------------------------------- run


def describe(events: pd.DataFrame, families: list[str], sequences: list[np.ndarray]) -> dict[str, pd.DataFrame]:
    """Counts that describe the analysed sequences."""
    k = len(families)
    counts = np.zeros((k, k), dtype=int)
    for seq in sequences:
        np.add.at(counts, (seq[:-1], seq[1:]), 1)
    matrix = pd.DataFrame(counts, columns=families)
    matrix.insert(0, "from", families)
    by_group = events.groupby(["class", "case", "family"]).size().unstack(fill_value=0).reindex(columns=families, fill_value=0)
    by_group["events"] = by_group.sum(axis=1)
    return {"transition_counts": matrix, "family_by_group": by_group.reset_index()}


def run(settings: Settings) -> dict:
    """Run the application and write its tables."""
    cfg = settings.pipeline
    analysis, sizes = cfg["analysis"], cfg["application"]
    families = cfg["families"]["order"]
    index = {name: i for i, name in enumerate(families)}
    targets = [(f"{x}>{y}", index[x], index[y]) for x, y in analysis["focal_transitions"]]
    events = load_events(settings)
    rng = streams(cfg["seed"], 9)

    variants = {
        "main": events,
        "without_longest_group": events[events["case"] != longest_case(events)].reset_index(drop=True),
        "rapid_repeats_merged": merge_rapid(events, analysis["rapid_repeat_s"]),
    }
    estimates, tests, sizes_rows = [], [], []
    for i, (name, frame) in enumerate(variants.items()):
        _, sequences = to_sequences(frame, families)
        est, tst = focal_analysis(sequences, targets, analysis, sizes, rng[2 * i], rng[2 * i + 1])
        estimates += [{"variant": name, **row} for row in est]
        tests += [{"variant": name, **row} for row in tst]
        sizes_rows.append(
            {"variant": name, "groups": len(sequences), "events": int(sum(len(s) for s in sequences)),
             "transitions": int(sum(len(s) - 1 for s in sequences))}
        )
    _, sequences = to_sequences(events, families)
    matrix = matrix_analysis(sequences, families, analysis, sizes["permutations"], rng[6])
    ties = tie_sensitivity(events, families, targets, analysis, analysis["tie_orderings"], rng[7])
    data = network_data(events, cfg["network"]["node_order"], cfg["classes"])
    net_tests, net_edges = network_analysis(data, cfg["network"]["node_order"], sizes["permutations"], rng[8])
    node_share = pd.DataFrame(
        {
            "node": cfg["network"]["node_order"],
            "share_of_segments_arm_0": data["presence"][data["arm"][data["owner"]] == 0].mean(axis=0),
            "share_of_segments_arm_1": data["presence"][data["arm"][data["owner"]] == 1].mean(axis=0),
        }
    )
    written = {
        "focal_estimates": pd.DataFrame(estimates),
        "focal_tests": pd.DataFrame(tests),
        "matrix_exploratory": pd.DataFrame(matrix),
        "tie_sensitivity": pd.DataFrame(ties),
        "network_tests": pd.DataFrame(net_tests),
        "network_edges": pd.DataFrame(net_edges),
        "network_nodes": node_share,
        "variant_sizes": pd.DataFrame(sizes_rows),
        **describe(events, families, sequences),
        "focal_added_intervals": pd.DataFrame(added_intervals(estimates, analysis["level"])),
    }
    outputs = []
    for name, frame in written.items():
        path = stage_dir(settings, "application") / f"{name}.csv"
        tables.write_csv(frame, path)
        outputs.append(path)
    report = {
        "groups": len(sequences),
        "events": int(len(events)),
        "transitions": int(sum(len(s) - 1 for s in sequences)),
        "tied_events": int((events["tied"] == "yes").sum()),
        "events_removed_as_rapid_repeats": int(len(events) - len(variants["rapid_repeats_merged"])),
        "longest_group_events": int(events.groupby("case").size().max()),
        "segments": int(len(data["owner"])),
        "arms": {cfg["classes"][k]: int((data["arm"] == k).sum()) for k in range(len(cfg["classes"]))},
        "sizes": sizes,
        "focal_transitions": [label for label, _, _ in targets],
    }
    path = stage_dir(settings, "application") / "application_report.json"
    tables.write_json(report, path)
    outputs.append(path)
    return {"outputs": outputs, "report": report}
