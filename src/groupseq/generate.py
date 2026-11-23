"""Generators of synthetic grouped event sequences and their true transition excess."""
from __future__ import annotations

import numpy as np

PAD = -1


def common_rates(codes: int, ratio: float) -> np.ndarray:
    """Code rates that fall geometrically from the most common code to the rarest."""
    weights = ratio ** np.arange(codes)
    return weights / weights.sum()


def draw_rates(rng: np.random.Generator, base: np.ndarray, concentration, size: int) -> np.ndarray:
    """Draw the code rates of each group; no concentration means every group has the base rates."""
    if concentration is None:
        return np.broadcast_to(base, (size, len(base))).copy()
    return rng.dirichlet(base * concentration, size=size)


def mixing_weight(base: np.ndarray, b: int, excess: float) -> float:
    """Weight of the point mass on the target code that yields the nominal excess at the base rates."""
    return excess / (1.0 - base[b])


def controlled_matrices(rates: np.ndarray, a: int, b: int, weight: float) -> np.ndarray:
    """Transition matrices whose rows all equal the group's rates, except the target row."""
    size, codes = rates.shape
    matrices = np.repeat(rates[:, None, :], codes, axis=1)
    matrices[:, a, :] *= 1.0 - weight
    matrices[:, a, b] += weight
    return matrices


def controlled_stationary(rates: np.ndarray, a: int, b: int, weight: float) -> np.ndarray:
    """Stationary distribution of each controlled chain, in closed form."""
    share = weight * rates[:, a] / (1.0 + weight * rates[:, a])
    out = (1.0 - share)[:, None] * rates
    out[:, b] += share
    return out


def stationary(matrices: np.ndarray) -> np.ndarray:
    """Stationary distribution of each transition matrix, by solving the balance equations."""
    size, codes, _ = matrices.shape
    system = np.transpose(matrices, (0, 2, 1)) - np.eye(codes)
    system[:, -1, :] = 1.0
    rhs = np.zeros((size, codes, 1))
    rhs[:, -1, 0] = 1.0
    solution = np.linalg.solve(system, rhs)[:, :, 0]
    solution = np.clip(solution, 0.0, None)
    return solution / solution.sum(axis=1, keepdims=True)


def rowwise_matrices(rng: np.random.Generator, common: np.ndarray, concentration, size: int) -> np.ndarray:
    """Draw each row of each group's matrix around the matching row of the common matrix."""
    codes = common.shape[0]
    if concentration is None:
        return np.broadcast_to(common, (size, codes, codes)).copy()
    out = np.empty((size, codes, codes))
    for row in range(codes):
        out[:, row, :] = rng.dirichlet(common[row] * concentration, size=size)
    return out


def draw_lengths(rng: np.random.Generator, profile: str, replications: int, groups: int, spec: dict) -> np.ndarray:
    """Sequence lengths in events for every group of every replication."""
    if profile == "equal":
        return np.full((replications, groups), int(spec["median_length"]))
    if profile not in ("skewed", "extreme"):
        raise ValueError(f"unknown length profile: {profile}")
    raw = np.exp(np.log(spec["median_length"]) + spec["log_sd"] * rng.standard_normal((replications, groups)))
    lengths = np.maximum(np.rint(raw).astype(int), int(spec["min_length"]))
    if profile == "extreme":
        lengths[:, -1] = int(spec["extreme_length"])
    return lengths


def simulate_chains(
    rng: np.random.Generator, matrices: np.ndarray, start: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    """Simulate one chain per matrix; rows are padded beyond each chain's length."""
    size, codes, _ = matrices.shape
    longest = int(lengths.max())
    cumulative = np.cumsum(matrices, axis=2)
    uniform = rng.random((size, longest))
    out = np.full((size, longest), PAD, dtype=np.int8)
    rows = np.arange(size)
    state = np.minimum((uniform[:, 0, None] > np.cumsum(start, axis=1)).sum(axis=1), codes - 1)
    out[:, 0] = state
    for step in range(1, longest):
        state = np.minimum((uniform[:, step, None] > cumulative[rows, state]).sum(axis=1), codes - 1)
        out[:, step] = state
    out[np.arange(longest)[None, :] >= lengths[:, None]] = PAD
    return out


def unpad(padded: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    """Return the sequences held in a padded matrix."""
    return [padded[i, : lengths[i]] for i in range(len(lengths))]


def excess_truth(matrices_ab: np.ndarray, stat_a: np.ndarray, stat_b: np.ndarray) -> dict:
    """True group-averaged and pooled excess over a population of groups.

    matrices_ab holds each group's probability of moving from a to b, stat_a and
    stat_b its stationary rates of a and b. Lengths are drawn independently of
    these, so they cancel from the pooled value.
    """
    return {
        "group": float(np.mean(matrices_ab - stat_b)),
        "pooled": float(np.sum(stat_a * matrices_ab) / np.sum(stat_a) - np.mean(stat_b)),
    }


def controlled_truth(
    rng: np.random.Generator, base: np.ndarray, concentration, a: int, b: int, excess: float, size: int
) -> dict:
    """True values of a controlled cell from a large population of simulated groups."""
    weight = mixing_weight(base, b, excess)
    rates = draw_rates(rng, base, concentration, 1 if concentration is None else size)
    stat = controlled_stationary(rates, a, b, weight)
    p_ab = (1.0 - weight) * rates[:, b] + weight
    truth = excess_truth(p_ab, stat[:, a], stat[:, b])
    if excess == 0.0:
        truth["group"] = 0.0
    return truth


def calibrated_truth(
    rng: np.random.Generator, common: np.ndarray, concentration, targets: list[tuple[int, int]], size: int,
    chunk: int = 100000,
) -> list[dict]:
    """True values of every target in a calibrated cell from a large population of simulated groups."""
    if concentration is None:
        stat = stationary(common[None, :, :])
        return [excess_truth(common[None, a, b], stat[:, a], stat[:, b]) for a, b in targets]
    sums = np.zeros((len(targets), 4))
    done = 0
    while done < size:
        count = min(chunk, size - done)
        matrices = rowwise_matrices(rng, common, concentration, count)
        stat = stationary(matrices)
        for i, (a, b) in enumerate(targets):
            p_ab = matrices[:, a, b]
            sums[i] += [np.sum(p_ab - stat[:, b]), np.sum(stat[:, a] * p_ab), np.sum(stat[:, a]), np.sum(stat[:, b])]
        done += count
    return [
        {"group": float(s[0] / size), "pooled": float(s[1] / s[2] - s[3] / size)} for s in sums
    ]
