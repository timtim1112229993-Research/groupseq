"""Co-occurrence networks of grouped segments and permutation tests of a two-arm contrast.

A segment records which nodes occur in it. Its vector holds one entry per pair
of nodes, equal to 1 when both occur. A group's vector is the mean over its
segments, optionally scaled to unit length.
"""
from __future__ import annotations

import numpy as np


def pair_index(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the two nodes of every pair, in a fixed order."""
    return np.triu_indices(nodes, k=1)


def segment_vectors(presence: np.ndarray) -> np.ndarray:
    """Pair co-occurrence vector of every segment from its node presence row."""
    first, second = pair_index(presence.shape[1])
    return (presence[:, first] * presence[:, second]).astype(float)


def group_vectors(vectors: np.ndarray, group_of_segment: np.ndarray, groups: int, unit: bool = True) -> np.ndarray:
    """Mean segment vector of every group; a zero vector stays zero under scaling."""
    sums = np.zeros((groups, vectors.shape[1]))
    np.add.at(sums, group_of_segment, vectors)
    counts = np.bincount(group_of_segment, minlength=groups).astype(float)
    means = sums / np.maximum(counts, 1.0)[:, None]
    if not unit:
        return means
    norm = np.sqrt((means**2).sum(axis=1))
    return means / np.where(norm > 0, norm, 1.0)[:, None]


def label_permutations(labels: np.ndarray, rng: np.random.Generator, draws: int) -> np.ndarray:
    """Random reassignments of the arm labels; one row per permutation."""
    return rng.permuted(np.broadcast_to(labels.astype(float), (draws, len(labels))), axis=1)


def _mean_difference(labels: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Difference of arm means (arm 1 minus arm 0) for one or many label rows."""
    n1 = labels.sum(axis=-1, keepdims=True)
    n0 = labels.shape[-1] - n1
    total = vectors.sum(axis=0)
    sum1 = labels @ vectors
    return sum1 / n1 - (total - sum1) / n0


def _welch(labels: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Unequal-variance t statistic of every pair for one or many label rows; zero where undefined."""
    n1 = labels.sum(axis=-1, keepdims=True)
    n0 = labels.shape[-1] - n1
    total, total_sq = vectors.sum(axis=0), (vectors**2).sum(axis=0)
    sum1, sq1 = labels @ vectors, labels @ vectors**2
    sum0, sq0 = total - sum1, total_sq - sq1
    mean1, mean0 = sum1 / n1, sum0 / n0
    var1 = np.maximum(sq1 - n1 * mean1**2, 0.0) / np.maximum(n1 - 1, 1)
    var0 = np.maximum(sq0 - n0 * mean0**2, 0.0) / np.maximum(n0 - 1, 1)
    scale = np.sqrt(var1 / n1 + var0 / n0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(scale > 1e-12, (mean1 - mean0) / np.where(scale > 1e-12, scale, 1.0), 0.0)


def _upper_p(observed: float, null: np.ndarray) -> float:
    return float((np.count_nonzero(null >= observed - 1e-12) + 1) / (null.size + 1))


def squared_difference_test(labels: np.ndarray, vectors: np.ndarray, permuted: np.ndarray) -> dict:
    """Permutation test on the sum of squared differences of arm means."""
    observed = float((_mean_difference(labels.astype(float), vectors) ** 2).sum())
    null = (_mean_difference(permuted, vectors) ** 2).sum(axis=1)
    return {"statistic": observed, "p": _upper_p(observed, null)}


def studentised_test(labels: np.ndarray, vectors: np.ndarray, permuted: np.ndarray) -> dict:
    """Permutation test on the largest absolute t statistic, with adjusted P values for every pair."""
    observed = np.abs(_welch(labels.astype(float), vectors))
    null_max = np.abs(_welch(permuted, vectors)).max(axis=1)
    adjusted = (np.count_nonzero(null_max[:, None] >= observed[None, :] - 1e-12, axis=0) + 1) / (null_max.size + 1)
    return {"statistic": float(observed.max()), "p": float(adjusted.min()), "pair_p": adjusted, "pair_t": observed}


def simulate_presence(
    rng: np.random.Generator, groups: int, arm_of_group: np.ndarray, spec: dict, group_sd: float, shifted: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate node presence in every segment of every group.

    Returns the presence matrix (segments by nodes) and the group of each segment.
    """
    nodes = int(spec["nodes"])
    high, low = spec["baseline"]
    probability = np.linspace(high, low, nodes)
    logit = np.log(probability / (1.0 - probability))
    low_seg, high_seg = spec["segments"]
    segments = rng.integers(low_seg, high_seg + 1, size=groups)
    effect = group_sd * rng.standard_normal((groups, nodes))
    group_logit = logit[None, :] + effect
    if shifted:
        group_logit[:, : int(spec["shift_nodes"])] += spec["shift"] * arm_of_group[:, None]
    group_of_segment = np.repeat(np.arange(groups), segments)
    chance = 1.0 / (1.0 + np.exp(-group_logit[group_of_segment]))
    presence = rng.random(chance.shape) < chance
    return presence, group_of_segment


def arms(groups: int) -> np.ndarray:
    """Split groups into two arms; with an odd count the second arm takes the extra group."""
    labels = np.zeros(groups, dtype=int)
    labels[groups // 2 :] = 1
    return labels
