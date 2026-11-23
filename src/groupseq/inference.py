"""Inference on a transition between two codes in grouped event sequences.

Every procedure works from four counts per group for a target transition a to b:
  nab  transitions from a to b
  na   transitions leaving a
  nb   transitions ending in b
  n    all transitions
Two statistics are defined on them. The group-averaged excess is the mean, over
contributing groups, of nab/na - nb/n. The pooled excess is the same contrast
computed from counts summed over groups.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np

NORMAL = NormalDist()
STATISTICS = ("group", "pooled")
SE_FLOOR = 1e-10


def transition_counts(sequences: list[np.ndarray], a: int, b: int) -> tuple[np.ndarray, ...]:
    """Count the four quantities for each sequence."""
    nab = np.zeros(len(sequences))
    na = np.zeros(len(sequences))
    nb = np.zeros(len(sequences))
    n = np.zeros(len(sequences))
    for g, seq in enumerate(sequences):
        left, right = seq[:-1] == a, seq[1:] == b
        nab[g] = np.count_nonzero(left & right)
        na[g] = np.count_nonzero(left)
        nb[g] = np.count_nonzero(right)
        n[g] = max(len(seq) - 1, 0)
    return nab, na, nb, n


def padded_counts(padded: np.ndarray, lengths: np.ndarray, a: int, b: int) -> tuple[np.ndarray, ...]:
    """Count the four quantities for sequences stored as rows of a padded matrix."""
    valid = np.arange(padded.shape[1] - 1)[None, :] < (lengths[:, None] - 1)
    left = (padded[:, :-1] == a) & valid
    right = (padded[:, 1:] == b) & valid
    nab = (left & right).sum(axis=1).astype(float)
    return nab, left.sum(axis=1).astype(float), right.sum(axis=1).astype(float), (lengths - 1).astype(float)


def group_terms(nab, na, nb, n, min_leaving: int):
    """Return each group's excess and whether the group contributes; the last axis indexes groups."""
    mask = (na >= min_leaving) & (n > 0)
    safe_na = np.where(mask, na, 1.0)
    safe_n = np.where(mask, n, 1.0)
    terms = np.where(mask, nab / safe_na - nb / safe_n, 0.0)
    return terms, mask


def group_excess(nab, na, nb, n, min_leaving: int):
    """Group-averaged excess, its standard error and the number of contributing groups."""
    terms, mask = group_terms(nab, na, nb, n, min_leaving)
    m = mask.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(m > 0, terms.sum(axis=-1) / np.maximum(m, 1), np.nan)
        centred = np.where(mask, terms - np.where(m > 0, mean, 0.0)[..., None], 0.0)
        var = np.where(m > 1, (centred**2).sum(axis=-1) / np.maximum(m - 1, 1), np.nan)
        se = np.sqrt(var / np.maximum(m, 1))
    return mean, se, m


def pooled_excess(nab, na, nb, n):
    """Pooled excess and its linearised cluster standard error."""
    groups = nab.shape[-1]
    sa, sn = na.sum(axis=-1), n.sum(axis=-1)
    ok = (sa > 0) & (sn > 0)
    safe_sa, safe_sn = np.where(ok, sa, 1.0), np.where(ok, sn, 1.0)
    p = nab.sum(axis=-1) / safe_sa
    q = nb.sum(axis=-1) / safe_sn
    value = np.where(ok, p - q, np.nan)
    influence = (nab - p[..., None] * na) / safe_sa[..., None] - (nb - q[..., None] * n) / safe_sn[..., None]
    with np.errstate(invalid="ignore"):
        se = np.where(ok, np.sqrt(groups / max(groups - 1, 1) * (influence**2).sum(axis=-1)), np.nan)
    return value, se


def adjusted_residual(nab, na, nb, n) -> float:
    """Adjusted residual of the pooled two-by-two table, treating transitions as independent."""
    obs, row, col, total = nab.sum(), na.sum(), nb.sum(), n.sum()
    if total <= 0:
        return float("nan")
    expected = row * col / total
    variance = expected * (1 - row / total) * (1 - col / total)
    if variance <= 0:
        return float("nan")
    return float((obs - expected) / np.sqrt(variance))


def percentile_interval(draws: np.ndarray, level: float) -> tuple[float, float]:
    """Percentile interval over the defined draws."""
    draws = draws[~np.isnan(draws)]
    if draws.size == 0:
        return float("nan"), float("nan")
    tail = (1 - level) / 2
    lo, hi = np.quantile(draws, [tail, 1 - tail])
    return float(lo), float(hi)


def bca_interval(estimate: float, draws: np.ndarray, jackknife: np.ndarray, level: float) -> tuple[float, float]:
    """Bias-corrected and accelerated interval."""
    draws = draws[~np.isnan(draws)]
    jackknife = jackknife[~np.isnan(jackknife)]
    if draws.size == 0 or np.isnan(estimate):
        return float("nan"), float("nan")
    count = draws.size
    below = (np.count_nonzero(draws < estimate) + 0.5 * np.count_nonzero(draws == estimate)) / count
    below = min(max(below, 1 / (count + 1)), count / (count + 1))
    z0 = NORMAL.inv_cdf(below)
    accel = 0.0
    if jackknife.size > 1:
        dev = jackknife.mean() - jackknife
        scale = (dev**2).sum()
        if scale > 0:
            accel = float((dev**3).sum() / (6 * scale**1.5))
    levels = []
    for z in (NORMAL.inv_cdf((1 - level) / 2), NORMAL.inv_cdf(1 - (1 - level) / 2)):
        denom = 1 - accel * (z0 + z)
        levels.append(NORMAL.cdf(z0 + (z0 + z) / denom) if denom > 0 else NORMAL.cdf(z))
    lo, hi = np.quantile(draws, sorted(levels))
    return float(lo), float(hi)


def studentised_interval(estimate: float, se: float, draws: np.ndarray, draw_se: np.ndarray, level: float):
    """Bootstrap-t interval; undefined when the standard error is zero or missing.

    A resample whose groups all carry the same value has a standard error of
    zero up to rounding; SE_FLOOR keeps such resamples out of the t distribution.
    """
    if np.isnan(estimate) or np.isnan(se) or se <= SE_FLOOR:
        return float("nan"), float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        usable = draw_se > SE_FLOOR
        t = np.where(usable, (draws - estimate) / np.where(usable, draw_se, 1.0), np.nan)
    t = t[~np.isnan(t)]
    if t.size == 0:
        return float("nan"), float("nan")
    tail = (1 - level) / 2
    q_lo, q_hi = np.quantile(t, [tail, 1 - tail])
    return float(estimate - q_hi * se), float(estimate - q_lo * se)


def event_bootstrap(nab, na, nb, n, rng: np.random.Generator, draws: int, level: float):
    """Percentile interval for the pooled excess when single transitions are resampled."""
    obs, row, col, total = nab.sum(), na.sum(), nb.sum(), n.sum()
    if total <= 0 or row <= 0:
        return float("nan"), float("nan")
    cells = np.array([obs, row - obs, col - obs, total - row - col + obs])
    sample = rng.multinomial(int(total), cells / total, size=draws).astype(float)
    leaving = sample[:, 0] + sample[:, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.where(leaving > 0, sample[:, 0] / leaving - (sample[:, 0] + sample[:, 2]) / total, np.nan)
    return percentile_interval(value, level)


def _jackknife_group(nab, na, nb, n, min_leaving):
    terms, mask = group_terms(nab, na, nb, n, min_leaving)
    m, total = mask.sum(), terms.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        left_out = np.where(mask, (total - terms) / (m - 1) if m > 1 else np.nan, total / m if m > 0 else np.nan)
    return left_out


def _jackknife_pooled(nab, na, nb, n):
    sab, sa, sb, sn = nab.sum() - nab, na.sum() - na, nb.sum() - nb, n.sum() - n
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((sa > 0) & (sn > 0), sab / sa - sb / sn, np.nan)


def group_resamples(groups: int, rng: np.random.Generator, draws: int) -> np.ndarray:
    """Draw the group indices of every bootstrap resample; one row per resample."""
    return rng.integers(0, groups, size=(draws, groups))


def group_bootstrap(nab, na, nb, n, pick: np.ndarray, level: float, min_leaving: int) -> dict:
    """Build three intervals for each statistic from one set of group resamples.

    Returns a dictionary keyed by statistic, each holding the estimate, its
    standard error, the number of contributing groups and the percentile, BCa
    and studentised intervals.
    """
    groups = len(nab)
    b_ab, b_a, b_b, b_n = nab[pick], na[pick], nb[pick], n[pick]
    out = {}
    est, se, m = group_excess(nab, na, nb, n, min_leaving)
    d_est, d_se, _ = group_excess(b_ab, b_a, b_b, b_n, min_leaving)
    est, se = float(est), float(se)
    out["group"] = {
        "estimate": est,
        "se": se,
        "contributing": int(m),
        "percentile": percentile_interval(d_est, level),
        "bca": bca_interval(est, d_est, _jackknife_group(nab, na, nb, n, min_leaving), level),
        "studentised": studentised_interval(est, se, d_est, d_se, level),
    }
    est, se = pooled_excess(nab, na, nb, n)
    d_est, d_se = pooled_excess(b_ab, b_a, b_b, b_n)
    est, se = float(est), float(se)
    out["pooled"] = {
        "estimate": est,
        "se": se,
        "contributing": groups,
        "percentile": percentile_interval(d_est, level),
        "bca": bca_interval(est, d_est, _jackknife_pooled(nab, na, nb, n), level),
        "studentised": studentised_interval(est, se, d_est, d_se, level),
    }
    return out


def _two_sided_p(observed: float, null: np.ndarray) -> float:
    """Two-sided permutation P value about the mean of the permutation distribution."""
    null = null[~np.isnan(null)]
    if null.size == 0 or np.isnan(observed):
        return float("nan")
    centre = null.mean()
    extreme = np.count_nonzero(np.abs(null - centre) >= abs(observed - centre) - 1e-12)
    return float((extreme + 1) / (null.size + 1))


def order_permutation(
    sequences: list[np.ndarray], targets: list[tuple[int, int]], rng: np.random.Generator, draws: int, min_leaving: int
) -> list[dict]:
    """Permute the order of events inside every sequence and test each target transition.

    The permutation keeps each group's code frequencies, so it tests the
    hypothesis that order carries no information in any group. One set of
    permutations serves every target.
    """
    groups = len(sequences)
    lengths = np.array([len(s) for s in sequences])
    if groups and np.all(lengths == lengths[0]):
        stacked = np.broadcast_to(np.stack(sequences).astype(np.int8), (draws, groups, int(lengths[0])))
        shuffled = [rng.permuted(stacked, axis=2)]
        blocks = [slice(0, groups)]
    else:
        shuffled = [
            rng.permuted(np.broadcast_to(seq.astype(np.int8), (draws, 1, len(seq))), axis=2) for seq in sequences
        ]
        blocks = [slice(g, g + 1) for g in range(groups)]
    n = (lengths - 1).astype(float)
    results = []
    for a, b in targets:
        p_ab = np.empty((draws, groups))
        p_a = np.empty((draws, groups))
        p_b = np.empty((draws, groups))
        for block, x in zip(blocks, shuffled):
            left = x[:, :, :-1] == a
            right = x[:, :, 1:] == b
            p_ab[:, block] = (left & right).sum(axis=2)
            p_a[:, block] = left.sum(axis=2)
            p_b[:, block] = right.sum(axis=2)
        nab, na, nb, _ = transition_counts(sequences, a, b)
        obs_group, _, _ = group_excess(nab, na, nb, n, min_leaving)
        obs_pooled, _ = pooled_excess(nab, na, nb, n)
        null_group, _, _ = group_excess(p_ab, p_a, p_b, n, min_leaving)
        null_pooled, _ = pooled_excess(p_ab, p_a, p_b, np.broadcast_to(n, (draws, groups)))
        results.append(
            {"group": _two_sided_p(float(obs_group), null_group), "pooled": _two_sided_p(float(obs_pooled), null_pooled)}
        )
    return results
