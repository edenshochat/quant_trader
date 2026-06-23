"""Elements 4-7: the Markov transition matrix and what we squeeze out of it.

Given a sequence of daily state labels we:

* count every observed ``state_t -> state_{t+1}`` transition (the "tally"),
* row-normalize the counts into a 3x3 probability matrix ``P`` where
  ``P[i, j] = P(tomorrow == j | today == i)`` and every row sums to 1
  (element 4),
* read the diagonal as the per-state *stickiness* / persistence (element 5),
* raise ``P`` to the n-th power for an n-day-ahead forecast (element 6 —
  "squaring the matrix"),
* solve for the stationary distribution ``pi`` (element 7 — the long-run mix
  that ``P^n`` converges to, where there is "no meaningful signal").
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .states import NUM_STATES, State


def count_transitions(states: pd.Series) -> np.ndarray:
    """Tally consecutive ``(today, tomorrow)`` state pairs into a 3x3 matrix.

    ``counts[i, j]`` is the number of times state ``i`` was followed by state
    ``j`` on the very next day. Pairs spanning a gap in the index (e.g. a
    backtest splice) are still counted positionally, which matches the
    transcript's "every time that transition took place".
    """
    codes = states.to_numpy(dtype=np.int64)
    counts = np.zeros((NUM_STATES, NUM_STATES), dtype=np.int64)
    if codes.size < 2:
        return counts
    np.add.at(counts, (codes[:-1], codes[1:]), 1)
    return counts


def transition_matrix(states: pd.Series) -> np.ndarray:
    """Row-normalized transition probability matrix ``P`` (rows sum to 1).

    A state that was never observed as a "today" (an all-zero row) falls back
    to a uniform distribution so ``P`` stays a valid stochastic matrix instead
    of producing NaNs.
    """
    counts = count_transitions(states).astype(np.float64)
    row_sums = counts.sum(axis=1, keepdims=True)

    empty = row_sums.ravel() == 0
    if empty.any():
        missing = [State(i).name for i in np.flatnonzero(empty)]
        warnings.warn(
            f"No outgoing transitions observed for state(s) {missing}; "
            "falling back to a uniform row.",
            stacklevel=2,
        )
        counts[empty] = 1.0
        row_sums[empty] = NUM_STATES

    return counts / row_sums


def stickiness(P: np.ndarray) -> np.ndarray:
    """Persistence score for each state: the diagonal of ``P``.

    ``stickiness(P)[BULL]`` is "if we're bullish today, the probability we're
    still bullish tomorrow".
    """
    return np.diag(P).copy()


def n_step(P: np.ndarray, n: int) -> np.ndarray:
    """``n``-day-ahead transition matrix, i.e. ``P`` raised to the n-th power.

    ``n=1`` is tomorrow, ``n=2`` is the "squared" two-day forecast, ``n=28`` is
    the four-week forecast that converges toward the stationary mix.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    return np.linalg.matrix_power(P, n)


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Long-run state distribution ``pi`` satisfying ``pi @ P == pi``.

    Computed as the left eigenvector of ``P`` for eigenvalue 1 (the right
    eigenvector of ``P.T``), normalized to sum to 1. This is the distribution
    ``P^n`` collapses to as ``n`` grows — element 7's "single sliver" with no
    actionable signal.
    """
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    vec = np.real(eigvecs[:, idx])
    total = vec.sum()
    if total == 0:
        # Degenerate; fall back to uniform.
        return np.full(NUM_STATES, 1.0 / NUM_STATES)
    pi = vec / total
    # Guard against tiny negative components from numerical noise.
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()
