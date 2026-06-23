"""Element 1 & 2 of the hedge-fund method: regime *states* and per-day labeling.

A "state" quantifies the market's recent direction so a vibe ("Bitcoin feels
bullish") becomes a number. We look at the trailing ``window`` trading days,
sum the daily percent returns, and bucket the result:

    cumulative trailing return >= bull_threshold   -> BULL
    cumulative trailing return <= bear_threshold   -> BEAR
    otherwise                                       -> SIDEWAYS

The defaults (20-day window, +/-5%) match the transcript. Every day from index
``window`` onward gets a label, so the whole price history is annotated with the
regime it was in.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd

# Ordered low -> high so the integer codes line up with "bear < sideways < bull"
# direction. This ordering is relied on by the HMM mapping (hmm.py) and by the
# transition matrix row/column layout (transition.py).
class State(IntEnum):
    BEAR = 0
    SIDEWAYS = 1
    BULL = 2


STATE_NAMES = [s.name.capitalize() for s in State]  # ["Bear", "Sideways", "Bull"]
NUM_STATES = len(State)

DEFAULT_WINDOW = 20
DEFAULT_BULL_THRESHOLD = 0.05
DEFAULT_BEAR_THRESHOLD = -0.05


def daily_returns(prices: pd.Series) -> pd.Series:
    """Simple daily percent returns from a price series."""
    return prices.pct_change()


def trailing_return(prices: pd.Series, window: int = DEFAULT_WINDOW) -> pd.Series:
    """Sum of the last ``window`` daily percent returns, per the transcript.

    The first ``window`` entries are NaN because there is no full look-back yet.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    return daily_returns(prices).rolling(window).sum()


def _region_loss(
    r: np.ndarray, bull_threshold: float, bear_threshold: float
) -> np.ndarray:
    """Per-day cost of assigning each state, ``shape (T, 3)``.

    Zero when the trailing return sits inside a state's region, growing linearly
    with the violation distance — a threshold-consistent analogue of the
    squared-distance loss in a jump model.
    """
    loss = np.empty((len(r), NUM_STATES), dtype=float)
    loss[:, State.BEAR] = np.maximum(0.0, r - bear_threshold)
    loss[:, State.BULL] = np.maximum(0.0, bull_threshold - r)
    loss[:, State.SIDEWAYS] = np.maximum(0.0, r - bull_threshold) + np.maximum(
        0.0, bear_threshold - r
    )
    return loss


def _jump_label(loss: np.ndarray, jump_penalty: float) -> np.ndarray:
    """Viterbi-style DP: minimize sum(loss) + jump_penalty * (#state switches).

    This is the jump-model objective (Bemporad/Nystrup): a fit loss plus an
    explicit penalty on each regime change, yielding persistent, low-whipsaw
    labels. With ``jump_penalty == 0`` the optimal path is the per-day argmin
    loss — i.e. the greedy threshold labeling.
    """
    t_len = len(loss)
    states = range(NUM_STATES)
    cost = loss[0].astype(float).copy()
    back = np.empty((t_len, NUM_STATES), dtype=np.int8)
    for t in range(1, t_len):
        prev = cost
        new_cost = np.empty(NUM_STATES)
        for s in states:
            best_src, best = s, prev[s]  # staying in s costs no penalty
            for sp in states:
                c = prev[sp] + (0.0 if sp == s else jump_penalty)
                if c < best:
                    best, best_src = c, sp
            new_cost[s] = loss[t, s] + best
            back[t, s] = best_src
        cost = new_cost

    codes = np.empty(t_len, dtype=np.int8)
    codes[-1] = int(np.argmin(cost))
    for t in range(t_len - 1, 0, -1):
        codes[t - 1] = back[t, codes[t]]
    return codes


def label_states(
    prices: pd.Series,
    window: int = DEFAULT_WINDOW,
    bull_threshold: float = DEFAULT_BULL_THRESHOLD,
    bear_threshold: float = DEFAULT_BEAR_THRESHOLD,
    jump_penalty: float = 0.0,
) -> pd.Series:
    """Label every day with its :class:`State`.

    Returns an ``int8`` Series aligned to ``prices``. Days without a full
    ``window`` look-back are dropped (they cannot be labeled).

    ``jump_penalty`` (>= 0) adds a jump-model switch penalty: instead of
    labeling each day greedily by its trailing return, the whole regime
    *sequence* is chosen to minimize fit loss plus ``jump_penalty`` per regime
    change, absorbing brief excursions into the surrounding regime. The default
    of 0 reproduces the greedy threshold labeling exactly.
    """
    if bear_threshold >= bull_threshold:
        raise ValueError("bear_threshold must be < bull_threshold")
    if jump_penalty < 0:
        raise ValueError("jump_penalty must be >= 0")

    cum = trailing_return(prices, window)
    valid = cum.notna().to_numpy()
    r = cum.to_numpy()[valid]
    if len(r) == 0:
        return pd.Series([], index=cum.index[valid], name="state", dtype=np.int8)

    if jump_penalty <= 0.0:
        codes = np.full(len(r), State.SIDEWAYS, dtype=np.int8)
        codes[r >= bull_threshold] = State.BULL
        codes[r <= bear_threshold] = State.BEAR
    else:
        codes = _jump_label(_region_loss(r, bull_threshold, bear_threshold), jump_penalty)

    return pd.Series(codes, index=cum.index[valid], name="state")
