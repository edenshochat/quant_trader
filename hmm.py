"""Element 10: a Hidden Markov Model cross-check of the hand-drawn states.

The +/-5% thresholds in states.py are a *subjective* human choice. The hidden
Markov model removes the labels entirely and lets an unsupervised model discover
the regimes from the return series itself (the "babysitter watching the kids
before naming their personalities"). We then:

1. fit a 3-state Gaussian HMM on daily returns,
2. relabel its hidden states as BEAR/SIDEWAYS/BULL by sorting on mean return,
3. compare the HMM labels to the rule-based labels — the *agreement rate* and a
   per-day *confirmation* mask. Where both agree, that's the transcript's
   "green light".

``hmmlearn`` is an optional dependency (install with ``pip install '.[quant]'``);
this module raises a clear error if it is missing rather than at import time, so
the rest of the engine works without it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .states import (
    DEFAULT_BEAR_THRESHOLD,
    DEFAULT_BULL_THRESHOLD,
    DEFAULT_WINDOW,
    NUM_STATES,
    label_states,
)


@dataclass
class HMMResult:
    """HMM regime labels and how well they confirm the rule-based labels."""

    hmm_states: pd.Series          # int-coded BEAR/SIDEWAYS/BULL per day
    rule_states: pd.Series         # the threshold-based labels, aligned
    agreement_rate: float          # fraction of aligned days where both agree
    confirmation: pd.Series        # bool mask: True where both agree
    state_means: np.ndarray        # mean daily return of each remapped state


def _require_hmmlearn():
    try:
        from hmmlearn.hmm import GaussianHMM  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            "hmmlearn is required for the HMM cross-check. "
            "Install it with: pip install 'ray-backend[quant]' "
            "(or: pip install hmmlearn)."
        ) from exc
    return GaussianHMM


def fit_hmm_states(
    prices: pd.Series,
    n_iter: int = 100,
    random_state: int = 42,
    n_restarts: int = 10,
) -> tuple[pd.Series, np.ndarray]:
    """Fit a 3-state Gaussian HMM on daily returns and relabel by mean return.

    EM is sensitive to initialization and can collapse into a poor local optimum
    (all means near zero). We therefore fit ``n_restarts`` models from different
    seeds and keep the highest log-likelihood fit, which makes the result both
    robust and deterministic.

    Returns the per-day HMM state Series (BEAR=0/SIDEWAYS=1/BULL=2) and the
    sorted per-state mean daily return.
    """
    GaussianHMM = _require_hmmlearn()

    rets = prices.pct_change().dropna()
    X = rets.to_numpy().reshape(-1, 1)

    best_model = None
    best_score = -np.inf
    for offset in range(max(1, n_restarts)):
        model = GaussianHMM(
            n_components=NUM_STATES,
            covariance_type="diag",
            n_iter=n_iter,
            random_state=random_state + offset,
        )
        try:
            model.fit(X)
            score = float(model.score(X))
        except (ValueError, FloatingPointError):
            continue
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        raise RuntimeError("Gaussian HMM failed to converge on the return series")

    raw = best_model.predict(X)

    # Remap raw hidden-state ids to BEAR<SIDEWAYS<BULL by ascending mean return.
    means = best_model.means_.ravel()
    order = np.argsort(means)            # raw ids sorted low->high mean
    remap = np.empty(NUM_STATES, dtype=np.int64)
    remap[order] = np.arange(NUM_STATES)  # lowest-mean raw id -> 0 (BEAR), etc.

    mapped = remap[raw]
    return (
        pd.Series(mapped, index=rets.index, name="hmm_state"),
        means[order],
    )


def cross_check(
    prices: pd.Series,
    window: int = DEFAULT_WINDOW,
    bull_threshold: float = DEFAULT_BULL_THRESHOLD,
    bear_threshold: float = DEFAULT_BEAR_THRESHOLD,
    n_iter: int = 100,
    random_state: int = 42,
) -> HMMResult:
    """Compare HMM-discovered regimes against the rule-based labels."""
    hmm_states, state_means = fit_hmm_states(prices, n_iter, random_state)
    rule_states = label_states(prices, window, bull_threshold, bear_threshold)

    # Align on the common dates (HMM drops day 1; rules drop the look-back).
    common = hmm_states.index.intersection(rule_states.index)
    hmm_aligned = hmm_states.loc[common]
    rule_aligned = rule_states.loc[common]

    confirmation = hmm_aligned == rule_aligned
    agreement_rate = float(confirmation.mean()) if len(common) else 0.0

    return HMMResult(
        hmm_states=hmm_aligned,
        rule_states=rule_aligned,
        agreement_rate=agreement_rate,
        confirmation=confirmation.rename("confirmation"),
        state_means=state_means,
    )
